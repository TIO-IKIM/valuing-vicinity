from collections import defaultdict
import logging
from pathlib import Path
import pandas as pd
import torch

from src.exp_management.run_experiment import reload_model
from src.exp_management.experiment.Experiment import Experiment
from src.exp_management.tracking import Visualizer
from src.deephist.segmentation.attention_segmentation.attention_inference import memory_inference
from src.deephist.segmentation.attention_segmentation.AttentionPatchesDistAnalysisDataset import AttentionPatchesDistAnalysisDataset

def run_attention_analysis(exp: Experiment):
    """Anaylse attention 

    Args:
        exp (Experiment): _description_
    """

    model = exp.get_model()
    if exp.args.nfold is not None:
        att_sum_per_dist = defaultdict(int)
        neighbour_count_per_dist = defaultdict(int)
        central_patch_counter = defaultdict(int)
    
        for fold in range(exp.args.nfold):
            logging.getLogger('exp').info(f"Inference for fold {fold}")
            
            reload_from = Path(exp.args.reload_model_folder) / f"fold_{fold}"
            exp.log_path = reload_from #str(Path(exp.args.logdir) / f"fold_{fold}")
            
            reload_model(model=model,
                         model_path=reload_from,
                         gpu=exp.args.gpu)
            
            # if explicit test data is provided:       
            if exp.args.test_data is not None:
                wsis = exp.data_provider.test_wsi_dataset.wsis
            else:
                # else, take test set per fold from training time
                wsis = exp.data_provider.cv_set.holdout_sets[fold].test_wsi_dataset.wsis
            
            a, n, c = attention_analysis(exp=exp, 
                                         model=model, 
                                         wsis=wsis)
            for cls in a.keys():
                att_sum_per_dist[cls] += a[cls]
                neighbour_count_per_dist[cls] += n[cls]
                central_patch_counter[cls] += c[cls]
                        
        # aggregate attentions
        #center_att_per_dist = defaultdict(int)  
        #center_to_neighbour_cls_att = defaultdict(int)
        attention_list = []
        
        for c_n_cls in att_sum_per_dist.keys():
            c_name = exp.data_provider.label_handler.decode(c_n_cls[0])
            n_name= exp.data_provider.label_handler.decode(c_n_cls[1])
            
            # normalized with mean of patches per dist
            att_distance_mean =  att_sum_per_dist[c_n_cls] / neighbour_count_per_dist[c_n_cls] * (neighbour_count_per_dist[c_n_cls] / central_patch_counter[c_n_cls])
            att_distance_mean = torch.nan_to_num(att_distance_mean)
            
            for dist, att in enumerate(att_distance_mean):
                attention_list.append({
                'center_cls': c_name,
                'neighbour_cls': n_name,
                'attention': att.item(),
                'distance': dist+1
                })
            # sum up attention over distances
            #center_to_neighbour_cls_att[(c_name,n_name)] = torch.sum(att_distance_mean).item()
            # sum up attention for center cls
            #center_att_per_dist[c_name] += att_distance_mean.numpy()
            
            print(f"{c_name} center to {n_name} neighbour distance attention:")
            print(att_distance_mean)
        df = pd.DataFrame(attention_list, columns=attention_list[0].keys())
        df.to_csv(f"attention_analysis_{exp.model_name}.csv", index=False)       
            
            
def attention_analysis(exp,
                       model,
                       wsis):
          
    k = exp.args.k_neighbours  
    l = k*2+1

    # distribute over distances
    dist_matrix = get_dist_matrix(k=exp.args.k_neighbours, dist='manhattan')
    
    att_sum_per_dist = defaultdict(int)
    neighbour_count_per_dist = defaultdict(int)
    central_patch_counter = defaultdict(int)
    
    model.eval()
    for wsi in wsis:
        with wsi.inference_mode(): # initializes memory
            logging.getLogger('exp').info("Building memory")
            # loader wsi with special dataset including neighbourhood patch distribution 
            wsi_loader = exp.data_provider.get_wsi_loader(wsi=wsi, 
                                                          dataset_type=AttentionPatchesDistAnalysisDataset)
            
            # fill memory
            model.initialize_memory(**wsi.meta_data['memory'], gpu=exp.args.gpu if not exp.args.memory_to_cpu else None)
            model.fill_memory(data_loader=wsi_loader, gpu=exp.args.gpu)
            
            # get neighbourhood class distribution (_dists)
            outputs, labels, attentions, n_masks, n_dists = memory_inference(data_loader=wsi_loader,
                                                                             model=model,
                                                                             gpu=exp.args.gpu,
                                                                             return_cls_dist=True)
    
            # merge "batches"
            outputs, labels, attention_per_head, n_masks, n_dists = \
                torch.cat(outputs), torch.cat(labels), torch.cat(attentions), torch.cat(n_masks), torch.cat(n_dists)
            # select center patch class distributions 
            center_dists = n_dists[:,k,k,:]
               
             # attention dim: patches, heads, 1, token
            n_patches, n_heads, _, _ = attention_per_head.shape
            
            # mean over heads per patch
            attentions = torch.mean(attention_per_head.view((n_patches, n_heads, (k*2+1),(k*2+1))), dim=1)
                
            # determine attention per center patch cls
            for center_cls in range(n_dists.shape[-1]):
                
                for neighbour_cls in range(n_dists.shape[-1]):
                    # % of cls in center patch 
                    center_patch_mask = center_dists[:,center_cls].unsqueeze(-1).unsqueeze(-1).expand(n_patches,l,l)
                    # % of cls in neighborur pathces
                                    
                    neighbour_patch_mask = n_dists[...,neighbour_cls]

                    # weight / select attentions by center patch cls
                    weighted_attentions = attentions * center_patch_mask * neighbour_patch_mask
                    # weight / select neighbourmasks for center patch cls
                    weighted_masks = n_masks * center_patch_mask * neighbour_patch_mask
                    
                    # dot-product to apply distance masks
                    dist_cube = dist_matrix.unsqueeze(0).expand((n_patches,l,l,k)) 
                    attention_cube = weighted_attentions.unsqueeze(-1).expand((n_patches,l,l,k))
                    neighbour_masks_cube = weighted_masks.unsqueeze(-1).expand((n_patches,l,l,k))
                    
                    attention_per_dist = attention_cube * dist_cube # select attention per distance dimension
                    
                    # count neighbour patches per distance / filter the distance cube by existing neighbours
                    neighbour_per_dist =  dist_cube * neighbour_masks_cube
                    
                    # sum over all patches for each dist
                    count_neighbours_per_dist = torch.sum(neighbour_per_dist, dim=(0,1,2))
                    sum_att_per_dist = torch.sum(attention_per_dist, dim=(0,1,2))
                    
                    # cumulate for all WSIs
                    att_sum_per_dist[(center_cls, neighbour_cls)] += sum_att_per_dist
                    neighbour_count_per_dist[(center_cls, neighbour_cls)] += count_neighbours_per_dist
                    central_patch_counter[(center_cls, neighbour_cls)] += torch.sum(center_dists[:,center_cls])
                     
    return att_sum_per_dist, neighbour_count_per_dist, central_patch_counter  
        
def get_dist_matrix(k=8, dist='manhattan'):
    assert dist in ['manhattan'], 'dist must be one of [manhattan]'
    
    if dist == 'manhattan':
        # create distance mask depeneding on neighbourhood size
        l = k*2+1
        dist_mask = torch.zeros(size=(l, l, k))
        # fill distance mask: for each k, value is 1 when x or/and y coord equals k
        for dist in range(1, k+1):
            # e.g. fill like this for dist=1 and k=2
            # 0 0 0 0 0
            # 0 1 1 1 0
            # 0 1 0 1 0
            # 0 1 1 1 0
            # 0 0 0 0 0
            for x in range(k-dist,l-(k-dist)):
                y1 = k-dist
                y2 = (k+dist)

                dist_mask[x,y1,dist-1] = 1
                dist_mask[x,y2,dist-1] = 1

            for y in range(k-dist,l-(k-dist)):
                x1 = (k-dist)
                x2 = (k+dist)
                
                dist_mask[x1,y,dist-1] = 1
                dist_mask[x2,y,dist-1] = 1
    
    return dist_mask