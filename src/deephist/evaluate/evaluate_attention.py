from genericpath import exists
import logging
from pathlib import Path
import warnings

import numpy as np
import torch

from src.deephist.segmentation.attention_segmentation.attention_inference import memory_inference
from src.exp_management.helper import getcov
from src.exp_management.tracking import Visualizer


def evaluate_details(patch_coordinates,
                     include_radius,
                     exp,
                     model,
                     wsis):
    
        viz = Visualizer(save_to_folder=True)

        model.eval()
        for wsi_name, patch_coordinates in patch_coordinates.items():
            # select WSI from wsis
            try:
                selected_wsi = [wsi for wsi in wsis if wsi.name == wsi_name][0]
            except Exception as e:
                logging.getLogger('exp').error(f"Warning: Cannot find WSI {wsi_name}. Continuing")
                continue
            
            log_path = Path(exp.log_path) / 'inference' / wsi_name
            log_path.mkdir(exist_ok=True, parents=True)
            
            # build memory on that WSI
            with selected_wsi.inference_mode(): # initializes memory
                logging.getLogger('exp').info("Building memory")
                wsi_loader = exp.data_provider.get_wsi_loader(wsi=selected_wsi)
                
                # fill memory
                model.initialize_memory(**selected_wsi.meta_data['memory'], gpu=exp.args.gpu if not exp.args.memory_to_cpu else None)
                model.fill_memory(data_loader=wsi_loader, gpu=exp.args.gpu)
                   
                outputs, labels, attentions, n_masks, _ = memory_inference(data_loader=wsi_loader,
                                                                           model=model,
                                                                           gpu=exp.args.gpu)  
                # merge "batches"
                outputs, labels, attentions, n_masks = torch.cat(outputs), torch.cat(labels), torch.cat(attentions), torch.cat(n_masks)
                
                # append results to patch object
                avg_estimates = 0
                patches = selected_wsi.get_patches()
                assert outputs.shape[0] == len(patches)
                
                for i, patch in enumerate(patches):
                    patch.prediction = exp.mask_to_img(mask=outputs[i],
                                                    label_handler=exp.data_provider.label_handler,
                                                    org_size=True) 
                    patch.mask = exp.mask_to_img(mask=labels[i],
                                                label_handler=exp.data_provider.label_handler,
                                                org_size=True)
                    
                    att = attentions[i]
                    n_patches, n_heads, _, = att.shape
                    k = exp.args.k_neighbours  
                    att_kernel = att.view((n_patches, n_heads, (k*2+1),(k*2+1)))
                    # hack because the memory has x and y changed..
                    att_kernel = torch.moveaxis(att_kernel, -2, -1)
                    att_kernel = torch.mean(att_kernel, dim=0).squeeze(dim=0)
                    
                    #     # test attention
                    #     att_kernel[:,:] = 0 
                    #     att_kernel[2:7, 12:17] = 0.1
                    #     att_kernel[3:6, 13:16] = 0.2
                    #     att_kernel[4:5, 14:15] = 0.3

                    #     att_kernel = torch.moveaxis(att_kernel, -1, 0)
                    
                    estimates = find_estimates(att_kernel)
                    avg_estimates += np.array(estimates)
                    
                    patch.estimates = estimates
                    patch.attention = att_kernel
                    
                    patch.neighbour_mask = n_masks[i]

                # store avg estimates per WSI
                selected_wsi.avg_estimates = avg_estimates / len(patches)
                     
                # select patches from inference definition
                for x, y in patch_coordinates:
                    # if None -> total WSI 
                    if x is None and y is None:
                        context_patches = selected_wsi._patch_map
                    else:
                        patch = selected_wsi.get_patch_from_position(x,y)
                        if patch is None:
                            logging.getLogger('exp').info(f"Patch {x}, {y} does not exist.")
                            continue
                        
                        context_patches, _  = patch.get_neighbours(k=include_radius)
                        context_patches_list = [p for p in list(context_patches.flatten()) if p is not None]
                        
                        if len(context_patches_list) == 0:
                            continue
                    
                    # pred + est
                    viz.plot_wsi_section(section=context_patches,
                                         mode='pred',
                                         att_estimates=True,
                                         log_path=log_path)
                    # gt
                    viz.plot_wsi_section(section=context_patches,
                                         mode='gt',
                                         log_path=log_path)
                    # tissue
                    viz.plot_wsi_section(section=context_patches,
                                         mode='org',
                                         log_path=log_path)
                    # pred
                    viz.plot_wsi_section(section=context_patches,
                                        mode='pred',
                                        log_path=log_path)

                    # att + gt
                    viz.plot_wsi_section(section=context_patches,
                                        mode='gt',
                                        attention=True,
                                        log_path=log_path)
                    # att + pred
                    viz.plot_wsi_section(section=context_patches,
                                        mode='pred',
                                        attention=True,
                                        log_path=log_path)
                    
                    

def find_estimates(agg_att):
    import numpy as np
    from scipy.optimize import leastsq

    def _2d_gaussian_residual(p, prob, coords):
        from scipy.stats import multivariate_normal

        mu = [p[0], p[1]]
        cov = getcov(radius=p[2], scale=p[3], theta=p[4])
        try:
            rv = multivariate_normal(mean=mu, cov=cov)
        except Exception as e:
            warnings.warn("Fall-back because of non-pos. semidefinite matrix")
            # in case, we do not get a pos. semidefinite matrix, fall-back 
            rv = multivariate_normal(mean=np.array([-100,-100]),
                                     cov=np.array([[25, 0],
                                                   [0, 25]]))
        
        residual = prob - rv.pdf(coords)
        return residual
    
    l = agg_att.shape[0]
    k = (l-1)//2
    # mean (x,y), radius, scale, theta
    params = [k, k, k/2, 0, 0]
        
    x, y = np.meshgrid(range(l), range(l))
    # filter center coord + attention because it is zero and might confuse parameter estimation.
    observed_att, coords = zip(*[(agg, coord) for agg, coord 
                                 in zip(list(agg_att.flatten().numpy()), 
                                     list(zip(y.flatten(), x.flatten()))
                                     ) if coord != (k,k)])
    # add 0.5 position bias for mismatch of patch count vs patch pos.
    coords = [(coord[0]+0.5, coord[1]+0.5) for coord in coords]
    res = leastsq(_2d_gaussian_residual, params, (observed_att, coords), maxfev=1000, full_output=1)
    estimates = tuple(res[0])
    print(res[2]['nfev'])
    print(estimates)
    return estimates
                        