import warnings
import torch
from torch import nn


def do_inference(data_loader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 gpu: int = None,
                 return_attention: bool = False,
                 args = None):
    """Apply model to data to receive model output

    Args:
        data_loader (torch.utils.data.DataLoader): A pytorch DataLoader
            that holds the inference data
        model (torch.nn.Module): A pytorch model
        args (Dict): args

    Returns:
        [type]: [description]
    """
    
    # first loop: create neighbourhood embedding memory
    with torch.no_grad():
        if not args.wsi_batch:
            model.eval()
        if not model.block_memory:
            model.initialize_memory(**data_loader.dataset.wsi_dataset.meta_data['memory'], gpu=gpu if not args.memory_to_cpu else None)
            model.fill_memory(data_loader=data_loader, gpu=gpu)
        else:
            warnings.warn("Attention inference but memory is blocked")
            
        outputs, labels, attentions, n_masks, _ = memory_inference(data_loader=data_loader,
                                                                   model=model,
                                                                   gpu=gpu)
        
    if return_attention:
        return outputs, labels, attentions, n_masks
    else:
        return outputs, labels
  
  
def memory_inference(data_loader,
                     model,
                     gpu, 
                     return_cls_dist=False):
    outputs = []
    labels = []
    attentions = []
    neighbour_masks = []
    neighbour_dists = []
    
    m = nn.Softmax(dim=1).cuda(gpu)
    model.cuda(gpu)

    # second loop: attend freezed neighbourhood memory   
    with torch.no_grad():
        for batch in data_loader:
            
            images = batch['img']
            targets = batch['mask']
            neighbours_idx = batch['patch_neighbour_idxs']
            if return_cls_dist is True:
                neighbour_dists.append(batch['neighbour_dist'])
            
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
        
            result = model(images, 
                           neighbours_idx) 
            
            # access results
            logits= result['logits']
            attention= result['attention']
            k_neighbour_mask= result['neighbour_masks']
 
            probs = m(logits)

            outputs.append(torch.argmax(probs,dim=1).cpu())
            labels.append(targets.cpu())
            attentions.append(attention.cpu() if attention is not None else None)
            neighbour_masks.append(k_neighbour_mask.cpu() if k_neighbour_mask is not None else None)
            
    return outputs, labels, attentions, neighbour_masks, neighbour_dists
    