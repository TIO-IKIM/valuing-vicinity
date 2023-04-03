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
    
         outputs = []
    labels = []
    attentions = []
    neighbour_masks = []
    
    m = nn.Softmax(dim=1).cuda(gpu)
    # second loop: attend freezed neighbourhood memory   
    with torch.no_grad():
        model.eval()  
        
        for  images, targets, neighbour_imgs, neighbours_masks in data_loader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
                neighbour_imgs = neighbour_imgs.cuda(gpu, non_blocking=True)
                k_neighbour_masks = neighbours_masks.cuda(gpu, non_blocking=True)    
        
            logits, attention = model(images, 
                                      neighbour_masks=k_neighbour_masks,
                                      neighbour_imgs=neighbour_imgs,
                                      return_attention=True)  
            probs = m(logits)

            outputs.append(torch.argmax(probs,dim=1).cpu())
            labels.append(targets.cpu())
            attentions.append(attention.cpu())
            neighbour_masks.append(k_neighbour_masks.cpu())
    if return_attention:
        return outputs, labels, attentions, neighbour_masks
    else:
        return outputs, labels