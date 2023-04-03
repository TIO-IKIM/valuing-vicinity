import torch
from torch import nn

def do_inference(data_loader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 gpu: int = None,
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

    outputs = []
    labels = []
    # switch to evaluate mode
    model.eval()
    m = nn.Softmax(dim=1).cuda(gpu)

    with torch.no_grad():
        for batch in data_loader:
            
            images = batch['img']
            targets = batch['mask']
            
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            # compute output
            result = model(images)
            
            logits = result['logits']
            
            probs = m(logits)
            
            outputs.append(torch.argmax(probs,dim=1).cpu())
            labels.append(targets.cpu())

    return outputs, labels