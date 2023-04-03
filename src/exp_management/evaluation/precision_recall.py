import torch
from torch.functional import Tensor 
from torch.nn.functional import one_hot

def precision(true_positives: Tensor,
              pred_positives: Tensor,
              n_classes: int,
              exclude_cls=None):
    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    precision_per_class = (true_positives / pred_positives)[selected_classes]
    mean_precision = precision_per_class[~torch.isnan(precision_per_class)].mean()
    return mean_precision.item(), {cls: dice.item() for dice, cls in zip(precision_per_class, selected_classes)}

def recall(true_positives: Tensor,
           positives: Tensor,
           n_classes: int,
           exclude_cls=None):
    
    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    recall_per_class = (true_positives / positives)[selected_classes]
    mean_recall = recall_per_class[~torch.isnan(recall_per_class)].mean()
    return mean_recall.item(), {cls: dice.item() for dice, cls in zip(recall_per_class, selected_classes)}
    
def true_positives(y_true,
                   y_pred,
                   n_classes,
                   exclude_cls=None):
    
    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    y_true_oh = one_hot(y_true, num_classes=n_classes)[..., selected_classes]
    y_pred_oh = one_hot(y_pred, num_classes=n_classes)[..., selected_classes]
    true_positives_per_class = torch.sum(y_true_oh * y_pred_oh, dim=[-2,-3,-4]) # sum over x,y image dim
    return true_positives_per_class

def positives(y_true,
              n_classes,
              exclude_cls=None):
    
    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    y_true_oh = one_hot(y_true, num_classes=n_classes)[..., selected_classes]
    postives_per_class = torch.sum(y_true_oh, dim=[-2,-3,-4]) # sum over batch,x,y image dim
    return postives_per_class


def pred_positives(y_pred,
                   n_classes,
                   exclude_cls=None):
    
    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    y_pred_oh = one_hot(y_pred, num_classes=n_classes)[..., selected_classes]
    pred_positives_per_class = torch.sum(y_pred_oh, dim=[-2,-3,-4]) # sum over batch,x,y image dim
    return pred_positives_per_class