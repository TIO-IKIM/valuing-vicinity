from typing import Dict, List, Tuple
import torch 
from torch.nn.functional import one_hot

def jaccard(jaccard_nominator: torch.Tensor,
            jaccard_denominator: torch.Tensor,
            n_classes: int,
            exclude_cls: List[int] = None) -> Tuple[float, Dict[int, float]]:
    """
    Jaccard index for multiclass. 
    
    Args:
        jaccard_nominator (torch.Tensor): Class-wise Jaccard nominator
        jaccard_denominator (torch.Tensor): Class-wise Jaccard denominator
        n_classes (int): number of classes
        exclude_cls (List[int], optional): List of excluded classes. Defaults to None.

    Returns:
        Tuple[float, Dict[int, float]]: Tuple of total jaccard score and dict of jaccard scores per class
    """

    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    # if class is neither predicted / nor in true: denominator => 0 -> nan dice score for this class.
    jaccard_per_class = (jaccard_nominator/jaccard_denominator)[selected_classes]
    # dice score mean over all non-nan scores
    mean_jaccard = jaccard_per_class[~torch.isnan(jaccard_per_class)].mean()
    
    return mean_jaccard.item(), {cls: jaccard.item() for jaccard, cls in zip(jaccard_per_class, selected_classes)}


def jaccard_nominator(y_true: torch.Tensor,
                      y_pred: torch.Tensor,
                      n_classes: int,
                      exclude_cls=None) -> torch.Tensor:
    """
    Determines the class-wise Jaccard nominator for n classes. 
    Expects a 3-dimensional tensor with first dim batch, second and third dim image dimensions and
    values are either the true class or the predicted class.

    Args:
        y_true (torch.Tensor): [description]
        y_pred (torch.Tensor): [description]
        n_classes (int): [description]
        exclude_cls ([type], optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: class-wise Jaccard nominator
    """
    
    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    y_true_oh = one_hot(y_true, num_classes=n_classes)[..., selected_classes]
    y_pred_oh = one_hot(y_pred, num_classes=n_classes)[..., selected_classes]
    intersect = torch.sum(y_true_oh * y_pred_oh, dim=[-2,-3]) # sum over x,y image dim
    jaccard_nominator_per_classes = torch.sum(1. * intersect, dim=0)
    return jaccard_nominator_per_classes

def jaccard_denominator(y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        n_classes: int,
                        exclude_cls: List[int]=None) -> torch.Tensor:
    """
    Determines the class-wise jaccard denominator for n classes. 
    Expects a 3-dimensional tensor with first dim batch, second and third dim image dimensions and
    values are either the true class or the predicted class.

    Args:
        y_true (torch.Tensor): 3-dim tensor with 1: batch, 2 + 3 image dim and value true pixel class
        y_pred (torch.Tensor): 3-dim tensor with 1: batch, 2 + 3 image dim and value predicted pixel class
        n_classes (int): number of existing classes (might neither be in predicted or true tensor)
        exclude_cls (List[int], optional): Exclude class. Defaults to None.

    Returns:
        torch.Tensor: class-wise Jacccard denominator
    """

    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    y_true_oh = one_hot(y_true, num_classes=n_classes)[..., selected_classes]
    y_pred_oh = one_hot(y_pred, num_classes=n_classes)[..., selected_classes]
    # union = set1 + set2 - intersect(set1, set2)
    denom = torch.sum(y_true_oh + y_pred_oh, dim=[-2,-3]) - torch.sum(y_true_oh * y_pred_oh, dim=[-2,-3]) # sum over x,y image dim
    dice_denominator_per_classes = torch.sum(denom.double(), dim=0) # no mean for ints
    return dice_denominator_per_classes

