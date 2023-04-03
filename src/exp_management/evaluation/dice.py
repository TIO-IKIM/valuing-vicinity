from typing import Dict, List, Tuple
import torch 
from torch.nn.functional import one_hot

def dice_coef(dice_nominator: torch.Tensor,
              dice_denominator: torch.Tensor,
              n_classes: int,
              exclude_cls: List[int] = None) -> Tuple[float, Dict[int, float]]:
    """
    Dice coefficient for multiclass. 
    
    Args:
        dice_nominator (torch.Tensor): Class-wise Dice nominator
        dice_denominator (torch.Tensor): Class-wise Dice denominator
        n_classes (int): number of classes
        exclude_cls (List[int], optional): List of excluded classes. Defaults to None.

    Returns:
        Tuple[float, Dict[int, float]]: Tuple of total dice score and dict of dice scores per class
    """

    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    # if class is neither predicted / nor in true: denominator => 0 -> nan Dice score for this class.
    dice_per_class = (dice_nominator/dice_denominator)[selected_classes]
    # dice score mean over all non-nan scores
    mean_dice = dice_per_class[~torch.isnan(dice_per_class)].mean()
    
    return mean_dice.item(), {cls: dice.item() for dice, cls in zip(dice_per_class, selected_classes)}


def dice_nominator(y_true: torch.Tensor,
                   y_pred: torch.Tensor,
                   n_classes: int,
                   exclude_cls=None) -> torch.Tensor:
    """
    Determines the class-wise Dice nominator for n classes. 
    Expects a 3-dimensional tensor with first dim batch, second and third dim image dimensions and
    values are either the true class or the predicted class.

    Args:
        y_true (torch.Tensor): [description]
        y_pred (torch.Tensor): [description]
        n_classes (int): [description]
        exclude_cls ([type], optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: class-wise Dice nominator
    """
    
    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    y_true_oh = one_hot(y_true, num_classes=n_classes)[..., selected_classes]
    y_pred_oh = one_hot(y_pred, num_classes=n_classes)[..., selected_classes]
    intersect = torch.sum(y_true_oh * y_pred_oh, dim=[-2,-3]) # sum over x,y image dim
    dice_nominator_per_classes = torch.sum(2. * intersect, dim=0)
    return dice_nominator_per_classes

def dice_denominator(y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     n_classes: int,
                     exclude_cls: List[int]=None) -> torch.Tensor:
    """
    Determines the class-wise Dice denominator for n classes. 
    Expects a 3-dimensional tensor with first dim batch, second and third dim image dimensions and
    values are either the true class or the predicted class.

    Args:
        y_true (torch.Tensor): 3-dim tensor with 1: batch, 2 + 3 image dim and value true pixel class
        y_pred (torch.Tensor): 3-dim tensor with 1: batch, 2 + 3 image dim and value predicted pixel class
        n_classes (int): number of existing classes (might neither be in predicted or true tensor)
        exclude_cls (List[int], optional): Exclude class. Defaults to None.

    Returns:
        torch.Tensor: class-wise Dice denominator
    """

    if exclude_cls is None:
        exclude_cls = []
    selected_classes = [cls for cls in range(n_classes) if cls not in exclude_cls]
    
    y_true_oh = one_hot(y_true, num_classes=n_classes)[..., selected_classes]
    y_pred_oh = one_hot(y_pred, num_classes=n_classes)[..., selected_classes]
    denom = torch.sum(y_true_oh + y_pred_oh, dim=[-2,-3]) # sum over x,y image dim
    dice_denominator_per_classes = torch.sum(denom.double(), dim=0)
    return dice_denominator_per_classes

