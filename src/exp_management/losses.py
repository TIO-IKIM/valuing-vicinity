import torch
from torch import nn
from torch.nn.functional import softmax


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, axis=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.axis = axis

    def forward(self, inputs, targets, smooth=1e-6):
        """[summary]

        Args:
            inputs (Tensor): logits
            targets (Tensor): probabiliy per class
            smooth (float, optional): Defaults to 1e-6.

        Returns:
            [type]: [description]
        """
    
        inputs = self.activation(inputs) # softmax
        n_classes = inputs.shape[1]
        targets_onehot = self._one_hot(targets, classes=n_classes)
        #flatten label and prediction tensors
        targets_flat = torch.flatten(targets_onehot, start_dim=1, end_dim=3)
        inputs_flat = torch.flatten(inputs, start_dim=1, end_dim=3)
        
        #True Positives, False Positives & False Negatives
        TP = torch.sum((inputs_flat * targets_flat), dim=0)
        FP = torch.sum(((1-targets_flat) * inputs_flat), dim=0)
        FN = torch.sum((targets_flat * (1-inputs_flat)), dim=0)
               
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        FocalTversky = torch.pow((1 - Tversky), self.gamma)
        meanFocalTversky = torch.mean(FocalTversky)
        return meanFocalTversky

    def activation(self, x): 
        return softmax(x, dim=self.axis)
    
    @staticmethod
    def _one_hot(x, classes, axis=1):
        "Creates one binay mask per class"
        return torch.stack([torch.where(x==c, 1, 0) for c in range(classes)], axis=axis)


class CombinedLoss(nn.Module):
    def __init__(self, l1, l2, weight2, add_l2_at_epoch=None):
        super(CombinedLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2    
        self.weight2 = weight2 
        self.start_epoch = add_l2_at_epoch
        
    def forward(self, inputs, targets, epoch=None):
        
        l1 = self.l1(inputs, targets)
        if epoch > self.start_epoch:
            l2 = self.l2(inputs, targets)
            return (1-self.weight2) * l1 + self.weight2 * l2
        else:
            return l1
        
class HelperLoss(nn.Module):
    def __init__(self, base_loss, base_weight):
        super(HelperLoss, self).__init__()
        self.base_loss = base_loss
        self.helper = nn.CrossEntropyLoss()   
        self.base_weight = base_weight 
        
    def forward(self, inputs, cls_logits, targets, cls_targets):
        
        l = self.base_loss(inputs, targets)
        if cls_logits is not None:
            helper_loss = self.helper(cls_logits, cls_targets)
            return {'loss': (self.base_weight) * l + (1-self.base_weight) * helper_loss,
                    'cls_loss': helper_loss,
                    'mask_loss': l}
        else:
            return {'loss': l,
                    'cls_loss': None,
                    'mask_loss': l}