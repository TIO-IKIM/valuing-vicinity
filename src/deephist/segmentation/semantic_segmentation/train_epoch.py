"""
Run supervised ML-experiment
"""
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from src.deephist.segmentation.attention_segmentation.logging import initialize_logging, log_epoch, log_step
from src.exp_management import tracking
from src.exp_management.data_provider import HoldoutSet
from src.exp_management.experiment.Experiment import Experiment
from src.pytorch_datasets.label_handler import LabelHandler


def train_epoch(exp: Experiment,
                holdout_set: HoldoutSet,
                model: nn.Module,
                criterion: _Loss,
                optimizer: torch.optim.Optimizer,
                label_handler: LabelHandler,
                epoch: int,
                args: Dict,
                writer: SummaryWriter) -> float:
    """Train the model on train-dataloader. Evaluate on val-dataloader

    Args:
        data_loaders (List[DataLoader]): [description]
        model (nn.Module): [description]
        criterion (_Loss): [description]
        optimizer (torch.optim.Optimizer): [description]
        label_handler (LabelHandler): [description]
        epoch (int): [description]
        args (Dict): [description]
        writer (writer): [description]

    Returns:
        float: Average validation loss after training step
    """
    
    for phase in ['train', 'vali']:
        
        viz = tracking.Visualizer(writer=writer)
        metric_logger = tracking.MetricLogger(delimiter="  ",
                                              tensorboard_writer=writer,
                                              args=args)
        
        initialize_logging(metric_logger=metric_logger,
                           phase=phase,
                           args=args)
        
        header = f'{phase} GPU {args.gpu} Epoch: [{epoch}]'

        if phase == 'train':
            # switch to train mode
            model.train()
            torch.set_grad_enabled(True)
            
            data_loader = holdout_set.train_loader
        else:
            model.eval()
            torch.set_grad_enabled(False)
            data_loader = holdout_set.vali_loader

        epoch_dice_nominator = 0
        epoch_dice_denominator = 0
        sample_images = None
        sample_labels = None
        sample_preds = None
        
        for batch in metric_logger.log_every(data_loader, args.print_freq, epoch, header, phase):
            
            images = batch['img']
            labels = batch['mask']
        
            if args.gpu is not None:
                images_gpu = images.cuda(args.gpu, non_blocking=True)
                labels_gpu = labels.cuda(args.gpu, non_blocking=True)
            # compute output and loss
            result = model(images_gpu)
            
            logits = result['logits']
            
            if args.combine_criterion_after_epoch is not None:
                loss = criterion(logits, labels_gpu, epoch)
            else:
                loss = criterion(logits, labels_gpu)
                
            # compute gradiemt and do SGD step
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if sample_images is None:
                sample_images = exp.unnormalize(images)
                sample_labels = labels
                sample_preds = logits.cpu().argmax(axis=1)
                
            step_dice_nominator, step_dice_denominator = log_step(phase=phase,   
                                                                  metric_logger=metric_logger,
                                                                  loss=loss,
                                                                  logits_gpu=logits,
                                                                  labels_gpu=labels_gpu,
                                                                  images=images,
                                                                  args=args)
            
            # add up dice nom and denom over one epoch to get "epoch-dice-score" - different to WSI-dice score!
            epoch_dice_nominator += step_dice_nominator
            epoch_dice_denominator += step_dice_denominator
        
        #after epoch is finished:        
        log_epoch(phase=phase,
                  metric_logger=metric_logger, 
                  viz=viz, 
                  epoch_dice_nominator=epoch_dice_nominator,
                  epoch_dice_denominator=epoch_dice_denominator,
                  model=model,
                  sample_images=sample_images,
                  sample_labels=sample_labels,
                  sample_preds=sample_preds,
                  label_handler=label_handler,
                  epoch=epoch,
                  args=args)
            
        logging.getLogger('exp').info(f"Averaged {phase} stats: {metric_logger.global_str()}")

    if args.performance_metric == 'dice':
        # performance set to (negative) Dice 
        performance_metric = -1 * metric_logger.vali_dice_coef.global_avg
    elif args.performance_metric == 'loss':
        performance_metric = metric_logger.vali_loss.global_avg
    return performance_metric
