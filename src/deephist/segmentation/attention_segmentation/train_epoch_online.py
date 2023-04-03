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


def train_epoch_online(exp: Experiment,
                       holdout_set: HoldoutSet,
                       model: nn.Module,
                       criterion: _Loss,
                       optimizer: torch.optim.Optimizer,
                       label_handler: LabelHandler,
                       epoch: int,
                       args: Dict,
                       writer: SummaryWriter) -> float:
    """Given the holdout-set: Train the model on train-dataloader. Evaluate on val-dataloader.
    For online, the neighbourhood is trained simultaneously instead of building up a memory.

    Args:
        holdout_set (holdout_set]): [description]
        model (nn.Module): [description]
        criterion (_Loss): [description]
        optimizer (torch.optim.Optimizer): [description]
        label_handler (LabelHandler): [description]
        epoch (int): [description]
        args (Dict): [description]
        writer (writer): [description]

    Returns:
        float: Average validation performance (smaller is better) after training step
    """
    
    for phase in ['train', 'vali']:
   
        metric_logger = tracking.MetricLogger(delimiter="  ",
                                              tensorboard_writer=writer,
                                              args=args)
        viz = tracking.Visualizer(writer=writer)
        
        if phase == 'train':
            data_loader = holdout_set.train_loader
            # for fast embedding inference
            wsi_dataset = holdout_set.train_wsi_dataset
        else:
            data_loader = holdout_set.vali_loader
            wsi_dataset= holdout_set.vali_wsi_dataset

        initialize_logging(metric_logger=metric_logger,
                           phase=phase,
                           num_heads=args.num_attention_heads,
                           args=args)

        header = f'{phase} GPU {args.gpu} Epoch: [{epoch}]'

        if phase == 'train':
            # switch to train mode
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        epoch_dice_nominator = 0
        epoch_dice_denominator = 0
        sample_images = None
        sample_labels = None
        sample_preds = None
        
        for batch in metric_logger.log_every(data_loader, args.print_freq, epoch, header, phase):
            
            images = batch['img']
            labels = batch['mask']
            neighbour_imgs = batch['neighbour_img']
            neighbour_mask = batch['neighbour_mask']
           
            if args.gpu is not None:
                images_gpu = images.cuda(args.gpu, non_blocking=True)
                labels_gpu = labels.cuda(args.gpu, non_blocking=True)
                neighbour_imgs_gpu = neighbour_imgs.cuda(args.gpu, non_blocking=True)
                neighbour_mask_gpu = neighbour_mask.cuda(args.gpu, non_blocking=True)
            
            result = model(images=images_gpu, 
                                      neighbour_masks=neighbour_mask_gpu,
                                      neighbour_imgs=neighbour_imgs_gpu,
                                      return_attention=True) 
            logits= result['logits']
            attention= result['attention']

            if args.combine_criterion_after_epoch is not None:
                loss = criterion(logits, labels_gpu, epoch)
            else:
                loss = criterion(logits, labels_gpu)
                
            # compute gradient and do SGD step
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                            
            step_dice_nominator, step_dice_denominator = log_step(phase=phase,   
                                                                  metric_logger=metric_logger,
                                                                  loss=loss,
                                                                  logits_gpu=logits,
                                                                  labels_gpu=labels_gpu,
                                                                  images=images,
                                                                  attention_gpu=attention, 
                                                                  neighbour_masks=neighbour_mask_gpu,
                                                                  args=args)
            
            # add up dice nom and denom over one epoch to get "epoch-dice-score" - different to WSI-dice score!
            epoch_dice_nominator += step_dice_nominator
            epoch_dice_denominator += step_dice_denominator
                
            if sample_images is None:
                sample_images = exp.unnormalize(images)
                sample_labels = labels
                sample_preds = logits.detach().argmax(axis=1).cpu()
                
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

