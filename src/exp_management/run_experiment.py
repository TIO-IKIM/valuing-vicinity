"""
Run supervised ML-experiment
"""

import copy
import logging 
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
from typing import List

from prettytable import PrettyTable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from src.exp_management.helper import set_seed
from src.exp_management.experiment.Experiment import Experiment
from src.exp_management.data_provider import CvSet, HoldoutSet
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder


def run_experiment(exp: Experiment
                   ) -> None:
    """Run a supervised ML-experiment from config file

    Args:
        config_path (str): path to config file
    """
    
    # only use half of cpu cores
    #torch.set_num_threads(int(os.cpu_count() * 0.5))
    
    args = exp.args
    
    # prepare data
    data_provider = exp.data_provider

    args.number_of_classes = data_provider.number_classes
    exp.exp_log(number_of_classes=args.number_of_classes)
    
    # either holdout or cv (/monte-carlo)
    if data_provider.holdout_set is not None:
        run_holdout(holdout_set=data_provider.holdout_set,
                    exp=exp)
    else:
        run_kfold_model(cv_set=data_provider.cv_set,
                        exp=exp)
        

def run_kfold_model(cv_set: CvSet,
                    exp: Experiment
                    ) -> None:
    """Runs multiple folds in parallel on given GPUs.

    Args:
        cv_set (CvSet): _description_
        exp (Experiment): _description_
    """
    
    if exp.args.gpus is not None:
        m = mp.Manager()
        gpu_queue = m.Queue()
        # initialize queue
        for gpu_id in exp.args.gpus:
            gpu_queue.put(gpu_id)
        
        torch.multiprocessing.spawn(run_holdout_model_in_parallel, 
                                    args=(cv_set.holdout_sets, gpu_queue, exp), 
                                    nprocs=len(exp.args.gpus), 
                                    join=True, 
                                    daemon=False,
                                    start_method='spawn')
    else:
        main_exp = copy.deepcopy(exp)
        fold_logs = list()
        for fold_no in range(exp.args.nfold):
            if exp.args.folds is not None:
                # check fold is defined:
                if fold_no not in exp.args.folds:
                    # jump to next fold
                    continue
            holdout_set = cv_set.holdout_sets[fold_no]
            
            # settings for folds
            exp.set_fold(fold=holdout_set.fold)
            
            exp.args.fold = holdout_set.fold
            logging.getLogger('exp').info(f"Starting {holdout_set.fold}. fold run on gpu {exp.args.gpu}")
            
            run_holdout(holdout_set=holdout_set,
                        exp=exp)
            # store logs 
            fold_logs.append(exp.get_log())
            logging.getLogger('exp').info(f"Finished fold {holdout_set.fold}")
                   
    # here aggregate kfold results
    main_exp.merge_fold_logs(fold_logs)

def run_holdout_model_in_parallel(proc_idx: int,
                                  holdout_sets: List[HoldoutSet],
                                  gpu_queue: mp.Queue,
                                  exp: Experiment
                                  ) -> None:
    """Wrapper to run a holdout set in parallel

    Args:
        proc_idx (int): Fold-id
        holdout_sets (List[HoldoutSet]): List of Holdout sets
        gpu_queue (mp.Queue): a queue to store the gpu tokens
        exp (Experiment): the Experiment
    """
    
    holdout_set = holdout_sets[proc_idx]
    # tread safe gpu id:
    exp.args.gpu = gpu_queue.get()
    
    # settings for folds
    exp.set_fold(fold=holdout_set.fold)
    
    exp.args.fold = holdout_set.fold
    logging.getLogger('exp').info(f"Starting {holdout_set.fold}. fold run on gpu {exp.args.gpu}")
    
    run_holdout(holdout_set=holdout_set,
                exp=exp)
    
    logging.getLogger('exp').info(f"Finished fold {holdout_set.fold}")
    # free gpu again
    gpu_queue.put(exp.args.gpu)

def run_holdout(exp: Experiment,
                holdout_set: HoldoutSet
                ) -> None:
    """
    Runs a holdout set. First train, second evaluate.

    Args:
        exp (Experiment): the Exeperiment
        holdout_set (HoldoutSet): the HoldoutSet
    """
    set_seed(exp.args.seed)

    #log holdout details to experiment
    exp.exp_log(holdout_set = holdout_set.metadata)
    
    writer = SummaryWriter(exp.log_path)
        
    if exp.args.reload_model_folder is None:
        train_holdout_model(holdout_set=holdout_set,
                            exp=exp,
                            writer=writer)
        reload_from = exp.log_path
    else:
        logging.getLogger('exp').info(f"Skipping training step and restoring best model from {exp.args.reload_model_folder}")
        reload_from = exp.args.reload_model_folder
        
    eval_holdout_model(holdout_set=holdout_set,
                       exp=exp,
                       reload_from=reload_from,
                       writer=writer)
    
    writer.close()
    
def eval_holdout_model(holdout_set: HoldoutSet,
                       exp: Experiment,
                       reload_from: str,
                       writer: SummaryWriter
                       ) -> None:
    """
    Evaluates a holdout set from a reloaded model - optionally with test data.

    Args:
        holdout_set (HoldoutSet): the HoldoutSet
        exp (Experiment): the Experiment
        reload_from (str): the path to the stored model
        writer (SummaryWriter): a tensorboard Writer
    """
    
    # reset model
    model = exp.get_model()

    reload_epoch = reload_model(model=model,
                                model_path=reload_from,
                                gpu=exp.args.gpu)
    
    # evaluate best model on val set
    if exp.args.reload_model_folder is None or exp.args.include_val_inference is True:
        logging.getLogger('exp').info("Evaluating final validation wsis")
        evaluate_model(exp=exp,
                       model=model,
                       wsis=holdout_set.vali_wsi_dataset.wsis,
                       writer=writer,
                       epoch=reload_epoch,
                       save_to_folder=True,
                       log_metrics=True,
                       tag='vali_best')
    
    # test model on testset
    if holdout_set.test_wsi_dataset is not None:
        ## evaluate WSI
        evaluate_model(exp=exp,
                       model=model,
                       wsis=holdout_set.test_wsi_dataset.wsis,
                       writer=writer,
                       epoch=reload_epoch,
                       tag='test',
                       save_to_folder=True,
                       log_metrics=True)
        
def train_holdout_model(holdout_set: HoldoutSet,
                        exp: Experiment,
                        writer: SummaryWriter
                        ) -> None:
    """
    Train a model on given data.

    Args:
        model (nn.Module): Pytorch module
        data_provider (DataProvider): DataProvider holding the train and val dataloader
        args (Dict): Train parameters
    """
    
    #torch.cuda.set_device(exp.args.gpu)
    model = exp.get_model()

    model = model.cuda(exp.args.gpu)

    logging.getLogger('exp').info(model) # print model after SyncBatchNorm  
    count_parameters(model)
    
    if exp.args.gpu is not None:
        logging.getLogger('exp').info("Use GPU: {} for training".format(exp.args.gpu))

    # define loss function (criterion) and optimizer
    criterion = exp.get_criterion()
    optimizer = exp.get_optimizer(model)
    
    if exp.args.adjust_lr is True:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                           gamma=exp.args.lr_gamma, 
                                                           verbose=False)

    cudnn.benchmark = True

    val_performance =  float("inf")
    bad_epochs = 0 # counter for early stopping

    for epoch in range(exp.args.epochs):
      
        new_val_performance = exp.run_train_vali_epoch(holdout_set=holdout_set,
                                                       model=model,
                                                       criterion=criterion,
                                                       optimizer=optimizer,
                                                       label_handler=holdout_set.data_provider.label_handler,
                                                       epoch=epoch,
                                                       writer=writer,
                                                       args=exp.args)
        if exp.args.adjust_lr is True:
            scheduler.step()
            logging.getLogger('exp').info(f"Updating learning rate to: {scheduler.get_last_lr()}")
                
        if (exp.args.warm_up_epochs is None \
                or epoch >= exp.args.warm_up_epochs):
            if new_val_performance < val_performance:
                
                bad_epochs = 0 # reset early stopping
                val_performance = new_val_performance
                exp.exp_log(best_epoch = epoch)
                exp.exp_log(best_val_performance = new_val_performance)
                torch.save(obj={'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'label_handler': holdout_set.data_provider.label_handler,
                                'args': exp.args
                                },
                        f=os.path.join(exp.checkpoint_path,
                                       'model_best.pth.tar'))
            else: 
                bad_epochs += 1
                if exp.args.early_stopping_epochs is not None and bad_epochs >= exp.args.early_stopping_epochs:
                    logging.getLogger('exp').info(f"Early stopping at epoch {epoch}. {bad_epochs} times no val loss improvement.")
                    exp.exp_log(early_stopping_epoch=epoch)
                    break # stop epoch loop
            
        # evaluation during training
        if exp.args.evaluate_every != 0 and (epoch+1) % exp.args.evaluate_every == 0:
            
            # sample wsis for evaluation once 
            if exp.args.n_eval_wsis is not None:
                
                n_train_wsis = len(holdout_set.train_wsi_dataset.wsis)
                train_wsi_idx = list(range(n_train_wsis))
                random.Random(exp.args.seed).shuffle(train_wsi_idx)
                train_wsis = [holdout_set.train_wsi_dataset.wsis[idx] for idx in 
                            train_wsi_idx[:min(exp.args.n_eval_wsis, n_train_wsis)]]
                
                n_val_wsis = len(holdout_set.vali_wsi_dataset.wsis)
                val_wsi_idx = list(range(n_val_wsis))
                random.Random(exp.args.seed).shuffle(val_wsi_idx)
                val_wsis = [holdout_set.vali_wsi_dataset.wsis[idx] for idx in 
                            val_wsi_idx[:min(exp.args.n_eval_wsis, n_val_wsis)]]
                
            else:
                train_wsis = holdout_set.train_wsi_dataset.wsis
                val_wsis = holdout_set.vali_wsi_dataset.wsis
                
            # evaluate validation set
            evaluate_model(exp=exp,
                           model=model,
                           wsis=train_wsis,
                           writer=writer,
                           epoch=epoch,
                           log_metrics=True,
                           tag='sample_train' if exp.args.n_eval_wsis else 'train')

            evaluate_model(exp=exp,
                           model=model,
                           wsis=val_wsis,
                           writer=writer,
                           epoch=epoch,
                           log_metrics=True,
                           tag='sample_vali' if exp.args.n_eval_wsis else 'vali')


def evaluate_model(exp: Experiment,
                   model: nn.modules,
                   wsis: List[WSIFromFolder],
                   writer: SummaryWriter,
                   tag: str,
                   epoch: int = None,
                   save_to_folder: bool = False,
                   log_metrics: bool = False
                   ) -> None:
    """ Given a model, the WSIs are first inferenced patch-wise and then evaluated (viz, metrics) 

    Args:
        exp (Experiment): the Experiment
        model (nn.modules): the model
        wsis (List[WSIFromFolder]): a List of WSI objects to evaluate
        writer (SummaryWriter): a tensorboard Writer
        tag (str): a name tag for the evaluation.
        epoch (int, optional): the epoch of the model. Defaults to None.
        save_to_folder (bool, optional): target folder to save the evaluation results to. Defaults to False.
        log_metrics (bool, optional): Set to True if you want to log tensorboard. Defaults to False.
    """
    
    logging.getLogger('exp').info(f"Evaluating {tag} WSIs.")
          
    # inference for all patches
    global_evaluation =  exp.wsi_inference(wsis=wsis,
                                           model=model,
                                           data_provider=exp.data_provider,
                                           gpu=exp.args.gpu)
    
    # evaluate on WSI level
    wsi_evaluation = exp.evaluate_wsis(wsis=wsis,
                                       data_provider=exp.data_provider,
                                       log_path=Path(exp.log_path),
                                       tag=tag,
                                       writer=writer,
                                       save_to_folder=save_to_folder,
                                       epoch=epoch)  
    
    if log_metrics == True:
        writer.add_scalar(tag=f'evaluation/{tag}_global_dice', scalar_value=global_evaluation['global_dice_score'], global_step=epoch)
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_dice', scalar_value=wsi_evaluation['wsi_mean_dice_scores'], global_step=epoch)
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_precision', scalar_value=wsi_evaluation['wsi_mean_precision'], global_step=epoch)
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_recall', scalar_value=wsi_evaluation['wsi_mean_recall'], global_step=epoch)
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_jaccard', scalar_value=wsi_evaluation['wsi_mean_jaccard_scores'], global_step=epoch)
    
    exp.exp_log(key=f'evaluation_wsi_{tag}_set',
                value={**wsi_evaluation, **global_evaluation})
    

def reload_model(model: nn.Module,
                 model_path: str,
                 gpu: int) -> int:
    """
    Reload the model in the best val loss state

    Args:
        model (nn.Module): module
        model_path (Path): path to model log folder (must contain 'checkpoints/model_best.pth.tar')

    Raises:
        Exception: Raised if model path cannot be found.
        
    Returns: epoch of reloaded model
    """
    # try to detect checkpoint files in model_path
    checkpoint_path = Path(model_path) / 'checkpoints' / 'model_best.pth.tar'
    if not checkpoint_path.exists():
        raise Exception(f"Cannot find best checkpoint file {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    return checkpoint['epoch']


def adjust_learning_rate(optimizer,
                         init_lr,
                         epoch,
                         args) -> None:
    """
    Decay the learning rate based on schedule
    """
    #cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    cur_lr = init_lr * math.pow((1 - epoch/args.epochs),0.9) 
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def count_parameters(model: torch.nn.Module) -> int:
    """Counts the model parameters

    Args:
        model (torch.nn.Module): a torch model

    Returns:
        int: number of model parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    logging.getLogger('exp').info(table)
    logging.getLogger('exp').info(f"Total Trainable Params: {total_params}")
    return total_params