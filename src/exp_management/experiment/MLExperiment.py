"""
Experiment offers help to manage configs and tracking of ML-experiments
"""
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import torch
from torchvision import transforms
from torch import nn

from src.exp_management.experiment.Experiment import Experiment
from src.exp_management import tracking
from src.exp_management.config import Config
from src.exp_management.data_provider import DataProvider
from src.lib.better_abc import ABCMeta
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder


class MLExperiment(Experiment, metaclass=ABCMeta):
    """
    Create an Experiment instance to log configs of ML-experiment
    """

    def __init__(self,
                 config_path: str,
                 config_parser: Type[Config],
                 testmode: bool = False,
                 prefix: str = 'exp',
                 **kwargs
                 ) -> None:
        super().__init__(config_path=config_path,
                         config_parser=config_parser,
                         prefix=prefix,
                         testmode=testmode,
                         **kwargs
                         )
        

    @abstractmethod
    def get_model(self) -> nn.Module:
        """
        Creates nn.Module for ml experiment

        Returns
            nn.Module: pytorch module
        """

    @abstractmethod
    def run_train_vali_epoch(self,
                             train_loader,
                             val_loader,
                             label_handler,
                             model,
                             criterion,
                             optimizer,
                             epoch,
                             writer,
                             args):
        """Provide a train & validation epoch function

        Args:
            train_loader ([type]): [description]
            val_loader ([type]): [description]
            label_handler  ([type]): [description]
            model ([type]): [description]
            criterion ([type]): [description]
            optimizer ([type]): [description]
            epoch ([type]): [description]
            writer ([type]): [description]
            args ([type]): [description]
        """
        
    @abstractmethod
    def wsi_inference(self,
                        wsis: List[WSIFromFolder],
                        model: nn.Module,
                        data_provider: DataProvider,
                        gpu: int):
        """
        Provide a function to inference wsis given a trained model.
        Provide patch predictions and wsi predictions to use later for evaluation.

        Args:
            wsis (List[WSIFromFolder]): [description]
            model (nn.Module): [description]
            data_provider (DataProvider): [description]
            gpu (int): [description]
        """

    def get_optimizer(self, 
                      model: torch.nn.Module
                      ) -> torch.optim.Optimizer:
        """Basic optimizer

        Args:
            model (torch.nn.Module): torch model

        Returns:
            torch.optim.Optimizer: A torch optimizer
        """
        optim_params = model.parameters()
        optimizer = torch.optim.SGD(optim_params,
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        return optimizer
    
    def get_criterion(self, weight=None):
        """Basic criterion

        Returns:
            [type]: [description]
        """
        return nn.CrossEntropyLoss(weight=weight).cuda(self.args.gpu)
    
    
    def evaluate_wsis(exp: Experiment,
                    wsis: List[WSIFromFolder],
                    data_provider: DataProvider,
                    log_path: Path,
                    epoch: int = None,
                    writer = None,
                    save_to_folder = False,
                    tag: str = ""
                    ) -> List[Dict]:
        """
        Run patch inference and returns WSI predictions

        Args:
            config_path (str): Path to yaml-config in PatchInferenceConfig style
        """

        viz = tracking.Visualizer(writer=writer,
                                save_to_folder=save_to_folder)

        log_wsi_preds = []

        wsi_predictions = []
        wsi_labels = []

        patch_prediction = []
        patch_label = []
        patch_wsi = []

        for wsi in wsis:
            
            if data_provider.patch_label_type not in ['mask']:
                
                viz.wsi_plot(tag=tag + "_wsi",
                            mode='wsi',
                            wsi=wsi,
                            log_path=log_path,
                            epoch=epoch)
                
                if data_provider.patch_label_type not in ['distribution']:
                    viz.wsi_plot(tag=tag + "_worst_patches",
                                wsi=wsi,
                                mode='worst_patches',
                                log_path=log_path,
                                epoch=epoch)
                
                # evalute meterics with drawn patches (to equal training)
                logging.getLogger('exp').info(f"Processing WSI {wsi.name}")
                patch_prediction.extend(wsi.get_patch_predictions())
                patch_label.extend(wsi.get_patch_labels(org=False))
                patch_wsi.extend([wsi.name] * len(wsi.get_patch_predictions()))

                wsi_labels.append(wsi.get_label(org=False))
                wsi_predictions.append(wsi.get_prediction())

                tmp_dict = dict()
                tmp_dict[wsi.name] = wsi.get_pred_dict()
                log_wsi_preds.append(tmp_dict)
                logging.getLogger('exp').info(f"{wsi.name} image prediction is {tmp_dict[wsi.name]} - real: {wsi.get_label()}")


        if exp.args.number_of_classes == 2 and data_provider.patch_label_type != 'mask':
            viz.roc_auc(tag= "roc_auc/" + tag,
                        predictions=wsi_predictions,
                        labels=wsi_labels,
                        label_handler=data_provider.image_label_handler,
                        log_path=log_path,
                        epoch=epoch)

            viz.probability_hist(tag="prob_hist/" + tag + "_wsi_level",
                                predictions=wsi_predictions,
                                labels=wsi_labels,
                                label_handler=data_provider.image_label_handler,
                                log_path=log_path,
                                epoch=epoch)

            viz.probability_hist(tag="prob_hist/" + tag + "_patch_level",
                                predictions=patch_prediction,
                                labels=patch_label,
                                wsis=patch_wsi,
                                label_handler=data_provider.label_handler,
                                log_path=log_path,
                                epoch=epoch)
                
        return log_wsi_preds


    def get_augmention(self) -> Tuple[Union[None, transforms.Compose],
                                      Union[None, transforms.Compose]
                                      ]:
        return None, None
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, batch):
        """
        Args:
            tensor (Tensor): batch of image of size (B,C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for i in range(batch.shape[0]):
            for t, m, s in zip(batch[i], self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return batch