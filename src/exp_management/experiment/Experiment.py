"""
Experiment offers help to manage configs and tracking of ML-experiments
"""

from collections import defaultdict
import datetime
import logging
from pathlib import Path
from typing import Counter, Dict, List, Type
import warnings
import yaml

import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from src.exp_management import tracking
from src.exp_management.config import Config
from src.exp_management.helper import set_seed
from src.lib.better_abc import ABCMeta, abstract_attribute


class Experiment(metaclass=ABCMeta):
    """
    Create an Experiment instance to log configs of ML-experiment
    """

    def __init__(self,
                 config_path: str,
                 config_parser: Type[Config],
                 testmode: bool = False,
                 prefix: str = 'exp',
                 log_level: int = logging.INFO,
                 **kwargs
                 ) -> None:
        
        self.testmode = testmode
        config = config_parser(config_paths=[config_path])
        
        self.config_path = config_path
        self.prefix = prefix
        self.args = config.parse_config(testmode=testmode)

        #if kwargs exist, try to replace in config:
        for key, value in kwargs.items():
            if key in self.args:
                setattr(self.args, key, value)
            else:
                raise Exception(f"Key {key} does not exist in config. Cannot replace")
        
        # if reload-model exists, reload config
        if self.args.reload_model_folder is not None:
            self.new = False
            config = config_parser(config_paths= [Path(self.args.reload_model_folder) / 'config.yml', config_path])
            self.args = config.parse_config(testmode=testmode)
            self.model_name = Path(self.args.reload_model_folder).stem
            self.base_log_path = Path(self.args.reload_model_folder)
        else:
            self.new = True
            self.model_name = Path(self.config_path).stem + "-{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.datetime.now())       
            # if parallel at the same time, tag with gpu to avoid overwriting
            self.model_name += f"_{str(self.args.gpu)}"
            self.base_log_path = str(Path(self.args.logdir) / str(Path(self.config_path).parent) / self.model_name)

        if 'logdir' in self.args and not testmode:
            
            self.set_log_path(log_path=Path(self.base_log_path))
            self._init_logging(log_level)
            
        if self.args.seed is not None:  
            set_seed(self.args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
            
        
    def _init_logging(self, log_level):
            
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(log_level)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(processName)-5s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        
        file = logging.FileHandler(filename=Path(self.log_path) / 'training.log', mode='w')
        file.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M')
        file.setFormatter(formatter)
        
        logging.getLogger('exp').setLevel(logging.INFO)
        
        # remove "old" handlers
        logging.getLogger('exp').handlers.clear()
        # add the handler to the root logger
        logging.getLogger('exp').addHandler(console)      
        logging.getLogger('exp').addHandler(file)

    def set_fold(self, fold: int):

        log_path=Path(self.base_log_path) / f"fold_{fold}"
        if self.args.reload_model_folder is not None:
            self.args.reload_model_folder =  str(self.args.reload_model_folder / f"fold_{fold}")

        self.set_log_path(log_path=log_path)

    def set_log_path(self, log_path: Path):
        if self.new:
                
            log_path.mkdir(parents=True, exist_ok=True)

            tracking.log_config(log_path, self.config_path)
            tracking.log_args(log_path / (self.prefix + "_args.yml"), self.args)

        # compatable with previous version
        global LOGFILE_PATH
        LOGFILE_PATH = log_path / (self.prefix + "_log.yml")

        self.logfile_path = log_path / (self.prefix + "_log.yml")

        self.log_path = str(log_path)
        self.checkpoint_path = str(log_path / 'checkpoints')
        
        if self.new:
            Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    @abstract_attribute
    def data_provider(self):
        """Returns a DataProvider object

        Returns:
            DataProvider: WSI DataProvider
        """
    
    def merge_fold_logs(self, fold_logs: List[Dict]):
        logging.getLogger('exp').info("Merge folds..")
        
        metrics = ['wsi_mean_dice_scores', 
                    'class_mean_dice_scores', 
                    'wsi_mean_jaccard_scores', 
                    'class_mean_jaccard_scores', 
                    'wsi_mean_precision',
                    'class_mean_precision',
                    'wsi_mean_recall',
                    'class_mean_recall', 
                    'global_dice_score']
        
        test_scores = defaultdict(list)
        val_scores = defaultdict(list)
        scores = {'vali_best': val_scores,
                  'test': test_scores}
        
        # fuse vali_best + test scores
        for fold in range(len(fold_logs)):
            for phase in ['vali_best', 'test']:
                for metric in metrics:
                    scores[phase][metric].append(fold_logs[fold][f'evaluation_wsi_{phase}_set'][metric])
      
        # log score per class
        cls_test_scores = defaultdict(list)
        cls_val_scores = defaultdict(list)    
        cls_scores = {'vali_best': cls_val_scores,
                      'test': cls_test_scores}
        
        for fold in range(len(fold_logs)):
            for phase in ['vali_best', 'test']:
                for cls, score in fold_logs[fold][f'evaluation_wsi_{phase}_set']['mean_dice_scores_per_class'].items():
                    metric = f'wsi_{cls}_dice_scores'
                    cls_scores[phase][metric].append(score)
                
        # mean / std
        for phase in ['vali_best', 'test']:
            # back to dict
            cls_scores[phase] = dict(cls_scores[phase])
            scores[phase] = dict(scores[phase])
            for score in scores[phase].keys():
                assert len(scores[phase][score]) == len(fold_logs) , 'not all metrics of all folds found'
                scores[phase][score] = {'mean': np.round(np.nanmean(np.array(scores[phase][score])),4).item(),
                                        'std': np.round(np.nanstd(np.array(scores[phase][score])),4).item()}
            for score in cls_scores[phase].keys():
                assert len(cls_scores[phase][score]) == len(fold_logs) , 'not all metrics of all folds found'
                cls_scores[phase][score] = {'mean': np.round(np.nanmean(np.array(cls_scores[phase][score])),4).item(),
                                        'std': np.round(np.nanstd(np.array(cls_scores[phase][score])),4).item()}
        
        # log to tensorboard
        writer = SummaryWriter(self.log_path)
        for phase in ['vali_best', 'test']:
            for score in scores[phase].keys():
                writer.add_scalar(tag=f'fold_aggregation/{phase}_fold_mean_{score}',
                                  scalar_value=scores[phase][score]['mean'])
            for score in cls_scores[phase].keys():
                writer.add_scalar(tag=f'fold_aggregation_class/{phase}_fold_mean_{score}',
                                  scalar_value=cls_scores[phase][score]['mean'])
        
        writer.close()
        
        self.exp_log(fold_aggregates=scores) 
        self.exp_log(fold_class_aggregates=cls_scores) 

        
    def get_log(self) -> Dict:
        if self.logfile_path.exists():
            with self.logfile_path.open('r') as yamlfile:
                log_yaml = yaml.safe_load(yamlfile) # Note the safe_load
        return log_yaml
        
    def exp_log(self,
                **kwargs):
        """
        Store arbitrary information in a log file.
        Either provide a key and value argument or named arguments.
        """
        if not self.testmode:     
            if len(kwargs.keys()) == 2 and 'key' in kwargs.keys() and 'value' in kwargs.keys():
                keyval = True
            else:
                keyval = False
                
            if self.logfile_path.exists():
                with self.logfile_path.open('r') as yamlfile:
                    log_yaml = yaml.safe_load(yamlfile) # Note the safe_load
                    if keyval:
                        add_to_dict(log_yaml, kwargs['key'], kwargs['value'])
                    else:
                        for key, value in kwargs.items():
                            add_to_dict(log_yaml, key, value)
            else:
                log_yaml = dict()
                if keyval:
                    add_to_dict(log_yaml, kwargs['key'], kwargs['value'])
                else:
                    for key, value in kwargs.items():
                        add_to_dict(log_yaml, key, value)

            with self.logfile_path.open('w') as yamlfile:
                yaml.safe_dump(log_yaml, yamlfile, sort_keys=False)

def add_to_dict(dic, key, val):
    if isinstance(val, list):
        dic[key] = val
    elif isinstance(val, dict):
        dic[key] = val
    elif isinstance(val, Counter):
        dic[key] = dict(val)
    elif isinstance(val, object):
        dic[key] = str(val)
    else:
        dic[key] = val