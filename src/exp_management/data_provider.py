import json
import logging
import math
import random
from pathlib import Path
from typing import List, Tuple, Type, Union

import torch
import yaml
from src.deephist.segmentation.semantic_segmentation.CustomPatchesDataset import \
    CustomPatchesDataset
from src.deephist.segmentation.attention_segmentation.wsis_dataloader import WsisDataLoader
from src.exp_management.experiment.Experiment import Experiment
from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.patch.patch_from_file import PatchFromFile
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder
from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import \
    WSIDatasetFolder


class DataProvider():
    """
    Create Training and Validation DataLoader given the configs. Handles WSI splits.
    """

    def __init__(self,
                 exp: Experiment = None,
                 train_data: str = None,
                 test_data: str = None,
                 embeddings_root: str = None,
                 overlay_polygons: bool = False,
                 image_label_in_path: bool = False,
                 patch_sampling: str = None,
                 patch_label_type: str = 'patch',
                 vali_split: float = None,
                 exclude_classes: List[int] = None,
                 include_classes: List[int] = None,
                 merge_classes: List[List[int]] = None,
                 draw_patches_per_class: int = None,
                 draw_patches_per_wsi: int = None,
                 label_map_file: str = None,
                 batch_size: int = None,
                 val_batch_size: int = None,
                 test_batch_size: int = None,
                 nfold: int = None,
                 mc_runs: int = None,
                 workers: int = 16,
                 distributed: bool = False,
                 gpu: int = None,
                 train_dataset_type = CustomPatchesDataset,
                 inference_dataset_type = None,
                 attention_on: bool = False,
                 embedding_dim: int = None,
                 k_neighbours: int = None,
                 multiscale_on: bool = False,
                 sample_size: int = None
                 ):

        # avoid holdout and cv at the same time
        assert(not (nfold is not None and mc_runs is not None)), 'either cross validation or monte carlo'
        assert(patch_label_type in ['patch', 'image', 'mask', 'distribution'])
        if exclude_classes is not None:
            assert(all([isinstance(ex, str) for ex in exclude_classes]))
        if include_classes is not None:
            assert(all([isinstance(inc, str) for inc in include_classes]))
        
        # detect label mapping
        if label_map_file is not None:
            with Path(label_map_file).open("r") as jfile:
                self.label_map = json.load(jfile)
        else:
            self.label_map = None

        self.exp = exp
        
        self.patch_label_handler = LabelHandler(prehist_label_map = self.label_map,
                                                merge_classes=merge_classes,
                                                include_classes=include_classes)
        self.image_label_handler = LabelHandler(include_classes=[0])
        self.mask_label_handler = LabelHandler(prehist_label_map = self.label_map, 
                                               merge_classes=merge_classes,
                                               include_classes=include_classes)
        
        self.patch_label_handler.lock() # to ensure no new labels are added
        self.image_label_handler.lock() # to ensure no new labels are added
        self.mask_label_handler.lock() # to ensure no new labels are added


        self.image_label_in_path = image_label_in_path
        self.patch_sampling = patch_sampling
        self.patch_label_type = patch_label_type
        self.embeddings_root = embeddings_root
        
        # check for pre-histo config
        try:
            prehisto_config_path = Path(train_data) / "prehisto_config.yml"
            with open(prehisto_config_path) as f:
                # use safe_load instead load
                self.prehisto_config = yaml.safe_load(f)
            if overlay_polygons is True:   
                self.polygons_root = self.prehisto_config['annotation_paths']
            else:
                self.polygons_root = None

            # determine thumbnail adjustment ratio to align patch plotting
            # (rescaling patches with fixed thumbnail size can result in rounding errors)
            expected_patch_img_size = (self.prehisto_config['patch_size'] * self.prehisto_config['downsample']) / 100
            self.exp.args.thumbnail_correction_ratio =  1 / (expected_patch_img_size / round(expected_patch_img_size))
        except Exception as e:
            if overlay_polygons is True:
                raise Exception("Cannot find prehist_config.yml in data root. \
                                Please provide to allow overlay_polygon-option.")
            else:
                self.polygons_root = None
                self.prehisto_config = None
                self.exp.args.thumbnail_correction_ratio = 1

        self.sample_size = sample_size
        self.train_data = train_data
        self.test_data = test_data
        self.val_ratio = vali_split
        self.exclude_patch_class = exclude_classes
        self.include_patch_class = include_classes
        self.merge_classes = merge_classes
        self.draw_patches_per_class = draw_patches_per_class
        self.draw_patches_per_wsi = draw_patches_per_wsi
        
        self.use_wsi_batching = True if exp.args.wsi_batch else False
         
        self.batch_size = batch_size
        
        if val_batch_size is None:
            self.val_batch_size = batch_size
        else:
            self.val_batch_size = val_batch_size
        if test_batch_size is None:
            self.test_batch_size = self.val_batch_size
        else:
            self.test_batch_size = test_batch_size
        self.nfold = nfold
        self.mc_runs = mc_runs

        self.workers = workers
        self.distributed = distributed
        self.gpu = gpu

        self.dataset_type = train_dataset_type
        
        if inference_dataset_type is None:
            self.inference_dataset_type = train_dataset_type
        else:
            self.inference_dataset_type = inference_dataset_type
        
        #attention
        self.attention_on = attention_on
        self.embedding_dim = embedding_dim
        self.k_neighbours = k_neighbours

        #multiscale
        self.multiscale_on = multiscale_on
        
        #augmentation
        self._set_augmentation()

        # on reload with provided test data, do not prepare train data
        if self.exp.args.reload_model_folder is None or self.exp.args.test_data is None:
            self._setup_data()
            self.holdout_set = self._set_holdout_set()
            self.train_set = self._set_train_set()
            self.cv_set = self._set_cv_set()
            
        # on inference use test_data
        elif self.exp.args.test_data is not None:
            self.test_wsi_dataset = self._set_test_wsis()
        else:
            raise Exception("Please provide a test_data path to do inference on.")
        
        if patch_label_type in ['patch', 'distribution']:
            self.label_handler = self.patch_label_handler
            self.number_classes = len(self.label_handler.classes)
        elif patch_label_type == 'image':
            self.label_handler = self.image_label_handler
            self.number_classes = len(self.label_handler.classes)
        elif patch_label_type == 'mask':
            self.label_handler = self.mask_label_handler
            self.number_classes = len(self.mask_label_handler.classes)

    def _set_augmentation(self):
        if self.train_data is not None:
            self.train_aug_transform, self.vali_aug_transform = self.exp.get_augmention()

    def _setup_data(self):
        if self.train_data is not None:

            self.wsi_dataset = WSIDatasetFolder(dataset_root=self.train_data,
                                                embeddings_root=self.embeddings_root,
                                                polygons_root=self.polygons_root,
                                                root_contains_wsi_label=self.image_label_in_path,
                                                patch_sampling=self.patch_sampling,
                                                exclude_patch_class=self.exclude_patch_class,
                                                include_patch_class=self.include_patch_class,
                                                merge_classes = self.merge_classes,
                                                patch_label_handler=self.patch_label_handler,
                                                image_label_handler=self.image_label_handler,
                                                mask_label_handler=self.mask_label_handler,
                                                patch_label_type=self.patch_label_type,
                                                draw_patches_per_class=self.draw_patches_per_class,
                                                draw_patches_per_wsi=self.draw_patches_per_wsi,
                                                prehisto_config=self.prehisto_config,
                                                attention_on=self.attention_on,
                                                embedding_dim=self.embedding_dim,
                                                k_neighbours=self.k_neighbours,
                                                multiscale_on=self.multiscale_on,
                                                sample_size=self.sample_size,
                                                exp=self.exp)
        

    def _set_cv_set(self):
        if self.nfold is not None:
            return CvSet(wsi_dataset=self.wsi_dataset,
                         data_provider=self,
                         nfold=self.nfold,
                         val_ratio=self.val_ratio)
        elif self.mc_runs is not None and self.val_ratio is not None:
            return McSet(wsi_dataset=self.wsi_dataset,
                         data_provider=self,
                         runs=self.mc_runs,
                         val_ratio=self.val_ratio)
        else:
            return None

    def _set_holdout_set(self):
        if self.train_data is not None and \
           self.val_ratio is not None and \
           self.nfold is None and \
           self.mc_runs is None:
               
            train_wsi_dataset, vali_wsi_dataset = self.wsi_dataset.split_wsi_dataset_by_ratios(split_ratios= [1-self.val_ratio,
                                                                                                              self.val_ratio])
            test_wsi_dataset = self._set_test_wsis()
            
            holdout_set =  HoldoutSet(train_wsi_dataset=train_wsi_dataset,
                                      vali_wsi_dataset=vali_wsi_dataset,
                                      test_wsi_dataset=test_wsi_dataset,
                                      data_provider=self)

            return holdout_set
        else:
            return None

    def _set_train_set(self):
        if self.train_data is not None and \
           self.val_ratio is  None and \
           self.nfold is None:
               return self.wsi_dataset
        else:
            return None

    def _set_test_wsis(self):
        if self.test_data is not None:
            test_dataset = WSIDatasetFolder(dataset_root=self.test_data,
                                            embeddings_root=self.embeddings_root,
                                            polygons_root=self.polygons_root,
                                            root_contains_wsi_label=self.image_label_in_path,
                                            exclude_patch_class=self.exclude_patch_class,
                                            include_patch_class=self.include_patch_class,
                                            merge_classes = self.merge_classes,
                                            patch_label_handler=self.patch_label_handler,
                                            image_label_handler=self.image_label_handler,
                                            mask_label_handler=self.mask_label_handler,
                                            patch_label_type=self.patch_label_type,
                                            prehisto_config=self.prehisto_config,
                                            attention_on=self.attention_on,
                                            embedding_dim=self.embedding_dim,
                                            k_neighbours=self.k_neighbours,
                                            multiscale_on=self.multiscale_on,
                                            exp=self.exp)
            
            return test_dataset
        else:
            return None

    def get_train_loader(self):
        # Data loading code
        if self.train_set is not None:

            train_dataset = self.dataset_type(wsi_dataset=self.train_set,
                                              transform=self.train_aug_transform,
                                              patch_label_type=self.patch_label_type
                                              )

            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                logging.getLogger('exp').info(f"GPU {self.gpu}")
            else:
                train_sampler = None

            logging.getLogger('exp').info(f"Train Data set length {len(train_dataset)}")
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=int(self.batch_size),
                                                       shuffle=True, #(train_sampler is None),
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       sampler=train_sampler,
                                                       drop_last=True,
                                                       collate_fn=self.dataset_type.collate,
                                                       )

            return train_loader, train_sampler, train_dataset.get_label_handler()

    def get_test_loader(self):

        if self.test_set is not None:

            test_dataset = self.dataset_type(wsi_dataset=self.test_set,
                                             transform=self.vali_aug_transform,
                                             patch_label_type=self.patch_label_type
                                             )
            # validation loader does not have distributed loader. So, each GPU runs a full validation run. However, only rank 0 prints
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.test_batch_size,
                                                      shuffle=False,
                                                      num_workers=self.workers,
                                                      pin_memory=True,
                                                      collate_fn=self.dataset_type.collate,
                                                      )

            return test_loader, test_dataset.get_label_handler()


    def get_wsi_loader(self,
                       wsi: Union[WSIFromFolder, WSIDatasetFolder] = None,
                       patches: List[PatchFromFile] = None,
                       dataset_type: Type[torch.utils.data.Dataset] = None
                       ) -> torch.utils.data.DataLoader:
        """Generates a DataLoader from a WSI object (generated by pre-histo) or a WSI dataset object.
           Used for inference.

        Returns:
            torch.utils.data.DataLoader: DataLoader
        """
        if dataset_type is None:
            dataset_type = self.inference_dataset_type
            
        wsi_pytorch_dataset = dataset_type(wsi_dataset=wsi,
                                          patches=patches,
                                          transform=self.vali_aug_transform)
        # validation loader does not have distributed loader. So, each GPU runs a full validation run. However, only rank 0 prints
        wsi_loader = torch.utils.data.DataLoader(wsi_pytorch_dataset,
                                                 batch_size=self.val_batch_size,
                                                 shuffle=False, # IMPORTANT, do not change for CLAM / and for patch inference order
                                                 num_workers=self.workers,
                                                 pin_memory=True,
                                                 collate_fn=dataset_type.collate,
                                                 #persistent_workers=True if self.workers > 0 and self.nfold is None else False
                                                 )

        return wsi_loader


class HoldoutSet():

    def __init__(self,
                 train_wsi_dataset: WSIDatasetFolder,
                 vali_wsi_dataset: WSIDatasetFolder,
                 data_provider: DataProvider,
                 test_wsi_dataset: WSIDatasetFolder = None,
                 fold: int = None
                ):
        self.metadata = dict()
        
        self.train_wsi_dataset = train_wsi_dataset
        self.vali_wsi_dataset = vali_wsi_dataset
        self.test_wsi_dataset = test_wsi_dataset
        self.data_provider = data_provider
        self.fold = fold
        
        self._create_loader()


    def _create_loader(self):
        
        # log all metadata of wsi dataset
        self.metadata['train_wsi_dataset'] = self.train_wsi_dataset.metadata
        self.metadata['vali_wsi_dataset'] = self.vali_wsi_dataset.metadata

        logging.getLogger('exp').info(f"Train Data set length {self.train_wsi_dataset.metadata['n_drawn_patches']}"
            f" patches from {self.train_wsi_dataset.metadata['n_wsis']} wsis")
        logging.getLogger('exp').info(f"Vali Data set length {self.vali_wsi_dataset.metadata['n_drawn_patches']}"
            f" patches from {self.vali_wsi_dataset.metadata['n_wsis']} wsis")
        
        if self.test_wsi_dataset is not None:
            self.metadata["test_wsi_dataset"] = self.test_wsi_dataset.metadata
            
            logging.getLogger('exp').info(f"Test Data set length {self.test_wsi_dataset.metadata['n_drawn_patches']}"
                f" patches from {self.test_wsi_dataset.metadata['n_wsis']} wsis")
              
        if self.data_provider.use_wsi_batching:
            self.train_loader = WsisDataLoader(wsi_dataset=self.train_wsi_dataset,
                                               transform=self.data_provider.train_aug_transform,
                                               batch_size=int(self.data_provider.batch_size),
                                               pin_memory=True,
                                               drop_last=True, # important in train_loader because of batch norm
                                               collate_fn=self.data_provider.dataset_type.collate)
            
            self.big_train_loader = WsisDataLoader(wsi_dataset=self.train_wsi_dataset,
                                               transform=self.data_provider.train_aug_transform,
                                               batch_size=int(self.data_provider.val_batch_size),
                                               pin_memory=True,
                                               drop_last=False, # important in train_loader because of batch norm
                                               collate_fn=self.data_provider.dataset_type.collate)
            
            self.vali_loader = WsisDataLoader(wsi_dataset=self.vali_wsi_dataset,
                                              transform=self.data_provider.vali_aug_transform,
                                              batch_size=int(self.data_provider.val_batch_size),
                                              pin_memory=True,
                                              drop_last=False,
                                              collate_fn=self.data_provider.dataset_type.collate)
            
            if self.test_wsi_dataset is not None: 
                self.test_loader = WsisDataLoader(wsi_dataset=self.test_wsi_dataset,
                                                  transform=self.data_provider.vali_aug_transform,
                                                  batch_size=int(self.data_provider.test_batch_size),
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  collate_fn=self.data_provider.dataset_type.collate)
        else:
            self.train_torch_dataset = self.data_provider.dataset_type(
                wsi_dataset=self.train_wsi_dataset,
                transform=self.data_provider.train_aug_transform)
            
            self.vali_torch_dataset = self.data_provider.dataset_type(
                wsi_dataset=self.vali_wsi_dataset,
                transform=self.data_provider.vali_aug_transform)
            
            self.train_loader = torch.utils.data.DataLoader(
                self.train_torch_dataset,
                batch_size=int(self.data_provider.batch_size),
                shuffle=True,
                num_workers=self.data_provider.workers,
                pin_memory=True,
                drop_last=True, # important in train_loader because of batch norm
                collate_fn=self.data_provider.dataset_type.collate
            )
            
            # used to quickly fill memory with big batch size (no drop_last)
            self.big_train_loader = torch.utils.data.DataLoader(
                self.train_torch_dataset,
                batch_size=int(self.data_provider.val_batch_size),
                num_workers=self.data_provider.workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=self.data_provider.dataset_type.collate
            )

            self.vali_loader = torch.utils.data.DataLoader(
                self.vali_torch_dataset,
                batch_size=int(self.data_provider.val_batch_size),
                num_workers=self.data_provider.workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=self.data_provider.dataset_type.collate
            )
        
            if self.test_wsi_dataset is not None:
                                
                self.test_torch_dataset = self.data_provider.dataset_type(
                    wsi_dataset=self.test_wsi_dataset,
                    transform=self.data_provider.vali_aug_transform)
               
                self.test_loader = torch.utils.data.DataLoader(
                    self.test_torch_dataset,
                    batch_size=int(self.data_provider.test_batch_size),
                    num_workers=self.data_provider.workers,
                    pin_memory=True,
                    collate_fn=self.data_provider.dataset_type.collate
                )

class McSet():
    """
    Create a Monte-Carlo data set
    """
    def __init__(self,
                 data_provider: DataProvider,
                 wsi_dataset: WSIDatasetFolder,
                 runs: int = 3,
                 val_ratio: float = 0.3):

        self.runs = runs
        self.val_ratio = val_ratio
        self.wsi_dataset= wsi_dataset
        self.data_provider = data_provider
        self.holdout_sets = self._create_mc_set()

    def _create_mc_set(self):
        logging.getLogger('exp').info("Creating Monte-Carlo-Sets")
        runs = []
        wsi_labels = self.wsi_dataset.get_wsi_labels()
        wsis = self.wsi_dataset.get_wsis()

        train_wsis = [ [] for _ in range(self.runs) ] 
        val_wsis = [ [] for _ in range(self.runs) ] 
        
        for lbl in set(wsi_labels):
            lbl_wsis = [wsi for wsi in wsis if wsi.get_label() == lbl]
            # shuffle list
      
            for run in range(self.runs):
                random.Random(5).shuffle(lbl_wsis)

                cutoff = int(round(self.val_ratio*len(lbl_wsis)))
                val_wsis[run].extend(lbl_wsis[:cutoff])
                train_wsis[run].extend(lbl_wsis[cutoff:])
                
        for run in range(self.runs):
            logging.getLogger('exp').info(f"Monte-Carlo-Set {run}")
            runs.append(HoldoutSet(
                train_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(train_wsis[run]),
                vali_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(val_wsis[run]),
                data_provider= self.data_provider,
                fold = run))

        #self.data_provider.exp.exp_log(splitting=fold_splits)
        return runs


class CvSet():
    """
    Create a K-Fold-Cross-Validation data set
    """

    def __init__(self,
                 data_provider: DataProvider,
                 wsi_dataset: WSIDatasetFolder,
                 nfold: int = 3,
                 val_ratio: float = 0.1):
        self.nfold = nfold
        self.val_ratio = val_ratio
        self.wsi_dataset= wsi_dataset
        self.data_provider = data_provider
        self.holdout_sets = self._create_kfold_set()

    def _create_kfold_set(self):
        """
        Create stratified cross validation datasets.
        """
        logging.getLogger('exp').info("Creating Cross-Validation-Sets")
        folds = []
        wsi_labels = self.wsi_dataset.get_wsi_labels()
        wsis = self.wsi_dataset.get_wsis()

        label_splits = dict()
        label_wsis = dict()
        for lbl in set(wsi_labels):
            lbl_wsis = [wsi for wsi in wsis if wsi.get_label() == lbl]
            # shuffle list
            random.Random(5).shuffle(lbl_wsis)
            # save the shuffled list of wsis
            label_wsis[lbl] = lbl_wsis

            # draw cutoffs for folds
            number_wsis = len(lbl_wsis)
            # find split points
            split_points = [int(round(fold / self.nfold * number_wsis)) for fold in list(range(1,self.nfold+1))]
            split_points = zip(([0]+split_points[:-1]), split_points) # zip with starting index
            label_splits[lbl] = list(split_points)

        fold_splits = dict()
        
        all_test_wsis = []
        
        for fold in range(self.nfold):
            logging.getLogger('exp').info(f"CV-Fold {fold}")
            
            train_wsis = []
            val_wsis = []
            test_wsis = []
            fold_splits[fold] = dict()
            fold_splits[fold]['train'] = dict()
            fold_splits[fold]['val'] = dict()
            fold_splits[fold]['test'] = dict()

            for lbl, lbl_split in label_splits.items():
                # add test indices per lbl
                lbl_test_wsis = label_wsis[lbl][lbl_split[fold][0]:lbl_split[fold][1]]
                test_wsis.extend(lbl_test_wsis)
                # all other incides per lbl go into the train set
                lbl_train_wsis = [lbl_idx for idx, lbl_idx in enumerate(label_wsis[lbl])
                                   if idx not in range(lbl_split[fold][0],lbl_split[fold][1])]
                
                # now split a validation set using val-split:
                if self.val_ratio is not None:
                    lbl_val_wsis = random.sample(population=lbl_train_wsis, k = math.ceil(len(lbl_train_wsis) * self.val_ratio))
                    val_wsis.extend(lbl_val_wsis)
                else:
                    # in special case for CY16, we want to use the same WSIs for validation/testing due to little data
                    lbl_val_wsis = lbl_test_wsis
                    val_wsis.extend(lbl_val_wsis)
                    
                train_wsis.extend([wsi for wsi in lbl_train_wsis if wsi not in lbl_val_wsis])

                fold_splits[fold]['train'][lbl] = [wsi.name for wsi in lbl_train_wsis]
                fold_splits[fold]['val'][lbl] = [wsi.name for wsi in lbl_val_wsis]
                fold_splits[fold]['test'][lbl] = [wsi.name for wsi in lbl_test_wsis]
            
            folds.append(HoldoutSet(
                train_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(train_wsis),
                test_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(test_wsis),
                vali_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(val_wsis),
                data_provider= self.data_provider,
                fold = fold))

            # collect val incides to assert uniqueness later
            all_test_wsis.extend(test_wsis)
            # no val/test data in train data
            assert(all(val_wsi not in train_wsis for val_wsi in val_wsis))
            assert(all(test_wsi not in train_wsis for test_wsi in test_wsis))

        # every wsi only once in test set
        assert(len(all_test_wsis) == len(set(all_test_wsis)))
        assert(len(all_test_wsis) == len(wsi_labels))

        self.data_provider.exp.exp_log(splitting=fold_splits)
        return folds


def split_wsi_dataset_root(dataset_root: str,
                           val_ratio: List[int],
                           image_label_in_path: bool
                           ) -> Tuple[List[str]]:
    """
    Providing a WSI dataset root, the WSI roots are split into N lists of WSI roots.

    Args:
        dataset_root (str): root of prehist-outcome WSI dataset
        val_ratio (float): validation ratio
        image_label_in_path (bool): Weither or not the image label is included in the WSI path

    Returns:
        Tuple[List[str]]: Returns a tuple of lists of WSI roots.
    """

    dataset_path = Path(dataset_root)

    if image_label_in_path:
        wsi_root_paths = [d for d in dataset_path.glob('*/*') if d.is_dir()]
    else:
        wsi_root_paths = [d for d in dataset_path.iterdir() if d.is_dir()]

    n_wsis = len(wsi_root_paths)
    random.Random(10).shuffle(wsi_root_paths)
    train_vali_split_index = round(n_wsis * (1-val_ratio))
    train_roots = wsi_root_paths[:train_vali_split_index]
    val_roots = wsi_root_paths[train_vali_split_index:]

    assert(len(train_roots) > 0 and len(val_roots) > 0)

    return train_roots, val_roots
