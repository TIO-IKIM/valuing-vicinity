"""
WSIDatasetFolder specializes the AbstractWSIDataset.
It wraps all WSIs of a prehist-repo preprocessing folder into one WSI dataset.
"""

from __future__ import annotations
from contextlib import contextmanager
import collections
from copy import copy, deepcopy
import logging
import pandas as pd
from pathlib import Path
from random import choices, sample, shuffle
import random
from typing import Dict, List, Tuple

import numpy as np

from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.patch.patch_from_file import PatchFromFile
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder


class WSIDatasetFolder:
    """
    WSIDatasetFolder provides access to all WSis and their patches.
    """

    def __init__(self,
                 dataset_root: str = None,
                 wsi_roots: List[str] = None,
                 embeddings_root: str = None,
                 polygons_root: str = None,
                 representation: str = 'pixel',
                 root_contains_wsi_label: bool = True,
                 patch_sampling: str = None,
                 draw_patches_per_class: int = None,
                 draw_patches_per_wsi: int = None,
                 exclude_patch_class: List[int] = None,
                 include_patch_class: List[int] = None,
                 merge_classes: List[List[int]] = None,
                 prehist_label_map: Dict[str, int] = None,
                 patch_label_type: str = 'patch',
                 patch_label_handler: LabelHandler = None,
                 image_label_handler: LabelHandler = None,
                 mask_label_handler: LabelHandler = None,
                 prehisto_config: Dict[str, int] = None,
                 attention_on: bool = False,
                 embedding_dim = None,
                 k_neighbours = None,
                 multiscale_on: bool = False,
                 sample_size: int = None,
                 fold=0,
                 exp = None
                 ) -> None:
        """
        WSIDatasetFolder provides access to all WSIs and their patches
        for a folder of preprocessed WSIs from the prehist-repo.

        Args:
            dataset_root (str): Provide the root of the folder of the preprocessed WSIs.
                The root can contain the WSI's label.
                Either dataset_root or list_of_wsi_roots must be provided.
            wsi_roots (List[str]): A list of prehist-output WSI folders.
                Either dataset_root or list_of_wsi_roots must be provided.
            embeddings_root (str): Provide a root of a folder with created embeddings.
            polygons_root (str): Provide a root to a folder with wsi polygon annotations.
                Names must be aligend with WSIs.            
            representation (str): Decide wether to use patch 'pixel' or 'embedding'.
            root_contains_wsi_label (bool, optional): Set to True if the root path contains
                the wsi label to provide a WSI label. Defaults to True.
            patch_sampling (str, optional): For all WSIs: To balance patch classes, select between
                over, under, overunder-sampling.
                For oversampling, patches of each class will be drawn as often as the highest
                class occurence (with repetition).
                For undersampling, patches of each class will be drawn as often as the lowest
                class occurence (without repetition).
                For overundersamping, a balanced draw number between highest and lowest class
                occurence is determined by N_draw = (ln(N_highest/N_lowest) / ln(2)) / 2 * N_lowest.
                N_draw patches are then drawn from each class with repetition.
                If None, all patches are included.
            draw_patches_per_class (int, optional): For each WSI, N patches per patch class will be drawn.
                If N exceeds the number of existing patches in one class in one WSI,
                all existing ones are taken. Defaults to None.
            draw_patches_per_wsi (int, optional): For each WSI, N patches will be drawn. Defaults to None.
            exclude_patch_class (list, optional): Define list of (folder) patch
                classes to exlude. Either exclude or include can be provided. Defaults to None.
            include_patch_class (list, optional): Define list of (folder) patch
                classes to include. Either exclude or include can be provided. Defaults to None.
            prehist_label_map ([Dict[str: int], optional): Provides a mapping table from medical label
                to the prehist label. If provided, medical label will be reported. If None,
                prehist labels will be reported.
            patch_label_type (str): Choose between patch, image, mask label to be set as patch label.
            patch_label_handler (LabelHandler): LabelHandler manages patch label mapping.
                To ensure labels are aligned over muliple WSI Datasets,
                create a LabelHandler outside and pass it to all WSI Datasets.
            mask_label_handler (LabelHandler): LabelHandler manages mask label mapping.
                To ensure labels are aligned over muliple WSI Datasets,
                create a LabelHandler outside and pass it to all WSI Datasets.
            image_label_handler (LabelHandler): LabelHandler manages image (wsi) label mapping.
                To ensure labels are aligned over muliple WSI Datasets,
                create a LabelHandler outside and pass it to all WSI Datasets.
            sample_size (int): Number of WSIs to random sample from the dataset. Default None.
            fold (int): Set the fold this wsi dataset belongs to. Default 0.
            embedding_dim (int): Set the dimension of the patch embedding memory. 
                If None, meomory is deactivated. Default to None. Optional.
        """

        assert(patch_sampling in [None, 'over', 'under', 'overunder'])
        assert(patch_label_type in ['patch', 'image', 'mask', 'distribution'])
        assert(representation in ['pixel', 'embedding'])
        assert(bool(exclude_patch_class) + bool(include_patch_class) <= 1)
        assert(bool(wsi_roots) + bool(dataset_root) == 1) # exactly one not None
        assert(bool(draw_patches_per_class) + bool(draw_patches_per_wsi) <= 1) # only one at most not None

        self.patch_sampling = patch_sampling

        if dataset_root is not None:
            self.root_path = Path(dataset_root)
            self.wsi_root_paths = None
        else:
            self.wsi_root_paths = [Path(wsi_root) for wsi_root in wsi_roots]
            self.root_path = None

        self.root_contains_wsi_label=root_contains_wsi_label
      
        if polygons_root is not None:
            self.polygon_paths = Path(polygons_root)
        else:
            self.polygon_paths = None

        self.prehisto_config=prehisto_config
        
        self.draw_patches_per_class = draw_patches_per_class
        self.draw_patches_per_wsi = draw_patches_per_wsi
        self.exclude_patch_class = exclude_patch_class
        self.include_patch_class = include_patch_class
        self.merge_classes = merge_classes
        self.embedding_root = embeddings_root
        self.representation = representation
        
        # attention
        self.attention_on = attention_on
        self.embedding_dim = embedding_dim
        self.k_neighbours = k_neighbours
        
        # multi scale:
        self.multiscale_on = multiscale_on
        
        self.sample_size = sample_size

        self.fold = fold
        
        self.exp = exp
        
        self._drawn_patches = None
        self.metadata = None
        
        self._all_patch_mode = False
        self._embedding_mode = False
        
        # instance to manage labels
        if patch_label_handler is None:
            self.patch_label_handler = LabelHandler(prehist_label_map,
                                                    merge_classes)
        else:
            self.patch_label_handler = patch_label_handler
        if image_label_handler is None:
            self.image_label_handler = LabelHandler()
        else:
            self.image_label_handler = image_label_handler
        if mask_label_handler is None:
            self.mask_label_handler = LabelHandler(merge_classes)
        else:
            self.mask_label_handler = mask_label_handler

        self.set_patch_label_type(patch_label_type)

        self.wsis = self._create_wsis()
        self._collect_patches()
        self._initialize_memory()
        
    def _initialize_memory(self):
        if self.attention_on:
            n_wsis = self.metadata['n_wsis']
            n_patches = self.metadata['n_patches']
            n_x = self.metadata['n_x_max']
            n_y = self.metadata['n_y_max']
            
            self.memory_params = dict(n_x=n_x,
                                      n_y=n_y,
                                      n_w=n_wsis,
                                      n_p=n_patches,
                                      D=self.embedding_dim,
                                      k=self.k_neighbours)
            
            self.metadata['memory'] = self.memory_params

    def set_patch_label_type(self, patch_label_type: str):
        if patch_label_type == 'img':
            self.label_handler = self.image_label_handler
        elif patch_label_type in ['patch','distribution']:
            self.label_handler = self.patch_label_handler
        elif  patch_label_type == 'mask':
            self.label_handler = self.mask_label_handler
        else:
            raise Exception("Label type must be on of [img, patch, mask]")
        self.patch_label_type = patch_label_type

    def _set_metadata(self) -> Dict:
        """
        Collects some metadata about the WSI Dataset

        Returns:
            Dict: meta data dictionary
        """
        metadata = {}
        wsis = []
        
        n_y_max = 0
        n_x_max = 0
        
        for wsi in self.wsis:
            wsis.append({'wsi_name': wsi.name,
                             'wsi_label': wsi.get_label()
            })
            n_y_max = max(wsi.n_patches_row, n_y_max)
            n_x_max = max(wsi.n_patches_col, n_x_max)
               
        metadata['n_y_max'] = n_y_max 
        metadata['n_x_max'] = n_x_max                    
        metadata['wsis'] = wsis
        metadata['n_wsis'] = len(self.wsis)
        metadata['wsi_label_dist'] = dict(collections.Counter(self.get_wsi_labels()))
        metadata['n_patches'] = len(self._all_patches)
        metadata['n_drawn_patches'] = len(self._drawn_patches)
        if self.patch_label_type in ['img', 'patch']:
            metadata['patch_label_dist'] = dict(collections.Counter(self.all_patch_labels))
            metadata['drawn_patch_label_dist'] = dict(collections.Counter(self.drawn_patch_labels))

            metadata['org_patch_label_dist'] = dict(collections.Counter(self.org_all_patch_labels))
            metadata['drawn_org_patch_label_dist'] = dict(collections.Counter(self.drawn_org_patch_labels))

        return metadata

    def _collect_patches(self) -> None:
        """
        Collects patch objects and their labels from the WSIs.
        """
        patch_list = list() # draw affects in WSI show up here
        all_patch_list = list() # includes all patches from each WSI
        for wsi in self.wsis:
            patch_list.extend(wsi.get_patches())
            with wsi.all_patch_mode():
                all_patch_list.extend(wsi.get_patches())
        self._patches = patch_list
        self._all_patches = all_patch_list
        # apply patch sampling
        self._drawn_patches = self._draw_patches()

        self.metadata = self._set_metadata()
        
    def get_patch_dist(self, relative=False) -> pd.DataFrame:
        
        wsi_patch_distributions = list()
        wsi_names = list()
        
        for wsi in self.wsis:
            wsi_patch_dist = wsi.get_patch_distribution(relative=relative)
            wsi_patch_distributions.append(wsi_patch_dist)
            wsi_names.append(wsi.name)
            
        dist_df = pd.DataFrame(wsi_patch_distributions, index=wsi_names)
        dist_df[np.isnan] = 0 
        dist_df['n_patches'], dist_df['n_unique_patches'] = dist_df.pop('n_patches'), dist_df.pop('n_unique_patches')
        dist_df = dist_df.sort_index()
        agg = dist_df.aggregate([np.mean, np.std])
        nonzero = dist_df.astype(bool).sum(axis=0)
        nonzero.name = 'count_wsi'
        if not relative:
            agg2 = dist_df.aggregate([sum])
            agg2 = agg2.rename(index={'sum': 'count_patches'})
            dist_df = dist_df.append(agg2)
        
        dist_df = dist_df.append(agg)
        dist_df = dist_df.append(nonzero)
            
        return dist_df
                        
    def get_patches(self) -> List[PatchFromFile]:
        if self._all_patch_mode:
            return self._all_patches
        else:
            return self._drawn_patches

    def get_patch_embeddings(self) -> List[List[np.ndarray]]:
        if self._all_patch_mode:
            patch_embeddings = []
            for patch in self._all_patches:
                    patch_embeddings.append(patch.get_embedding())
            return patch_embeddings
        else:
            return [patch.get_embedding() for patch in self._drawn_patches]
    
    
    def _create_wsis(self) -> List[WSIFromFolder]:
        """
        Creates all WSI objects.

        Returns:
            List[AbstractWSI]: List of WSI objects
        """

        wsis = list()
        wsi_labels = list()

        if self.wsi_root_paths is  None:
            if self.root_contains_wsi_label:
                self.wsi_root_paths = [d for d in self.root_path.glob('*/*') if d.is_dir()]
            else:
                self.wsi_root_paths = [d for d in self.root_path.iterdir() if d.is_dir()]

        if self.sample_size is not None:
            self.wsi_root_paths = random.sample(population=self.wsi_root_paths,
                                                k=self.sample_size)
            
        for idx, wsi_root in enumerate(self.wsi_root_paths):
            logging.getLogger('exp').info(f"Creating WSI object for {wsi_root.name}")
            wsi_label = wsi_root.parts[-2] if self.root_contains_wsi_label else -1
            
            if self.embedding_root is not None:
                # finde wsi embedding file in embedding_root folder:
                embedding_path = self._get_embedding_path(wsi_name = wsi_root.name)
            else:
                embedding_path = None
                
            if self.polygon_paths is not None:
                matched_paths = [poly_file for poly_file in self.polygon_paths.glob('**/*')
                                 if poly_file.stem == wsi_root.name]
                if len(matched_paths) > 1 or len(matched_paths) == 0:
                    raise Exception(f"Cannot find WSI annotation file for {wsi_root.name}.")
                else:
                    annotation_path = matched_paths[0]
            else: 
                annotation_path = None
                  
            wsi = WSIFromFolder(root=wsi_root,
                                idx=idx,
                                wsi_label= wsi_label,
                                embedding_path=embedding_path,
                                annotation_path = annotation_path,
                                wsi_dataset=self)
            
            ## check unique name constraint:
            if wsi.name in [wsi.name for wsi in wsis]:
                raise Exception(f"WSI name must be unique: Please change name of {wsi.name}")
            
            ## if there are not relevant patches - omit wsi
            if len(wsi.get_patches()) > 0:
                wsis.append(wsi)
                wsi_labels.append(wsi_label)
            else:
                logging.getLogger('exp').info(f"Ommitting wsi {wsi.name} because of no relevant patches")

        # order wsis list by name
        wsis.sort(key=lambda x: x.name)

        return wsis

    def _get_embedding_path(self, wsi_name: str) -> str:
        """
        Get embedding file for a given wsi_name. Embedding file must start with wsi_name

        Args:
            wsi_name (str): Unique wsi name

        Returns:
            str: embedding path 
        """

        embedding_files = [f for f in Path(self.embedding_root).iterdir() if f.name.startswith(wsi_name)]
        if len(embedding_files) > 1:
            raise Exception(f"Unambigous embedding files found: {embedding_files} for wsi {wsi_name}.")
        elif len(embedding_files) == 0:
            raise Exception(f"No embedding files found for wsi {wsi_name}.")
        else:
            return embedding_files[0]
            
            
    def get_wsi_embeddings(self) -> List[List[np.ndarray]]:
        if self._all_patch_mode:
            wsi_embeddings = []
            for wsi in self.wsis:
                with wsi.all_patch_mode():
                    wsi_embeddings.append(wsi.get_embeddings())
            return wsi_embeddings
        else:
            return [wsi.get_embeddings() for wsi in self.wsis]
    
    def get_wsis(self) -> List[WSIFromFolder]: 
        return self.wsis
    
    def get_wsi_labels(self, org=True) -> List[str]:
        return [wsi.get_label(org=org) for wsi in self.wsis]

    def _draw_patches(self) -> List[PatchFromFile]:
        """
        Returns a list of patches for all WSIs in this dataset.
        For oversampling, patches of each class will be drawn as often as the highest
        class occurence (with repetition).
        For undersampling, patches of each class will be drawn as often as the lowest
        class occurence (without repetition).
        For overundersamping, a balanced draw number between highest and lowest class
        occurence is determined by N_draw = (ln(N_highest/N_lowest) / ln(2)) / 2 * N_lowest.
        N_draw patches are then drawn from each class with repetition.
        If sampling is None, all patches are included.
        """

        if self.patch_sampling is not None:
            with self.patch_label_mode('patch'):
                tmp_patch_labels = self.patch_labels
                
                logging.getLogger('exp').info(f"sampling modus: {self.patch_sampling}sampling.")
                label_dist=collections.Counter(tmp_patch_labels)
                logging.getLogger('exp').info(f"pre-class-distribution: {label_dist}")
                idx_sampling = list()

                if self.patch_sampling == 'under':
                    sample_size = min(label_dist.values()) # cls with lowest occurence
                    for cls in set(tmp_patch_labels):
                        idx_sampling.extend(sample([idx for idx, lbl in enumerate(tmp_patch_labels)
                                                    if lbl == cls], sample_size))
                elif self.patch_sampling == 'over':
                    sample_size = max(label_dist.values()) # cls with highest occurence
                    for cls in set(tmp_patch_labels):
                        idx_sampling.extend(choices([idx for idx, lbl in enumerate(tmp_patch_labels)
                                                    if lbl == cls], k=sample_size))
                elif self.patch_sampling == 'overunder':
                    under_sample_size = min(label_dist.values())
                    over_sample_size = max(label_dist.values())
                    _factor = round(np.log(over_sample_size/ under_sample_size) / np.log(2)/2, 2).item()
                    logging.getLogger('exp').info(f"Over-sampling factor: {_factor}")
                    sample_size = round(under_sample_size * np.power(2, _factor)) # balance size
                    
                    if self.exp:
                        self.exp.exp_log(over_under_sampling_factor=_factor)

                    for cls in set(tmp_patch_labels):
                        idx_sampling.extend(choices([idx for idx, lbl in enumerate(tmp_patch_labels)
                                                    if lbl == cls], k=sample_size))
                else:
                    raise Exception("sampling must be one of over, under or overunder")

                # random shuffle of indices
                shuffle(idx_sampling)
                label_dist=collections.Counter([tmp_patch_labels[idx] for idx in idx_sampling])
                logging.getLogger('exp').info(f"post-class-distribution: {label_dist}")

                patches = [self._patches[idx] for idx in idx_sampling]
        else:
            with self.patch_label_mode('patch'):
                label_dist=collections.Counter(self.patch_labels)
                logging.getLogger('exp').info(f"class-distribution: {label_dist}")
            patches =  self._patches

        return patches
    
    @property 
    def patch_labels(self):
        return self._get_patch_labels(self._patches, org=False)
    
    @property 
    def org_patch_labels(self):
        return self._get_patch_labels(self._patches, org=True)
    
    @property 
    def all_patch_labels(self):
        return self._get_patch_labels(self._all_patches, org=False)
    
    @property 
    def org_all_patch_labels(self):
        return self._get_patch_labels(self._all_patches, org=True)
            
    @property 
    def drawn_patch_labels(self):
        return self._get_patch_labels(self._drawn_patches, org=False)
        
    @property 
    def drawn_org_patch_labels(self):
        return self._get_patch_labels(self._drawn_patches, org=True)
        
    def _get_patch_labels(self,
                         patches,
                         org=True):
        return [patch.get_label(org=org) for patch in patches]
    
    def get_patch_labels(self,
                         org=True):
        if self._all_patch_mode:
            if org:
                return self.org_all_patch_labels
            else:
                return self.all_patch_labels
        else:
            if org:
                return self.drawn_org_patch_labels
            else:
                return self.drawn_patch_labels
        
    def get_wsi_dataset_subset(self,
                               wsis: List[WSIFromFolder],
                               dcopy: bool = True):
        """Createas a subset of the wsi data given the wsis. By default, deep-copies the referenced WSIDataset.

        Args:
            wsis (List[WSIFromFolder]): [description]

        Returns:
            [type]: [description]
        """
        assert(all([wsi in self.wsis for wsi in wsis]))
        
        if dcopy:
            wsi_dataset_subset = deepcopy(self)
        else:
            wsi_dataset_subset = copy(self)
        
        # remove wsis that are not in list of wsis
        wsi_names = [wsi.name for wsi in wsis]
        wsi_dataset_subset.wsis = [wsi for wsi in wsi_dataset_subset.wsis if wsi.name in wsi_names]
        #sanity check
        assert len(wsi_dataset_subset.wsis) == len(wsis), "Missing WSI in new data set"
        
        wsi_dataset_subset._collect_patches()
        wsi_dataset_subset._initialize_memory()
        # reset wsi indices
        for idx in range(len(wsi_dataset_subset.wsis)):
            wsi_dataset_subset.wsis[idx].idx = idx
        
        return wsi_dataset_subset

        
    def split_wsi_dataset_by_ratios(self,
                                    split_ratios: List[float] = None
                                    ) -> Tuple[WSIDatasetFolder]:
        """
        Splits WSIDataset whole-slide-wise given a list of ratios.

        Args:
            wsi_dataset (WSIDatasetFolder): WSIDataset
            split_ratios (list, optional): List of split ratios -
                must sum up to 1. Defaults to [0.8, 0.2].

        Returns:
            List[WSIDatasetFolder]: List of WSI Datasets in sequence of split-ratio list
        """

        if split_ratios is None:
            split_ratios = [0.8, 0.2]
        assert(isinstance(split_ratios, list) and sum(split_ratios) == 1)

        logging.getLogger('exp').info("Splitting WSI dataset.")
        wsi_datasets = []
        
        wsi_labels = self.get_wsi_labels()
        wsis = self.get_wsis()
        # to ensure reproducable splits, order wsis by name
        wsis, wsi_labels = (zip(*sorted(zip(wsis, wsi_labels), key=lambda wsi_and_lbl: wsi_and_lbl[0].name)))
        
        wsis_sets = [ [] for _ in range(len(split_ratios)) ] 
        
        label_wsis = dict()
        
        for lbl in set(wsi_labels): 
            lbl_wsis = [wsi for wsi in wsis if wsi.get_label() == lbl]
            # shuffle list
            random.Random(4).shuffle(lbl_wsis)
            # save the shuffled list of wsis 
            label_wsis[lbl] = lbl_wsis
            
            # draw cutoffs for folds
            number_wsis = len(lbl_wsis)
            # find split points
            split_points = list(np.round(np.cumsum(split_ratios[:-1])*number_wsis, 0).astype(int))+[number_wsis]
            split_points = zip(([0]+split_points[:-1]), split_points) # zip with starting index
            lbl_splits = list(split_points)
            
            for set_idx, lbl_split in enumerate(lbl_splits):
                # add val indices per lbl
                lbl_wsis = label_wsis[lbl][lbl_split[0]:lbl_split[1]]
                wsis_sets[set_idx].extend(lbl_wsis)
                
        wsi_datasets = []
        for wsis_set in wsis_sets:
            if len(wsis_set) == 0:
                raise Exception("Splitting results in empty dataset.")
            else:
                # make a copy of wsi images to ensure independence in cv runs
                wsi_datasets.append(self.get_wsi_dataset_subset(wsis_set))
                
        return wsi_datasets

    @contextmanager
    def all_patch_mode(self):
        """Set context to receive all patches from wsi (instead of sampled ones).
        """
        self._all_patch_mode = True
        yield(self)
        self._all_patch_mode = False
        
    @contextmanager
    def patch_label_mode(self, label_type):
        """Set context to receive all patches from wsi (instead of sampled ones).
        """
        assert(label_type in ['patch', 'image', 'mask', 'distribution'])

        default_type = self.patch_label_type
        self.patch_label_type = label_type
        yield(self)
        self.patch_label_type = default_type

    @contextmanager
    def embedding_mode(self):
        """Set context to receive wsi embeddings instead of wsis.
        """
        self._embedding_mode = True
        yield(self)
        self._embedding_mode = False