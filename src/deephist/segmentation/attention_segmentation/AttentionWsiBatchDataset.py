"""
Provide a CustomPatchesDataset to work with WSI and Patches objects.

"""

from contextlib import contextmanager
import random
from typing import List, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.deephist.segmentation.attention_segmentation.models.memory import Memory
from src.pytorch_datasets.patch.patch_from_file import PatchFromFile
from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder
from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import WSIDatasetFolder


class AttentionWsiBatchDataset(Dataset):
    """
    AttentionWsiBatchDataset is a pytorch dataset to provide patch batches of an identical WSI.
    """

    def __init__(self,
                 wsi_dataset: WSIDatasetFolder,
                 patch_batchsize: int,
                 transform: transforms.Compose = None):
        """
        Create a AttentionWsiBatchDataset from a WSI dataset.

        Args:
            wsi_dataset (WSIDatasetFolder): A WSIDataset
                to get wsis from.
            transform (transforms.Compose, optional): Augmentation pipeline. Defaults to None.
        """
    
            
        self.patch_batchsize = patch_batchsize
        self.wsi_dataset = wsi_dataset
        self.transform = transform
                
    def get_label_handler(self) -> LabelHandler:
        """Get the label handler of the WSIs to access map to original labels.

        Returns:
            LabelHandler: LabelHandler that was created during the WSI datset building.
        """
        return self.wsi_dataset.label_handler

    def __len__(self):
        
        n_wsi = len(self.wsi_dataset.wsis)
        
        return n_wsi
    
    def __getitem__(self, idx):
        """
        Draw 'batch_size' patches from 'idx' wsi. Consider class imbalance. 

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
                
        if torch.is_tensor(idx):
            idx = idx.tolist()

        wsi = self.wsi_dataset.wsis[idx]
        
        #draw number batch size patches
        patch_batch = random.sample(wsi.get_patches(), self.patch_batchsize)
        
        # get patch idx in memory
        patch_batch_idx = [Memory.get_memory_idx(patch=patch,
                                          k=self.wsi_dataset.k_neighbours) 
                           for patch in patch_batch]
        
        # get k-neighbourhood patch idxs in memory
        patch_batch_neighbour_idxs = [Memory.get_neighbour_memory_idxs(k=self.wsi_dataset.k_neighbours,
                                                                patch=patch)
                                      for patch in patch_batch]
        
        patch_batch_img, batch_label = zip(*[patch() for patch in patch_batch])
      
        if self.transform is not None:
            patch_batch_img = [self.transform(patch_img) for patch_img in patch_batch_img]

        return patch_batch_img, batch_label, patch_batch_idx, patch_batch_neighbour_idxs
    
    # needed for memory fill-up
    def all_patch_mode(self):
        """Set context to receive all patches from wsi (instead of sampled ones).
        """
        with self.wsi_dataset.all_patch_mode():
            yield(self)
