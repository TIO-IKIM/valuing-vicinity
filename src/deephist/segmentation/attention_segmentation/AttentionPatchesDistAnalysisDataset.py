"""
Provide a CustomPatchesDataset to work with WSI and Patches objects.

"""

from contextlib import contextmanager
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.deephist.segmentation.attention_segmentation.models.memory import Memory
from src.pytorch_datasets.patch.patch_from_file import PatchFromFile
from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder
from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import WSIDatasetFolder


class AttentionPatchesDistAnalysisDataset(Dataset):
    """
    AttentionPatchesDistAnalysisDataset is a pytorch dataset to provide patches.
    Here, we include the class distribution of all neighbourhood patches for attention analysis.
    
    """
    @staticmethod
    def collate(batch):
        return NeighbourBatchDistAnalysis(batch)

    def __init__(self,
                 wsi_dataset: Union[WSIDatasetFolder, WSIFromFolder] = None,
                 patches: List[PatchFromFile] = None,
                 transform: transforms.Compose = None):
        """
        Create a AttentionPatchesDataset from a WSI or WSI dataset.

        Args:
            wsi_dataset Union[AbstractWSIDataset, AbstractWSI]: Either a WSI or a WSIDataset
                to get patches from.
            transform (transforms.Compose, optional): Augmentation pipeline. Defaults to None.
        """
        assert patches is not None or wsi_dataset is not None, "Either provide patches or wsi_dataset."

        if patches is None and wsi_dataset is not None:
            self.use_patches = False
        else:
            self.use_patches = True
        
        self.wsi_dataset = wsi_dataset 
        self.patches = patches
        self.transform = transform
        self.n_classes = self.wsi_dataset.label_handler.n_classes
                
    def get_label_handler(self) -> LabelHandler:
        """Get the label handler of the WSIs to access map to original labels.

        Returns:
            LabelHandler: LabelHandler that was created during the WSI datset building.
        """
        return self.wsi_dataset.label_handler

    def __len__(self):
        
        if self.use_patches:
            n_patches = len(self.patches)
        else:
            n_patches = len(self.wsi_dataset.get_patches())
        return n_patches
    
    def __getitem__(self, idx):
                
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # __call__ of patch provides image and label
        if self.use_patches:
            patch = self.patches[idx]
        else:
            if idx >= len(self.wsi_dataset.get_patches()):
                print("stop") 
            patch = self.wsi_dataset.get_patches()[idx]
        
        # get patch idx in memory
        patch_idx = Memory.get_memory_idx(patch=patch,
                                          k=self.wsi_dataset.k_neighbours)
        # get k-neighbourhood patch idxs in memory
        patch_neighbour_idxs = Memory.get_neighbour_memory_idxs(k=self.wsi_dataset.k_neighbours,
                                                                patch=patch)
        
        patch_img, label = patch()
        
        neighbour_patches , _ = patch.get_k_neighbours()
        # initialize empty neighbourhood class distribution 
        neighbour_dist = np.zeros(neighbour_patches.shape + (self.n_classes,))
        
        with self.wsi_dataset.patch_label_mode('distribution'):
            # construct class distribution of neighbourhood
            for x in range(neighbour_patches.shape[0]):
                for y in range(neighbour_patches.shape[0]):
                    patch = neighbour_patches[x,y]
                    if patch is not None:
                        neighbour_dist[x,y] = patch.get_label()
            
        if self.transform is not None:
            patch_img = self.transform(patch_img)

        return patch_img, label, neighbour_dist, patch_idx, patch_neighbour_idxs
    
    @contextmanager
    def all_patch_mode(self):
        """Set context to receive all patches from wsi (instead of sampled ones).
        """
        with self.wsi_dataset.all_patch_mode():
            yield(self)


class NeighbourBatchDistAnalysis:
    def __init__(self, batch) -> None:    
        self.img = torch.stack([item[0] for item in batch])
        self.mask = torch.stack([torch.LongTensor(item[1]) for item in batch])
        self.neighbour_dist = torch.stack([torch.FloatTensor(item[2]) for item in batch])
        self.patch_idx = torch.stack([torch.LongTensor(l) for l in list(zip(*[(item[3]) for item in batch]))])
        self.patch_neighbour_idxs = torch.stack([torch.LongTensor(l) for l in list(zip(*[(item[4]) for item in batch]))]) 
        
    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.mask = self.mask.pin_memory()
        self.neighbour_dist = self.neighbour_dist.pin_memory()
        self.patch_idx = self.patch_idx.pin_memory()
        self.patch_neighbour_idxs = self.patch_neighbour_idxs.pin_memory()
        return {'img': self.img,
                'mask': self.mask,
                'neighbour_dist': self.neighbour_dist,
                'patch_idx': self.patch_idx, 
                'patch_neighbour_idxs': self.patch_neighbour_idxs
                }