"""
Provide a CustomPatchesDataset to work with WSI and Patches objects.

"""
from typing import List, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.patch.patch_from_file import PatchFromFile
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder
from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import WSIDatasetFolder

class CustomPatchesDataset(Dataset):
    """
    CustomPatchesDataset is a pytorch dataset to work with WSI patches
    provides by eihter WSI objects or WSIDatset objects.
    """

    @staticmethod
    def collate(batch):
            return Batch(batch)
        
    def __init__(self,
                 wsi_dataset: Union[WSIDatasetFolder, WSIFromFolder] = None,
                 patches: List[PatchFromFile] = None,
                 transform: transforms.Compose = None):
        """
        Create a CustomPatchesDataset from a WSI or WSI dataset.

        Args:
            wsi_dataset Union[AbstractWSIDataset, AbstractWSI]: Either a WSI or a WSIDataset
                to get patches from.
            transform (transforms.Compose, optional): Augmentation pipeline. Defaults to None.
        """
        if patches is not None and wsi_dataset is not None:
            raise Exception("Either provide patches or wsis.")
        elif patches is None and wsi_dataset is not None:
            self.use_patches = False
        else:
            self.use_patches = True
            
        self.wsi_dataset = wsi_dataset
        if self.use_patches:
            self._patches = patches
        else:
            self._patches = self.wsi_dataset.get_patches()
        self.transform = transform

    def get_label_handler(self) -> LabelHandler:
        """Get the label handler of the WSIs to access map to original labels.

        Returns:
            LabelHandler: LabelHandler that was created during the WSI datset building.
        """
        return self.wsi_dataset.label_handler

    def __len__(self):
        
        return len(self._patches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # __call__ of patch provides image and label
        patch_img, label = self._patches[idx]()
        
        if self.transform is not None:
            patch_img = self.transform(patch_img)

        return patch_img, label

class Batch:
    def __init__(self, batch) -> None:    
        self.img = torch.stack([item[0] for item in batch])
        self.mask = torch.stack([torch.LongTensor(item[1]) for item in batch])
        
    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.mask = self.mask.pin_memory()
        return {'img': self.img,
                'mask': self.mask,
                }
