from collections import Iterator
from contextlib import contextmanager
import random
import torch 

from src.deephist.segmentation.attention_segmentation.AttentionPatchesDataset import AttentionPatchesDataset
from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import WSIDatasetFolder


class WsisDataLoader(Iterator):
    
    def __init__(self,
                 wsi_dataset: WSIDatasetFolder,
                 transform,
                 batch_size: int,
                 pin_memory: int,
                 drop_last: int,
                 collate_fn,
                 shuffle=True):
        
        self.dataset = DummyDataset(wsi_dataset,transform=transform) # dummy dataset to mock dataset for memory 
        # wsis
    
        self.dataloaders = []
        
        for wsi_dataset in self.dataset.wsi_datasets:
            
            # create dataloader:
            loader = torch.utils.data.DataLoader(
                wsi_dataset,
                batch_size=int(batch_size),
                shuffle=True, # shuffle within one WSI
                num_workers=0,
                pin_memory=pin_memory,
                drop_last=drop_last, # important in train_loader because of batch norm
                collate_fn=collate_fn
            )
            self.dataloaders.append(loader)
        
        self.shuffle = shuffle 
        self._reset()
        
    def _reset(self):
        self.wsi_idx = 0
        self.total_wsis = len(self.dataloaders)
        # reset next iterator to first wsi
        self.current_dl_iter = None
        self.dl_iters = None
        self.draw_wsi_idxs = list(range(self.total_wsis))
        
    def __next__(self):
        if self.shuffle:
            return self.__random_next()
        else:
            return self.__deterministic_next()
        
    def __random_next(self):
        if self.dl_iters is None:
            self.dl_iters = [iter(self.dataloaders[wsi_idx]) for wsi_idx in range(self.total_wsis)]
        try:
            # draw a wsi
            wsi_idx = random.choice(self.draw_wsi_idxs)
            batch = next(self.dl_iters[wsi_idx])
        except StopIteration as e:
            # then, for this wsi there is no more batch - remove from draw list
            self.draw_wsi_idxs = [idx for idx in self.draw_wsi_idxs if idx != wsi_idx]
            
            if len(self.draw_wsi_idxs) == 0 :
                # last wsi was done
                self._reset()
                raise StopIteration
            # else draw again - but now, draw list is reduced 
            batch = self.__random_next()
        return batch
    
    def __deterministic_next(self):
        if self.current_dl_iter is None:
            self.current_dl_iter = iter(self.dataloaders[0])
        try:
            batch = next(self.current_dl_iter)
        except StopIteration as e:
            self.wsi_idx += 1
            if self.wsi_idx >= self.total_wsis:
                # last wsi was done
                self._reset()
                raise StopIteration
            self.current_dl_iter = iter(self.dataloaders[self.wsi_idx])
            batch = self.__deterministic_next()
        return batch
    
    def __len__(self):
        # return number of total batches
        return sum([len(dl) for dl in self.dataloaders])

class DummyDataset():
    
    def __init__(self, 
                 wsi_dataset: WSIDatasetFolder, 
                 transform):
        self.wsi_dataset = wsi_dataset # needed for memory information
        self.wsis = wsi_dataset.wsis
        
        self.wsi_datasets = []
        
        for wsi in self.wsis:
            self.wsi_datasets.append(AttentionPatchesDataset(wsi_dataset=wsi,
                                                             transform=transform)
                                     )
        
    def __len__(self):
        # return number of total samples
        return sum([len(wsi_dataset) for wsi_dataset in self.wsi_datasets])
    
    @contextmanager
    def all_patch_mode(self):
        # remember each mode of every wsi_dataset
        pre_modes = [wsi._all_patch_mode for wsi in self.wsis]
        for wsi in self.wsis:
            wsi._all_patch_mode = True        
        yield(self)
        # reset old modes
        for wsi, pre_mode in zip(self.wsis, pre_modes):
            wsi._all_patch_mode = pre_mode
            