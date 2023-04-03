"""
Provides the LabelHandler to manage mapping between original
label and pytorch label
"""
import logging
from typing import Dict, List, Union

import numpy as np
from torch.functional import Tensor


class LabelHandler():
    """
    Collects and stores labels and their decoded pytorch mapping.
    Can be used to decode label to original label.
    Label mapping is successivley build while creating the patches (labels are unkown in advance).
    To ensure a consistent label mapping, mapping is sorted by orginal labels and therefore can
    change while building up the WSI patch objects.
    You must lock the handler after creating all patches to use it for label encoding/decoding.
    """

    def __init__(self,
                 prehist_label_map: Dict[str, int] = None,
                 merge_classes: List[List[int]] = None,
                 include_classes: List[int] = None,
                 ) -> None:
        """
        Handles the mapping between pytorch labels, prehist labels and orginal labels

        Args:
            label_map ([Dict], optional): Provides a mapping tabel between medical labels
            and prehist labels. If not provided, mapping results in prehist labels. Otherwise
            mapping results in medical labels.
        """
        if prehist_label_map is not None:
            self.medical_to_prehist_label_map = prehist_label_map
            self.prehist_to_medical_label_map = {str(v): k for k, v in self.medical_to_prehist_label_map.items()}
            self.medical_label = True
        else:
            self.medical_label = False

        if merge_classes is not None:
            self.merge_classes = {merge_cls[0]:merge_cls[1:] for merge_cls in merge_classes}
            # list of all classes, that are "merged away"
            self.merged_away_classes = [cls for merge_cls in [merge_cls[1:] for merge_cls in merge_classes] for cls in merge_cls] 
            assert(len(self.merged_away_classes) == len(set(self.merged_away_classes))), 'merged classes seem to be double'
            self.replace_class_with = {cls[1]: cls[0] for merge_cls in [list(zip(merge_cls[:1]*len(merge_cls[1:]), merge_cls[1:])) for merge_cls in merge_classes] for cls in merge_cls}
        else:
            self.merge_classes = {}
            self.merged_away_classes = []
            self.replace_class_with = {}
            
        self.classes = list()
        self.np_classes = None
        self.np_classes_idx = None
        self.locked=False
        
        if include_classes is not None:
            self.add_labels(include_classes)

    def add_label(self,
                  org_label: int):
        """
        Add label to labelhandler and assign encoded label - if not yet exists

        Args:
            org_label (int): original (prehist) label
        """
        if org_label not in self.classes and org_label not in self.merged_away_classes:
            logging.getLogger('exp').info(f"Adding new label '{org_label}' to LabelHandler")
            if not self.locked:
                self.classes.append(int(org_label))
                self.classes.sort() # ensure dict labels are in descending order
            else:
                raise Exception("Illegal new label adding after lock")
        
    def add_labels(self,
                   org_labels: int):
        for org_lbl in org_labels:
            self.add_label(org_label=int(org_lbl))
            
            
    def get_color_array(self,
                        alpha: float = None,
                        type: str = 'float') -> np.array:
        from src.settings import COLOR_PALETTE
        
        assert(type in ['float', 'int', 'hex']) , 'type must be one of int, float'
        
        if alpha is not None:
            assert alpha >= 0 and alpha <= 1, 'alpha must be between 0 and 1.'
        
        if type == 'hex':
            internal_color_map = ['#{:02x}{:02x}{:02x}'.format(*tuple(int(col*255) for col in COLOR_PALETTE[cl])) for cl in self.classes]
        else:
            # float or int 
            if alpha is not None:   
                internal_color_map = [COLOR_PALETTE[cl] + (alpha,) for cl in self.classes]
            else:
                internal_color_map = [COLOR_PALETTE[cl] for cl in self.classes]
            if type == 'int':
                internal_color_map = (np.array(internal_color_map) * 255).astype(np.uint8)
        
        return np.array(internal_color_map)
        
    def translate(self,
                  org_label: Union[str, List[str]]) -> Union[str, List[str]]:
        """Translates prehist label to medical label - if mapping table is provided.

        Args:
            org_label (str): Prehist label

        Returns:
            str: medical label, if mapping table is provided. Else prehist label.
        """
        if isinstance(org_label, list):
            if self.medical_label:
                return [self.prehist_to_medical_label_map[str(lbl)] for lbl in org_label]
            else:
                return org_label
        else:
            if self.medical_label:
                return self.prehist_to_medical_label_map[str(org_label)]
            else:
                return org_label
    def reverse_translate(self,
                          medical_label: Union[str, List[str]]) -> Union[str, List[str]]:
        """Translates medical label to prehist label - if mapping table is provided.

        Args:
            medical_label (str): Medical label

        Returns:
            str: prehist label, if mapping table is provided. Else medical label.
        """
        if isinstance(medical_label, list):
            if self.medical_label:
                return [self.medical_to_prehist_label_map[str(lbl)] for lbl in medical_label]
            else:
                return medical_label
        else:
            if self.medical_label:
                return self.medical_to_prehist_label_map[str(medical_label)]
            else:
                return medical_label   
             
    def encode(self,
               org_label: Union[str, np.ndarray],
               medical: bool = False
               ) -> int:
        """
        Translates prehist label into pytorch label by providing the index of
        the org_label in the classes list.
        Can only be used after locking the label handler.
        E.g. for classes [-1, 1, 2] that represent the prehist labels the
        encoding of -1 results into the pytorch label 0.
        If medical is true, org_label expects the medical label used in label map.
        Then, encode translate e.g. tumor (medical label) -> 6 (prehist label) -> 0 (internal label)

        Args:
            org_label (int, nd.array): Original prehist label.

        Returns:
            int: Pytorch label
        """
        if medical:
            org_label = self.medical_to_prehist_label_map[org_label]
            
        if not self.locked:
            raise Exception("LabelHandler is not locked yet. \
                            Please call lock() before to ensure stable encoding")
        
        if isinstance(org_label, str) and org_label not in self.classes:
            raise Exception("Orginal label does not exists in LabelHandler. \
                            Please call add_label beforehand")

        if isinstance(org_label, str):
            return self.classes.index(org_label)
        else:
            #numpy array for masks
            if self.merge_classes is None:
                return self.np_classes_idx[org_label]
            else:
                # first replace org_label, than map to internal class
                return self.np_classes_idx[self.np_replace_classes_idx[org_label]] 
            

    def decode(self,
               label: int
               ) -> str:
        """
        Translates pytorch label to prehist (or medical) label.
        Can only be used after locking the label handler.
        If the LabelHandler knows the medical classes (prehist_label_map provided),
        then the pytorch label is decoded to the medical label.
        E.g. for classes [-1, 1] and prehist mapping {'normal': -1, 'tumor': 1,}
        the pytorch label 0 is firstly mapped to the prehist label -1 and secondly
        mapped to the medical label 'normal'

        Args:
            label (int): Pytorch label

        Returns:
            str: Either prehist label or medical label
        """

        if not self.locked:
            raise Exception("LabelHandler is not locked yet. \
                Please call lock() before to ensure stable decoding")

        if not isinstance(label, (list, np.ndarray, Tensor)):
            return self.translate(org_label=self.classes[label])
        else:
            return [self.translate(self.classes[lbl]) for lbl in label]


    def lock(self):
        """
        Set lock to close the LabelHandler for excepting new labels to ensure stable label encoding
        """
        logging.getLogger('exp').info(f'LabelHandler: {self}')
        self.locked = True
        self.np_classes = np.array(self.classes)
        self.n_classes = len(self.classes)
        self.internal_classes = list(range(self.n_classes))
        # construct idx lookup list for fast encoding
        if len(self.np_classes) > 0:
            self.np_classes_idx = np.array([self.classes.index(org_label) if org_label in self.classes 
                                            else 99 for org_label in range(self.np_classes.min(),
                                                                            self.np_classes.max()+1)])
            # construct merge classes lookup list for fast encoding
            self.np_replace_classes_idx = np.array([self.replace_label(org_label) 
                                                    if self.replace_label(org_label) in self.classes 
                                            else 99 for org_label in range(self.np_classes.min(),
                                                                            self.np_classes.max()+1)])

    def replace_label(self, org_label):
        if org_label in self.replace_class_with.keys():
            return self.replace_class_with[org_label]
        else:
            return org_label
        
    def __str__(self) -> str:
        return str({lbl: self.translate(org_lbl) for lbl, org_lbl in enumerate(self.classes)})
