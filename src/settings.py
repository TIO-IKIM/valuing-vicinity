
from typing import List, Tuple

from matplotlib import cm
import torch

from src.pytorch_datasets.label_handler import LabelHandler

def get_class_weights(label_handler: LabelHandler) -> torch.Tensor:
    
    class_weights = {'background': 1/2,
                     'cyst': 1/0.125,
                     'tumor_bleeding': 1/0.25,
                     'tumor_necrosis': 1/0.5,
                     'tumor_vital': 1/2,
                     'extrarenal': 1/1,
                     'cortex': 1/0.8,
                     'angioinvasion': 1/0.125,
                     'mark': 1/0.8,
                     'tumor_regression': 1/0.8,
                     'capsule': 1/0.25
                     }
    
    weights = torch.Tensor([class_weights[label_handler.decode(cls)] 
                            for cls in label_handler.internal_classes])
    return weights

def get_custom_palette() -> List[Tuple[float, float, float]]:
    # define colors in RGB / 255 
    
    custom_pal = []
    set3_cmap = cm.get_cmap("Set3").colors
    set1_cmap = cm.get_cmap("Set1").colors
    
    custom_pal.append((0.901, 0.901, 0.901)) #0 background
    custom_pal.append(set1_cmap[1]) #1 cyst
    custom_pal.append(set1_cmap[2]) #2 tumor_bleeding
    custom_pal.append(set1_cmap[3]) #3 tumor_necrosis
    custom_pal.append((0,0,0)) #4 papillom
    custom_pal.append((0,0,0)) #5 lymph node
    custom_pal.append((0,0,0)) #6 diffuse tumor growth in soft tissue
    custom_pal.append(set3_cmap[4]) #7 cortex_atrophy
    custom_pal.append(set1_cmap[4]) #8 tumor_vital
    custom_pal.append(set3_cmap[6]) #9 extrarenal
    custom_pal.append(set3_cmap[7]) #10 cortex
    custom_pal.append((0,0,0)) #11 tissue
    custom_pal.append((0,0,0)) #12 tumor
    custom_pal.append(set1_cmap[5]) #13 angioinvasion
    custom_pal.append((0,0,0)) #14 medullary spindel cell nodule
    custom_pal.append((0.823, 0.294, 0.309)) #15 mark 
    custom_pal.append(set1_cmap[6]) #16 tumor_regression
    custom_pal.append(set1_cmap[7]) #17 capsule
    
    return custom_pal

COLOR_PALETTE = get_custom_palette()    