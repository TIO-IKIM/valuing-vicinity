import PIL
from matplotlib import pyplot as plt
import numpy as np
from numpy import argmax

from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder


def plot_patches(wsi: WSIFromFolder,
                 mode: str = 'worst',
                 top: int = 10,
                 per: str = 'class') -> PIL.Image:
    """
    Creates an PIL Image with a plot of top worst patch predictions per class for one WSI object.

    Returns:
        PIL.Image: Image with worst patches and their predictions
    """
    
    assert(mode in ['worst', 'best']), 'only worst mode is implemented'
    assert(per in ['class']), 'only class mode is implemented'
    
    patch_labels = wsi.get_patch_labels()
    classes = set(patch_labels)
    patch_predictions = wsi.get_patch_predictions()
    scores = [pred[lbl] for pred, lbl in zip(patch_predictions, patch_labels)]
    patches = wsi.get_patches()
    
    worst_indices = dict()
    for cls in classes:
        if mode == 'worst':
            # get all patch indices for current label, where prediction does not match lbl
            tmp_index_and_score = [(idx, scores[idx]) for idx, lbl in enumerate(patch_labels) if lbl == cls and lbl != argmax(patch_predictions[idx])]
        elif mode == 'best':
            tmp_index_and_score = [(idx, scores[idx]) for idx, lbl in enumerate(patch_labels) if lbl == cls and lbl == argmax(patch_predictions[idx])]
        # sort by score ascending:
        tmp_index_and_score.sort(key=lambda x:x[1], reverse= True if mode == 'best' else False)
        # now, worst patches are the first ones with smallest score
        top_indices = [idx for idx, _ in tmp_index_and_score[:top]]
        worst_indices[cls] = top_indices
        
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(6*top, 6*len(classes)))
    
    for cls_pos, cls in enumerate(classes):
        for ptc_pos, patch_idx in enumerate(worst_indices[cls]): 
            ax = fig.add_subplot(len(classes), top, (cls_pos*top) + (ptc_pos+1) , xticks=[], yticks=[])
            
            plt.imshow(np.asarray(patches[patch_idx].get_image()))
            probs = patches[patch_idx].get_prediction()
            cls_prob = probs[patches[patch_idx].get_label()]
            pred_prob = max(probs)
            pred_lbl = wsi.wsi_dataset.label_handler.decode(argmax(probs))
            
            ax.set_title(f"Label: {patches[patch_idx].get_label(org=True)} ({str(round(cls_prob,3))})\n Pred: {pred_lbl} ({str(round(pred_prob,3))})")
        
    fig.canvas.draw()
    img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    
    return img
    
