import logging
from typing import List, Union,Tuple

from matplotlib import pyplot as plt, cm
import numpy as np
from sklearn.manifold import TSNE


def visualize_tsne(embeddings: List[np.ndarray],
                   label_handler,
                   save_to: str = None,
                   labels: List[Union[str, Tuple[str]]] = None,
                   color_labels: List[Union[int, Tuple[int]]] = None):
    """
    Creates tsne embedding plot. Provides one- or two-dimensional lables.
    Second dimension is plotted as symbol.

    Args:
        embeddings (List[np.ndarray]): Embeddings of patches
        save_to (str): Save path of tsne plot
        labels (List[Union[str, Tuple[str]]], optional): Either list of text labels or list of tuples of two text labels. Defaults to None.
        color_labels (List[Union[int, Tuple[int]]], optional): Either list of integer labels or list of tuples of two integer labels. Defaults to None.
    """

    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
        
    cmap = cm.get_cmap("Set3").colors

    if labels is None:
        labels = np.array([0]*len(embeddings))

    tsne = TSNE(n_components=2).fit_transform(embeddings) #.view(-1,1024))

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

        # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if not isinstance(labels[0], tuple):
        # for every class, we'll add a scatter plot separately
        for idx, label in enumerate(set(labels)):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            if color_labels is None:
                color = cmap[idx % 12] # 12 differnet colors at most
            else:
                cmap = label_handler.get_color_array()
                color = cmap[color_labels[indices[0]]] # take first index and get its integer-label

            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, color=color, label=label, alpha=0.5, )

    else:
        ## to plot a tsne of image and patch label concurrently:
        markers = ['o', 'P']
         # for every class, we'll add a scatter plot separately
        for idx, label in enumerate(set(labels)):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            if color_labels is None:
                color = cmap[idx % 12] # 12 differnet colors at most
            else:
                # take first index and first element of tuple and get its integer-label
                color = cmap[color_labels[indices[0]][0]]
            # add a scatter plot with the corresponding color and label

            ax.scatter(current_tx, current_ty, color=color, label=label, marker= markers[color_labels[indices[0]][1]], alpha=0.5, )

    
    if save_to is not None:
        # build a legend using the labels we set previously
        lgd = ax.legend(loc=9, bbox_to_anchor=(0.5,0))

        # finally, show the plot
        logging.getLogger('exp').info(f"Saving tsne result to {save_to}")
        plt.savefig(save_to, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    else:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig


    # scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range