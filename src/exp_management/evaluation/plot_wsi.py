"""
Visulazations of WSIs and their patch (classification)
"""

import copy

import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from shapely.geometry.multipolygon import MultiPolygon

from src.exp_management.helper import get_concat_h
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder

def plot_wsi(wsi: WSIFromFolder,
             alpha=0.5,
             patch_heatmap=False,
             use_true_mask=False,
             thumbnail_only=False,
             annotation=False
             ):
    """
    Plots wsi from its patches and predictions
    
    Args:
        wsi (AbstractWSI): A WSI object
        mode (str, optional): tbd. Defaults to 'binary_prediction'.
    """
    
    color_array = wsi.wsi_dataset.label_handler.get_color_array(alpha=1)
    #append -1 color as transparent color 
    color_array = np.append(color_array, np.array([[0,0,0,0]]), axis=0)
    
    if len(wsi.wsi_dataset.label_handler.classes) == 2:
        mode = 'binary'
    else:
        mode = 'multiclass'
        classes = list(range(len(wsi.wsi_dataset.label_handler.classes)))
            
    if not thumbnail_only:
        
        # First columns then rows because of OpenSlide order
        n_col = wsi.get_metadata('org_n_tiles_col')
        n_row = wsi.get_metadata('org_n_tiles_row')
       
        if wsi.wsi_dataset.patch_label_type in ['img', 'patch', 'distribution']:
            
            patch_matrix = np.full(
                (n_row, n_col),
                fill_value=-1, dtype=np.float16)

            with wsi.all_patch_mode():
                # Go through the patches and add them to the original array
                for patch in wsi.get_patches():

                    if not patch.has_prediction():
                        continue
                    else:
                        prediction = patch.get_prediction()

                    row_coord = patch.x_coord 
                    col_coord = patch.y_coord

                    if mode == 'binary':
                        patch_matrix[
                            col_coord, row_coord
                        ] = prediction[1] # 2nd class: good for now..
                    elif mode == 'multiclass':
                        patch_matrix[
                            col_coord, row_coord
                        ] = np.argmax(prediction)
            
            if mode == 'binary': 
                color_map = get_cmap("coolwarm_r").copy()
                color_map.set_under("w", alpha=0) # sets all other pixel to transparent
                color_array = color_map.colors
                patch_matrix = (color_map(patch_matrix)[:, :, :] * 255 ).astype(np.uint8)
                # hacky way to set alpha channel for pixel with color.. 
                patch_matrix_alpha_dim = patch_matrix[:,:,3]
                patch_matrix_alpha_dim[patch_matrix_alpha_dim>0]=patch_matrix_alpha_dim[patch_matrix_alpha_dim>0]*alpha
            elif mode == 'multiclass':
                patch_matrix = patch_matrix.astype(np.int8)
                patch_matrix = (color_array[patch_matrix] * 255).astype(np.uint8)
            
            wsi_heatmap = Image.fromarray(patch_matrix, 'RGBA')
            # get thumbnail and overlay it with heatmap
            #wsi_thumbnail = copy.deepcopy(wsi.thumbnail)
            
            wsi_heatmap = wsi_heatmap.resize(wsi.thumbnail.size, Image.ANTIALIAS)    
            #wsi_thumbnail.paste(wsi_heatmap, mask=wsi_heatmap)
        
        elif wsi.wsi_dataset.patch_label_type == 'mask':
            # mask prediction as patch thumbnail exists - construct as image
            wsi_canvas = Image.new('RGB', wsi.thumbnail.size, color='white')
            with wsi.all_patch_mode():
                # Go through the patches and add them to the original array
                for patch in wsi.get_patches():
                    if use_true_mask:
                        patch_thumbnail = patch.true_mask
                    elif not patch_heatmap:
                        if not patch.has_prediction():
                            continue
                        else:
                            patch_thumbnail = patch.get_prediction()
                    else:
                            patch_thumbnail = patch.heatmap
                            
                    row_coord = patch.x_coord 
                    col_coord = patch.y_coord
                    
                    patch_thumbnail_x, patch_thumbnail_y  = patch_thumbnail.size
                                    
                    # now paste patch thumbnail onto wsi thumbnail
                    wsi_canvas.paste(patch_thumbnail,(row_coord*patch_thumbnail_x,
                                                        col_coord*patch_thumbnail_y),
                                        )
                # resize to original thumbnail size
                wsi_thumbnail = wsi_canvas.resize(wsi.thumbnail.size, Image.ANTIALIAS) 
        else:
            raise Exception("Patch label type must either be img, patch or mask")
            
    else:
        wsi_thumbnail = copy.deepcopy(wsi.thumbnail)
    
    if annotation is True: 
        add_polygons(wsi_thumbnail=wsi_thumbnail, 
                    wsi = wsi)
        
    if mode == 'multiclass':
        # add legend:
        legend_img = create_legend_img(colors = [color_array[cls] for cls in classes],
                                       labels = [wsi.wsi_dataset.label_handler.decode(cls) for cls in classes])
        
        wsi_thumbnail = get_concat_h(im1=legend_img,
                                    im2=wsi_thumbnail)

    return wsi_thumbnail

def create_legend_img(colors, labels, loc=3, dpi=150, **kwargs):
    import io
    
    expand=[-5,-5,5,5]
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(len(labels))]
    plt.figure().clear()
    legend = plt.legend(handles, labels, loc=loc, framealpha=1, frameon=True, **kwargs)
    
    fig = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches=bbox, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img


def add_polygons(wsi_thumbnail: Image, 
                 wsi: WSIFromFolder):
    """Adds polygon annotations to wsi thumbnail. Annotations must be loaded to wsi object.

    Args:
        wsi_thumbnail (Image): WSI Thumbnail
        wsi (WSIFromFolder): WSI object with annotations
    """
    
    color_array = wsi.wsi_dataset.label_handler.get_color_array(alpha=0.5, type="int")

    if wsi.annotations is not None:
        cls_draw = ImageDraw.Draw(wsi_thumbnail)

        for _, poly, medical_label in wsi.annotations:
            if isinstance(poly, MultiPolygon):
                polys = list(poly.geoms)
                medical_labels = [medical_label]*len(polys)
            else:
                polys = [poly]
                medical_labels = [medical_label]
            for poly, medical_label in zip(polys, medical_labels):
                label_color = tuple(color_array[wsi.wsi_dataset.label_handler.encode(medical_label, medical=True)])
                labels_ext = list(poly.exterior.coords) 
                labels_int = [list(interior.coords) for interior in poly.interiors] 
                
                cls_draw.line(labels_ext, fill=label_color, width=5, joint="curve")
                for poly_int in labels_int:
                    cls_draw.line(poly_int, fill=label_color, width=5, joint="curve")