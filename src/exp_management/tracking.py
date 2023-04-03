"""
Provides functionality to track metrics
and visualize artefacts
"""
from collections import deque, defaultdict
import datetime
import logging
from math import atan2, pi
import scipy
from pathlib import Path
import random
from typing import List
import numpy as np
from numpy import linalg as LA
import time
import yaml

import cv2
import matplotlib.pyplot as plt
import matplotlib
import PIL
import torch
import torch.nn as nn


from src.exp_management.helper import get_concat_v, getcov
from src.exp_management.evaluation.tsne import visualize_tsne
from src.exp_management.evaluation.plot_patches import plot_patches
from src.exp_management.evaluation.plot_wsi import plot_wsi
from src.exp_management.evaluation.confusion_matrix import plot_confusion_matrix_from_data
from src.exp_management.evaluation.roc import plot_roc_curve
from src.exp_management.evaluation.prob_hist import plot_prob_hist, plot_score_hist, plot_score_2dhist
from src.pytorch_datasets.label_handler import LabelHandler


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    @ heavily adapted from up-detr
    """

    def __init__(self, 
                 window_size=20, 
                 to_tensorboard=True,
                 type='avg'):

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.type = type
        self.to_tensorboard = to_tensorboard

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return None

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    @property
    def log(self):
         return round(getattr(self, self.type),2)

    def __str__(self):
        return f"{self.type}: {round(getattr(self, self.type),2)}"


class MetricLogger(object):
    """
    @ heavily adapted from up-detr
    """
    def __init__(self, delimiter="\t", tensorboard_writer=None, args=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.tensorboard_writer = tensorboard_writer
        self.args = args
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int, tuple))
            # if tuple - 2nd elment is count value
            if isinstance(v, tuple):
                assert(len(v) == 2)
                if isinstance(v[0], torch.Tensor):
                    v = list(v)
                    v[0] = v[0].item()
                self.meters[k].update(v[0], v[1])
            else:
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def global_str(self):
        loss_str = []
        max_metrics = 3
        counter = 0
        for name, meter in self.meters.items():
            if counter == max_metrics:
                break
            counter += 1
            loss_str.append(
                "{}: {}".format(name, round(meter.global_avg, 2))
            )
        return self.delimiter.join(loss_str)

    def __str__(self):
        loss_str = []
        max_metrics = 10
        counter = 0
        for name, meter in self.meters.items():
            if counter == max_metrics:
                break
            counter += 1
            loss_str.append(
                "{}: {}".format(name, round(meter.avg, 2))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def send_meters_to_tensorboard(self, step):
        if self.tensorboard_writer is None:
            Warning("No tensorboard writer attached to MetricLogger")
            return
        for name, meter in self.meters.items():
            if meter.to_tensorboard:
                if meter.global_avg is not None:
                    name = name.replace("_slash_", "/") # hack to handle subgrouping in tensorboard
                    self.tensorboard_writer.add_scalar(tag=name,
                                                       scalar_value=meter.global_avg,
                                                       global_step=step)


    def log_every(self, 
                  iterable, 
                  print_freq,
                  epoch=None, 
                  header=None, 
                  phase='train'):
        
        if len(iterable) == 0:
            raise Exception("Zero iterations in dataloader: probably drop_last = True and n_samples < batch_size")

        self.add_meter(f'total_time_{phase}', SmoothedValue(window_size=1, type='avg'))

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(type='avg')
        data_time = SmoothedValue(type='avg')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            # print every print_freq steps
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.getLogger('exp').info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logging.getLogger('exp').info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.getLogger('exp').info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        if phase == 'train':
            self.update(total_time_train=total_time)
        elif phase == 'vali':
            self.update(total_time_vali=total_time)
        self.send_meters_to_tensorboard(step=epoch)


class Visualizer():
    """ Visualizer wrapper based on Tensorboard.

    Returns:
        Visualizer: Class file.
    """
    def __init__(self, 
                 writer = None, 
                 save_to_folder: bool = False):
        self.writer = writer
        self.save_to_folder = save_to_folder

    def write_images(self,
                     image_tensor,
                     tag=None,
                     sample_size=None,
                     epoch=None):
        """Writes an image tensor to the current tensorboard run. Select sample size to draw samples from the tensor.
        Args:
            image_tensor ([type]): [description]
            label_tensor ([type], optional): [description]. Defaults to None.
            tag ([type], optional): [description]. Defaults to None.
            sample ([type], optional): [description]. Defaults to None.
        """

        if self.writer is None:
            return

        batch_size = image_tensor.size(0)

        if sample_size is not None:
            if sample_size > batch_size:
                sample_size = batch_size

            perm = torch.randperm(batch_size)
            idx = perm[:sample_size]
            image_samples = image_tensor[idx]
        else:
            image_samples = image_tensor

        if self.writer is not None:
            self.writer.add_images(tag,
                                image_samples,
                                epoch)
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")


    def compare_images(self,
                    image_tensor1,
                    image_tensor2,
                    tag=None,
                    sample_size=None,
                    epoch=None):
        """Compares two image tensors by writing to the current tensorboard run. Select sample size to draw samples from the tensor.
        Args:
            image_tensor ([type]): [description]
            label_tensor ([type], optional): [description]. Defaults to None.
            tag ([type], optional): [description]. Defaults to None.
            sample ([type], optional): [description]. Defaults to None.
        """

        if self.writer is None:
            return

        batch_size = image_tensor1.size(0)

        if sample_size is not None:
            if sample_size > batch_size:
                sample_size = batch_size

            perm = torch.randperm(batch_size)
            idx = perm[:sample_size]
            image_samples1 = image_tensor1[idx]
            image_samples2 = image_tensor2[idx]
        else:
            image_samples1 = image_tensor1
            image_samples2 = image_tensor2

        if self.writer is not None:

            self.writer.add_images(tag + " Tensor 1", image_samples1, epoch)
            self.writer.add_images(tag + " Tensor 2", image_samples2, epoch)
            
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")

        
    def plot_position_embeddings(self,
                                 tag,
                                 model,
                                 epoch=None):
        
        if hasattr(model, 'msa') and model.msa.learn_pos_encoding:
            pos_h=model.msa.pos_h
            pos_w=model.msa.pos_w
            pos_h = pos_h.squeeze()
            pos_w = pos_w.squeeze()
            
            k2 = pos_h.shape[0]
            k = (k2-1)//2
                    
            fig = plt.figure(figsize=(10,5))
            
            cos = nn.CosineSimilarity(dim=0)
            cos_h_i = []
            cos_w_i = []
        
            for i in range(pos_h.shape[0]):
                cos_h = []
                cos_w = []
                for j in range(pos_h.shape[0]):
                    cos_h.append(cos(pos_h[i], pos_h[j]).detach().cpu().item())
                    cos_w.append(cos(pos_w[i], pos_w[j]).detach().cpu().item())
                
                cos_w_i.append(cos_h)
                cos_h_i.append(cos_w)
            
            # horizontal neighbours
            ax1 = fig.add_subplot(1,2,1)
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.set_yticks([])
            ax1.set_xticks(list(range(0,k2)))
            ax1.set_xticklabels([str(i) for i in range(-k,k+1)])
            ax1.title.set_text('Height positional encodings')
            plt1 = ax1.imshow(np.array(cos_h_i).reshape(k2,k2), cmap='hot', interpolation='nearest')
            
            # vertical neighbours
            ax2 = fig.add_subplot(1,2,2)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.yaxis.set_ticks_position('right')
            ax2.set_xticks([])
            ax2.set_yticks(list(range(0,k2)))
            ax2.set_yticklabels([str(i) for i in range(-k,k+1)])
            ax2.title.set_text('Width positional encodings')
            ax2.imshow(np.array(cos_w_i).reshape(k2,k2), cmap='hot', interpolation='nearest')

            plt.colorbar(plt1, ax=fig.get_axes())

            if self.writer is not None:
                self.writer.add_figure(tag, fig, epoch)
                
            if self.save_to_folder:
                None
                #raise Exception("Not implemented yet.")


    def plot_tsne(self,
                  tag,
                  wsi_dataset,
                  memory,
                  sample_size,
                  label_handler,
                  epoch=None):
        
        patches = wsi_dataset.get_patches()
        rnd_idxs = random.sample(range(len(patches)), k=min(sample_size, len(patches)))
        sample_patches = [ptc for idx, ptc in enumerate(patches) if idx in rnd_idxs]
        sample_embeddings = memory.get_embeddings(patches=sample_patches)

        # sanity check: all embeddings should have values:
        assert torch.all(torch.max(sample_embeddings, dim=1)[0] != 0).item(), 'all embeddings should have values'
        
        with wsi_dataset.patch_label_mode('patch'):
            sample_labels = [patch.get_label() for patch in sample_patches]
            sample_org_labels = [patch.get_label(org=True) for patch in sample_patches]

        if sample_embeddings.is_cuda:
            sample_embeddings = sample_embeddings.cpu()
            
        tsne_fig = visualize_tsne(embeddings=sample_embeddings,
                                   color_labels=sample_labels,
                                   labels=sample_org_labels,
                                   label_handler=label_handler)
        
        if self.writer is not None:
            self.writer.add_figure(tag, tsne_fig, epoch)
            
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")

    def plot_samples(self,
                     tag: str,
                     images: np.ndarray,
                     labels: List[str] = None,
                     sample_size = None,
                     row_size = None,
                     col_size = None,
                     epoch=None):
        '''
        Generates matplotlib Figure with images
        and labels from a batch alongside the actual label.
        '''
        if sample_size is not None:
            col_size = sample_size
            row_size = 1
       
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(6*col_size, 6*row_size))
        for col_idx in np.arange(col_size):
            for row_idx in np.arange(row_size):
                idx = row_idx * col_size + col_idx
                if idx >= len(images):
                    break
                ax = fig.add_subplot(row_size, col_size, idx+1, xticks=[], yticks=[])
                i = images[idx]
                plt.imshow(np.transpose(i, (1, 2, 0)))
                if labels is not None:
                    ax.set_title(f"Label: {labels[idx]}")
            
        if self.writer is not None:
            self.writer.add_figure(tag, fig, epoch)
        
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")

    def plot_masks(
        self,
        tag,
        masks: np.ndarray,
        label_handler: LabelHandler,
        row_size=2,
        col_size=6,
        epoch=None
        ) -> None:
        
        sample_size = row_size * col_size
        
        color_array = label_handler.get_color_array()
        masks_rbg = list()
        for mask in masks[:min(sample_size, len(masks))]:
            masks_rbg.append(np.transpose(color_array[mask], (2, 0, 1)))
        
        self.plot_samples(tag=tag,
                          images=masks_rbg,
                          row_size=row_size,
                          col_size=col_size,
                          epoch=epoch)
        
    def confusion_matrix_img(self,
                             tag,
                             label_handler,
                             predictions=None,
                             labels=None,
                             confm=None,
                             epoch=None,
                             fz=11):
        conf_matrix_img = plot_confusion_matrix_from_data(y_test=labels,
                                                          predictions=predictions,
                                                          confm=confm,
                                                          label_handler=label_handler,
                                                          fz=fz)
        if self.writer is not None:
            self.writer.add_image(tag, conf_matrix_img, epoch)
            
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")
    
    def roc_auc(self,
                tag,
                predictions,
                labels,
                label_handler,
                log_path: str,
                epoch=None):
        if len(set(labels)) == 2:     
            roc_img_array = plot_roc_curve(y_true=labels,
                                           y_probas=predictions,
                                           label_handler=label_handler,
                                           log_path=log_path)
            
            if self.writer is not None:
                self.writer.add_image(tag, roc_img_array, epoch)
            
            if self.save_to_folder:
                None
                #raise Exception("Not implemented yet.")

    def probability_hist(self,
                         tag,
                         predictions,
                         labels,
                         label_handler,
                         log_path: str,
                         epoch,
                         wsis=None):
        prob_his_img_array = plot_prob_hist(y_true=labels,
                                            y_probas=predictions,
                                            y_wsi=wsis,
                                            label_handler=label_handler,
                                            log_path=log_path)
        if self.writer is not None:
            self.writer.add_image(tag, prob_his_img_array, epoch)
        
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")
        
    def score_hist(self,
                   tag,
                   score_value,
                   score_name,
                   label_handler,
                   log_path,
                   epoch,
                   score_classes=None):
        score_img_array = plot_score_hist(score_value=score_value,
                                          score_name=score_name,
                                          label_handler=label_handler,
                                          log_path=log_path,
                                          score_classes=score_classes)
        if self.writer is not None:
            self.writer.add_image(tag, score_img_array, epoch)
        
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")
        
    def score_2dhist(self,
                     tag,
                     score_value1,
                     score_name1,
                     score_value2,
                     score_name2,
                     label_handler,
                     log_path,
                     epoch,
                     score_classes=None):

        score_img_array = plot_score_2dhist(score_value1=score_value1,
                                            score_name1=score_name1,
                                            score_value2=score_value2,
                                            score_name2=score_name2,
                                            label_handler=label_handler,
                                            log_path=log_path,
                                            score_classes=score_classes)
        if self.writer is not None:
            self.writer.add_image(tag, score_img_array, epoch)
        
        if self.save_to_folder:
            None
            #raise Exception("Not implemented yet.")
    
    def plot_wsi_section(self,
                         section: np.ndarray,
                         log_path,
                         mode='gt',
                         attention=False,
                         att_estimates=False,
                         plot_coord=True):
        """ 
        Args:
            section (nd.Array): 2-D array of patches 
        """
        
        assert(mode in ['gt', 'pred', 'org'])
        
        # 1. plot groundtruth
        
        # determine output size via array size * patch size
        n_patches = section.shape[0]
        l = (n_patches-1)//2
        patch_size = section[l,l].mask.size[0]
        center_patch = section[l,l]
        k = center_patch.wsi.k_neighbours
        wsi_name = center_patch.wsi.name
        center_x = center_patch.x_coord
        center_y = center_patch.y_coord
       
         # if attention on, section area must fit attention area
        if attention is True and (l != k):
            # crop section to attention area:
            section = section[(l-k):(l+k+1),(l-k):(l+k+1)]
            n_patches = section.shape[0]
            l = (n_patches-1)//2

        patches_canvas = PIL.Image.new('RGB', (n_patches*patch_size, n_patches*patch_size), color='white')
         
        # paste patches 
        for x,y in np.ndindex(section.shape):
            if section[x,y] is None:
                continue
            
            if mode == 'gt':
                patch = section[x,y].mask
            elif mode == 'pred':
                patch = section[x,y].prediction
            elif mode == 'org':
                patch = section[x,y].get_image()
            patches_canvas.paste(patch,(patch_size*x,patch_size*y))

        patches_canvas_draw = PIL.ImageDraw.Draw(patches_canvas)
        
        # paste lines
        for x,y in np.ndindex(section.shape):
            if section[x,y] is None:
                continue
            shape = [(patch_size*x, patch_size*y), (patch_size*x + patch_size, patch_size*y + patch_size)]
            patches_canvas_draw.rectangle(shape, outline ="black")
        
        # plotting ellipsis
        # determine 90 % quantile of chi squared dist with df=2
        alpha=0.10
        error_factor = scipy.stats.chi2.ppf(1-alpha, df=2)
        
        if att_estimates is True:
               
            mode = mode + "_e"

            avg_estimates = center_patch.wsi.avg_estimates

            attention_img = PIL.Image.new('RGBA', (n_patches*patch_size, n_patches*patch_size), (255,255,255,180))
            attention_canvas_array = np.array(attention_img)
            
            for x,y in np.ndindex(section.shape):
                if section[x,y] is None:
                    continue
                # draw att. estimates
                estimates = section[x,y].estimates
                neighbourh_to_image = patch_size / (k*2+1)
                # y / x is flipped
                y_c = (estimates[0]) * neighbourh_to_image
                x_c = (estimates[1]) * neighbourh_to_image
                radius = (estimates[2]) * neighbourh_to_image 
                avg_radius = avg_estimates[2] * neighbourh_to_image
                
                cmap = matplotlib.cm.get_cmap('seismic')
                #high radius: red , small radius: blue
                heatmap_color = [int(255*c) for c in cmap(0.5 + (radius-avg_radius)/patch_size)]
                # getcov(radius, scale, rotate)
                cov = getcov(estimates[2], estimates[3], estimates[4])
                
                eigenvalues, eigenvectors  = LA.eig(cov)
                # determine angle to first eigenvector to x-axis:
                np.angle(eigenvectors[0], deg=True)
                # angle to x-axis: atans(y,x)
                angle = atan2(eigenvectors[0][1], eigenvectors[0][0]) * 180 / pi
                # now: subtract 90 degrees because of different coorodinates in opencv
                angle -= 90
                cv2.ellipse(img=attention_canvas_array,
                            center=(patch_size*x + int(x_c),
                                    patch_size*y + int(y_c)), 
                            axes=(int((error_factor*eigenvalues[0])**(1/2) *neighbourh_to_image), 
                                  int((error_factor*eigenvalues[1])**(1/2) *neighbourh_to_image)), 
                            angle=angle,
                            startAngle=0, 
                            endAngle=360,
                            color=heatmap_color,
                            thickness=-1)
                # ellipse border
                cv2.ellipse(img=attention_canvas_array,
                            center=(patch_size*x + int(x_c),
                                    patch_size*y+ int(y_c)), 
                            axes=(int((error_factor*eigenvalues[0])**(1/2) *neighbourh_to_image),
                                  int((error_factor*eigenvalues[1])**(1/2) *neighbourh_to_image)), 
                            angle=angle,
                            startAngle=0, 
                            endAngle=360,
                            color=(0,0,0,127),
                            thickness=1)
                cv2.arrowedLine(img=attention_canvas_array,
                                pt1=(int(patch_size*x+patch_size/2), 
                                     int(patch_size*y+patch_size/2)),
                                # strech arrow into direction:
                                pt2=(int(patch_size*x+x_c+5*(x_c-patch_size/2)), int(patch_size*y+y_c+5*(y_c-patch_size/2))),
                                color=(0,0,0,127),
                                thickness=5,
                                tipLength=0.4)
            attention_canvas = PIL.Image.fromarray(attention_canvas_array, mode='RGBA')
            patches_canvas.paste(attention_canvas, (0,0), attention_canvas)
    
        if attention is True:
            mode = mode + "_a"
            # attention masks over all heads
            attention = center_patch.attention.numpy()
            # enhance attentions by factor 10
            mask = cv2.resize((attention**(1/2)*4), patches_canvas.size)
            cmap = matplotlib.cm.get_cmap('afmhot')
            c_mask =  cmap(mask)
            # set alpha channel
            c_mask[:,:,3] = 0.8
            c_mask_canvas = PIL.Image.fromarray((c_mask*255).astype("uint8"), mode='RGBA')
            patches_canvas.paste(c_mask_canvas, (0,0), c_mask_canvas)
            
            # draw ellipse
            estimates = center_patch.estimates
            neighbourh_to_image = patch_size
            # x / y is flipped
            y_c = int((estimates[0]) * neighbourh_to_image)
            x_c = int((estimates[1]) * neighbourh_to_image)
            
            # getcov(radius, scale, rotate)
            cov = getcov(estimates[2], estimates[3], estimates[4])
            eigenvalues, eigenvectors  = LA.eig(cov)
            # angle to x-axis: atans(y,x)
            angle = atan2(eigenvectors[0][1], eigenvectors[0][0]) * 180 / pi
            # now: subtract 90 degrees because of different coorodinates in opencv
            angle -= 90
            patches_canvas_array = np.array(patches_canvas)

            cv2.ellipse(img=patches_canvas_array,
                        center=(x_c,
                                y_c), 
                        axes=(int((error_factor*eigenvalues[0])**(1/2) *neighbourh_to_image), 
                                int((error_factor*eigenvalues[1])**(1/2) *neighbourh_to_image)), 
                        angle=angle,
                        startAngle=0, 
                        endAngle=360,
                        color=(255,255,255,255),
                        thickness=8)
             # from center of center patch to center of distribtuon
            x_center = int(patch_size*k+patch_size/2)
            y_center = int(patch_size*k+patch_size/2)
            cv2.drawMarker(img=patches_canvas_array, 
                           position=(x_center,x_center),
                           color=(255, 204, 0), 
                           markerType=cv2.MARKER_TILTED_CROSS, 
                           markerSize=100,
                           thickness=15)
            # first ellipsis axis
            cv2.arrowedLine(img=patches_canvas_array,
                            pt1=(x_c,
                                 y_c), 
                            pt2=(int(x_c + eigenvectors[0][1] * (error_factor*eigenvalues[0])**(1/2) *neighbourh_to_image),
                                 int(y_c - eigenvectors[0][0] * (error_factor*eigenvalues[0])**(1/2) *neighbourh_to_image)), 
                            color=(255,255,255,255),
                            thickness=8)
            # second ellipsis axis
            cv2.arrowedLine(img=patches_canvas_array,
                            pt1=(x_c,
                                 y_c), 
                            pt2=(int(x_c + eigenvectors[1][1] * (error_factor*eigenvalues[1])**(1/2) *neighbourh_to_image),
                                 int(y_c - eigenvectors[1][0] * (error_factor*eigenvalues[1])**(1/2) *neighbourh_to_image)), 
                            color=(255,255,255,255),
                            thickness=8)
            cv2.arrowedLine(img=patches_canvas_array,
                            pt1=(x_center,
                                 y_center),
                            # strech arrow into direction:
                            pt2=(x_c, y_c),
                            color=(0,128,255,255),
                            thickness=16,
                            tipLength=0.4)
            
            
            patches_canvas = PIL.Image.fromarray(patches_canvas_array, mode='RGB')

        if plot_coord is True:
            for x,y in np.ndindex(section.shape):
                if section[x,y] is None:
                    continue
                patches_canvas_draw.text((patch_size*x, patch_size*y), str(section[x,y].get_coordinates()))
        
        
        patches_canvas.save(Path(log_path) / f'patch_area_{center_x}_{center_y}_{wsi_name}_k{l}_{mode}.png')        
        patches_canvas = patches_canvas.resize((100*l,100*l),PIL.Image.ANTIALIAS)
        patches_canvas.save(Path(log_path) / f'patch_area_{center_x}_{center_y}_{wsi_name}_k{l}_{mode}_small.png')

    def wsi_plot(self,
                 tag,
                 wsi,
                 log_path,
                 epoch,
                 mode='wsi'):
        
        if mode == 'wsi':
            plot = plot_wsi(wsi=wsi)
        elif mode == 'heatmap':
            plot = plot_wsi(wsi=wsi, patch_heatmap=True)   
        elif mode == 'wsi+heatmap':
            plot_mask = plot_wsi(wsi=wsi)
            plot_heatmap = plot_wsi(wsi=wsi, patch_heatmap=True)   
            #combine
            plot = get_concat_v(plot_mask, plot_heatmap)
        elif mode == 'wsi+heatmap+thumbnail':
            plot_mask = plot_wsi(wsi=wsi)
            plot_heatmap = plot_wsi(wsi=wsi, patch_heatmap=True)   
            plot_thumbnail = plot_wsi(wsi=wsi, thumbnail_only=True)
            #combine
            plot_tmp = get_concat_v(plot_mask, plot_heatmap)
            plot = get_concat_v(plot_tmp, plot_thumbnail)
        elif mode == 'truewsi+wsi+heatmap+thumbnail':
            plot_true_mask = plot_wsi(wsi=wsi, use_true_mask=True)
            plot_mask = plot_wsi(wsi=wsi)
            plot_heatmap = plot_wsi(wsi=wsi, patch_heatmap=True)   
            plot_thumbnail = plot_wsi(wsi=wsi, thumbnail_only=True)
            #combine
            plot_tmp = get_concat_v(plot_true_mask, plot_mask)
            plot_tmp = get_concat_v(plot_tmp, plot_heatmap)
            plot = get_concat_v(plot_tmp, plot_thumbnail)
            
        elif mode == 'worst_patches':
            plot = plot_patches(wsi=wsi,
                                top=10,
                                per='class',
                                mode='worst')
        
        if self.save_to_folder:
            plot_path= log_path / tag / str(wsi.get_label()) / f"{wsi.name}_{mode}.png"
            plot_path.parents[0].mkdir(parents=True, exist_ok=True)
            plot.save(plot_path, dpi=(200, 200))
            
        if self.writer is not None:
            plot.thumbnail(tuple(int(s*0.40) for s in plot.size), PIL.Image.ANTIALIAS)
            self.writer.add_image(tag=f"{tag}_viz/{wsi.get_label()}/{wsi.name}_{mode}",
                            img_tensor=np.transpose(np.array(plot),
                                                    (2, 0, 1)),
                            global_step=epoch)
            

def log_config(logpath, config_path):
    from shutil import copy
    copy(config_path, Path(logpath) / 'config.yml' )

def log_args(logpath, args):
    with logpath.open('w') as yamlfile:
        yaml.safe_dump(vars(args), yamlfile, default_flow_style=None)
    
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logging.getLogger('exp').info('%r %2.2f sec' % \
              (method.__name__, te-ts))
        return result

    return timed

class Timer():
    
    def __init__(self, verbose=True) -> None:
        self._start = 0
        self._sum_table = defaultdict(int)
        self._count_table = defaultdict(int)
        self.verbose = verbose
        
    def start(self):
        self._start = time.time()
        
    def stop(self, key):
        end_time = time.time()
        duration = end_time - self._start
        self._sum_table[key] += duration
        self._count_table[key] += 1
        
        if self.verbose:
            logging.getLogger('exp').info(f'{key}: {round(duration,4)} sec (avg: {round(self._sum_table[key]/ self._count_table[key],4)})')
        
        #reset start time
        self._start = end_time