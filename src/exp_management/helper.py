from collections import defaultdict
import random
from PIL import Image
import time
import numpy as np

import torch 

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (max(im1.width,im2.width), im1.height + im2.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height,im2.height)), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def rot(theta):
    theta = np.deg2rad(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
def getcov(radius=1, scale=1, theta=0):
    """
    Args:
        radius: The PDF will be dilated by this factor
        scale: The PDF be stretched by a factor of (scale + 1) in the x direction, and squashed by a factor of 1/(scale + 1) in the y direction
        theta: The PDF will be rotated by this many degrees
    """
    cov = np.array([
        [radius*(scale + 1), 0],
        [0, radius/(scale + 1)]
    ])
    r = rot(theta)
    return r @ cov @ r.T