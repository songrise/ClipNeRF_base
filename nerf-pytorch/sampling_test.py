# -*- coding : utf-8 -*-
# @FileName  : sampling_test.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Jun 09, 2022
# @Github    : https://github.com/songrise
# @Description: stride sampling a img 
#%%
from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

img_raw = Image.open('data/nerf_synthetic//lego/train/r_0.png')
img_raw = np.array(img_raw)

#%%
def stride_sampled(H,W,batch_size):
    """
    Stride-sample the raw rays.
    Params:
        raw_rays: [N, H, W, ro+rd+rgb, 3]
        batch_size: int, the size of batch used for one train iteration, assume always power of two.
        stride: int, the stride used for sampling.
    return 
        The index of the sampled rays. each batch is a semantically meaningful img.
    """

    all_idx = np.ones(H*W,dtype=np.int)
    #evenly crop the image according to the batch_size
    i_0,j_0 = 0,0
    stride = H*W//batch_size
    # for i in range(stride**2-1)
    #todo implement sampling all pixels
    sample_idx = np.meshgrid(np.arange(0,H,stride),np.arange(0,W,stride))

    return sample_idx


sample_idx = stride_sampled(400,400,batch_size=64000)
img_sampled = img_raw[sample_idx[0][:40,:40],sample_idx[1][:40,:40]]
plt.imshow(img_sampled)

# %%
#sampling a stride
# i_0,j_0 = np.random.randint(800,2)
i_0,j_0 = 400,400
stride = 4
#sample pixel on the grid, skip per stride
sample_i = np.meshgrid(np.arange(i_0,i_0+32*stride,stride),np.arange(j_0,j_0+32*stride,stride))
img_sampled = img_raw[sample_i[0],sample_i[1]]
# %%
plt.imshow(img_sampled)
# %%
