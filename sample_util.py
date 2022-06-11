# -*- coding : utf-8 -*-
# @FileName  : sampling.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Jun 09, 2022
# @Github    : https://github.com/songrise
# @Description: Sampling utils for generating rays.
#%%

import torch
import numpy as np

import matplotlib.pyplot as plt


def sample_rays(raw_rays:torch.Tensor,stride:int,retain:bool=False):
    """
    Stride-sample the raw rays.
    Params:
        raw_rays: [N, H, W, ro+rd+rgb, 3]
        batch_size: int, the size of batch used for one train iteration, assume always power of two.
        stride: int, the stride used for sampling.
        retain: bool, if True, return the number of pixels sampled.
    return 
        concatenated rays [(N-1)*H*W, ro+rd+rgb, 3], must be used per batch to ensure sematic info.
    """

    # assert batch_size > 0 

    # assume H*W divisible by batch_size
    H, W = raw_rays.shape[1:3]

    sample_idx = stride_sampled(H,W,stride,retain)
    if not retain:
        sampled = raw_rays[:,sample_idx[0],sample_idx[1],:,:]
    else:
        sampled = []
        for i in range(0,stride**2,2):
            sampled.append(raw_rays[:,sample_idx[i],sample_idx[i+1],:,:])
        sampled = np.concatenate(sampled,axis=0)
    return sampled,sample_idx

def sample_img(raw_img,stride:int,retain = False):
    """
    Stride-sample the raw rays.
    Params:
        raw_rays: [N, H, W, ro+rd+rgb, 3]
        batch_size: int, the size of batch used for one train iteration, assume always power of two.
        stride: int, the stride used for sampling.
        retain: bool, if True, return the number of pixels sampled.
    return 
        concatenated rays [(N-1)*H*W, ro+rd+rgb, 3], must be used per batch to ensure sematic info.
    """

    # assert batch_size > 0 

    # assume H*W divisible by batch_size
    H, W = raw_img.shape[1:3]

    sample_idx = stride_sampled(H,W,stride,retain)
    if not retain:
        sampled = raw_img[:,sample_idx[0],sample_idx[1],:]
    else:
        sampled = []
        for i in range(0,stride**2,2):
            sampled.append(raw_img[:,sample_idx[i],sample_idx[i+1],:])
        sampled = np.concatenate(sampled,axis=0)
    return sampled,sample_idx

def stride_sampled(H,W,stride,retain=False):
    """
    Stride-sample the raw rays.
    Params:
        raw_rays: [N, H, W, ro+rd+rgb, 3]
        batch_size: int, the size of batch used for one train iteration, assume always power of two.
        stride: int, the stride used for sampling.
    return 
        The index of the sampled rays. each batch is a semantically meaningful img.
    """
    # for i in range(stride**2-1)
    sample_idx = np.meshgrid(range(0,H,stride),range(0,W,stride))
    if not retain:
        return sample_idx

    for i in range(0,stride):
        for j in range(0,stride):
            if i == j == 0:
                continue
            sample_idx += np.meshgrid(range(-i,H-i,stride),range(-j,W-j,stride))
    return sample_idx

def patchify_ray(rays, patch_size):
    """
    Patchify the rays.
    Params:
        raw_rays: [N, H, W, ro+rd+rgb, 3]
        batch_size: int, the size of batch used for one train iteration, assume always power of two.
    return 
        concatenated rays [(N-1)*H*W, ro+rd+rgb, 3], must be used per batch to ensure sematic info.
    """
    #? ray rotated
    H, W = rays.shape[1:3]
    result = np.ones_like(rays)
    L = int(np.sqrt(patch_size)) #length of the patch,assume equal to width
    result = np.ones((rays.shape[0]*(H//L)*(W//L),L,L,rays.shape[3],rays.shape[4]))
    n_patch = 0
    for i in range(rays.shape[0]):
        for j in range(H//L):
            for k in range(W//L):
                result[n_patch,:,:,:] = rays[i,j*L:(j+1)*L,k*L:(k+1)*L,:]
                n_patch+=1
    result = np.transpose(result,(0,2,1,3,4))
    result = result.reshape(result.shape[0]*L*L,rays.shape[3],rays.shape[4])
    return result

def patchify_img(img, patch_size):
    """
    Patchify the img.
    Params:
        img: [N,H, W, C]
        patch_size: int, the size of batch used for one train iteration, assume always power of two.
    return 
        concatenated rays [(N-1)*H*W, rgb], must be used per batch to ensure sematic info.
    """
    H, W = img.shape[1:3]

    L = int(np.sqrt(patch_size)) #length of the patch,assume equal to width
    result = np.ones((img.shape[0]*(H//L)*(W//L),L,L,img.shape[3]))
    n_patch = 0
    for i in range(img.shape[0]):
        for j in range(H//L):
            for k in range(W//L):
                result[n_patch,:,:,:] = img[i,j*L:(j+1)*L,k*L:(k+1)*L,:]
                n_patch+=1
    result = np.transpose(result,(0,2,1,3))
    result = result.reshape(result.shape[0]*L*L,img.shape[3])
    return result

#%%
if __name__ == '__main__':
    # raw_rays = torch.randn(1,800,800,6,3)

    # sampled,i = sample_rays(raw_rays,2)
    # patch = patchify_ray(sampled,1600)
    # plt.imshow(sampled[0,:,:,0,0])
    # print(sampled.shape)
    # print(patch.shape)

    from PIL import Image
    import numpy as np
    import torch
    import cv2
    import matplotlib.pyplot as plt
    import torchvision
    img_raw = Image.open('/root/ClipNeRF_base/data/flower/images_8/image000.png')
    img_raw = np.array(img_raw)
    img_raw = np.expand_dims(img_raw,0)
    sampled, i = sample_img(img_raw,3,retain=True)
    img_patch = patchify_img(sampled,2500)

    # plt.imshow(first_patch[:,:,:])

    # grid visiualization
    patches = img_patch.reshape(img_patch.shape[0]//2500,50,50,-1)
    patches = patches.astype(np.uint8)
    patches = torch.Tensor(patches)
    patches = patches.permute(0,3,1,2)
    patches_img=torchvision.utils.make_grid(patches[:,:,:,:],nrow=6,pad_value=3)
    patches_img = patches_img.numpy()
    patches_img = patches_img.transpose(1,2,0)
    patches_img = patches_img.astype(np.uint8)
    plt.imshow(patches_img)

    #forge into ray shape
    # img_raw = np.expand_dims(img_raw,axis = 3)
    # sampled,i = sample_rays(img_raw,2,retain=True)
    # print(sampled.shape)
    # patch = patchify_ray(sampled,2500)

    # # grid visiualization
    # patches = patch.reshape(patch.shape[0]//2500,50,50,-1)
    # patches = patches.astype(np.uint8)
    # patches = torch.Tensor(patches)
    # patches = patches.permute(0,3,1,2)
    # patches_img=torchvision.utils.make_grid(patches[:,:,:,:],nrow=6,pad_value=3)
    # patches_img = patches_img.numpy()
    # patches_img = patches_img.transpose(1,2,0)
    # patches_img = patches_img.astype(np.uint8)
    # plt.imshow(patches_img)
    # plt.show()

    
# %%
