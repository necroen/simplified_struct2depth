# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:38:27 2019

@author: x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import util

# # 传进来的 depth 明明是 disp 视差图
# def depth_smoothness(depth, img):
#     """Computes image-aware depth smoothness loss."""

#     def gradient_x(img):
#         return img[:, :, :-1, :] - img[:, :, 1:, :]

#     def gradient_y(img):
#         return img[:, :-1, :, :] - img[:, 1:, :, :]

#     depth_dx = gradient_x(depth)
#     depth_dy = gradient_y(depth)
#     image_dx = gradient_x(img)
#     image_dy = gradient_y(img)
#     weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
#     weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
#     smoothness_x = depth_dx * weights_x
#     smoothness_y = depth_dy * weights_y

#     return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))


def gradient_x(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def gradient_y(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy


def disp_smoothness(disp, img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return torch.mean( torch.abs(smoothness_x) ) + torch.mean( torch.abs(smoothness_y) )



def SSIM(x, y): 
    # structural similarity index  结构相似性，是一种衡量两幅图像相似度的指标
    # https://blog.csdn.net/kevin_cc98/article/details/79028507
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


from params import args
seq_length = args.seq_length

num_scales = 4

# total_loss, reconstr_loss, smooth_loss, ssim_loss = \
# calc_total_loss(image_stack, disp, depth, depth_upsampled, egomotion, intrinsic_mat)
def calc_total_loss(image_stack, disp, depth, depth_upsampled, egomotion, intrinsic_mat):

    middle_frame_index = (seq_length-1)//2   # 0 1 2 中间是 1
    
    # self.images is organized by ...[scale][B, h, w, seq_len * 3].
    images = [None for _ in range(num_scales)]

    # 先把图片缩放，为后续计算loss做准备
    for s in range(num_scales):
        height_s = int( 128 / (2**s) )
        width_s = int( 416 / (2**s) )
        
        images[s] = [nn.functional.interpolate(x,
                                                size=[height_s, width_s], 
                                                mode='bilinear', 
                                                align_corners=True)
                                                for x in image_stack]
        
    smooth_loss = 0 # 计算各个尺度的 smooth_loss
    for s in range(num_scales):
        # Smoothness.
        for i in range(seq_length):
            compute_minimum_loss = True
            if not compute_minimum_loss or i == middle_frame_index:
                disp_smoothing = disp[i][s]
                mean_disp = torch.mean(disp_smoothing, (1, 2, 3), True)
                disp_input = disp_smoothing / mean_disp
                smooth_loss += ( 1.0 / (2**s) ) * disp_smoothness(disp_input, images[s][i])
        
    # Following nested lists are organized by ...[scale][source-target].
    warped_image = [{} for _ in range(num_scales)]
    warp_mask = [{} for _ in range(num_scales)]
    warp_error = [{} for _ in range(num_scales)]
    ssim_error = [{} for _ in range(num_scales)] 

    reconstr_loss = 0
    ssim_loss = 0

    for s in range(num_scales):

        for i in range(seq_length):
            for j in range(seq_length):
                if i == j:
                    continue
                
                # When computing minimum loss, only consider the middle frame as target.
                if compute_minimum_loss and j != middle_frame_index:
                    continue
                
                exhaustive_mode = False
                if (not compute_minimum_loss and not exhaustive_mode and abs(i - j) != 1):
                    continue
                
                depth_upsampling = True
                selected_scale = 0 if depth_upsampling else s
                source = images[selected_scale][i]
                target = images[selected_scale][j]
                
                if depth_upsampling:
                    target_depth = depth_upsampled[j][s]
                else:
                    target_depth = depth[j][s]
                    
                key = '%d-%d' % (i, j)
                # print("key:", key)
                
                # 这个时候传进来的egomotion的尺寸是 [batchsize, 2, 6]
                egomotion_mat_i_j = util.get_transform_mat(egomotion, i, j)
                # print("egomotion_mat_i_j size:\n", egomotion_mat_i_j.size() ) ([1, 4, 4])
                
                # print("intrinsic_mat size:", intrinsic_mat.size() )
                warped_image[s][key], warp_mask[s][key] = \
                    util.inverse_warp(source, 
                                        target_depth.squeeze(1), 
                                        egomotion_mat_i_j[:, 0:3, :],
                                        intrinsic_mat[:, selected_scale, :, :]
                                        )
                
                # Reconstruction loss.
                warp_error[s][key] = torch.abs(warped_image[s][key] - target) 
                if not compute_minimum_loss:
                    reconstr_loss += torch.mean(warp_error[s][key] * warp_mask[s][key])
  
                # SSIM.
                ssim_error[s][key] = SSIM(warped_image[s][key], target)
                
                # TODO(rezama): This should be min_pool2d().
                if not compute_minimum_loss:
                    # ssim_mask = slim.avg_pool2d(warp_mask[s][key], 3, 1, 'VALID')
                    ssim_mask = nn.AvgPool2d(3, 1)(warp_mask[s][key])
                    ssim_loss += torch.mean(ssim_error[s][key] * ssim_mask)

    for s in range(num_scales):
        # If the minimum loss should be computed, the loss calculation has been
        # postponed until here.
        if compute_minimum_loss:
            for frame_index in range(middle_frame_index):
                key1 = '%d-%d' % (frame_index, middle_frame_index)
                key2 = '%d-%d' % (seq_length - frame_index - 1, middle_frame_index)
                
                # print('computing min error between %s and %s', key1, key2)
                
                min_error = torch.min(warp_error[s][key1], warp_error[s][key2])
                reconstr_loss += torch.mean(min_error)
                
                # Also compute the minimum SSIM loss.
                min_error_ssim = torch.min(ssim_error[s][key1], ssim_error[s][key2])
                ssim_loss += torch.mean(min_error_ssim)
    
    
    total_loss = 0.85*reconstr_loss + 0.04*smooth_loss + 0.15*ssim_loss
    return total_loss, reconstr_loss, smooth_loss, ssim_loss