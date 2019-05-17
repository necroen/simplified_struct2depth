# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:20:03 2019

@author: x

This program may not work.

这个是用来保存历史状态的，未必跑的通，main.py才是有效的主程序

网络权重保存 和 optimizer 状态要保存，要支持从断点继续训练

还有数据增强没写
学习率衰减没写
计算valid loss没写
如果不把计算loss的部分封装起来，似乎很难方便的计算valid loss

另外还要弄清pose是谁对谁，inverse warp的含义，这部分应该是相当通用的
最好弄个虚拟视点的demo玩玩

"""

# C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe

import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from params import args
from data_loader import train_dataloader, valid_dataloader

from DispNetS import DispNetS     # 深度估计
from PoseExpNet import PoseExpNet # 姿态估计

def main():

    seq_length = args.seq_length
    num_scales = 4

    torch.manual_seed(0)

    device = args.device
    
    disp_net = DispNetS().to(device) 
    disp_net.init_weights()
    
    
    pose_exp_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_exp_net.init_weights()
    
    args_lr = args.learning_rate
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args_lr},
        {'params': pose_exp_net.parameters(), 'lr': args_lr}
    ]

    args_momentum = 0.9
    args_beta = 0.999
    args_weight_decay = 0

    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args_momentum, args_beta),
                                 weight_decay = args_weight_decay
                                 )

    # cudnn.benchmark = True
    
    best_loss = 10000000

    args_epochs = 300
    for epoch in range(args_epochs):
        print("============================== epoch:", epoch )
        
        disp_net.train()
        pose_exp_net.train()
        
        c_time = time.time()
        # 开始单个epoch

        running_loss = 0.0

        for loader_idx, (image_stack, image_stack_norm, intrinsic_mat, _) in enumerate(train_dataloader):
            
            image_stack = [img.to(device) for img in image_stack]
            image_stack_norm = [img.to(device) for img in image_stack_norm]
            intrinsic_mat = intrinsic_mat.to(device) # 1 4 3 3
            
            disp = {}
            depth = {}
            depth_upsampled = {}
            
            for seq_i in range(seq_length):
                multiscale_disps_i, _ = disp_net(image_stack[seq_i])
                # [1,1,128,416], [1,1,64,208],[1,1,32,104],[1,1,16,52]
                # print("mmultiscale_disps_i[0] size:", multiscale_disps_i[0].size() )
                # a = input("pasue ...")

                # if seq_i == 1:
                #     dd = multiscale_disps_i[0]
                #     dd = dd.detach().cpu().numpy()
                #     np.save( "./rst/" + str(loader_idx) + ".npy", dd)
                
                multiscale_depths_i = [1.0 / d for d in multiscale_disps_i]
                disp[seq_i] = multiscale_disps_i
                depth[seq_i] = multiscale_depths_i
                
                depth_upsampled[seq_i] = []
                
                for s in range(num_scales):
                    depth_upsampled[seq_i].append( nn.functional.interpolate(multiscale_depths_i[s],
                                   size=[128, 416], mode='bilinear', align_corners=True) )
                    
            egomotion = pose_exp_net(image_stack_norm[1], [ image_stack_norm[0], image_stack_norm[2] ])
            # torch.Size([1, 2, 6])

            # print("egomoiton size:", egomotion.size() )
            # a = input("pasue ...")
            
    
            # 开始build loss======================================
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
                        # print("mean disp:", mean_disp)
                        
                        disp_input = disp_smoothing / mean_disp
                        
                        from loss_func import disp_smoothness
                        smooth_loss += ( 1.0 / (2**s) ) * disp_smoothness(disp_input, images[s][i])

                        # print("smooth loss success")
                        # a = input("pasue ...")
    
             
        
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
                        
                        import util

                        # 这个时候传进来的egomotion的尺寸是 [batchsize, 2, 6]
                        egomotion_mat_i_j = util.get_transform_mat(egomotion, i, j)
                        # print("egomotion_mat_i_j size:\n", egomotion_mat_i_j.size() ) ([1, 4, 4])

                        
                        # print("egomotion_mat_i_j success!")
                        # a = input("pasue ...")
                        
                        # print("intrinsic_mat size:", intrinsic_mat.size() )
                        warped_image[s][key], warp_mask[s][key] = \
                            util.inverse_warp(source, 
                                              target_depth.squeeze(1), 
                                              egomotion_mat_i_j[:, 0:3, :],
                                              intrinsic_mat[:, selected_scale, :, :]
                                              )

                        # print("inverse_warp success!")
                        # a = input("pasue ...")
                        
                        # Reconstruction loss.
                        warp_error[s][key] = torch.abs(warped_image[s][key] - target) 
                        if not compute_minimum_loss:
                            reconstr_loss += torch.mean(warp_error[s][key] * warp_mask[s][key])
    
                            
                        # SSIM.
                        from loss_func import SSIM
                        ssim_error[s][key] = SSIM(warped_image[s][key], target)

                        # print("SSIM success!")
                        # a = input("pasue ...")
                        
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

            if loader_idx % 200 == 0:
            # if loader_idx % 10 == 0:
                print("idx: %4d reconstr: %.5f  smooth: %.5f  ssim: %.5f  total: %.5f" % \
                    (loader_idx, reconstr_loss, smooth_loss, ssim_loss, total_loss) )


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            #############单个epoch结束

        batch_size = 1

        if running_loss < best_loss:
            best_loss = running_loss
            print("\n")
            print("best loss:", best_loss / (len(train_loader) * batch_size) )
            print("\n")
            
            # torch.save(disp_net.state_dict(),     './wt_tiny/disp_net_best.pth' )
            # torch.save(pose_exp_net.state_dict(), './wt_tiny/pose_exp_net_best.pth' )

        
        running_loss /= len(train_dataloader) / args.batchsize
        print ( 'Epoch:', epoch, 
                'train_loss:', running_loss,
                'time:', round(time.time() - c_time, 3), 's')
        
        
if __name__ == '__main__':
    main()

#    key1 = '%d-%d' % (3, 2)
#    print(key1)