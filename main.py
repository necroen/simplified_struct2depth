# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:20:03 2019

@author: x


网络权重保存 和 optimizer 状态要保存，要支持从断点继续训练

还有数据增强没写
学习率衰减没写
计算valid loss没写
如果不把计算loss的部分封装起来，似乎很难方便的计算valid loss

另外还要弄清pose是谁对谁，inverse warp的含义，这部分应该是相当通用的
最好弄个虚拟视点合成的demo玩玩

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
from data_loader import train_dataloader, valid_dataloader, train_size, valid_size

from DispNetS import DispNetS     # 深度估计
from PoseExpNet import PoseExpNet # 姿态估计


#   0-100  1e-4
# 100-200  2e-5
# 200-end  1e-5
# def adjust_learning_rate(optimizer, epoch, learning_rate):
#     if epoch >= 100 and epoch < 200:
#         lr = 2e-5
#     elif epoch >= 200:
#         lr = 1e-5
#     else:
#         lr = learning_rate

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# lr = get_lr(epoch, learning_rate)
def get_lr(epoch, learning_rate):
    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    return lr



# 从断点处继续训练，需要载入之前保存的 模型 和 优化器 的参数，还有epoch，best_valid_loss

#save_path = './ckpt/monodepth_epoch_%s.pth' % epoch
#torch.save(save(model, optimizer, epoch, best_valid_loss), save_path)
def save(model, optimizer, epoch, best_valid_loss):
    save_dict = dict()
    
    save_dict['model'] = model.state_dict()
    save_dict['optimizer'] = optimizer.state_dict()
    save_dict['epoch'] = epoch
    save_dict['best_valid_loss'] = best_valid_loss
    
    return save_dict


def load(dict_path, model, optimizer):
    adict = torch.load(dict_path)
    
    model.load_state_dict(adict['model'])
    
    # https://github.com/pytorch/pytorch/issues/2830
    optimizer.load_state_dict( adict['optimizer'] )
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
            
    epoch = adict['epoch']
    best_valid_loss = adict['best_valid_loss']
    
    return model, optimizer, epoch, best_valid_loss


def set_lr(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate



def main():

    print("train_size:", train_size)
    print("valid_size:", valid_size)

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


    start_epoch = 0

    # continue_train = 1  # 是否是从 断点 开始训练  
    # if continue_train:  # 如果是继续训练，则载入之前的参数
    #     # 擦，这里还要考虑多个网络的权重如何保存和载入，还要改save和load，懒得写了
    #     # 还要搞个[model1, model2, ...] 之类的列表
    #     model, optimizer, epoch = load(args.previous_ckpt_path, model, optimizer)
    #     start_epoch = epoch + 1
    #     model = model.to(device)
    #     set_lr(optimizer, get_lr(start_epoch, args.learning_rate))
    #     print("\n")
    #     print("load previous checkpoint successfully!")
    #     print("start_epoch:", start_epoch)
    #     print("\n")
    # else:  # 如果不是，则是从头训练，什么都不用做
    #     model = model.to(device)
    #     print("\n")
    #     print("train from scrach!")
    #     print("start_epoch:", start_epoch)
    #     print("\n")


    # cudnn.benchmark = True
    
    best_loss = float('Inf')

    args_epochs = 300
    # for epoch in range(args_epochs):
    for epoch in range(start_epoch, args_epochs):  # 这样写是为了支持从断点开始继续训练
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

            # 开始build loss======================================
            from loss_func import calc_total_loss

            total_loss, reconstr_loss, smooth_loss, ssim_loss = \
            calc_total_loss(image_stack, disp, depth, depth_upsampled, egomotion, intrinsic_mat)
            # total loss 计算结束 ================================

            if loader_idx % (200/args.batchsize) == 0:
                print("idx: %4d reconstr: %.5f  smooth: %.5f  ssim: %.5f  total: %.5f" % \
                    (loader_idx, reconstr_loss, smooth_loss, ssim_loss, total_loss) )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        #############单个epoch结束

        running_loss /= (train_size/args.batchsize)

        if running_loss < best_loss:
            best_loss = running_loss

            print("* best loss:", best_loss )

            torch.save(disp_net.state_dict(),     './disp_net_best.pth' )
            torch.save(pose_exp_net.state_dict(), './pose_exp_net_best.pth' )

        print ( 'Epoch:', epoch, 
                'train_loss:', running_loss,
                'time:', round(time.time() - c_time, 3), 's')
        
        
if __name__ == '__main__':
    main()

#    key1 = '%d-%d' % (3, 2)
#    print(key1)