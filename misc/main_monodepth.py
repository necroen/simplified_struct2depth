# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:41:57 2019

@author: x

单独这个脚本跑不通，只是用来参考的


这个程序完成了从 断点 继续训练的功能

可以推广到其他模型的训练中

其实最好做一个 非常小的数据集 以方便调试

"""

import time
import collections
import os

import numpy as np
import matplotlib.pyplot as plt

from easydict import EasyDict as edict

import torch
import torch.optim as optim


args = edict({'train_data_dir':'D:\\py2019\\myMonoDepth\\train_data',
              'val_data_dir':'D:\\py2019\\myMonoDepth\\valid_data',
              'model_path':'./',
              'output_directory':'./ckpt',
              'input_height':256,
              'input_width':512,
              'model':'resnet18_md',
              'pretrained':True,
              'mode':'train',
              'epochs':200,
              'learning_rate':1e-4,
              'batch_size': 2,
              'adjust_lr':True,
              'device':'cuda:0',
              'do_augmentation':True,
              'augment_parameters':[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
              'print_images':False, # False  是否打印中间结果？ 彻夜训练的话一定要设为False
              'print_weights':False,
              'input_channels': 3,
              'num_workers': 2,
              'use_multiple_gpu': False,
              'previous_ckpt_path':"D:\\py2019\\myMonoDepth\\zModel\\ckpt\\model_best_cpt.pth"})

device = args.device

device_cpu = "cpu"


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")
        
        
def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

# lr = get_lr(epoch, learning_rate)
def get_lr(epoch, learning_rate):
    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    return lr


def set_lr(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        

def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


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
    
    return model, optimizer, epoch


#save_path = './ckpt/monodepth_epoch_%s.pth' % epoch
#torch.save(save(model, optimizer, epoch), save_path)
def save(model, optimizer, epoch):
    save_dict = dict()
    
#    model = model.to(device)
#    optimizer = to_device(optimizer, device_cpu) # 保留cpu模式下的权重
    
#    params = model.state_dict()  
#    for k, v in params.items():
#        print("key:", type(v) )
    
    save_dict['model'] = model.state_dict()
    save_dict['optimizer'] = optimizer.state_dict()
    save_dict['epoch'] = epoch  # epoch是 int 数字
    
    return save_dict


from data_loader_2 import KittiLoader, prepare_dataloader
from models_4 import Resnet18_md
from loss_5 import MonodepthLoss


def train():
    model = Resnet18_md(3)
    loss_function = MonodepthLoss(n=4, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    
    
    continue_train = 1  # 是否是从 断点 开始训练  ===========================
    if continue_train:  # 如果是继续训练，则载入之前的参数
        model, optimizer, epoch = load(args.previous_ckpt_path, model, optimizer)
        start_epoch = epoch + 1
        
        model = model.to(device)
        
        set_lr(optimizer, get_lr(start_epoch, args.learning_rate))
        
        print("\n")
        print("load previous checkpoint successfully!")
        print("start_epoch:", start_epoch)
        print("\n")
        
    else:  # 如果不是，则是从头训练，什么都不用做
        model = model.to(device)
        print("\n")
        print("train from scrach!")
        print("start_epoch:", start_epoch)
        print("\n")
    
    
    
    
    train_img_num, train_loader = prepare_dataloader(args.train_data_dir,  # 训练集
                                                    args.mode, 
                                                    args.augment_parameters,
                                                    args.do_augmentation, 
                                                    args.batch_size,
                                                    (args.input_height, args.input_width),
                                                    args.num_workers)
    
    
    valid_img_num, valid_loader = prepare_dataloader(args.val_data_dir,  # 验证集
                                                    args.mode,
                                                    args.augment_parameters,
                                                    False, 
                                                    args.batch_size,
                                                    (args.input_height, args.input_width),
                                                    args.num_workers) 
    
    losses = []
    val_losses = []
    best_loss = float('Inf')
    best_val_loss = float('Inf')
    
    running_val_loss = 0.0
    
    # 先计算 valid loss
    model.eval()
    for data in valid_loader:
        data = to_device(data, device)
        left  = data['left_image']
        right = data['right_image']
        disps = model(left)
    
        loss = loss_function(disps, [left, right])
        val_losses.append(loss.item())
        running_val_loss += loss.item()
    
    running_val_loss /= valid_img_num / args.batch_size
    print('Val_loss:', running_val_loss)
    
    # 开始迭代
    for epoch in range(start_epoch, args.epochs):
        print("=========================================  epoch:", epoch)
        set_lr(optimizer, get_lr(epoch, args.learning_rate))
        
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("optimizer learning rate:", lr)
        
        c_time = time.time()
        running_loss = 0.0
    
        model.train()
        i = 0
        len_loader = len(train_loader) 
        
        for data in train_loader:
            # print( '%.2f'% (i/len(loader)) )
            if i%100 == 0:
                print(i, "/", len_loader )
    
            i = i+1
    
            # Load data
            data = to_device(data, device)
            left = data['left_image']
            right = data['right_image']
    
            # One optimization iteration
            optimizer.zero_grad()
            disps = model(left)
    
            loss = loss_function(disps, [left, right])
            loss.backward()
            
            optimizer.step()
            losses.append(loss.item())
            
            if args.print_images and i%100 == 0:
                pass
            
            running_loss += loss.item()
            
        running_val_loss = 0.0
        
        print("train finish, start calculate valid loss")
        model.eval()
        for data in valid_loader:
            data = to_device(data, device)
            left = data['left_image']
            right = data['right_image']
            disps = model(left)
            loss = loss_function(disps, [left, right])
            val_losses.append(loss.item())
            running_val_loss += loss.item()
    
        # Estimate loss per image
        running_loss     /= train_img_num / args.batch_size
        running_val_loss /= valid_img_num / args.batch_size
        
        # print (
        #         'Epoch:',      epoch,
        #         'train_loss:', running_loss,
        #         'val_loss:',   running_val_loss,
        #         'time:',       round(time.time() - c_time, 3), 's',
        #         )
        
        fp = open("./summary.txt","a+",encoding="utf-8")
        
        ss = 'Epoch: %d train_loss: %.15f val_loss: %.15f time: %.3f s' % (epoch,
                                                                           running_loss, 
                                                                           running_val_loss, 
                                                            round(time.time() - c_time, 3))
        print(ss)
        fp.write(ss + "\n")
        fp.close()
        
        
        
        # torch.save(model.state_dict(), args.output_directory + '/model_last_cpt.pth')
        # save_path = './ckpt/monodepth_epoch_%s.pth' % epoch
        save_path = args.output_directory + '/model_last_cpt.pth'
        torch.save(save(model, optimizer, epoch), save_path)
        
        if running_val_loss < best_val_loss and running_val_loss < 0.692672904881631:
            # self.save(self.args.model_path[:-4] + '_cpt.pth')
            
            # torch.save(model.state_dict(), args.output_directory + '/model_best_cpt.pth')
            save_path = args.output_directory + '/model_best_cpt.pth'
            torch.save(save(model, optimizer, epoch), save_path)
        
            best_val_loss = running_val_loss
            print('Model_saved')
            
    
    #print ('Finished Training. Best loss:', best_loss)
    print ('Finished Training. Best val loss:', best_val_loss)
    
if __name__ == '__main__':
    train()
    