# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:42:36 2019

@author: x


self.intrinsics is get by calib.py, and write in this file by hand !!!!!!


有些数据增强需要同时对内参做操作，比如水平翻转
有些数据增强不需要对内参做操作，比如HSV的变化
变换后的内参可以先计算好，免得每次都计算
但是随机裁剪图片的时候，变换后的内参是不能预先计算好

数据增强单个操作传入的参数应该是(imgs, intrin)


if sequence_length = 3
tgt image is the i th image, ref is i-1 th and i+1 th image. 
ref = [i-1, i+1]

if sequence_length = 5
tgt = i, ref = [i-2, i-1, i+1, i+2]  

if sequence_length = 7
...

"""

import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
import random

from torch.utils.data import DataLoader, Dataset, random_split

import os

import custom_transforms

from params import args


sequence_length = 3 # 参数

demi_length = (sequence_length-1)//2

def get_lists(ds_path):
    # ds = './dataset'
    ll = os.listdir(ds_path)
    ll = [ds_path + '/' + x for x in ll]
    # ll = ['./dataset/video1', './dataset/video2', ...]
    
    seq_list = []
    
    for x in ll:
        imgs = os.listdir(x)
        imgs = [x + '/' + img for img in imgs]
        
        seq_list.append(imgs)
        
    return seq_list  # List[ List[img_path] ]


def get_shifts(sequence_length):
    # 3 -> [-1, 1]
    # 5 -> [-2, -1, 1, 2]
    assert sequence_length%2 == 1 and sequence_length > 1, \
    'sequence_length must be odd and > 1'
    
    shifts = list(range(-demi_length, demi_length + 1))
    shifts.pop(demi_length)
    
    # print(shifts)
    return shifts


def get_samples(seq_list, sequence_length):
    shifts = get_shifts(sequence_length)
    samples = []
    
    for imgs in seq_list:
        assert len(imgs) > sequence_length, 'dataset is too small!'
        
        for i in range(demi_length, len(imgs)-demi_length): # 1，321-1
            sample = {'tgt': imgs[i], 
                      'ref_imgs': []
                      }
            
            for j in shifts: # shifts = [-1, 1]
                sample['ref_imgs'].append(imgs[i+j])  # tgt 是 i   ref 是 i-1 和 i+1
                
            samples.append(sample) 
        random.shuffle(samples)
        
    return samples



def get_multi_scale_intrinsics(intrinsics, num_scales):
    """Returns multiple intrinsic matrices for different scales."""

    intrinsics_multi_scale = np.zeros( (4,3,3), dtype = np.float32)
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        intrinsics_copy = intrinsics.copy()
        
        intrinsics_copy[0] = intrinsics_copy[0]*( 1/ (2**s) )
        intrinsics_copy[1] = intrinsics_copy[1]*( 1/ (2**s) )
        
        intrinsics_multi_scale[s, :,:] = intrinsics_copy
    
    return intrinsics_multi_scale  # [4, 3, 3]


def get_multi_scale_inv_intrinsics(intrinsics, num_scales):
    inv_intrinsics_multi_scale = np.zeros( (4,3,3), dtype = np.float32)
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        intrinsics_copy = intrinsics.copy()
        
        intrinsics_copy[0] = intrinsics_copy[0]*( 1/ (2**s) )
        intrinsics_copy[1] = intrinsics_copy[1]*( 1/ (2**s) )
        
        inv_intrinsics_multi_scale[s, :,:] = np.linalg.inv(intrinsics_copy) 
        # 求了相机逆矩阵
    
    return inv_intrinsics_multi_scale



def load_as_float(path):
    return imread(path).astype(np.float32)




class SequenceFolder(data.Dataset):
    def __init__(self, 
                 ds_path, 
                 seed=0, 
                 sequence_length = 3,
                 num_scales = 4):
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.num_scales = num_scales
        
        seq_list = get_lists(ds_path)
        self.samples = get_samples(seq_list, sequence_length)
        
        # get by calib.py
        self.intrinsics = np.array([
                                 [1.14183754e+03, 0.00000000e+00, 6.28283670e+02],
                                 [0.00000000e+00, 1.13869492e+03, 3.56277189e+02],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 ]).astype(np.float32)
        
        # resize 1280 x 720 -> 416 x 128
        self.intrinsics[0] = self.intrinsics[0]*(416.0/1280.0)
        self.intrinsics[1] = self.intrinsics[1]*(128.0/720.0)
        
        self.ms_k     = get_multi_scale_intrinsics(self.intrinsics, self.num_scales)
        self.ms_inv_k = get_multi_scale_inv_intrinsics(self.intrinsics, self.num_scales)
        
        ######################
        self.to_tensor = custom_transforms.Compose([ custom_transforms.ArrayToTensor() ])
        self.to_tensor_norm = custom_transforms.Compose([ custom_transforms.ArrayToTensor(),
                                                     custom_transforms.Normalize(
                                                             mean=[0.485, 0.456, 0.406],
                                                             std =[0.229, 0.224, 0.225])
                                                  ])
        
    def __getitem__(self, index):
        sample = self.samples[index]
        # np.copy(sample['intrinsics'])
        
        tgt_img = load_as_float(sample['tgt'])   # 这里读入了图像, 之前是以图片路径保存的
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        # put the tgt_img in the center of image_stack_origin
        ref_imgs.insert(demi_length, tgt_img)
        image_stack_origin = ref_imgs  

        # seq = 3  
        # image_stack_origin = [ ref_imgs[0], tgt_img, ref_imgs[1] ]  
        
        # seq = 5
        # image_stack_origin = [ ref_imgs[0], ref_imgs[1], tgt_img, ref_imgs[2], ref_imgs[3] ]
        
        
        image_stack = self.to_tensor( image_stack_origin.copy() )
        image_stack_norm = self.to_tensor_norm( image_stack_origin.copy() )
        
        intrinsic_mat = torch.from_numpy(self.ms_k)
        intrinsic_mat_inv = torch.from_numpy(self.ms_inv_k)
        
        return image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv
        #      list 3 128 x 416               4 x 3 x 3

    def __len__(self):
        return len(self.samples)
    

ds_path = './dataset' 
ds = SequenceFolder(ds_path)

train_size = int(0.9 * len(ds))   # 方便的按比例分割数据集
valid_size = len(ds) - train_size

train_dataset, valid_dataset = random_split(ds, [train_size, valid_size]) # 注意这个函数


train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, 
                              shuffle=True, num_workers=4)

valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchsize, 
                              shuffle=True, num_workers=4)
                                                  
                                                     
        
if __name__ == '__main__':
    ds_path = './dataset'   # 要处理最后的一个斜杠，统一成没斜杠的情况，有斜杠的话就去掉
    # ds = SequenceFolder(ds_path)
    # print(len(ds.samples))
    
    # image_stack, image_stack_norm, intrinsic_mat, intrinsic_mat_inv = ds[0]

    # train_size = int(0.9 * len(ds))
    # test_size = len(ds) - train_size
    # train_dataset, test_dataset = random_split(ds, [train_size, test_size]) # 注意这个函数

    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # for test_batch in test_dataloader:
    #     print(test_batch)
    
    seqlen = 3
    a = [None]*(seqlen-1)

    demi_length = (seqlen-1)//2
    a.insert(demi_length, 6)
    
    print(a)
    