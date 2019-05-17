# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:23:20 2019

@author: x

用于保存训练参数

"""

from easydict import EasyDict as edict

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

args = edict({'dataset_path':'./dataset',
              'device':device,
              'learning_rate': 1e-4,
              'batchsize':2,
              'checkpoint_dir':'b',
              'seq_length':3
              })

# for k, v in args.items():
#     print(k, ': ', v)
