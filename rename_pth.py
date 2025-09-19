from pytorch_utils.common import creat_folder,thop_macs
from pytorch_utils.warmup_scheduler import GradualWarmupScheduler
from pytorch_utils.dataset import cifar_set,Dateset_dir
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from pytorch_utils.rand_augment import RandAugment
import torch
import torch.nn as nn
import argparse
import numpy as np
import pickle
import os
import copy

# a = torch.load('./sceneflow.tar')

# result = a['state_dict']

# check_point = {
#         'model_state_dict': result,
#         }

# torch.save(check_point,'sceneflow.pth')

a = torch.load('./sceneflow.pth')
a = a['model_state_dict']
new_pth  = copy.deepcopy(a)

for key in a.keys():
    if key[:7] == 'module.':
        new_pth[key[7:]] = a[key]
        del new_pth[key]

check_point = {
        'model_state_dict': new_pth,
        }

torch.save(check_point,'sceneflow_2.pth')