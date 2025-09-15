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

a = torch.load('./sf_binary_depth.pth')
b = torch.load('/home/liqi/Code/250729_Mystereo/pth/13_4591774.90.pth')

result = a['model_state_dict']['featnet.firstconv.0.0.weight'] == b['model_state_dict']['featnet.firstconv.0.0.weight']

print(result)