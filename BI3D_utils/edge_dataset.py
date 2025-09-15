import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class edge_dataset(Dataset):
    #用来把edge的输出输入到ViT中
    def __init__(self,edge_output=None):
        self.data_info = list()
        if edge_output:
            self.add_data_info(edge_output)

    def __getitem__(self, index): #训练时通过getitem读取样本
        img, pickup_idx, label = self.data_info[index]

        return (img, pickup_idx), label

    def __len__(self):
        return len(self.data_info)

    def add_data_info(self,edge_output):
        
        for i in range(len(edge_output)):
            img, pickup_idx, label = edge_output[i]
            img = torch.tensor(img,dtype=torch.float32)
            pickup_idx = torch.tensor(pickup_idx,dtype=torch.int16)
            label = torch.tensor(int(label)).unsqueeze(dim=0)
            self.data_info.append((img, pickup_idx, label))