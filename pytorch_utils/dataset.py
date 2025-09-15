#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:39:30 2021

@author: ihpc
"""

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle

persistent_workers = False
#第一个iter不用再重新初始化worker

# labellist = np.load('./cifar10/cifar10_label.npy',allow_pickle=True).item()
# train_set = Dateset_dir(data_dir='./cifar100/train/',transform=data_transform['train'],data_type='.png',labellist=labellist)
# val_set = Dateset_dir(data_dir='./cifar100/test/',transform=data_transform['val'],data_type='.png',labellist=labellist)
# num_classes = train_set.num_classes
class Dateset_dir(Dataset):
    def __init__(self,data_dir,transform=None,data_type='.JPEG',labellist={}):
        self.labellist = labellist
        self.num_classes = len(self.labellist)
        if not isinstance(data_type,list):
            data_type = [data_type]
        self.data_type = data_type
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        #如果换设备，labellist会变

    def __getitem__(self, index): #训练时通过getitem读取样本
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,data_dir):
        def fi(path):
            end = os.path.splitext(path)[1]
            if end in self.data_type:
                return True
            else:
                return False

        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                if sub_dir not in self.labellist:
                    #sub_dir是类名
                    self.labellist[sub_dir] = self.num_classes
                    self.num_classes +=1
                img_list = os.listdir(os.path.join(root, sub_dir))
                img_list = list(filter(fi, img_list))

                for i in range(len(img_list)):
                    img_name = img_list[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.labellist[sub_dir]
                    data_info.append((path_img, label))

        return data_info

def local_trainset(path,batch_size,data_type='PNG',num_workers=0,
                   transform = None,labellist = {}):

    if transform == None:
            transform=transforms.Compose(
                [
                    transforms.Resize((256,256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(60),
                    transforms.RandomResizedCrop((224,224),scale=(0.5,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

    dataset=Dateset_dir(
                    data_dir = path,
                    transform = transform,
                    data_type = data_type,
                    labellist = labellist,
                    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers = persistent_workers,
    )

    return train_loader

def local_valset(path,batch_size,data_type='PNG',num_workers=0,
                   transform = None,labellist = {}):
    #testset is the same
    if transform == None:
            transform=transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

    dataset=Dateset_dir(
                    data_dir = path,
                    transform = transform,
                    data_type = data_type,
                    labellist = labellist,
                    )

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers = persistent_workers,
    )

    return val_loader


def cifar_set(dataset_name,train_transform=None,val_transform=None):
    CIFAR_PATH = './'

    if train_transform == None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

    if val_transform == None:
        val_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])


    if dataset_name == 'cifar10':
        num_classes = 10
        train_set = torchvision.datasets.CIFAR10(root=CIFAR_PATH,
                                                          train=True,
                                                          download=True,
                                                          transform=train_transform)

        val_set = torchvision.datasets.CIFAR10(root=CIFAR_PATH,
                                                     train=False,
                                                     download=True,
                                                     transform=val_transform)

    elif dataset_name == 'cifar100':
        num_classes = 100
        train_set = torchvision.datasets.CIFAR100(root=CIFAR_PATH,
                                                          train=True,
                                                          download=True,
                                                          transform=train_transform)

        val_set = torchvision.datasets.CIFAR100(root=CIFAR_PATH,
                                                     train=False,
                                                     download=True,
                                                     transform=val_transform)

    elif dataset_name == 'cifar10_debug':

        num_classes = 10
        train_set = torchvision.datasets.CIFAR10(root=CIFAR_PATH,
                                                          train=True,
                                                          download=True,
                                                          transform=train_transform)


        val_set = torchvision.datasets.CIFAR10(root=CIFAR_PATH,
                                                     train=False,
                                                     download=True,
                                                     transform=val_transform)


        factor = 100
        #等距序列做索引下标
        indices = [x for x in range(0, len(val_set)) if x % factor == 0]
        train_set = torch.utils.data.Subset(train_set, indices)
        val_set = torch.utils.data.Subset(val_set, indices)
    else:
        return None

    return train_set, val_set, num_classes


def MNIST(batch_size,num_workers=0,data_PATH = './',
                    train_transform = None,val_transform = None):
    if train_transform == None:
            train_transform=transforms.Compose(
               [transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5])]
            )
    if val_transform == None:
            val_transform=transforms.Compose(
               [transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5])]
            )

    train_dataset = torchvision.datasets.MNIST(root=data_PATH,
                                   train=True,transform=train_transform,download=True)

    train_loader = torch.utils.data.DataLoader(
                                                dataset=train_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=True,
                                                pin_memory=True,
                                                persistent_workers = persistent_workers,
                                                )
    val_dataset = torchvision.datasets.MNIST(root=data_PATH,
                                   train=False,transform=val_transform,download=True)

    val_loader = torch.utils.data.DataLoader(
                                                dataset=val_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=False,
                                                pin_memory=True,
                                                persistent_workers = persistent_workers,
                                                )

    return train_loader,val_loader

class Dateset_imgenet100(Dataset):
    def __init__(self,data_info,transform=None):

        self.data_info = data_info
        self.transform = transform

    def __getitem__(self, index): #训练时通过getitem读取样本
        img, label = self.data_info[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)


def imagenet100_set(path='./ImageNet100/',mode = 'train'):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
