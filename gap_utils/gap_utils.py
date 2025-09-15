#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:15:32 2023

@author: liqi
"""
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

transform_test = {
    'train' : transforms.Compose([
    transforms.Resize(16),
    transforms.ToTensor(),
    transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ]),
    'val' : transforms.Compose([
    transforms.Resize(16),
    transforms.ToTensor(),
    ]),
    }

transform_32 = {
    'train' : transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ]),
    'val' : transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    ]),
    }

transform_64 = {
    'train' : transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    ]),
    'val' : transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    ]),
    }

transform_224 = {
    'train' : transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ]),
    'val' : transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ]),
    }


class gap_loader():
    def __init__(self, data, with_label=True, transpose_to_hwc=False):
        self._idx = 0
        self._transpose_to_hwc = transpose_to_hwc
        self.data = list(data)
        self._max_idx = len(self.data)
        self._label = with_label

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < self._max_idx:

            image, label = self.data[self._idx]
            #(3, 32, 32)
            image = image.to("cpu").detach().numpy()

            # 转换到 HWC
            if self._transpose_to_hwc:
                image_data = image.transpose(1, 2, 0)
            else:
                image_data = image

            self._idx += 1
            if self._label:
                return image_data, label
            else:
                return image_data
        else:
            raise StopIteration()

@torch.no_grad()
def gap_test(model,test_loader,quantize=True,dequantize=True):

    correct = total = 0
    for _, (inputs, target) in enumerate(test_loader):

        # inputs = inputs.item()
        outputs = model.execute([inputs], quantize=quantize, dequantize = dequantize)
        outputs = outputs[-1][0]
        total += 1
        correct += np.equal(outputs.argmax(), target).sum()

    test_acc = correct / total *100.0

    return test_acc


#废弃
class gap_dataset():
    def __init__(self, data_dir, max_idx=None, transpose_to_chw=False):
        self._idx = 0
        self._transpose_to_chw = transpose_to_chw
        self.data_type = ['.JPEG','.png','.jpg']
        self.data_info = self.get_image_info(data_dir)
        self._max_idx = max_idx if max_idx is not None else (len(self.data_info)-1)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx > self._max_idx:
            raise StopIteration()
        filename = self.data_info[self._idx]

        image = Image.open(filename)
        #(32, 32, 3)
        img_array = np.array(image)

        # 预处理
        img_array = img_array / 255.0

        # 转换到 CHW
        if self._transpose_to_chw:
            img_array = img_array.transpose(2, 0, 1)

        self._idx += 1
        return img_array

    def get_image_info(self,data_dir):
        def fi(path):
            end = os.path.splitext(path)[1]
            if end in self.data_type:
                return True
            else:
                return False

        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:

                img_list = os.listdir(os.path.join(root, sub_dir))
                img_list = list(filter(fi, img_list))

                for i in range(len(img_list)):
                    img_name = img_list[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    data_info.append(path_img)
        return data_info


#废弃
class gap_dataset_label():
    def __init__(self, data_dir,labellist=dict(), max_idx=None, transpose_to_chw=False):
        self._idx = 0
        self.labellist = labellist.item()
        self.count     = len(self.labellist)
        self._transpose_to_chw = transpose_to_chw
        self.data_type = ['.JPEG','.png','.jpg']
        self.data_info = self.get_image_info(data_dir)
        self._max_idx = max_idx if max_idx is not None else (len(self.data_info)-1)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx > self._max_idx:
            raise StopIteration()

        filename, label = self.data_info[self._idx]

        # 读一个图片，转换为np array
        image = Image.open(filename)
        img_array = np.array(image)

        # 预处理
        img_array = img_array / 255.0

        # 转换到 CHW
        if self._transpose_to_chw:
            img_array = img_array.transpose(2, 0, 1)

        self._idx += 1
        return img_array, label

    def get_image_info(self,data_dir):
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
                    self.labellist[sub_dir] = self.count
                    self.count+=1
                img_list = os.listdir(os.path.join(root, sub_dir))
                img_list = list(filter(fi, img_list))

                for i in range(len(img_list)):
                    img_name = img_list[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.labellist[sub_dir]
                    data_info.append((path_img, label))

        return data_info
