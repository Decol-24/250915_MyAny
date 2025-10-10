#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:16:28 2022

@author: liqi
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from itertools import repeat
import collections.abc
import time

def parallel_setting():
    import torch.distributed as dist
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    #CUDA_VISIBLE_DEVICES=0,1 torchrun main.py -para
    return args

def parameter_to_cpu(parameter):
    from collections import OrderedDict

    new_parameter = OrderedDict()
    for k, v in parameter.items():
        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        new_parameter[name] = v

    return new_parameter

def parameter_to_para(parameter):
    from collections import OrderedDict

    new_parameter = OrderedDict()
    for k, v in parameter.items():
        name = 'module.' + k # add 'module.'
        new_parameter[name] = v

    return new_parameter

def clever_format(nums, format="%.2f"):
    from collections.abc import Iterable

    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            format="%.5f"
            clever_nums.append(format % (num / 1e6) + "M")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums

def thop_macs(net,input=None,device=None):
    import thop

    if input is not None:
        x = input
    else:
        x = torch.randn(1,3,32,32)

    if device:
        x = x.to(device)
    else:
        net.to('cpu')
    macs, params = thop.profile(net,inputs=(x,x))
    macs, params = clever_format([macs, params], "%.2f")
    # print('macs:',macs, 'params:', params,)
    return macs, params

def get_logger(log_path,formator=None):
    import logging
    # 创建一个日志器
    logger = logging.getLogger("logger")

    # 设置日志输出的最低等级,低于当前等级则会被忽略
    logger.setLevel(logging.INFO)

    while(logger.handlers):
        logger.handlers.pop()
    # 判断当前日志对象中是否有处理器，如果没有，则添加处理器
    if not logger.handlers:
        # 创建处理器：sh为控制台处理器，fh为文件处理器
        sh = logging.StreamHandler()

        # 创建处理器：sh为控制台处理器，fh为文件处理器,log_file为日志存放的文件夹
        # log_file = os.path.join(log_dir,"{}_log".format(time.strftime("%Y/%m/%d",time.localtime())))
        if os.path.splitext(log_path)[1] != '.log':
            log_path = log_path+'.log'
        fh = logging.FileHandler(log_path,mode="w")

        # 创建格式器,并将sh，fh设置对应的格式
        if not formator:

            # formator = logging.Formatter(fmt = "%(asctime)s %(filename)s %(levelname)s %(message)s",
            #                          datefmt="%Y/%m/%d %X")
            formator = logging.Formatter(fmt = "%(asctime)s %(levelname)s %(message)s",
                                      datefmt="%m/%d %H:%M")
        sh.setFormatter(formator)
        fh.setFormatter(formator)

        # 将处理器，添加至日志器中
        logger.addHandler(sh)
        logger.addHandler(fh)

    return logger

def creat_folder(folder):
    import shutil

    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)  #不存在则创建
    return folder

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

import time
class time_counter():
    #记录start 和 end 之间的时间和次数，得到平均一次消耗的时间
    def __init__(self,num=1):
        self.num = num
        self.time_temp = [0 for x in range(self.num)]
        self.time_result = [0 for x in range(self.num)]
    
    def start(self,id=0):
        self.time_temp[id] -= time.perf_counter()
    
    def end(self,id=0):
        self.time_temp[id] += time.perf_counter()
    
    def avg_time(self,id=0,count=1):
        return self.time_temp[id]/count
    
    def avg_time_str(self,id=0,count=1):
        return "{:4f}".format(self.avg_time_str(id,count))

    def reset(self):
        self.time_temp = [0 for x in range(self.num)]
        self.time_result = [0 for x in range(self.num)]