#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:24:43 2021

@author: ihpc
"""

import torch
import copy


def del_tensor_num(input , dim , index_1 , index_2):
    #根据下标删除index1到index2的内容
    if dim == 0:
        a = input[:index_1]
        b = input[index_2+1:]
        return torch.cat((a,b),dim=0)
    if dim == 1:
        a = input[:,:index_1]
        b = input[:,index_2+1:]
        return torch.cat((a,b),dim=1)
    if dim == 2:
        a = input[:,:,:index_1]
        b = input[:,:,index_2+1:]
        return torch.cat((a,b),dim=2)
    if dim == 3:
        a = input[:,:,:,:index_1]
        b = input[:,:,:,index_2+1:]
        return torch.cat((a,b),dim=3)

def del_tensor_list(input ,dim, del_list):
    #反向遍历
    index_2 = len(del_list)-1
    while(index_2>=0):
        index_1 = index_2
        while(index_1>=0):
            if del_list[index_1-1] == del_list[index_1]-1:#如果数字连续，则index_2为连续的终点
                index_1 = index_1-1
            else:
                break
        input = del_tensor_num(input,dim, del_list[index_1] , del_list[index_2])
        index_2 = index_1-1
    return input

def add_tensor(input, dim, init_mode=0, width=1, position=-1):
    #按维度添加size的tensor，填充到width宽
    #init_mode: 0-填充0; 1-填充1; 2-凯明初始化
    #position填充开始下标
    assert dim in [0,1]

    if width <= 0:
        return input

    def constant_0(m):
        torch.nn.init.constant_(m, 0)
    def constant_1(m):
        torch.nn.init.constant_(m, 1)
    def kaiming(m):
        torch.nn.init.kaiming_normal_(m)

    if init_mode == 0:
        init_method = constant_0

    elif init_mode == 1:
        init_method = constant_1

    elif init_mode == 2:
        init_method = kaiming

    #conv
    if len(input.shape) == 4:
        if dim == 0:
            temp = torch.empty(width,input.shape[1],input.shape[2],input.shape[3])

        elif dim == 1:
            temp = torch.empty(input.shape[0],width,input.shape[2],input.shape[3])

    elif len(input.shape) == 2:
        if dim == 0:
            temp = torch.empty(width,input.shape[1])

        elif dim == 1:
            temp = torch.empty(input.shape[0],width)

    elif len(input.shape) == 1:
        temp = torch.empty(width)

    init_method(temp)

    if position == 0:
        return torch.cat((temp, input),dim=dim)
    elif position > 0:
        if dim == 0:
            a = input[:position]
            b = input[position:]
            return torch.cat((a, temp, b),dim=dim)
        elif dim == 1:
            a = input[:,:position]
            b = input[:,position:]
            return torch.cat((a, temp, b),dim=dim)
    else:
        return torch.cat((input,temp),dim=dim)


def tensor_expend(input, dim, init_mode=0, width=1):
    #在dim维度添加size的tensor，填充到width宽
    #init_mode: 0-填充0; 1-填充1; 2-凯明初始化
    #统一填充到最后
    assert dim in [0,1]

    add_width = width - input.shape[dim]


    if add_width <= 0:
        return input

    if init_mode == 0:
        def constant_0(m):
            torch.nn.init.constant_(m, 0)
        init_method = constant_0

    elif init_mode == 1:
        def constant_1(m):
            torch.nn.init.constant_(m, 1)
        init_method = constant_1

    elif init_mode == 2:
        def kaiming(m):
            torch.nn.init.kaiming_normal_(m)
        init_method = kaiming

    #conv
    if len(input.shape) == 4:
        if dim == 0:
            temp = torch.empty(add_width,input.shape[1],input.shape[2],input.shape[3])

        elif dim == 1:
            temp = torch.empty(input.shape[0],add_width,input.shape[2],input.shape[3])
    #linear
    elif len(input.shape) == 2:
        if dim == 0:
            temp = torch.empty(add_width,input.shape[1])

        elif dim == 1:
            temp = torch.empty(input.shape[0],add_width)
    #bn
    elif len(input.shape) == 1:
        temp = torch.empty(add_width)

    init_method(temp)
    temp.to(input.device)

    return torch.cat((input, temp),dim=dim)



def del_tensor_bool(input, dim, mask):
    #按bool列表删除，mask必须和input同形状
    #删除False的项
    if dim == 0:
        return input[mask]
    if dim == 1:
        return input[:,mask]
    if dim == 2:
        return input[:,:,mask]
    if dim == 3:
        return input[:,:,:,mask]

#======================================================
def prune_channel_vgg(parameter,channel):#根据parameter得到通道数
    channel_s = channel.copy()
    weight_size = []
    count = 0
    for k in parameter:#下标对应层数
        if k[-6:] == 'weight' and len(parameter[k].shape) == 4:
            weight_size.append(parameter[k].shape[0])
    for x in range(len(channel)):#count表示第几层卷积层
        if channel[x] == 'M':
            continue
        else :
            channel_s[x] = weight_size[count]
            count += 1
    return channel_s

def prune_channel_resnet(parameter):#根据parameter得到通道数

    channel = []

    for idx in range(1,10):
        layer_temp = []
        layer = 'layer'+str(idx)
        bottleneck = 0
        while(1):
            if layer+'.'+str(bottleneck)+'.conv1.weight' in parameter:
                b_temp = []
                b_conv = 1
                while(1):
                    key = layer+'.'+str(bottleneck)+'.conv'+str(b_conv)+'.weight'
                    if key in parameter:
                        b_temp.append(parameter[key].shape[0])
                        b_conv+=1
                    else:
                        break
                layer_temp.append(b_temp)
                bottleneck += 1
            else :
                break

        if not layer_temp :
            break
        channel.append(layer_temp)

    return channel

def get_block_number(key):
    for idx in range(len(key)):
        if key[idx] == '.':
            start = idx+1
            for idx_1 in range(start,len(key)):
                if key[idx_1] == '.':
                    end = idx_1
                    break
            break

    return int(key[start:end])
def prune_channel_mobilenetV2(parameter):#根据parameter得到通道数

    output=[]
    end = block = -1
    for key in parameter:
        if key[:8] == 'features' and key[-6:] == 'weight' and len(parameter[key].shape) == 4:
            if block == get_block_number(key):
                output[end].extend([parameter[key].shape[0]])
            else :
                block = get_block_number(key)
                end += 1
                output.append([parameter[key].shape[0]])

    del output[0]
    del output[-1]
    for idx in range(len(output)):#修整
        if output[idx][0] == output[idx][1]:
            del output[idx][0]

    return output