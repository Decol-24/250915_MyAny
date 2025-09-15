#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:11:47 2023

@author: liqi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class zero_out_conv(nn.Module):
    def __init__(self, out_channel, kernel_size=1, stride=1, padding=0):
        super(zero_out_conv, self).__init__()
        self.conv = nn.Conv2d(out_channel//2, out_channel//2, kernel_size, stride=stride, padding=padding, bias=False, groups=out_channel//2)
        torch.nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class DownsampleA(nn.Module):
    #featuremap size根据stride减半，再拼接同shape的元素为0的tensor
    def __init__(self, in_channel, out_channel, stride):
        super(DownsampleA, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
        self.zero_out = zero_out_conv(out_channel)

    def forward(self, x):

        x = self.avg(x)
        x_2 = self.zero_out(x)
        return torch.cat((x, x_2), 1)

class BasicBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 hidden_channel,
                 out_channel,
                 stride = 1,
                 use_downsample = False,):

        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.stride = stride
        self.use_downsample = use_downsample
        self.input_size = 0

        self.conv1 = nn.Conv2d(self.in_channel, self.hidden_channel, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.hidden_channel)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.hidden_channel, self.out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        self.ReLU2 = nn.ReLU(inplace=True)

        if use_downsample:
            # self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
            self.downsample = DownsampleA(self.in_channel, self.out_channel, self.stride)


    def forward(self, x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.ReLU1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.use_downsample:
            x = self.downsample(x)

        return self.ReLU2(x + y)

class ConvBNActivation(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) // 2 * dilation
        norm_layer = nn.BatchNorm2d
        activation_layer = nn.ReLU


        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,bias=False)
        self.bn = norm_layer(out_planes)
        self.activation = activation_layer(inplace=True)

        self.in_channels = in_planes
        self.out_channels = out_planes
        self.stride = stride

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#cifar
class VGGNet(nn.Module):
    def __init__(self, arch_config, num_classes=10):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, arch_config['in_channel'], kernel_size=3, stride=arch_config['in_stride'], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(arch_config['in_channel'])
        layers = []

        front_out_channel = arch_config['in_channel']
        for (output_channel,stride) in arch_config['block_arch']:
            new_block = ConvBNActivation(front_out_channel,output_channel,stride=stride)
            layers.append(new_block)
            front_out_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(16,stride=16)
        self.linear = nn.Linear(7*7*output_channel, num_classes)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_flops(self):
        import thop
        x = torch.randn(1,32,112,112)

        self.to('cpu')
        macs, params = thop.profile(self.features,inputs=(x,))
        macs, params = self.clever_format([macs, params], "%.2f")
        # print('macs:',macs, 'params:', params,)
        return macs, params

    def clever_format(self, nums, format="%.2f"):
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


    def get_param(self):
        param_group = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                param_group.append(m.weight)
                if m.bias is not None:
                    param_group.append(m.bias)
            elif isinstance(m, nn.Linear):
                param_group.append(m.weight)
                if m.bias is not None:
                    param_group.append(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                param_group.append(m.weight)
                param_group.append(m.bias)

        return [{'params': param_group}]


def resnet56_mod(num_classes):
    #IR: [hidden_channel, output_channel, stride, use_downsample]
    #追加实验用
    #仅修改hidden_channel
    arch_frame =[
                [15, 1],
                ]

    arch_config = {
        'in_channel': 32,
        'in_stride' : 2,
        'block_arch' : arch_frame,
        'Res_mode' : 'Basic',
    }

    net = VGGNet(arch_config, num_classes)
    return net


if __name__ == '__main__':

    net = resnet56_mod(10)