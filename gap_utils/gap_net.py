#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:11:47 2023

@author: liqi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List
from torch import Tensor

# class demoNET(nn.Module):
#     def __init__(self, num_classes):
#         super(demoNET, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(3)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.bn2 = nn.BatchNorm2d(3)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(8*8*3, num_classes)


#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x

class demoNET(nn.Module):
    def __init__(self, num_classes):
        super(demoNET, self).__init__()
        out_channel = 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channel, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8*8*out_channel, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class demoNET2(nn.Module):
    def __init__(self, num_classes):
        super(demoNET2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(8192, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int,
        hidden_channel: int,
        use_res_connect: bool,
        first_IR:bool = False,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        norm_layer = nn.BatchNorm2d

        self.use_res_connect = use_res_connect
        layers: List[nn.Module] = []

        if not first_IR:
            # pw
            layers.append(ConvBNReLU(input_channel, hidden_channel, kernel_size=1))

        layers.extend([
            # dw
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # pw-linear
            nn.Conv2d(hidden_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(output_channel),
        ])

        self.conv = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:

        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_mod(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        layers: dict = None,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2_mod, self).__init__()

        block = InvertedResidual

        # building first layer

        features: List[nn.Module] = [ConvBNReLU(3, layers['input_channel'], stride=layers['in_stride'])]
        hidden_channel, output_channel, stride, use_res_connect = layers['IR'][0]
        features.append(block(input_channel = layers['input_channel'], hidden_channel=hidden_channel, output_channel=output_channel,
                              stride = stride, first_IR=True, use_res_connect=use_res_connect))
        input_channel = output_channel

        # building inverted residual blocks
        for hidden_channel, output_channel, stride, use_res_connect in layers['IR'][1:]:
            features.append(block(input_channel = input_channel, hidden_channel=hidden_channel, output_channel=output_channel,
                                  stride = stride, use_res_connect=use_res_connect))
            input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, layers['last_channel'], kernel_size=1))

        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(layers['last_channel'], num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def MobilenetV2():
    #IR: [hidden_channel, output_channel, stride, use_res]
    layers = {
        'input_channel' : 32,
        'in_stride' : 2,
        'last_channel' : 1280,
        'IR' : [[32, 16, 1, False],
                [96, 24, 2, False],
                [144, 24, 1, True],
                [144, 32, 2, False],
                [192, 32, 1, True],
                [192, 32, 1, True],
                [192, 64, 2, False],
                [384, 64, 1, True],
                [384, 64, 1, True],
                [384, 64, 1, True],
                [384, 96, 1, False],
                [576, 96, 1, True],
                [576, 96, 1, True],
                [576, 160, 2, False],
                [960, 160, 1, True],
                [960, 160, 1, True],
                [960, 320, 1, False]],
    }
    return layers

def mob_t1(num_classes : int , parameter = None) -> MobileNetV2_mod:

    #IR: [hidden_channel, output_channel, stride, use_res]
    layers = {
        'input_channel' : 32,
        'in_stride' : 1,
        'last_channel' : 320,
        'IR' : [[32, 16, 1, False],
                [24, 12, 2, False],
                [18, 16, 2, False],
                [24, 32, 2, False],
                [48, 48, 1, False],
                [72, 80, 2, False],
                [120, 160, 1, False],]
            }

    model = MobileNetV2_mod(num_classes = num_classes, layers = layers)
    if parameter :
        model.load_state_dict(parameter)

    return model

def MobilenetV2_tight(num_classes : int , parameter = None) -> MobileNetV2_mod:

    #IR: [hidden_channel, output_channel, stride, use_res]
    layers = {
            'input_channel' : 32,
            'in_stride' : 1,
            'last_channel' : 320,
            'IR' : [[32, 16, 1, False],#!
                    [16, 24, 2, False],#!
                    [16, 24, 1, True],
                    [24, 32, 2, False],#!
                    [32, 32, 1, True],
                    [32, 32, 1, True],
                    [32, 64, 2, False],#!
                    [64, 64, 1, True],
                    [64, 64, 1, True],
                    [64, 64, 1, True],
                    [64, 96, 1, False],#!
                    # [96, 96, 1, True],
                    # [96, 96, 1, True],
                    [96, 160, 2, False],#!
                    # [160, 160, 1, True],
                    # [160, 160, 1, True],
                    [160, 320, 1, False]],#!
        }

    model = MobileNetV2_mod(num_classes = num_classes, layers = layers)
    if parameter :
        model.load_state_dict(parameter)

    return model

def mob_t_test(num_classes : int , parameter = None) -> MobileNetV2_mod:

    #IR: [hidden_channel, output_channel, stride, use_res]
    layers = {
            'input_channel' : 32,
            'in_stride' : 2,
            'last_channel' : 320,
            'IR' : [[32, 16, 1, False],#!
                    [16, 24, 2, False],#!
                    [16, 24, 1, True],
                    [24, 32, 2, False],#!
                    [64, 32, 1, True],
                    [512, 32, 1, True], #5
                    [32, 64, 2, False],#!
                    [64, 64, 1, True],
                    [64, 64, 1, True],
                    [64, 64, 1, True],
                    [64, 96, 1, False],#!
                    [16, 96, 1, True],
                    [64, 96, 1, True],
                    [96, 160, 2, False],#!
                    [256, 160, 1, True],
                    [16, 160, 1, True],
                    [160, 320, 1, False]],#!
        }

    model = MobileNetV2_mod(num_classes = num_classes, layers = layers)
    if parameter :
        model.load_state_dict(parameter)

    return model


net = demoNET2(10)

if __name__ == '__main__':
    from pytorch_utils.common import thop_macs
    # net = demoNET2(10)
    x = torch.randn(1,3,32,32)
    macs, params = thop_macs(net)
    print('macs:',macs, 'params:', params,)
    y = net(x)