#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: liqi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List
from torch import Tensor

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
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
        first_IR:bool = False,
        use_res_connect = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = use_res_connect
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        if not first_IR:
            # pw
            layers.append(ConvBNReLU(input_channel, hidden_channel, kernel_size=1, norm_layer=norm_layer))

        layers.extend([
            # dw
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel, norm_layer=norm_layer),
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

class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        layers: dict = None,
    ) -> None:

        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        # building first layer
        features: List[nn.Module] = [ConvBNReLU(3, layers['input_channel'], stride=layers['in_stride'])]
        hidden_channel, output_channel, stride, use_res_connect = layers['IR'][0]
        features.append(block(input_channel = layers['input_channel'], hidden_channel=hidden_channel, output_channel=output_channel, stride = stride, first_IR=True, use_res_connect=use_res_connect))
        input_channel = output_channel

        # building inverted residual blocks
        for hidden_channel, output_channel, stride, use_res_connect in layers['IR'][1:]:
            features.append(block(input_channel = input_channel, hidden_channel=hidden_channel, output_channel = output_channel, stride = stride, use_res_connect=use_res_connect))
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

def MobilenetV2(num_classes : int , parameter = None) -> MobileNetV2:

    #IR: [hidden_channel, output_channel, stride, use_res_connect]
    layers = {
        'IR_mode' : 'IR',
        'input_channel' : 32,
        'in_stride' : 2, #modfied from 2
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

    model = MobileNetV2(num_classes = num_classes, layers = layers)
    if parameter :
        model.load_state_dict(parameter)

    return model

if __name__ == '__main__':
    net = MobilenetV2(10)
    print(net)
    x = torch.randn(10,3,32,32)
    y = net(x)