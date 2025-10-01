import torch
import torch.nn as nn
import torch.nn.functional as F


#构成conv-bn-relu结构，根据参数有很多变种
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        #        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)

        return x
    

# 一般的Resnet-cifar，但用的是GN层
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
    

class Guidance(nn.Module):
    def __init__(self, output_dim=64, norm_fn='batch'):
        super(Guidance, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        ## 4x use ##
        self.conv_start = nn.Sequential(nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3), self.norm1, nn.ReLU(inplace=True))
        
        ## 2x use ##
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        # self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)

        # output convolution
        self.conv_g0 = nn.Sequential(BasicConv(64, 64, kernel_size=3, padding=1), #俩基础2dconv-bn-relu结构
                                     BasicConv(64, 64, kernel_size=3, padding=1))

        self.guidance = nn.Conv2d(64, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv_g0(x)
        g = self.guidance(x)

        return g
    
#来自论文RAFT[41]的凸上采样模块,固定4倍上采样
class PropgationNet_4x(nn.Module):
    def __init__(self, base_channels):
        super(PropgationNet_4x, self).__init__()
        self.base_channels = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1,padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, 9 * 16, kernel_size=(3, 3), stride=(1, 1), padding=1,dilation=(1, 1), bias=False)
            )

    def forward(self, guidance, disp):
        # guidance为下采样之前的左图，所以可以用来指导disp上采样
        b, c, h, w = disp.shape
        disp = F.unfold(4 * disp, [3, 3], padding=1).view(b, 1, 9, 1, 1, h, w) #首先视差*4，然后按3*3卷积方式展开，再view为[B,1,9,1,1,h,w]
        mask = self.conv(guidance).view(b, 1, 9, 4, 4, h, w) #reshape到可以广播的维度
        mask = F.softmax(mask, dim=2)
        up_disp = torch.sum(mask * disp, dim=2) #逐元素相乘后在dim=2上求和，得到 [b,1,4,4,h,w]
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3) #转置矩阵，得到 [b,1,h,4,w,4]
        return up_disp.reshape(b, 1, 4 * h, 4 * w) #reshape（把h维度和4维度合并）为 [b,1,4h,4w]