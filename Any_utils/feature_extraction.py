import torch
import torch.nn as nn
import torch.nn.functional as F

def preconv2d(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bn=True):
    if bn:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        if is_deconv:
            self.up = nn.Sequential(
                nn.BatchNorm2d(in_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
            )
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2) #上采样层
            in_size = int(in_size * 1.5)

        self.conv = nn.Sequential(
            preconv2d(in_size, out_size, 3, 1, 1),
            preconv2d(out_size, out_size, 3, 1, 1),
        )

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        buttom, right = inputs1.size(2)%2, inputs1.size(3)%2
        outputs2 = F.pad(outputs2, (0, -right, 0, -buttom)) #如果w是奇数，right 就是 -1，表示裁掉右边 1 列。以此类推
        return self.conv(torch.cat([inputs1, outputs2], 1))

# U-Net 特征提取器 from anynet
class feature_extraction_conv(nn.Module):
    def __init__(self, init_channels,  nblock=2):
        super(feature_extraction_conv, self).__init__()

        self.init_channels = init_channels
        nC = self.init_channels
        downsample_conv = [nn.Conv2d(3,  nC, 3, 1, 1), # 512x256
                                    preconv2d(nC, nC, 3, 2, 1)]
        downsample_conv = nn.Sequential(*downsample_conv)

        inC = nC
        outC = 2*nC
        block0 = self._make_block(inC, outC, nblock)
        self.block0 = nn.Sequential(downsample_conv, block0) #block == BN-ReLU-Conv2d  ;  block0 == block[1,2]-block[2,2]

        nC = 2*nC
        self.blocks = []
        for i in range(2):
            self.blocks.append(self._make_block((2**i)*nC,  (2**(i+1))*nC, nblock)) #0: block[2,4]-block[4,4] ; 1: block[4,8]-block[8,8]

        self.upblocks = []
        for i in reversed(range(2)):
            self.upblocks.append(unetUp(nC*2**(i+1), nC*2**i, False)) #上采样模块

        self.blocks = nn.ModuleList(self.blocks)
        self.upblocks = nn.ModuleList(self.upblocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_block(self, inC, outC, nblock ):
        model = []
        model.append(nn.MaxPool2d(2,2))
        for i in range(nblock):
            model.append(preconv2d(inC, outC, 3, 1, 1))
            inC = outC
        return nn.Sequential(*model)


    def forward(self, x):
        downs = [self.block0(x)]
        for i in range(2):
            downs.append(self.blocks[i](downs[-1]))
        downs = list(reversed(downs))
        for i in range(1,3):
            downs[i] = self.upblocks[i-1](downs[i], downs[i-1])
        return downs

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                                    nn.BatchNorm2d(out_channels),)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out
    
# 参考 GwcNet [12] from DCANet
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 2, 1, 1, 1)
        #BasicBlock 是 Convbn(k=步长) - ReLU - Convbn(k固定为1) 的结构，输入和输出之间有残差连接。比Resnet-cifar的一个块的结构更加简单
        #_make_layer() 的参数是：块的类型，输出通道，块的堆叠次数，步长，pad，空洞卷积
        self.layer2 = self._make_layer(BasicBlock, 64, 2, 2, 1, 1) #输出图片的size相比输入减少一半
        self.layer3 = self._make_layer(BasicBlock, 128, 2, 1, 1, 1) #后面的输出size不变
        # self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        
        # self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1,
        #                                         bias=False))
        #对最后的拼接的输出在通道上进行压缩，压缩到concat_feature_channel

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        # 仅在stride不是1的时候用downsample层，是一个ConvBN结构
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer2(x)

        gwc_feature = torch.cat((l2, l3), dim=1)

        return gwc_feature