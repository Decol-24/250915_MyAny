import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from DCA_utils.cva import cva
from DCA_utils.submodule import convbn,BasicBlock,convbn_3d,Guidance,build_gwc_volume,build_concat_volume,disparity_regression


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        #BasicBlock 是 Convbn(k=步长) - ReLU - Convbn(k固定为1) 的结构，输入和输出之间有残差连接。比Resnet-cifar的一个块的结构更加简单
        #_make_layer() 的参数是：块的类型，输出通道，块的堆叠次数，步长，pad，空洞卷积
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1) #输出图片的size相比输入减少一半
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1) #后面的输出size不变
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
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
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        #返回值是由后三个块的输出拼接构成，channel为320

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}
        #两个值都返回gwc_feature为320，concat_feature为12

#来自论文RAFT[41]的凸上采样模块
class PropgationNet_4x(nn.Module):
    def __init__(self, base_channels):
        super(PropgationNet_4x, self).__init__()
        self.base_channels = base_channels
        self.conv = nn.Sequential(convbn(base_channels, base_channels * 2, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(base_channels * 2, 9 * 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                            dilation=(1, 1), bias=False))
                                #输出通道是固定的 9 * 16

    def forward(self, guidance, disp):
        # guidance为下采样之前的左图，所以可以用来指导disp上采样
        b, c, h, w = disp.shape
        disp = F.unfold(4 * disp, [3, 3], padding=1).view(b, 1, 9, 1, 1, h, w) #首先视差*4，然后按3*3卷积方式展开，再view为[B,1,9,1,1,h,w]
        mask = self.conv(guidance).view(b, 1, 9, 4, 4, h, w) #reshape到可以广播的维度
        mask = F.softmax(mask, dim=2)
        up_disp = torch.sum(mask * disp, dim=2) #逐元素相乘后在dim=2上求和，得到 [b,1,4,4,h,w]
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3) #转置矩阵，得到 [b,1,h,4,w,4]
        return up_disp.reshape(b, 1, 4 * h, 4 * w) #reshape（把h维度和4维度合并）为 [b,1,4h,4w]

class GwcNet(nn.Module):
    def __init__(self, start_disp, end_disp, use_concat_volume=True):
        super(GwcNet, self).__init__()
        self.use_concat_volume = use_concat_volume
        self.max_disp = end_disp - start_disp
        self.num_groups = 40
        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
            #经过压缩后最后的输出通道是12
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1), #输入是40+24
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.cva1 = cva(32, downsample=True)
        self.cva2 = cva(32, downsample=True)
        self.cva3 = cva(32, downsample=True)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.guidance = Guidance(64) #类似Resnet
        self.prop = PropgationNet_4x(64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, left, right):

        features_left = self.feature_extraction(left)
        # ['gwc_feature'].shape = [1,320,64,128]    ['concat_feature'].shape = [1, 12, 64, 128]
        features_right = self.feature_extraction(right)

        guidance = self.guidance(left) #根据左图构建指导体

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.max_disp // 4, self.num_groups)
        #[B, num_groups, maxdisp, H, W]
        #构造分组代价体，遵循GWC模式
        #从这里开始考虑最大视差就只有原来的1/4，来减少计算量
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.max_disp // 4) #[B, 2 * C, maxdisp, H, W]
            #用提取的拼接用的特征体构造拼接代价体，遵循PSMNet。拼接用的特征更小
            volume = torch.cat((gwc_volume, concat_volume), 1)
            #把两个代价体拼接
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume) #一个3D卷积提取左右特征
        cost0 = self.dres1(cost0) + cost0 #残差链接

        #### augment cv ####
        prob_volume1, augmented_cost = self.cva1(cost0)
        out1 = cost0 + augmented_cost

        prob_volume2, out2 = self.cva2(out1) #prob_volume是分类结果
        prob_volume3, out3 = self.cva3(out2) #[1,32,48,64,128]

        # convex upsample
        out3 = self.classif3(out3) #[1,1,48,64,128]
        # cost2 = F.upsample(out2, scale_factor=(4, 4, 4), mode='trilinear')
        cost3 = torch.squeeze(out3, 1)
        cost3 = F.softmax(cost3, dim=1)

        pred4 = disparity_regression(cost3, self.max_disp//4) #[1,1,64,128]
        pred4 = self.prop(guidance['g'], pred4) #[1,1,256,512]

        # 构建多个输出头，单独指导CVA层的训练。仅在训练时使用
        if self.training:
            out0 = self.classif0(cost0) # 刚聚合后3D卷积的结果
            # out0 = F.upsample(out0, scale_factor=(4, 4, 4), mode='trilinear')
            out0 = torch.squeeze(out0, 1)
            pred0 = F.softmax(out0, dim=1) #[1,48,64,128]
            # pred0 = disparity_regression(pred0, self.maxdisp)

            out_dca1 = F.interpolate(prob_volume1, scale_factor=(2, 2, 2), mode='trilinear') # 第一个CVA层的分类结果
            out_dca1 = torch.squeeze(out_dca1, 1)
            pred_dca1 = F.softmax(out_dca1, dim=1)#[1,48,64,128]
            # pred_dca0 = disparity_regression(pred_dca0, self.maxdisp)

            out_dca2 = F.interpolate(prob_volume2, scale_factor=(2, 2, 2), mode='trilinear') # 第二个CVA层的分类结果
            out_dca2 = torch.squeeze(out_dca2, 1)
            pred_dca2 = F.softmax(out_dca2, dim=1) #[1,48,64,128]
            # pred_dca1 = disparity_regression(pred_dca1, self.maxdisp)

            out_dca3 = F.interpolate(prob_volume3, scale_factor=(8, 8, 8), mode='trilinear') # 第三个CVA层的分类结果，用了和最终输出接近的处理后再计算loss
            out_dca3 = torch.squeeze(out_dca3, 1)
            pred_dca3 = F.softmax(out_dca3, dim=1) #[1,192,256,512]
            pred_dca3 = disparity_regression(pred_dca3, self.max_disp) #[1,1,256,512]

            out1 = self.classif1(out1) # 第一个CVA层的识别结果
            # out1 = F.upsample(out1, scale_factor=(4, 4, 4), mode='trilinear')
            cost1 = torch.squeeze(out1, 1)
            pred1 = F.softmax(cost1, dim=1) #[1,48,64,128]
            # pred1 = disparity_regression(pred1, self.maxdisp)

            out2 = self.classif2(out2) # 第二个CVA层的识别结果
            # out2 = F.upsample(out2, scale_factor=(4, 4, 4), mode='trilinear')
            cost2 = torch.squeeze(out2, 1)
            pred2 = F.softmax(cost2, dim=1) #[1,48,64,128]

            return [pred0, pred_dca1, pred_dca2,
                    pred1, pred2], [pred_dca3, pred4]
            # return [pred0, pred_dca0, pred_dca1, pred_dca2, pred1, pred2, pred3]

        else:
            return pred4