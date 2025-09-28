from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from DCA_utils.Anynet_submodule import post_3dconvs,feature_extraction,Guidance

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
        return "{:4f}".format(self.avg_time(id,count))

    def reset(self):
        self.time_temp = [0 for x in range(self.num)]
        self.time_result = [0 for x in range(self.num)]

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential()

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


class AnyNet(nn.Module):
    def __init__(self):
        super(AnyNet, self).__init__()

        self.refine_spn = None

        self.feature_extraction = feature_extraction() #Unet

        self.volume_postprocess = []

        # for i in range(3):
        #     net3d = post_3dconvs(self.layers_3d, self.channels_3d*self.growth_rate[i]) #Conv3D[1,16] - Conv3D[16,16] * $ - Conv3D[16,1]
        #     self.volume_postprocess.append(net3d)
        # self.volume_postprocess = nn.ModuleList(self.volume_postprocess) # 3个post_3dconvs

        self.guidance = Guidance(64) #类似Resnet
        self.prop = PropgationNet_4x(64)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        # x 是扩展后的右图，disp是上层的视差
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1) # 数列[0-64] 在第一个维度复制到[32,64]
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W) # 数列[0-32] 在第二个维度复制到[32,64]
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1) #复制到[30,1,32,64]
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float() #拼接 [30,2,32,64]

        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp #vgrid的第1维度的第一项减去disp

        # scale grid to [-1,1] 缩放回[-1,1] 第1维度的第一项因为减过disp，所以会比-1更小
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1) #[30,32,64,2]
        output = nn.functional.grid_sample(x, vgrid) #按照vgrid从x中采样，vgrid的含义是[N, H_out, W_out, 2]，最后两个维度是坐标，范围在[-1,1]。越界的采样点直接补0
        # 对每个像素都进行修正，竖方向不修改，横方向根据disp中对应的值移动
        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
        # feat_l [6,8,16,32]
        cost = torch.zeros((feat_l.size()[0], maxdisp//stride, feat_l.size()[2], feat_l.size()[3]), device='cuda') #[6, 12, 16, 32]
        for i in range(0, maxdisp, stride):
            cost[:, i//stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1) #原始左图放到代价体左边。i==0的时候实际上什么都没做
            if i > 0:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1) # 右图往右平移后，左图减右图。表示两个特征向量之间的L1距离
            else:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1) # i==0的时候左右图特征重叠

        return cost.contiguous()

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        # disp 为上一层的视差输出 [1]:[6,1,32,64]
        size = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,size[-2], size[-1]) #在batch维度之后的维度复制(maxdisp*2-1)份，然后和batch维度合为一个维度 [30,1,32,64]
        batch_shift = torch.arange(-maxdisp+1, maxdisp, device='cuda').repeat(size[0])[:,None,None,None] * stride #创建视差偏移[-2:2]，然后扩展 [30,1,1,1]
        batch_disp = batch_disp - batch_shift.float() #减去视差偏移，表示在前级的视差结果再尝试多个偏移，如果偏移后的结果是正确，那么左右图将完全匹配
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1]) #对这层的输出进行同样的复制操作 [30,4,32,64]
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0],-1, size[2],size[3])
        return cost.contiguous()


    def forward(self, left, right):

        feats_l = self.feature_extraction(left) #[0]：[1,32,64,128]
        feats_r = self.feature_extraction(right)

        guidance = self.guidance(left) #根据左图构建指导体

        pred = [feats_l]
        
        # for scale in range(len(feats_l)): #scale表示阶段 [0-2]
        #     if scale > 0:
        #         wflow = F.upsample(pred[scale-1], (feats_l[scale].size(2), feats_l[scale].size(3)),
        #                            mode='bilinear') * feats_l[scale].size(2) / img_size[2] #把上一层的结果上采样
        #         cost = self._build_volume_2d3(feats_l[scale], feats_r[scale],
        #                                  self.maxdisplist[scale], wflow, stride=1)
        #     else:
        #         cost = self._build_volume_2d(feats_l[scale], feats_r[scale],
        #                                      self.maxdisplist[scale], stride=1)
        #     cost = torch.unsqueeze(cost, 1) #[0]:(6,1,12,16,32)
        #     cost = self.volume_postprocess[scale](cost) #用第scale个post_3dconvs处理cost
        #     cost = cost.squeeze(1)  #[0]:(6,12,16,32)
        #     if scale == 0:
        #         pred_low_res = disparityregression2(0, self.maxdisplist[0])(F.softmax(-cost, dim=1))
        #         pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2) #按图片shape压缩的比例对数值放大，以配合下一步的上采样
        #         disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
        #         pred.append(disp_up)
        #     else:
        #         pred_low_res = disparityregression2(-self.maxdisplist[scale]+1, self.maxdisplist[scale], stride=1)(F.softmax(-cost, dim=1))
        #         pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
        #         disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
        #         pred.append(disp_up+pred[scale-1])


        pred4 = self.prop(guidance, pred4) #[1,1,256,512]

        return pred

#和GC-net一样的softmax
class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1).float() #[1,12,1,1] 视差回归用数列。[1,i,1,1] = i

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3]) #扩张到(6,12,16,32)
        out = torch.sum(x * disp, 1, keepdim=True) #然后乘到x上
        return out