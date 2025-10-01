from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from DCA_utils.Anynet_submodule import feature_extraction,cross_attention
from DCA_utils.upsample_net import Guidance,PropgationNet_4x

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


class AnyNet(nn.Module):
    def __init__(self, start_disp, end_disp):
        super(AnyNet,self).__init__()

        self.refine_spn = None
        self.disparity_arange = self.disparity_segmentation(start_disp//4,end_disp//4)

        self.feature_extraction = feature_extraction() #Unet

        self.attention_1 = cross_attention(64)
        self.attention_2 = cross_attention(64)
        self.attention_3 = cross_attention(64)

        self.guidance = Guidance(64) #类似Resnet
        self.up = PropgationNet_4x(64)


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

    def disparity_segmentation(self,start_disp,end_disp,step=3):
        temp = []
        temp.append(torch.arange(start_disp, end_disp, step=step, device='cuda',requires_grad=False))
        temp.append(torch.arange(-step+1, step, device='cuda',requires_grad=False))
        return temp


    def forward(self, left, right):

        feats_l = self.feature_extraction(left) #[0]：[1,32,64,128]
        feats_r = self.feature_extraction(right)

        guidance = self.guidance(left) #根据左图构建指导体

        preds = []

        cost = self._build_volume_2d(feats_l, feats_r, self.disparity_arange[0])
        disp_1 = self.attention_1(cost,cost)
        disp_re_1 = disparity_regression2(disp_1, self.disparity_arange[0])
        
        preds.append(disp_re_1)

        cost = self._build_volume_2d(feats_l, feats_r, self.disparity_arange[1])
        disp_2 = self.attention_2(cost,disp_1)
        disp_re_2 = disparity_regression2(disp_2, self.disparity_arange[1])
        preds.append(disp_re_1+disp_re_2)

        preds.append(self.up(guidance, preds[-1])) #[1,1,256,512]

        return preds
    
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


    def _build_volume_2d(self, feat_l, feat_r, disparity_arange):

        num_disp = disparity_arange.shape[0]
        max_disp = disparity_arange.max()
        B,feature_size,H,W = feat_l.shape

        cost = torch.zeros((feat_l.shape[0], num_disp, feature_size*2, feat_l.shape[2], feat_l.shape[3]+max_disp), device='cuda') #[B, disp, feature *2, W, H]
        for i,dis in enumerate(disparity_arange):
            cost[:, i, :feature_size, :, 0:W] = feat_l[:, :, :, :]
            cost[:, i, feature_size:, :, dis:W+dis] = feat_r[:, :, :, :]

        return cost.contiguous()

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        # disp 为上一层的视差输出
        size = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,size[-2], size[-1]) #在batch维度之后的维度复制(maxdisp*2-1)份，然后和batch维度合为一个维度 [30,1,32,64]
        batch_shift = torch.arange(-maxdisp+1, maxdisp, device='cuda').repeat(size[0])[:,None,None,None] * stride #创建视差偏移[-2:2]，然后扩展 [30,1,1,1]
        batch_disp = batch_disp - batch_shift.float() #减去视差偏移，表示在前级的视差结果再尝试多个偏移，如果偏移后的结果是正确，那么左右图将完全匹配
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1]) #对这层的输出进行同样的复制操作 [30,4,32,64]
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0],-1, size[2],size[3])
        return cost.contiguous()

#和GC-net一样的softmax TODO 特征维度应该降为1
def disparity_regression2(x, disparity_arange):
    disp = disparity_arange.view(1, -1, 1, 1, 1).float()
    disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3], x.size()[4]) #扩张到(6,12,16,32)
    x = F.softmax(x, dim=1)
    x = x * disp
    out = torch.sum(x, 2, keepdim=True) #乘到x上后求和
    return out