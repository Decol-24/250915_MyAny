from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from Any_utils.Anynet_submodule import cross_attention
from Any_utils.feature_extraction import feature_extraction
from Any_utils.upsample_net import Guidance,PropgationNet_4x

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
    
    def all_avg_time_str(self,count=1):
        str = ""
        for i in range(self.num):
            str = str+"Work {} is {:4f}s.\n".format(i,self.avg_time(i,count))
        return str

    def reset(self):
        self.time_temp = [0 for x in range(self.num)]
        self.time_result = [0 for x in range(self.num)]


class AnyNet(nn.Module):
    def __init__(self, start_disp, end_disp, device, **kwargs):
        super(AnyNet,self).__init__()

        self.start_disp = start_disp
        self.end_disp = end_disp
        self.device = device
        self.disparity_arange = None
        self.num_groups = 32

        self.feature_extraction = feature_extraction() #Unet
        self.feature_extraction_2 = feature_extraction() #Unet

        self.attention_1 = cross_attention(self.num_groups,key_query_num_convs=1)
        self.attention_2 = cross_attention(self.num_groups,key_query_num_convs=1)
        self.attention_3 = cross_attention(self.num_groups,key_query_num_convs=1)

        self.classif_1 = nn.Sequential(nn.Conv3d(self.num_groups, self.num_groups, kernel_size=3, stride=1,padding=1, bias=False),
                                nn.BatchNorm3d(self.num_groups),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(self.num_groups, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.guidance = Guidance(self.num_groups) #类似Resnet
        self.up = PropgationNet_4x(self.num_groups)

        self.t = time_counter(num=7)

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
    
    def set_disparity_arange(self,disparity_arange):
        self.disparity_arange = disparity_arange

    def forward(self, left, right):
        self.t.start(0)

        feats_l = self.feature_extraction(left) #[0]：[1,320,64,128]
        feats_r = self.feature_extraction(right)

        guidance = self.guidance(left) #根据左图构建指导体

        self.t.end(0)
        self.t.start(1)

        disps = []

        cost = self._build_gwc_volume(feats_l, feats_r, self.disparity_arange[0], self.num_groups) #[B,32,disp,H,W]

        self.t.end(1)
        self.t.start(2)

        disp_1 = self.attention_1(cost,cost)  #[B,32,disp,H,W]

        self.t.end(2)
        self.t.start(3)

        disps.append(disp_1)

        cost = self._build_gwc_volume(feats_l, feats_r, self.disparity_arange[1], self.num_groups) #[B,32,disp,H,W]
        disp_2 = self.attention_2(cost,cost)  #[B,32,disp,H,W]

        self.t.end(3)
        self.t.start(4)

        disps.append(disp_2)

        cost = self._build_gwc_volume(feats_l, feats_r, self.disparity_arange[2], self.num_groups) #[B,32,disp,H,W]
        disp_3 = self.attention_3(cost,cost)  #[B,32,disp,H,W]

        self.t.end(4)
        self.t.start(5)

        disps.append(disp_3)

        merged_disp = self._merge_volume(disps,self.disparity_arange)

        self.t.end(5)
        self.t.start(6)


        pred = self.classif_1(merged_disp).squeeze(1) #[B,1,maxdisp,H,W]
        pred = self.disparity_regression3(pred,self.start_disp//4,self.end_disp//4)
        self.t.end(6)

        disp_up = self.up(guidance, pred) #[B,256,512]

        # self.t.end(7)

        return disp_up

    def _build_volume_2d(self, feat_l, feat_r, disparity_arange):

        num_disp = disparity_arange.shape[0]
        B,feature_size,H,W = feat_l.shape

        cost = torch.zeros((feat_l.shape[0], num_disp, feature_size*2, feat_l.shape[2], feat_l.shape[3]), device=feat_l.device) #[B, disp, feature *2, W, H]
        for i,dis in enumerate(disparity_arange):
            if dis == 0:
                cost[:, i, :feature_size, :, :] = feat_l[:, :, :, :]
                cost[:, i, feature_size:, :, :] = feat_r[:, :, :, :]
            else:
                cost[:, i, :feature_size, :, dis:] = feat_l[:, :, :, dis:]
                cost[:, i, feature_size:, :, dis:] = feat_r[:, :, :, :-dis]

        return cost.permute(0,2,1,3,4).contiguous() #[B, feature*2, disp, W, H]
    
        #构造gwc 代价体
    def _build_gwc_volume(self, feat_l, feat_r, disparity_arange, num_groups):
        # num_groups为组的数列
        B, C, H, W = feat_l.shape
        maxdisp = len(disparity_arange)
        volume = feat_l.new_zeros([B, num_groups, maxdisp, H, W])
        for i,dis in enumerate(disparity_arange):
            if dis == 0:
                volume[:, :, i, :, :] = self._groupwise_correlation(feat_l, feat_r, num_groups)
            else:
                volume[:, :, i, :, dis:] = self._groupwise_correlation(feat_l[:, :, :, dis:], feat_r[:, :, :, :-dis], num_groups)

        volume = volume.contiguous() #[B,num_groups,disp,H,W]
        return volume
    
    def _merge_volume(self, est_disp_list, disparity_arange_list):

        B,num_groups,disp,H,W = est_disp_list[0].shape
        maxdisp = sum([len(l) for l in disparity_arange_list])

        merged_disp = torch.zeros((B, num_groups, maxdisp, H, W), device=est_disp_list[0].device,dtype=est_disp_list[0].dtype) #[B, disp, feature *2, W, H]
        for idx,disparity_arange in enumerate(disparity_arange_list):
            merged_disp[:,:,disparity_arange,:,:] = est_disp_list[idx]

        return merged_disp.contiguous() #[B, feature*2, disp, W, H]
    
    def _groupwise_correlation(self,fea1, fea2, num_groups):
        #fea.shape = [B,C,H,W]
        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2) 
        # fea1 和 fea2 按元素相乘得到[B, C, H, W]，然后分成多个组，在每个组内求平均来压缩channel维度
        assert cost.shape == (B, num_groups, H, W)
        return cost

    #和GC-net一样的softmax
    def disparity_regression2(self, x, disparity_arange):
        disp = disparity_arange.view(1, -1, 1, 1).float()
        disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3]) #扩张到(B,disp,H,W)
        x = F.softmax(x, dim=1)
        x = x * disp
        out = torch.sum(x, 1, keepdim=False) #乘到x上后求和[B,H,W]
        return out
    
    #全回归
    def disparity_regression3(self, x, start_disp, end_disp):
        disparity_arange = torch.arange(start_disp,end_disp,device=x.device).view(1, -1, 1, 1)
        disparity_arange = disparity_arange.repeat(x.size()[0], 1, x.size()[2], x.size()[3]) #扩张到(B,disp,H,W)
        x = F.softmax(x, dim=1)
        x = x * disparity_arange
        out = torch.sum(x, 1, keepdim=False) #乘到x上后求和[B,H,W]
        return out