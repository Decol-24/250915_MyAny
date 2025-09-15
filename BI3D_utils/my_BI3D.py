# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from BI3D_utils import FeatExtractNet
from BI3D_utils import SegNet2D
from BI3D_utils import RefineNet2D
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
        return "{:4f}".format(self.avg_time_str(id,count))

    def reset(self):
        self.time_temp = [0 for x in range(self.num)]
        self.time_result = [0 for x in range(self.num)]


__all__ = ["bi3dnet_binary_depth", "bi3dnet_continuous_depth_2D", "bi3dnet_continuous_depth_3D"]


def compute_cost_volume(features_left, features_right, disp_ids, max_disp, is_disps_per_example):

    batch_size = features_left.shape[0]
    feature_size = features_left.shape[1]
    H = features_left.shape[2]
    W = features_left.shape[3]
    device = features_left.device

    psv_size = 1

    psv = Variable(features_left.new_zeros(batch_size, psv_size, feature_size * 2, H, W + max_disp)).to(device) # [1,psv,64,192,384]全是0。w维度考虑到了视差平移

    if is_disps_per_example: #第一维度psv维度只有1,只考虑一个视差平面下的结果
        for i in range(batch_size):
            psv[i, 0, :feature_size, :, 0:W] = features_left[i]
            psv[i, 0, feature_size:, :, disp_ids : W + disp_ids] = features_right[i]
        psv = psv.contiguous() #左图从左边开始直接放入psv，右图往右平移后放入psv（也就是论文中的把右图映射到左视角），两图在特征维度拼接。[B,1,64,192,384]
    else:
        for i in range(psv_size):
            psv[:, i, :feature_size, :, 0:W] = features_left
            psv[:, i, feature_size:, :, disp_ids[0, i] : W + disp_ids[0, i]] = features_right
        psv = psv.contiguous() #按每个psv平面拼接，也是在特征维度拼接。但是每个平面拼接的数据源是一样的，仅有平移程度不一样。[1,65,64,192,384]

    return psv

"""
Bi3DNet for binary depthmap generation.
"""

class Bi3DNetBinaryDepth(nn.Module):
    def __init__(
        self,
        max_disparity=192,
        is_refine=True,
        is_disps_per_example=True,
    ):

        super(Bi3DNetBinaryDepth, self).__init__()

        self.max_disparity = max_disparity
        self.max_disparity_seg = int(max_disparity / 3)
        self.is_disps_per_example = is_disps_per_example

        self.is_refine = is_refine

        self.featnet = FeatExtractNet.FeatExtractNetSPP()
        self.featnethr = FeatExtractNet.FeatExtractNetHR()
        self.segnet = SegNet2D.SegNet2D()
        if self.is_refine:
            self.refinenet = RefineNet2D.SegRefineNet()

        # self.times = time_counter(num=5)
        return

    def forward(self, img_left, img_right, psv_disp):

        # self.times.start(0)
        batch_size = img_left.shape[0]
        psv_size = 1

        if psv_size == 1:
            self.is_disps_per_example = True #psv只设置为1，只考虑一个平面
        else:
            self.is_disps_per_example = False

        # 提取特征
        features = self.featnet(torch.cat((img_left, img_right), dim=0)) # 用PSMNet提取特征。在batch维度上拼接，所以各个模块部分不需要特别处理

        features_left = features[:batch_size, :, :, :]
        features_right = features[batch_size:, :, :, :] #拆分左右代价体 [1,32,180,320]

        # self.times.end(0)
        # self.times.start(1)

        if self.is_refine:
            features_lefthr = self.featnethr(img_left) # [1,16,546,960] 从左图原图提取的特征，用于最后的细化
        feature_size = features_left.shape[1]
        H = features_left.shape[2]
        W = features_left.shape[3]

        # self.times.end(1)
        # self.times.start(2)

        # Cost Volume Generation
        psv = compute_cost_volume(
            features_left, features_right, psv_disp, self.max_disparity_seg, self.is_disps_per_example
        )

        psv = psv.view(batch_size * psv_size, feature_size * 2, H, W + self.max_disparity_seg) #把batch和psv维度合并到一个维度
        
        # self.times.end(2)
        # self.times.start(3)

        # Segmentation Network
        seg_raw_low_res = self.segnet(psv)[:, :, :, :W]  # segnet模型的左右图不重合的部分舍弃，减少计算消耗 [1,192,320]
        seg_prob_low_res = torch.sigmoid(seg_raw_low_res)
        seg_prob_low_res = seg_prob_low_res.view(batch_size, psv_size, H, W) # 把batch和psv拆开

        seg_prob_low_res_up = F.interpolate(
            seg_prob_low_res, size=img_left.size()[-2:], mode="bilinear", align_corners=False
        ) # 上采样到 [1,576,960]
        out = []
        out.append(seg_prob_low_res_up)

        # self.times.end(3)
        # self.times.start(4)

        # Refinement
        if self.is_refine:
            seg_raw_high_res = F.interpolate(
                seg_raw_low_res, size=img_left.size()[-2:], mode="bilinear", align_corners=False
            ) #用sigmoid处理之前的segnet的结果上采样 [1,1,576,960]
            # Refine Net
            features_left_expand = (
                features_lefthr[:, None, :, :, :].expand(-1, psv_size, -1, -1, -1).contiguous() # 这里None是增加一个维度，然后把这个维度扩张到psv_size
            ) #[1,16,576,960]
            features_left_expand = features_left_expand.view(
                -1, features_lefthr.size()[1], features_lefthr.size()[2], features_lefthr.size()[3]
            ) #合并batch和psv维度
            refine_net_input = torch.cat((seg_raw_high_res, features_left_expand), dim=1) #[1,17,576,960]

            seg_raw_high_res = self.refinenet(refine_net_input) #refinenet是2个Conv2d构成

            seg_prob_high_res = torch.sigmoid(seg_raw_high_res) # [1,576,960]
            seg_prob_high_res = seg_prob_high_res.view(
                batch_size, psv_size, img_left.size()[-2], img_left.size()[-1]
            ) ##拆分batch和psv维度
            out.append(seg_prob_high_res)
        else:
            out.append(seg_prob_low_res_up)
            
        # self.times.end(4)

        return out


def bi3dnet_binary_depth(options, data=None):

    print("==> USING Bi3DNetBinaryDepth")
    for key in options:
        if "bi3dnet" in key:
            print("{} : {}".format(key, options[key]))

    model = Bi3DNetBinaryDepth(
        options,
        featnet_arch=options["bi3dnet_featnet_arch"],
        segnet_arch=options["bi3dnet_segnet_arch"],
        refinenet_arch=options["bi3dnet_refinenet_arch"],
        featnethr_arch=options["bi3dnet_featnethr_arch"],
        max_disparity=options["bi3dnet_max_disparity"],
        is_disps_per_example=options["bi3dnet_disps_per_example_true"],
    )

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model
