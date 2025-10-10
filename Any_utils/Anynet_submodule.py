import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
    
'''self attention block'''
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query,
                 query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm,
                 value_out_norm, matmul_norm, with_out_project, **kwargs):
        super(SelfAttentionBlock, self).__init__()
        # norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        # key project
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
            use_norm=key_query_norm,
        )
        # query project
        if share_key_query:
            assert key_in_channels == query_in_channels
            self.query_project = self.key_project
        else:
            self.query_project = self.buildproject(
                in_channels=query_in_channels,
                out_channels=transform_channels,
                num_convs=key_query_num_convs,
                use_norm=key_query_norm,
            )
        # value project
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels if with_out_project else out_channels,
            num_convs=value_out_num_convs,
            use_norm=value_out_norm,
        )
        # Q,K,V用三个project转换

        # out project
        self.out_project = None
        if with_out_project:
            self.out_project = self.buildproject(
                in_channels=transform_channels,
                out_channels=out_channels,
                num_convs=value_out_num_convs,
                use_norm=value_out_norm,
            )

        # downsample
        self.query_downsample = query_downsample #False
        self.key_downsample = key_downsample #False
        self.matmul_norm = matmul_norm #True
        self.transform_channels = transform_channels

    #修改过的多头自注意力机制，Q和K不同源
    def forward(self, query_feats, key_feats):
        # query_feats: [B,64,disp1,H,W]
        # key_feats: [B,64,disp2,H,W]
        head_dim = 8
        B, C, H, W = query_feats.shape[0], query_feats.shape[1], query_feats.shape[3], query_feats.shape[4]
        hw = H*W

        query = self.query_project(query_feats) #3D卷积，输入的第二维度要应该是channels

        if self.query_downsample is not None: query = self.query_downsample(query)

        query = query.reshape(B, C//head_dim, head_dim, query.shape[2], hw) # batch, heads, head_c, disp1, h*w
        query = query.permute(0, 4, 1, 3, 2).contiguous()  # batch, h*w, heads, disp1, head_c

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.reshape(B, C//head_dim, head_dim, key.shape[2], hw)
        key = key.permute(0, 4, 1, 2, 3).contiguous()  # batch, h*w, heads, head_c, disp2
        value = value.reshape(B, C//head_dim, head_dim, value.shape[2], hw)
        value = value.permute(0, 4, 1, 3, 2).contiguous()  # batch, h*w, heads, disp2, head_c

        sim_map = torch.matmul(query, key)  # batch, h*w, head, disp1, disp2
        #Q和K先结合得到关注度sim_map
        #得到query_feats的每个视差级对key_feats的每个视差级的关注度

        if self.matmul_norm: #True
            sim_map = (head_dim ** -0.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)  # batch, h*w, head, disp1, head_c
        # sim_map是[query_feats每个视差级，query_feats每个视差级对key_feats的每个视差级的关注度]
        # value是[key_feats的每个视差级，每个视差级下的特征向量]
        # 第一矩阵第一行相关的点乘结果相当于 query_feats的第一视差级对key_feats所有视差级的关注度 * 对应视差级的特征向量 => query_feats的第一视差级对key_feats所有视差级的关注结果的合

        context = context.permute(0, 1, 2, 4, 3).flatten(2, 3) # batch, h*w, channels, disp1

        context = context.permute(0, 2, 3, 1).contiguous()  # batch, channels, disp1, h*w
        context = context.reshape(B, C, -1, H, W)  # batch, channels, disp1, h, w
        if self.out_project is not None:
            context = self.out_project(context)

        return context
    
    def buildproject(self, in_channels, out_channels, num_convs, use_norm):
        if use_norm: #都是True
            convs = [
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm3d(out_channels),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
        else:
            convs = [nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]
    
class cross_attention(nn.Module):
    def __init__(self, feats_channels, key_query_num_convs=1, **kwargs):
        super(cross_attention, self).__init__()
        
        self.cross_attention = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            transform_channels=feats_channels, #所有project的输出通道
            out_channels=feats_channels, #最后的输出通道
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=key_query_num_convs,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
        )

    '''forward'''
    def forward(self, inputs, value):
        feats_sl = self.cross_attention(inputs, value)
        return feats_sl