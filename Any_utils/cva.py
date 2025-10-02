import torch
import torch.nn as nn
import torch.nn.functional as F
from DCA_utils.DCA_submodule import convbn,BasicBlock,convbn_3d

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
    #Q来自C',K和V来自C_cls
    def forward(self, query_feats, key_feats):
        # query_feats: [batch, channels, disparity, height ,width]
        head_dim = 8
        batch_size, channels, disparity, height, width = query_feats.shape
        hw = height*width
        query = self.query_project(query_feats)
        if self.query_downsample is not None: query = self.query_downsample(query)
        #query: b, h, hc, d, h*w
        query = query.reshape(batch_size, channels//head_dim, head_dim, disparity, hw)
        query = query.permute(0, 4, 1, 3, 2).contiguous()  # batch, h*w, head, disparity, head_c

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.reshape(batch_size, channels//head_dim, head_dim, disparity, hw)
        key = key.permute(0, 4, 1, 2, 3)  # batch, h*w, head, head_c, disparity
        value = value.reshape(batch_size, channels//head_dim, head_dim, disparity, hw)
        value = value.permute(0, 4, 1, 3, 2)  # batch, h*w, head, disparity, head_c

        sim_map = torch.matmul(query, key)  # batch, h*w, head, disparity, disparity
        #Q和K先结合得到关注度sim_map
        #对所有在[h,w]上的元素，对每个head内，计算Q的[视差d，此视差下所有通道构成的特征向量] @ K的[此视差下所有通道构成的特征向量，视差d]
        #相当于第一步是用Q的disparity_0的特征向量，点乘K的每个视差的特征向量，依次往后面计算disparity_1，disparity_2，……
        #得到的结果为[disparity, disparity]，也就是 [Q的disparity，此disparity下对K的每个disparity的关注度] 

        if self.matmul_norm:
            sim_map = (head_dim ** -0.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)    # batch, h*w, head, disparity, disparity
        context = torch.matmul(sim_map, value)  # batch, h*w, head, disparity, head_c
        #再把sim_map和V矩阵相乘
        #相当于是用Q的disparity下对K的每个disparity的关注度，点乘V的对应视差下的特征向量，得到包含关注度的V值。
        context = context.permute(0, 1, 2, 4, 3).flatten(2, 3)         # batch, h*w, channels, disparity

        context = context.permute(0, 2, 3, 1).contiguous()  # batch, channels, disparity, h*w
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # batch, channels, disparity, h, w
        if self.out_project is not None:
            context = self.out_project(context)
        return context

    '''build project'''
    # 生成project，就是堆叠的1x1卷积层
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


'''semantic-level context module'''
class SemanticLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, reduction=8, concat_input=True, **kwargs):
        super(SemanticLevelContext, self).__init__()
        
        # 创建多头自注意力机制层
        # 设置上 feats_channels == transform_channels
        self.cross_attention = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
        )

    '''forward'''
    def forward(self, x, preds):
        inputs = x
        preds = F.softmax(preds, dim=1) #正则化
        batch_size, num_channels, disparity_planes, h, w = x.size()
        feats_sl = torch.zeros(batch_size, h*w, disparity_planes, num_channels).type_as(x) # (1,2048,24,32)

        # 对应论文中的同质区域表示生成
        for batch_idx in range(batch_size):

            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(num_channels, disparity_planes, -1), preds_iter.reshape(disparity_planes, -1)
            feats_iter, preds_iter = feats_iter.permute(2, 1, 0).contiguous(), preds_iter.permute(1, 0).contiguous()
            #   feats_iter: (num_channels, D, H, W)  --> (H*W, D, num_channels)
            #   preds_iter: (D, H, W) --> (H*W, D)
            argmax = preds_iter.argmax(1)
            # argmax[2024] 取到preds中每个像素的视差分类的最大值的下标作为此像素的视差
            # 在每个视差内进行聚合处理
            for disp_id in range(disparity_planes):
                mask = (argmax == disp_id)
                # 只在preds中视差值等于当前disp_id的像素点之间进行操作。mask记录对应的像素的下标
                if mask.sum() == 0: continue #此视差级内无结果，跳过
                feats_iter_cls = feats_iter[mask, disp_id]  # [mask, num_channels] 取出mask对应像素点，并用对应channel向量当特征向量
                preds_iter_cls = preds_iter[:, disp_id][mask]  # 等效于preds_iter[mask, disp_id]，shape为[mask]，取出mask对应像素点
                weight = F.softmax(preds_iter_cls, dim=0)
                feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1) # preds_iter_cls作为权重乘到feats_iter_cls的对应特征向量上
                feats_sl[batch_idx][mask, disp_id] = feats_iter_cls # 处理结果存储到feats_sl
        
        feats_sl = feats_sl.reshape(batch_size, h, w, disparity_planes, num_channels)
        feats_sl = feats_sl.permute(0, 4, 3, 1, 2).contiguous()
        # feats_sl: [batch, num_channels, d, h, w]
        # 这个矩阵相当稀疏啊，每个d维度下仅存分类结果=d的h*w位置的特征信息，所以所有d维度下的所有h*w信息合起来才相当于原来一张图。(b != 0).sum()==2048

        # x做query，(加权处理后的x + x)做key和value得到对(加权处理后的x + x)的自注意力结果
        # 也就是根据当前x来选择哪些带分类结果的特征是重要的
        # 对应论文中的同质区域引导聚合
        feats_sl = self.cross_attention(inputs, feats_sl+inputs)
        # feats_sl = self.cross_attention(inputs, inputs)
        #怀疑这个的作用，画图分析

        return feats_sl

#多个3D卷积来聚合
class Multi_Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Multi_Aggregation, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels*2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = F.relu(self.conv3(conv2) + self.redir(x), inplace=True)

        return conv3

class cva(nn.Module):
    def __init__(self, channel=32, downsample=True):
        super(cva, self).__init__()
        self.channel = channel
        if downsample:
            self.downsample = nn.Sequential(nn.AvgPool3d((3, 3, 3), stride=2, padding=1),
                                            convbn_3d(self.channel, 32, 3, 1, 1),
                                            nn.ReLU(inplace=True))

        self.slc_net = SemanticLevelContext(feats_channels=self.channel,transform_channels=self.channel,concat_input=True)
        # build image-level context module
        self.classify = nn.Sequential(convbn_3d(self.channel, self.channel, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(self.channel, 1, kernel_size=3, padding=1, stride=1, bias=False)) # 最后输出通道为1，表示分类结果

        self.fuse = nn.Sequential(convbn_3d(64, 32, 1, 1, 0),)
        self.cost_agg = Multi_Aggregation(self.channel)


    def forward(self, cost_volume, downsample=True):
        if downsample:
            cost_down = self.downsample(cost_volume)
            prob_volume = self.classify(cost_down).squeeze(1) #prob_volume是分类结果，[1,24,32,64]，第一维度的24是表示每个点属于每个视差级的可能性
            augmented_cost_down = self.slc_net(cost_down, prob_volume) #根据分类结果指导得到对代价体的自注意力结果
            augmented_cost = F.interpolate(augmented_cost_down, scale_factor=(2,2,2), mode='trilinear') #三线性插值，上采样到depth–height–width是原来的两倍
        else:
            prob_volume = self.classify(cost_volume).squeeze(1)
            augmented_cost = self.slc_net(cost_volume, prob_volume)

        augmented_cost = self.fuse(torch.cat([augmented_cost, cost_volume], dim=1)) #把自注意力结果和元素结果拼接，然后多个3D卷积提取特征，对应论文中的3D CNN层
        augmented_cost = self.cost_agg(augmented_cost)

        return prob_volume.unsqueeze(1), augmented_cost