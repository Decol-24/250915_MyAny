import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

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

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        #BasicBlock 是 Convbn(k=步长) - ReLU - Convbn(k固定为1) 的结构，输入和输出之间有残差连接。比Resnet-cifar的一个块的结构更加简单
        #_make_layer() 的参数是：块的类型，输出通道，块的堆叠次数，步长，pad，空洞卷积
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1) #输出图片的size相比输入减少一半
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1) #后面的输出size不变
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        
        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1,
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

        out_feature = self.lastconv(gwc_feature)
        return out_feature

    
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
        # query_feats: [B,64,disp,H,W+disp]
        head_dim = 8
        B, C, H, W = query_feats.shape[0], query_feats.shape[1], query_feats.shape[3], query_feats.shape[4]
        hw = H*W

        query = self.query_project(query_feats) #3D卷积，输入的第一维度要应该是channels

        if self.query_downsample is not None: query = self.query_downsample(query)

        query = query.reshape(B, C//head_dim, head_dim, query.shape[2], hw) # batch, heads, head_c, disparity, h*w
        query = query.permute(0, 4, 1, 3, 2).contiguous()  # batch, h*w, heads, disparity, head_c

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.reshape(B, C//head_dim, head_dim, key.shape[2], hw)
        key = key.permute(0, 4, 1, 2, 3).contiguous()  # batch, h*w, heads, head_c, disparity
        value = value.reshape(B, C//head_dim, head_dim, value.shape[2], hw)
        value = value.permute(0, 4, 1, 3, 2).contiguous()  # batch, h*w, heads, disparity, head_c

        sim_map = torch.matmul(query, key)  # batch, h*w, head, disparity, disparity
        #Q和K先结合得到关注度sim_map
        #对所有在[h,w]上的元素，对每个head内，计算Q的[视差d，此视差下所有通道构成的特征向量] @ K的[此视差下所有通道构成的特征向量，视差d]
        #相当于第一步是用Q的disparity_0的特征向量，点乘K的每个视差的特征向量，依次往后面计算disparity_1，disparity_2，……
        #得到的结果为[disparity, disparity]，也就是 [Q的disparity，此disparity下对K的每个disparity的关注度] 

        if self.matmul_norm: #True
            sim_map = (head_dim ** -0.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)    # batch, h*w, head, disparity, disparity
        context = torch.matmul(sim_map, value)  # batch, h*w, head, disparity, head_c
        #再把sim_map和V矩阵相乘
        #相当于是用Q的disparity下对K的每个disparity的关注度，点乘V的对应视差下的特征向量，得到包含关注度的V值。 #TODO
        context = context.permute(0, 1, 2, 4, 3).flatten(2, 3) # batch, h*w, channels, disparity

        context = context.permute(0, 2, 3, 1).contiguous()  # batch, channels, disparity, h*w
        context = context.reshape(B, C, -1, H, W)  # batch, channels, disparity, h, w
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
    def __init__(self, feats_channels, **kwargs):
        super(cross_attention, self).__init__()
        
        self.cross_attention = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            transform_channels=feats_channels, #所有project的输出通道
            out_channels=feats_channels, #最后的输出通道
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
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