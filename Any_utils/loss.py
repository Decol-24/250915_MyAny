import torch.nn.functional as F
import torch
def isNaN(x):
    return x != x

class l1_loss(object):
    def __init__(self,start_disp, end_disp, logger, sparse=False):
        self.start_disp = start_disp
        self.end_disp = end_disp
        self.max_disp = end_disp - start_disp
        self.logger = logger
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d
            #缩放函数
    #输入同一为list
    def __call__(self, disp_ests, disp_gt):

        all_losses = []

        for disp_est in disp_ests:
            B,H,W = disp_est.shape

            if disp_gt[0].shape[-2] != H or disp_gt[0].shape[-1] != W:
                # 当此级别的预测代价体的高宽与真实视差图的高宽不一致时，计算缩放比例并缩放真实视差图
                scale = disp_gt[0].shape[-1] / (W * 1.0)
                scaled_gtDisp = disp_gt / scale

                scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))
                scaled_max_disp = int(self.max_disp/scale)
                lower_bound = self.start_disp
                upper_bound = lower_bound + scaled_max_disp
                mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound).detach_().byte().bool()
            else:
                scaled_gtDisp = disp_gt
                mask = (scaled_gtDisp >= self.start_disp) & (scaled_gtDisp < self.end_disp).detach_().byte().bool()

            loss_b = []
            for batch in range(B):

                if mask[batch].sum() < 1.0:
                    loss = disp_est[batch].sum() * 0.0  # 为了正确检测此项，需要分batch计算mask
                # self.logger.info('No point in range!')
                else:
                    mask_scaled_gtDisp = scaled_gtDisp[batch] * mask[batch]
                    loss = F.smooth_l1_loss(disp_est[batch], mask_scaled_gtDisp, reduction='mean')

                loss_b.append(loss)

            all_losses.append(sum(loss_b))

        return all_losses
    
class loss_sparse(object):
    def __init__(self,start_disp, end_disp, logger, disparity_arange, sparse=False):
        self.start_disp = start_disp
        self.end_disp = end_disp
        self.max_disp = end_disp - start_disp
        self.logger = logger
        self.disparity_arange = disparity_arange
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d
            #缩放函数

    def __call__(self, disp_est_sparse, disp_gt):

        B,H,W = disp_est_sparse.shape

        if disp_gt[0].shape[-2] != H or disp_gt[0].shape[-1] != W:
            # 当此级别的预测代价体的高宽与真实视差图的高宽不一致时，计算缩放比例并缩放真实视差图
            scale = disp_gt[0].shape[-1] / (W * 1.0)
            scaled_gtDisp = disp_gt / scale

            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))
            scaled_max_disp = int(self.max_disp/scale)
            lower_bound = self.start_disp
            upper_bound = lower_bound + scaled_max_disp
            mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound).detach_().byte().bool()
        else:
            scaled_gtDisp = disp_gt
            mask = (scaled_gtDisp >= self.start_disp) & (scaled_gtDisp < self.end_disp).detach_().byte().bool()

        loss_sparse_b = []
        for batch in range(B):

            if mask[batch].sum() < 1.0:
                loss = disp_est_sparse[batch].sum() * 0.0  # 为了正确检测此项，需要分batch计算mask
            # self.logger.info('No point in range!')
            else:
                mask_scaled_gtDisp = scaled_gtDisp[batch] * mask[batch]
                idx = torch.argmin(torch.abs(mask_scaled_gtDisp.view(-1, 1) - self.disparity_arange), dim=1)
                loss = F.smooth_l1_loss(disp_est_sparse[batch].view(-1), self.disparity_arange[idx], reduction='mean')

            loss_sparse_b.append(loss)

        return sum(loss_sparse_b)

def model_loss(disp_est, disp_gt, mask):

    return F.smooth_l1_loss(disp_est[:,mask].squeeze(), disp_gt[mask], reduction='mean')

def focal_loss(disp_ests, disp_gt, start_disp, end_disp, focal_coefficient, sparse):
    # 多阶段视差估计网络的加权 Focal Loss 总和
    # disp_ests是一个列表，包含多个阶段的视差估计结果

    focal_loss_evaluator = StereoFocalLoss(start_disp, end_disp, focal_coefficient=focal_coefficient, sparse=sparse)

    weights = [0.5, 0.7]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        loss = weight * focal_loss_evaluator(disp_est, disp_gt, variance=1)
        all_losses.append(loss)
    return sum(all_losses)

class Disp2Prob(object):
    """
    Convert disparity map to matching probability volume
    将视差图转换为匹配概率体
        Args:
            maxDisp, (int): the maximum of disparity
            gtDisp, (torch.Tensor): in (..., Height, Width) layout
            start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index

        Outputs:
            probability, (torch.Tensor): in [BatchSize, maxDisp, Height, Width] layout
    """
    def __init__(self, dilation=1):

        self.dilation = dilation
        self.eps = 1e-40

    def getProb(self,gtDisp,variance,max_disp):
        # [BatchSize, Height, Width]
        self.max_disp = max_disp

        B, H, W = gtDisp.shape
        gtDisp = gtDisp.view(B, 1, H, W)

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        self.index = torch.arange(0, self.max_disp, dtype=gtDisp.dtype, device=gtDisp.device)
        self.index = self.index.view(1, self.max_disp, 1, 1)

        # [BatchSize, maxDisp, Height, Width]
        self.index = self.index.repeat(B, 1, H, W).contiguous()

        probability = self.calProb(gtDisp,variance)

        # let the outliers' probability to be 0
        # in case divide or log 0, we plus a tiny constant value
        probability = probability + self.eps
        #概率分布也同样需要mask处理

        # in case probability is NaN
        if isNaN(probability.min()) or isNaN(probability.max()):
            print('Probability ==> min: {}, max: {}'.format(probability.min(), probability.max()))
            print('Disparity Ground Truth after mask out ')
            raise ValueError(" \'probability contains NaN!")

        return probability

    def calProb(self,gtDisp,variance):
        #子类必须重写此方法
        raise NotImplementedError


class LaplaceDisp2Prob(Disp2Prob):
    # variance is the diversity of the Laplace distribution
    def __init__(self, dilation=1):
        super(LaplaceDisp2Prob, self).__init__(dilation)

    def calProb(self,gtDisp,variance):
        # 1/N * exp( - (d - d{gt}) / var), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        scaled_distance = ((-torch.abs(self.index - gtDisp))) #-|d - d_gt|
        probability = F.softmax(scaled_distance, dim=1)

        return probability

class GaussianDisp2Prob(Disp2Prob):
    # variance is the variance of the Gaussian distribution
    def __init__(self, dilation=1):
        super(GaussianDisp2Prob, self).__init__(dilation)

    def calProb(self,gtDisp,variance):
        # 1/N * exp( - (d - d{gt})^2 / b), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        distance = (torch.abs(self.index - gtDisp))
        scaled_distance = (- distance.pow(2.0) / variance)
        probability = F.softmax(scaled_distance, dim=1)

        return probability

class OneHotDisp2Prob(Disp2Prob):
    # variance is the variance of the OneHot distribution
    def __init__(self, dilation=1):
        super(OneHotDisp2Prob, self).__init__(dilation)

    def calProb(self,gtDisp,variance):
        # |d - d{gt}| < variance, [BatchSize, maxDisp, Height, Width]
        b, c, h, w = gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        self.index = torch.arange(0, self.max_Disp, dtype=gtDisp.dtype, device=gtDisp.device)
        self.index = self.index.view(1, self.max_Disp, 1, 1)

        # [BatchSize, maxDisp, Height, Width]
        self.index = self.index.repeat(b, 1, h, w).contiguous()
        probability = torch.lt(torch.abs(self.index - gtDisp), variance).type_as(gtDisp)

        return probability


class StereoFocalLoss(object):
    """
    Under the same start disparity and maximum disparity, calculating all estimated cost volumes' loss
    计算预测的代价体(cost volume)的损失
        Args:
            dilation (int): the step between near disparity index, it mainly used in gt probability volume generation 
            相邻视差索引之间的步长，主要用于生成真实概率体
            weights, (list of float or None): weight for each scale of estCost. 
            每个尺度的 estCost 对应的权重
            focal_coefficient, (float): stereo focal loss coefficient, details please refer to paper. default: 0.0
            立体焦点损失(stereo focal loss)的系数,详细内容请参考论文,默认值: 0.0  
            sparse, (bool): whether the ground-truth disparity is sparse, for example, KITTI is sparse, but SceneFlow is not. default: False
            表示真实视差(ground-truth disparity)是否稀疏,例如 KITTI 数据集是稀疏的，而 SceneFlow 数据集不是稀疏的

        Inputs:
            estCost, (Tensor or list of Tensor): the estimated cost volume, in (BatchSize, max_disp, Height, Width) layout
            预测的代价体，格式为 (BatchSize, max_disp, Height, Width)
            gtDisp, (Tensor): the ground truth disparity map, in (BatchSize, 1, Height, Width) layout.
            真实视差图，格式为 (BatchSize, 1, Height, Width)
            variance, (Tensor or list of Tensor): the variance of distribution, details please refer to paper, in (BatchSize, 1, Height, Width) layout.
            分布的方差，详细内容请参考论文，格式为 (BatchSize, 1, Height, Width)

        Outputs:
            loss, (dict), the loss of each level

        ..Note:
            Before calculate loss, the estCost shouldn't be normalized,
              because we will use softmax for normalization
    """

    def __init__(self, start_disp=0, end_disp=192, dilation=1, weights=None, focal_coefficient=0.0, sparse=False):
        self.end_disp = end_disp
        self.start_disp = start_disp
        self.max_disp = end_disp - start_disp
        self.dilation = dilation
        self.weights = weights
        self.focal_coefficient = focal_coefficient
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d
            #缩放函数
        self.LaplaceDisp2Prob = LaplaceDisp2Prob(dilation=dilation)

    def loss_per_level(self, estCost, gtDisp, variance=1.0, dilation=1):
        B, C, H, W = estCost.shape
        scaled_gtDisp = gtDisp.clone()

        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale per level and scale gtDisp
            # 当此级别的预测代价体的高宽与真实视差图的高宽不一致时，计算缩放比例并缩放真实视差图
            # 在网络中有两种尺寸的预测结果
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp.clone() / scale

            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))
            scaled_max_disp = int(self.max_disp/scale)

        # 有效视差的掩码
        # (起始视差, 最大视差 / 缩放因子)
        # 注意：KITTI 数据集中的无效视差值被设置为 0，一定要将其屏蔽掉
        lower_bound = self.start_disp
        upper_bound = lower_bound + scaled_max_disp
        mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound).detach_().byte().bool()

        if mask.sum() < 1.0:
            return estCost.sum() * 0.0  # let this sample have loss with 0
        else:
            # transfer disparity map to probability map
            mask_scaled_gtDisp = scaled_gtDisp * mask
            scaled_gtProb = self.LaplaceDisp2Prob.getProb(mask_scaled_gtDisp, variance=variance, max_disp=scaled_max_disp)
            #得到真实视差的概率曲线

        # stereo focal loss
        estProb = F.log_softmax(estCost, dim=1)
        # 预测视差的概率曲线
        weight = (1.0 - scaled_gtProb).pow(-self.focal_coefficient).type_as(scaled_gtProb)
        loss = -((scaled_gtProb * estProb) * weight * mask.float()).sum(dim=1, keepdim=True).mean()

        return loss

    def __call__(self, estCost, gtDisp, variance):
        # compute loss for per level
        loss_all_level = []
        loss_all_level = self.loss_per_level(estCost, gtDisp, variance)

        return loss_all_level