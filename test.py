from DCA_utils.DCA_runner import my_runner
from DCA_utils.my_DCA import GwcNet
from pytorch_utils.common import creat_folder
from pytorch_utils.warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import argparse
import DCA_utils.creat_loader as DATA
import pickle
import torch.nn.functional as F
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int, help='GPU number')
    parser.add_argument('-save_path', default='./pth', type=str)
    parser.add_argument('-save_epe', default=1., type=float)
    parser.add_argument('-data_path', default='/home/liqi/Code/Scene_Flow_Datasets/')
    parser.add_argument('-pth_load', default='./sceneflow_2.pth', type=str)

    parser.add_argument('-device', default='cuda:0', type=str) #cuda:0  cpu
    parser.add_argument('-seed', default=7777, type=int)
    parser.add_argument('-train_EPOCHS', default=500, type=int)
    parser.add_argument('-train_warm_up', default=30, type=int)
    parser.add_argument('-train_lr', default=1e-2, type=float)
    parser.add_argument('-start_disp', default=0, type=int)
    parser.add_argument('-end_disp', default=192, type=int)
    parser.add_argument('-focal_coefficient', default=5.0, type=float)
    parser.add_argument('-sparse', default=False, type=bool)

    parser.add_argument('-batch_size', default=1, type=int)

    parser.add_argument('-mixup_alpha', default=0.5, type=float)
    parser.add_argument('-grad_clip_value', default=1., type=float)
    args = parser.parse_args()

    #Dataset
    # scale = 4.0
    # scale_func = F.adaptive_avg_pool2d
    # max_disp = 192
    # train_loader, test_loader = DATA.creat_SceneFlow(args.data_path,batch_size=args.batch_size)
    # for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
    #     mask = ((disp_true < 192) & (disp_true > 0)).byte().bool() # 得到一个布尔张量，标记出 0 < disp_true < 192 的像素
    #     mask.detach_()
    #     if mask.sum() < 1.0:
    #         print("error_{}".format(batch_idx))

    #     scaled_gtDisp = disp_true / scale
    #     H, W = 576, 960

    #     scaled_gtDisp = scale_func(scaled_gtDisp, (H, W))
    #     scaled_max_disp = int(max_disp/scale)

    #     # 有效视差的掩码
    #     # (起始视差, 最大视差 / 缩放因子)
    #     # 注意：KITTI 数据集中的无效视差值被设置为 0，一定要将其屏蔽掉
    #     lower_bound = 0
    #     upper_bound = lower_bound + scaled_max_disp
    #     mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound).detach_().byte().bool()
    #     if mask.sum() < 1.0:
    #         print("scaled_error_{}".format(batch_idx))

    train_loader, test_loader = DATA.creat_SceneFlow(args.data_path,batch_size=args.batch_size)
    time_c = time.time()
    for batch_idx, (imgL, imgR, disp_L) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print((time.time() - time_c)/100)
            time_c = time.time()