from Any_utils.Any_runner import my_runner
from Any_utils.my_anynet_3 import AnyNet
from pytorch_utils.common import creat_folder
from pytorch_utils.warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import argparse
import Any_utils.creat_loader as DATA
import pickle

def test(args,Net,train_loader,val_loader,**kwargs):
    assert args.batch_size == 1

    Net.to(args.device)

    for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
        pass
        imgL, imgR = imgL.to(args.device), imgR.to(args.device)

    for i in range(20):
        preds = Net(imgL, imgR)

    Net.t.reset()

    for i in range(50):
        preds = Net(imgL, imgR)

    print(Net.t.all_avg_time_str(50))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int, help='GPU number')
    parser.add_argument('-save_path', default='./pth', type=str)
    parser.add_argument('-save_epe', default=1., type=float)
    parser.add_argument('-data_path', default='/home/liqi/Code/Scene_Flow_Datasets/')
    parser.add_argument('-pth_load', default='./69_4.69.pth', type=str)

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
    train_loader, test_loader = DATA.creat_mid_SceneFlow(args.data_path,batch_size=args.batch_size)

    #model
    Net = AnyNet(args.start_disp,args.end_disp)
    if "cuda:" in args.device:
        torch.cuda.manual_seed(args.seed)

    creat_folder(args.save_path)

    runner = my_runner(args)
    runner.set_model(Net)
    runner.load_pth(args.pth_load)

    Net.cuda()
    Net.eval()
    with torch.no_grad():
        def epe(pred, gt, maxdisp=None):
            mask = torch.isfinite(gt)
            if maxdisp is not None:
                mask = mask & (gt>0) & (gt<maxdisp)
            diff = (pred - gt).abs()
            return diff[mask].mean().item()

        # 取1个val batch
        imgL, imgR, gt = next(iter(test_loader))
        pred = Net(imgL.to(args.device), imgR.to(args.device))[-1]
        # 若val时做过resize，确保：gt *= scale_factor；pred同口径
        print('VAL stats:',
            'pred', pred.min().item(), pred.max().item(),
            'gt', gt.min().item(), gt.max().item())
        print('VAL epe(px)=', epe(pred.cpu(), gt, maxdisp=192))

        # 取1个train batch做同样口径
        imgL_t, imgR_t, gt_t = next(iter(train_loader))
        pred_t = Net(imgL_t.to(args.device), imgR_t.to(args.device))[-1]
        print('TRAIN epe(px)=', epe(pred_t.cpu(), gt_t, maxdisp=192))

    # test(args=args,Net=Net,train_loader=train_loader,val_loader=test_loader)