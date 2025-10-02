from DCA_utils.DCA_runner import my_runner
from DCA_utils.my_anynet import AnyNet
from pytorch_utils.common import creat_folder
from pytorch_utils.warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import argparse
import DCA_utils.creat_loader as DATA
import pickle

def train(args,Net,train_loader,val_loader,**kwargs):

    runner = my_runner(args)
    runner.set_model(Net)
    # runner.load_pth(args.pth_load)
    
    # val_acc = runner.test(val_loader)
    # runner.logger.info("val_acc: {}".format(val_acc))

    optimizer = torch.optim.SGD(
        runner.model.parameters(),
        lr=args.train_lr,
        momentum=0.9,
        weight_decay=0,
        nesterov=False,
    )

    scheduler_cos = CosineAnnealingWarmRestarts(optimizer, T_0 = int(args.train_EPOCHS), T_mult = 1)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.train_warm_up, after_scheduler=scheduler_cos)

    runner.train(train_loader, val_loader, scheduler_cos, optimizer) 

def test(args,Net,train_loader,val_loader,**kwargs):
    runner = my_runner(args)
    runner.set_model(Net)
    runner.load_pth(args.pth_load)

    runner.logger.info("Train start.")
    # val_acc = runner.test(val_loader)
    # runner.logger.info("val_acc: {}".format(val_acc))

    val_epe = runner.val_onece(val_loader,runner.setting.device)
    runner.logger.info('val_epe: {:.3f}'.format(val_epe))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int, help='GPU number')
    parser.add_argument('-save_path', default='./pth', type=str)
    parser.add_argument('-save_epe', default=8., type=float)
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

    parser.add_argument('-batch_size', default=8, type=int)

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

    train(args=args,Net=Net,train_loader=train_loader,val_loader=test_loader)