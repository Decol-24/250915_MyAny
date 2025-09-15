import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_utils.common import get_logger,creat_folder
import vai_utils
from pytorch_utils.dataset import cifar_set,Dateset_dir
import numpy as np


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', default='0', type=str)
    parser.add_argument('-pth_load', default='per_swim_img.pth', type=str)
    parser.add_argument('-save_path', default='./build', type=str)
    parser.add_argument('-dataset', default='cifar10', choices=['cifar10', 'cifar100', 'cifar10_debug'], type=str)
    parser.add_argument('-input_size', default=224, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-num_workers', default=16, type=int)
    parser.add_argument('-m', '--mode', dest='mode', choices=['1', '2', '3', '4', 'customize'], help='execution mode', type=str)
    parser.add_argument('-dp', '--deploy', dest='deploy', action='store_true')
    parser.add_argument('-ft', '--finetune', dest='finetune', action='store_true', help='fast finetune model before calibration')
    args = parser.parse_args()

    args.device = "cuda"

    creat_folder(args.save_path)
    logger = get_logger(os.path.join(args.save_path,'log.log'))
    #===dataset==========================
    #cifar10
    # args.num_classes = 10
    # data_transform = {
    # "val": transforms.Compose([transforms.Resize(224),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize( mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])}
    
    
    # labellist = np.load('./cifar10/label.npy',allow_pickle=True).item()
    # val_set = Dateset_dir(data_dir='./cifar10/val/',transform=data_transform['val'],data_type='.png',labellist=labellist)
    #imagenet
    args.num_classes = 100
    data_transform_imagenet = {
    "val": transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize( mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])}

    imagenet_dir = './ImageNet100/'
    labellist = np.load(imagenet_dir+'label.npy',allow_pickle=True).item()
    val_set = Dateset_dir(data_dir= imagenet_dir + 'val/',transform=data_transform_imagenet['val'],data_type='.JPEG',labellist=labellist)

    #===define network===================
    from vit_utils.my_vit import hybrid
    from vit_utils.my_swinT import Swim_T
    from vit_utils.vit_runner import vit_runner

    model = Swim_T(args.num_classes, deploy=True)
    runner = vit_runner()
    runner.initialize_runner(args, model)
    runner.load_vit(args.pth_load)
    net = model.get_edge_layer()

    #===quantize===
    if args.mode == '1':
        quantize_acc, finetuned_acc = vai_utils.verification(net, args, val_set)
        logger.info('quantize_acc: {}, finetuned_acc: {}'.format(quantize_acc, finetuned_acc))

    elif args.mode == '2':
        vai_utils.ft_deploy(net, args, val_set)

    elif args.mode == '3':
        vai_utils.deploy(net, args, val_set)

    elif args.mode == '4':
        vai_utils.deploy_hybird(net, args, val_set)

    elif args.mode == 'customize':
        test_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=20,
            pin_memory=False,
        )
        vai_utils.quantize(net,
                    args.save_path,
                    args.quant_mode,
                    test_loader,
                    args.batch_size,
                    args.input_size,
                    finetune = args.finetune,
                    deploy = args.deploy,
                    )

if __name__ == '__main__':
    run_main()
