import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

def verification(net, args, val_set):
    # finetune 测试精度

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    subset_len = int(len(val_set)/4)
    sub_set = torch.utils.data.Subset(val_set, random.sample(range(0, len(val_set)), subset_len))

    sub_loader = torch.utils.data.DataLoader(sub_set,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=False,)

    quantize(net,args.save_path,'calib',val_loader,args.batch_size,args.input_size,finetune = False,deploy = False,device=args.device) #验证

    quantize_acc = quantize(net,args.save_path,'test',val_loader,args.batch_size,args.input_size,finetune = False,deploy = False,device=args.device) #测试精度

    quantize(net,args.save_path,'calib',sub_loader,args.batch_size,args.input_size,finetune = True,deploy = False,device=args.device) #finetune

    finetuned_acc = quantize(net,args.save_path,'test',val_loader,args.batch_size,args.input_size,finetune = True,deploy = False,device=args.device) #载入finetune后的param，测试精度

    return quantize_acc, finetuned_acc

def ft_deploy(net, args, val_set):
    # finetune后部署
    subset_len = int(len(val_set)/4)
    sub_set = torch.utils.data.Subset(
        val_set, random.sample(range(0, len(val_set)), subset_len))

    sub_loader = torch.utils.data.DataLoader(
        sub_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    quantize(net,args.save_path,'calib',sub_loader,args.batch_size,args.input_size,finetune = True,deploy = False,device=args.device)

    sub_set_1 = torch.utils.data.Subset(val_set, random.sample(range(0, len(val_set)), 1))

    sub_loader_1 = torch.utils.data.DataLoader(sub_set_1,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=False,)

    quantize(net,args.save_path,'test',sub_loader_1,batchsize=1,image_size=args.input_size,finetune = True,deploy = True,device=args.device)
    
def deploy(net, args, val_set):
    # 仅部署

    sub_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    quantize(net,args.save_path,'calib',sub_loader,args.batch_size,args.input_size,finetune = False,deploy = False,device=args.device)

    sub_set_1 = torch.utils.data.Subset(val_set, random.sample(range(0, len(val_set)), 1))

    sub_loader_1 = torch.utils.data.DataLoader(sub_set_1,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=False,)

    quantize(net,args.save_path,'test',sub_loader_1,batchsize=1,image_size=args.input_size,finetune = False,deploy = True,device=args.device)

def deploy_hybird(net, args, val_set):
    # 部署 for Myhybird (ViT+IoT)
    
    subset_len = int(len(val_set)/8)
    sub_set = torch.utils.data.Subset(
        val_set, random.sample(range(0, len(val_set)), subset_len))
    
    sub_loader = torch.utils.data.DataLoader(
        sub_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    quantize_hybird(net,args.save_path,'calib',sub_loader,args.batch_size,args.input_size,finetune = False,deploy = False,device=args.device) #验证

    quantize_hybird(net,args.save_path,'calib',sub_loader,args.batch_size,args.input_size,finetune = True,deploy = False,device=args.device) #finetune

    sub_set_1 = torch.utils.data.Subset(val_set, random.sample(range(0, len(val_set)), 1))

    sub_loader_1 = torch.utils.data.DataLoader(sub_set_1,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=False,)

    quantize_hybird(net,args.save_path,'test',sub_loader_1,batchsize=1,image_size=args.input_size,finetune = True,deploy = True,device=args.device)
    # quantize_hybird(net,args.save_path,'test',sub_loader_1,batchsize=1,image_size=args.input_size,finetune = False,deploy = True,device=args.device)

@torch.no_grad()
def test(net,test_loader,device,criterion = None):

    test_loss = correct = total = 0
    net.eval()
    net.to(device=device)

    if criterion:
        for idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_idx = (idx + 1)
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

        test_loss /= test_idx
        test_acc = correct / total *100.0

        return test_acc, test_loss

    else:
        for idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

        test_acc = correct / total *100.0

        return test_acc


def quantize(net,build_dir,quant_mode,data_loader,batchsize,image_size,finetune,deploy,device):
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel
    device = torch.device(device)

    quantized_dir = os.path.join(build_dir, 'quant_model')

    net = net.to(device)

    rand_in = torch.randn([batchsize, 3, image_size, image_size])
    quantizer = torch_quantizer(quant_mode, net, (rand_in), bitwidth=8, output_dir=quantized_dir, device=device)
    quantized_model = quantizer.quant_model

    if finetune == True:
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        if quant_mode == 'calib':
          quantizer.fast_finetune(test, (quantized_model, data_loader, device, loss_fn))
        elif quant_mode == 'test' or deploy:
          quantizer.load_ft_param()

    acc = test(quantized_model, data_loader, device)

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()

    if  deploy:
        quantizer.export_xmodel(deploy_check=True, output_dir=quantized_dir)

    return acc

def quantize_hybird(net,build_dir,quant_mode,data_loader,batchsize,image_size,finetune,deploy,device):
    #for ViT + IoT
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

    device = torch.device(device)

    quantized_dir = os.path.join(build_dir, 'quant_model')
    net = net.to(device)

    rand_in = torch.randn([batchsize, 3, image_size, image_size])
    quantizer = torch_quantizer(quant_mode, net, (rand_in), bitwidth=8, output_dir=quantized_dir, device=device)
    quantized_model = quantizer.quant_model

    if finetune == True:
        # loss_fn = torch.nn.CrossEntropyLoss().to(device)
        if quant_mode == 'calib':
          quantizer.fast_finetune(test_run, (quantized_model, data_loader, device))
        elif quant_mode == 'test' or deploy:
          quantizer.load_ft_param()

    acc = test_run(quantized_model, data_loader, device)

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()

    if deploy:
        quantizer.export_xmodel(deploy_check=True, output_dir=quantized_dir)

    return acc

@torch.no_grad()
def test_run(net,test_loader,device):
    #only run
    net.eval()
    net.to(device=device)

    for idx, (inputs, targets) in enumerate(test_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
    return

@torch.no_grad()
def output_compare(net,quantized_model,test_loader,device):
    #量化前网络和量化后网络的输出之间计算cos相关性
    net.eval()
    net.to(device=device)
    cos_f = nn.CosineSimilarity(dim=-1, eps=1e-6)

    for idx, (inputs, targets) in enumerate(test_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        quantized_outputs = quantized_model(inputs)

        cos_sim = cos_f(outputs,quantized_outputs)

    return 
    