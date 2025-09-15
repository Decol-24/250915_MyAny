#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ihpc
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from pytorch_utils import train
from pytorch_utils.common import get_logger
from pytorch_utils.dataset import cifar_set
from pytorch_utils.warmup_scheduler import GradualWarmupScheduler

from gap_utils.gap_utils import transform_224
import os
import argparse

transform=transform_224
dummy_input = torch.zeros([1,3,224,224])
from gap_utils.gap_net2 import resnet56_mod
net = resnet56_mod(10)

macs, params = net.get_flops()
print('macs: {} , params: {}'.format(macs,params))

@torch.no_grad()
def val(net,val_loader,device,criterion):

    val_loss = correct = total = 0
    net.eval()
    for idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        val_idx = (idx + 1)
        total += targets.size(0)
        correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()
    val_loss /= val_idx
    val_acc = correct / total *100.0

    return val_acc, val_loss


def train_ep(ep,net,criterion,optimizer,train_loader,device,logger):

    train_loss = correct = total = 0
    net.train()

    for idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()


def train_stepLR(net,EPOCHS,train_loader,val_loader,criterion,optimizer,device,best_acc,logger,scheduler=None):

    best_loss = 0
    for ep in range(1, EPOCHS + 1):

        scheduler.step(ep)
        train_ep(ep,net,criterion,optimizer,train_loader,device,logger)

        val_acc, val_loss = val(net,val_loader,device,criterion)
        if val_acc >= best_acc:
            best_loss = val_loss
            best_acc = val_acc
            torch.save(net.state_dict(),'latest' +'.pth')

    logger.info('  ==Final loss is {:.3f} | Final acc is {:.3f} |'.format(best_loss,best_acc))


    return best_loss , best_acc

def model_train(net):
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, dest='gpu', type=int, help='GPU number')
    parser.add_argument('-SAVE_NAME', default='pth', type=str)
    args = parser.parse_args()
    num_workers = 20
    batch_size = 128
    EPOCHS = 2
    best_acc = 0
    learning_rate = 0.01
    warm_up = 5

    # 1. define dataloader ==================================================================
    logger = get_logger("log.log")
    train_set, val_set, num_classes = cifar_set('cifar10_debug',train_transform=transform['train'],val_transform=transform['val'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

    # 2. define network ==================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # 3. define loss and optimizer ===========================================================
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    # scheduler = lr_scheduler.StepLR(optimizer, step_size = 80, gamma = 0.1)
    scheduler_cos = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = int(EPOCHS/2), T_mult = 1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cos)
    # 4. start to train ======================================================================
    _,best_acc = train_stepLR(net,EPOCHS,train_loader,val_loader,criterion,optimizer,device,best_acc,logger,scheduler=scheduler)

model_train(net)
parameter = torch.load('latest' +'.pth',map_location=torch.device('cpu'))
net.load_state_dict(parameter)
# net.switch_to_deploy()
torch.onnx.export(net,args=dummy_input,f='mob.onnx',verbose=False)
