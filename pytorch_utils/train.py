#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 18:04:03 2021

@author: liqi
"""

import torch
import numpy as np
import random
import os

def plt(history):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    fontsize = 14

    fig = plt.figure()
    fig, ax = plt.subplots(2, 2, sharex='all', sharey='row', figsize=(12, 9))

    for i in range(2):
        for j in range(2):
            ax[i,j].grid(axis = 'y',linestyle = '--')
            ax[i,j].grid(axis = 'x',linestyle = '--')
            ax[i,j].tick_params("x",which="major",length=2,width = 1,colors = "black",direction='in',labelsize=fontsize)
            ax[i,j].tick_params("y",which="major",length=2,width = 1,colors = "black",direction='in',labelsize=fontsize)

    ax[0,0].set_ylabel('Accuracy',fontsize=fontsize)
    ax[1,0].set_ylabel('Loss',fontsize=fontsize)

    ax[1,0].set_xlabel('Epoch',fontsize=fontsize)
    ax[1,1].set_xlabel('Epoch',fontsize=fontsize)

    ax[0,0].plot(history['train_acc'], label = 'train acc')
    ax[0,1].plot(history['val_acc'], label = 'val acc')
    ax[1,0].plot(history['train_loss'], label = 'train loss',color='#ff7f0e')
    ax[1,1].plot(history['val_loss'], label = 'val loss',color='#ff7f0e')

    for i in range(2):
        for j in range(2):
            ax[i,j].legend(fontsize=fontsize,frameon=True)
    plt.tight_layout(pad = 1)
    plt.savefig('history.svg')
    plt.show()

    return

def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

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
            # if (idx+1)%5 == 0:
            #     break
        test_loss /= test_idx
        test_acc = correct / total *100.0

        return test_acc, test_loss

    else:
        for idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()
            # if (idx+1)%5 == 0:
            #     break
        test_acc = correct / total *100.0

        return test_acc

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


def train_ep(ep,net,criterion,optimizer,train_loader,device,logger,scheduler):

    train_loss = correct = total = 0
    net.train()

    for idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        scheduler.step(ep)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

        if (idx + 1) % 100 == 0 or (idx + 1) == len(train_loader):

            logger.info(
                "  ==step: [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                    idx + 1,
                    len(train_loader),
                    train_loss / (idx + 1),
                    100.0 * correct / total,
                ))


def train_stepLR(net,EPOCHS,train_loader,val_loader,criterion,optimizer,device,best_acc,SAVE_path,logger,scheduler=None):

    best_loss = 0
    optimizer.zero_grad()
    optimizer.step()
    logger.info("  ==Train start.")
    for ep in range(1, EPOCHS + 1):

        logger.info("")
        logger.info("  ==Epoches: [{}/{}] ============================".format(ep,EPOCHS))

        scheduler.step(ep)

        train_ep(ep,net,criterion,optimizer,train_loader,device,logger,scheduler)

        val_acc, val_loss = val(net,val_loader,device,criterion)

        logger.info('  ==val_loss: {:.3f} | val_acc: {:6.3f}%'.format(val_loss, val_acc))

        if val_acc >= best_acc:
            best_loss = val_loss
            best_acc = val_acc
            path = os.path.join(SAVE_path,'{:.2f}'.format(best_acc))
            torch.save(net.state_dict(),path +'.pth')
            path = os.path.join(SAVE_path,'latest')
            torch.save(net.state_dict(),path +'.pth')
            logger.info('  ==Parameter save in {}.'.format(ep))
            logger.info('  ==Now best loss is : {:.3f} | best acc is : {:.3f} |'.format(best_loss,best_acc))

        if (ep+1) % 50 == 0:
            logger.info('  ==Now best loss is {:.3f} | best acc is {:.3f} |'.format(best_loss,best_acc))

    logger.info('  ==Final loss is {:.3f} | Final acc is {:.3f} |'.format(best_loss,best_acc))
    logger.info('  ==Train end.')
    return best_loss , best_acc
