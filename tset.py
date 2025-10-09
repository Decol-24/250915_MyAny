import os
import torch
from pytorch_utils.mixup import mixup_data,mixup_criterion
from pytorch_utils.grad_scale import dispatch_clip_grad
import torch.nn as nn
from Any_utils.loss import l1_loss, loss_sparse
import gc

arrary = torch.arange(0,15).reshape(-1,1,1)
x = torch.ones(15,32,32)

x = x * arrary
mask = x < 13
b = x[mask]

print(b.shape)

print(x.shape)

