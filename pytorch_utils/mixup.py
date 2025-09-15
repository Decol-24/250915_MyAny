import numpy as np
import torch

def mixup_data(input, label, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        return input, label, label, 1

    batch_size = input.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_input = lam * input + (1 - lam) * input[index, :]
    label_a, label_b = label, label[index]
    return mixed_input, label_a, label_b, lam


def mixup_criterion(criterion, pred, label_a, label_b, lam):
    if lam == 1:
        return criterion(pred, label_a)
    else:
        return lam * criterion(pred, label_a) + (1 - lam) * criterion(pred, label_b)