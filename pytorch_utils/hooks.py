#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:56:58 2021

@author: ihpc
"""

import numpy as np

def output_hook(hook_temp):
    # hook_temp is list
    # [intput,output]
    def hook(model, input, output):
        nonlocal hook_temp
        hook_temp.append(input[0].detach())
        hook_temp.append(output[0].detach())

    return hook

# for idx, test_input in enumerate(inputs):

#     test_input = test_input.unsqueeze(dim=0)
#     out = runner.model(test_input)
#     for hook_idx in range(len(hook_temp)):
#         input = hook_temp[hook_idx][0].numpy()
#         output = hook_temp[hook_idx][1].numpy()
#         weight_output = output - input
#         input_temp[hook_idx].append((input.sum()))
#         weight_output_temp[hook_idx].append((weight_output.sum()))
#         hook_temp[hook_idx].pop()
#         hook_temp[hook_idx].pop()

# def register_hook(self):
#     from pytorch_code.hooks import output_hook
#     MKIR_layers = self.get_MKIR_layer()
#     IR_skip_layers = []
#     for idx_1,m in enumerate(MKIR_layers):
#         if m.use_res_connect:
#             IR_skip_layers.append(m)
#     hook_temp = [[] for x in range(len(IR_skip_layers))]
#     handles = []
#     for idx_1,m in enumerate(IR_skip_layers):
#         if m.use_res_connect:
#             h = m.register_forward_hook(output_hook(hook_temp[idx_1]))
#             handles.append(h)
#     self.handles = handles
#     return hook_temp

# def remove_hook(self):
#     for h in self.handles:
#         h.remove()