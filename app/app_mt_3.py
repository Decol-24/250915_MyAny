'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


'''
Author: Mark Harvey
Modified for Custom OP tutorial by Giovanni Guasti
'''

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import vitis_ai_library
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import random
import pickle

_divider = '-------------------------------'

#预处理

def normalize(img,mean,std):
    #模型训练时用Image.open(img), 运行时用cv2.imread(img)打开. 虽然是同一张图片,因为打开方式不同所以需要转换RGB通道
    #cv2.imread()打开时为BGR
    RGB = [0 for x in range(3)]
    RGB[2], RGB[1], RGB[0] = cv2.split(img)
    for idx,img in enumerate(RGB):
        img = img / 255.
        img = img - mean[idx]
        img = img / std[idx]
        RGB[idx] = img
    image = cv2.merge([RGB[0], RGB[1], RGB[2]])

    return image

def preprocess_fn(image_path, fix_scale):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image.reshape(224,224,3)

    image = normalize(image, mean = (0.507, 0.487, 0.441), std = (0.267, 0.256, 0.276)) #cifar
    # image = normalize(image, mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010))#kuzushiji
    image = image * fix_scale
    image = image.astype(np.int8)

    return image

def decode_5_7(np_input, cos_sim):

    # print(np_input.shape)
    # print(cos_sim.shape)

    cos_sim_idx = cos_sim > 0.9

    replaced = np.zeros(196, dtype=np.bool_) #初始化为False，被替换过则同下标为True，用来在循环中跳过

    replaced_idx = np.arange(196, dtype=np.int16) #复原用的列表
    
    for idx in range(196):
        if replaced[idx]:
            continue
        
        cos_sim_idx[idx][idx] = False #自身改为False
        np.logical_or(replaced,cos_sim_idx[idx],out=replaced)
        replaced_idx[cos_sim_idx[idx]] = idx
    
    replaced_idx[replaced] += 196 #被替换过则对应元素大于token_num，没被替换过则记录应该存放的下标
    np.logical_not(replaced,out=replaced) #被替换过则同下标为False，这样只在这里做一次取反就能避免在循环中使用xor

    return np_input[replaced], replaced_idx

def app(image_dir,data_id,model):

    listimage=os.listdir(image_dir)
    # random.shuffle(listimage)

    runTotal = 1000
    listimage = listimage[data_id*runTotal : (data_id+1)*runTotal]

    # 初始化存放输出的变量
    out_0 = [None] * runTotal #cos_sim
    out_1 = [None] * runTotal #output
    out_compress = [None] * runTotal
    out_replaced_idx = [None] * runTotal

    g = xir.Graph.deserialize(model)

    # 创建 graph runner
    runner = vitis_ai_library.GraphRunner.create_graph_runner(g)

    #np_output存放本次推论的输出，和输出buffer同地址
    input_tensor_buffers = runner.get_inputs()
    input_shape = input_tensor_buffers[0].get_tensor().dims

    output_tensor_buffers = [runner.get_outputs()[i] for i in range(len(runner.get_outputs()))]

    input_fixpos = runner.get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    np_output_0 = np.array(output_tensor_buffers[0],copy=False)
    np_output_1 = np.array(output_tensor_buffers[1],copy=False)

    # 预处理图像
    print('Pre-processing',runTotal,'images...')
    img = []
    
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path, input_scale))
    
    print('Processing')

    time_check = [time.perf_counter()]
    for i in range(runTotal):
        #把输入buffer的地址指向图片
        input_tensor_buffers[0] = img[i].reshape(input_shape[1:])
        #推论
        v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
        runner.wait(v)
        #np.copy重要，否则out_q[i]全部指向最新的np_output[0]值
        out_0[i] = np.copy(np_output_0[0])
        out_1[i] = np.copy(np_output_1[0])

    time_check.append(time.perf_counter())
    print('Compressing')
    for i in range(runTotal):
        out_compress[i],out_replaced_idx[i] = decode_5_7(out_0[i],out_1[i])

    time_check.append(time.perf_counter())

    out = [None] * runTotal
    for i in range(runTotal):
        label, _ = listimage[i].split('_',1)
        out[i] = (out_compress[i],out_replaced_idx[i],label)

    for time_idx in range(1,len(time_check)):
        # time_gap = (time_check[time_idx] - time_check[time_idx-1])
        time_gap = (time_check[time_idx] - time_check[time_idx-1]) / runTotal
        print(time_gap)
    
    with open('./out/out_fpga_{}'.format(data_id),'wb') as f:
        pickle.dump(out,f)

    return

def main():

  # 分成10批次测试
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='deploy_cifar10_224', help='Path to folder of images. Default is images')
  ap.add_argument('-m', '--model',     type=str, default='cos_5.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
  args = ap.parse_args()
  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --model     : ', args.model)

  for data_id in range(10):
    app(args.image_dir,data_id,args.model)

if __name__ == '__main__':
  main()