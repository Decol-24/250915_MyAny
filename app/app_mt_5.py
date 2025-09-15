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
from app_utils import app,preprocess_fn,creat_folder,time_counter


def np_cos_sim(A,B):
        
    temp_1 = np.dot(A, B)
    temp_2 = np.linalg.norm(A, axis=0) * np.linalg.norm (B, axis=0)
    cos = temp_1 / temp_2
    
    return cos

def cos_8(np_input):
    #计算cos sim
    # np_input [196,768]
    token_num = 196
    cos_sim = np.zeros((token_num,token_num))

    for i in range(token_num):
        for j in range(token_num):
            cos_sim[i,j] = np_cos_sim(np_input[i],np_input[j])
    return cos_sim

def encode_5(np_input, cos_sim, threshould):
    # for cos_5,cos_7,cos_8
    #根据计算的cos sim 压缩输出
    cos_sim_idx = cos_sim > threshould

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

def get_cos_8(data_id,inter_out_dir,inter_out_2_dir,runTotal,time_c):
    # 获得cos sim结果
    with open('log.txt','a') as f:
        print('Getting cos of data_{}'.format(data_id),file=f)
    
    path = os.path.join(inter_out_dir, 'inter_{}'.format(data_id))
    with open(path,'rb') as f:
        inter_out = pickle.load(f)

    if runTotal > len(inter_out):
        runTotal = len(inter_out)
    
    cos_sim_list = [None] * runTotal

    time_c.start()
    for i in range(runTotal):
        cos_sim_list[i] = cos_8(inter_out[i][0])
        if i % 100 == 0:
            print('{} / {}'.format(i,runTotal))

    time_c.end(runTotal)

    out = [None] * runTotal
    for i in range(runTotal):
        out[i] = (inter_out[i][0],inter_out[i][1],cos_sim_list[i])  #conv_out,label,cos_sim

    creat_folder(inter_out_2_dir)
    path = os.path.join(inter_out_2_dir, 'inter_2_{}'.format(data_id))
    with open(path,'wb') as f:
        pickle.dump(out,f)

    return

def encode(data_id,inter_2_out_dir,final_out_dir,runTotal,time_c,threshould):
    # 根据保存的cos sim，压缩输出
    with open('log.txt','a') as f:
        print('Encoding data_{}'.format(data_id),file=f)
    
    path = os.path.join(inter_2_out_dir, 'inter_2_{}'.format(data_id))
    with open(path,'rb') as f:
        inter_out = pickle.load(f)

    if runTotal > len(inter_out):
        runTotal = len(inter_out)
    
    out_compress = [None] * runTotal
    out_replaced_idx = [None] * runTotal

    time_c.start()

    for i in range(runTotal):
        out_compress[i],out_replaced_idx[i] = encode_5(inter_out[i][0],inter_out[i][2],threshould)
        if i % 100 == 0:
            print('{} / {}'.format(i,runTotal))

    time_c.end(runTotal)

    out = [None] * runTotal
    for i in range(runTotal):
        out[i] = (out_compress[i],out_replaced_idx[i],inter_out[i][1])

    creat_folder(final_out_dir)
    path = os.path.join(final_out_dir, 'out_fpga_{}'.format(data_id))
    with open(path,'wb') as f:
        pickle.dump(out,f)

    return

def main(args):
    # for cos_8
    # mode = 0: 卷积结果；mode = 1: cos结果；mode = 2: 压缩结果；
    
    with open('log.txt','w') as f:
        print ('Command line options:',file=f)
        print (' --image_dir: ', args.image_dir,file=f)
        print (' --xmodel: ', args.xmodel,file=f)
        print (' --mode: ', args.mode,file=f)

    time_c = time_counter()
    inter_out_dir = './inter_out_imagenet'
    inter_out_2_dir = './inter_out_2_img'
    final_out_dir = './tes_final_out'

    threshould = 0.84
    runTotal = 1000 #!
    
    for data_id in range(12,13): #!
        if args.mode == 0:
            app(args.image_dir,inter_out_dir,data_id,args.xmodel,runTotal,time_c)
        elif args.mode == 1:
            get_cos_8(data_id,inter_out_dir,inter_out_2_dir,runTotal,time_c)
        elif args.mode == 2:
            encode(data_id,inter_out_2_dir,final_out_dir,runTotal,time_c,threshould)

        with open('log.txt','a') as f:
            print('Now avg_time:',file=f)
            print('{}'.format(time_c.avg_time()),file=f)

def main_loop(args):
    # for cos_8
    # mode = 0: 卷积结果；mode = 1: cos结果；mode = 2: 压缩结果；
    
    with open('log.txt','w') as f:
        print ('Command line options:',file=f)
        print (' --image_dir: ', args.image_dir,file=f)
        print (' --xmodel: ', args.xmodel,file=f)
        print (' --mode: ', args.mode,file=f)

    time_c = time_counter()
    inter_out_2_dir = './inter_out_2_img'
    final_out_dir = './final_out_c8_img'
    creat_folder(final_out_dir)

    runTotal = 1000 #!
    for threshould in range(80,95):
    # for threshould in [85,90,95]: #!
        _threshould = threshould*0.01
        with open('log.txt','a') as f:
            print('------------------------------',file=f)
            print('Now threshould: {}'.format(_threshould),file=f)

        for data_id in range(10,13): #!

            path = os.path.join(final_out_dir,'{}'.format(_threshould))

            if args.mode == 2:
                
                encode(data_id,inter_out_2_dir,path,runTotal,time_c,_threshould)

            with open('log.txt','a') as f:
                print('Now cos_8 avg_time:',file=f)
                print('{}'.format(time_c.avg_time()),file=f)
        
        time_c.reset()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--image_dir', type=str, default='deploy_imagenet', help='Path to folder of images. Default is images')
    ap.add_argument('-x', '--xmodel',     type=str, default='cos_imagenet.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
    ap.add_argument('-m', dest='mode', type=int, default=1, help='mode=0,计算中间结果; mode=1,得到编码后结果')
    args = ap.parse_args()
    
    # main(args)
    main_loop(args)