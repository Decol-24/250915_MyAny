import os
import cv2
import numpy as np
import time
import vart
import vitis_ai_library
import os
import pathlib
import xir
import threading
import pickle

#正则化
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

#预处理
def preprocess_fn(image_path, fix_scale):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image.reshape(224,224,3)

    image = normalize(image, mean = (0.507, 0.487, 0.441), std = (0.267, 0.256, 0.276)) #cifar
    # image = normalize(image, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))#imagenet100
    image = image * fix_scale
    image = image.astype(np.int8)

    return image

def app(image_dir,inter_out_dir,data_id,model,runTotal,time_c):

    listimage=os.listdir(image_dir)
    start_id = data_id*runTotal

    if (data_id+1)*runTotal <= len(listimage):
        end_id = (data_id+1)*runTotal
    else: 
        end_id = len(listimage)
        runTotal = end_id - start_id

    listimage = listimage[start_id:end_id]

    # 初始化存放输出的变量
    out_0 = [None] * runTotal #cos_sim

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

    # 预处理图像
    with open('log.txt','a') as f:
        print('Pre-processing',runTotal,'images...',file=f)
    img = []
    
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path, input_scale))
    
    with open('log.txt','a') as f:
        print('Processing data_{}'.format(data_id),file=f)

    time_c.start()

    for i in range(runTotal):
        #把输入buffer的地址指向图片
        input_tensor_buffers[0] = img[i].reshape(input_shape[1:])
        #推论
        v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
        runner.wait(v)
        #np.copy重要，否则out_q[i]全部指向最新的np_output[0]值
        out_0[i] = np.copy(np_output_0[0])

    time_c.end(runTotal)

    out = [None] * runTotal
    for i in range(runTotal):
        label, _ = listimage[i].split('_',1)
        out[i] = (out_0[i],label)

    creat_folder(inter_out_dir)
    path = os.path.join(inter_out_dir, 'inter_{}'.format(data_id))
    with open(path,'wb') as f:
        pickle.dump(out,f)

def app_no_dump(image_dir,inter_out_dir,data_id,model,once_img_num,time_c):

    listimage=os.listdir(image_dir)
    start_id = data_id*once_img_num

    if (data_id+1)*once_img_num <= len(listimage):
        end_id = (data_id+1)*once_img_num
    else: 
        end_id = len(listimage)
        once_img_num = end_id - start_id

    listimage = listimage[start_id:end_id]

    # 初始化存放输出的变量
    out_0 = [None] * once_img_num #cos_sim

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

    # 预处理图像
    with open('log.txt','a') as f:
        print('Pre-processing',once_img_num,'images...',file=f)
    img = []
    
    for i in range(once_img_num):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path, input_scale))
    
    with open('log.txt','a') as f:
        print('Processing data_{}'.format(data_id),file=f)

    time_c.start()

    for i in range(once_img_num):
        #把输入buffer的地址指向图片
        input_tensor_buffers[0] = img[i].reshape(input_shape[1:])
        #推论
        v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
        runner.wait(v)
        #np.copy重要，否则out_q[i]全部指向最新的np_output[0]值
        out_0[i] = np.copy(np_output_0[0])

    time_c.end(once_img_num)

    out = [None] * once_img_num
    for i in range(once_img_num):
        label, _ = listimage[i].split('_',1)
        out[i] = (out_0[i],label)

    return out


def creat_folder(folder):
    #不存在则创建，存在不处理
    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder

class time_counter():
    #记录start 和 end 之间的时间和次数，得到平均一次消耗的时间
    def __init__(self):
        self.sum_time = 0
        self.count = 0
        self.time_temp = 0
    
    def start(self):
        self.time_temp = time.perf_counter()
    
    def end(self,count=1):
        time_gap = time.perf_counter() - self.time_temp
        self.sum_time += time_gap
        self.count += count
        self.time_temp = 0
    
    def avg_time(self):
        return self.sum_time/self.count
    
    def reset(self):
        self.sum_time = 0
        self.count = 0