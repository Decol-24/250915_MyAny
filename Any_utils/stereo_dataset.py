import os
import os.path
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import re
import numpy as np
import chardet
import torchvision.transforms as transforms
import torch
import time

def sub_dataset(dataset,factor=100):
    #等距序列做索引下标
    indices = [x for x in range(0, len(dataset)) if x % factor == 0]
    subdataset = torch.utils.data.Subset(dataset, indices)
    return subdataset

def is_image_file(filename):
    IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader_SceneFlow(filepath,select=[0,1,2]):
    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_finalpass') > -1] #找到所有包含原始图片的文件夹
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]  #找到所有包含视差文件的文件夹

    # monkaa_part
    if 0 in select:
        monkaa_path = filepath + [x for x in image if 'monkaa' in x][0] #找到monkaa原始图片的文件夹
        monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0] #找到monkaa视差文件的文件夹
        monkaa_path = monkaa_path + '/frames_finalpass/'
        monkaa_disp = monkaa_disp + '/disparity/'

        monkaa_dir = os.listdir(monkaa_path)
        for dd in monkaa_dir:
            for im in os.listdir(monkaa_path + '/' + dd + '/left/'):
                if is_image_file(monkaa_path + '/' + dd + '/left/' + im):
                    all_left_img.append(monkaa_path + '/' + dd + '/left/' + im) #对每个列出文件判断，如果是图片文件，则添加到列表中
                    all_left_disp.append(monkaa_disp + '/' + dd + '/left/' + im.split(".")[0] + '.pfm')  #对每个列出文件判断，如果是视差文件，则添加到此列表中

            for im in os.listdir(monkaa_path + '/' + dd + '/right/'):
                if is_image_file(monkaa_path + '/' + dd + '/right/' + im):
                    all_right_img.append(monkaa_path + '/' + dd + '/right/' + im) #对右侧图像做同处理。右侧图片不含视差文件

    if 1 in select:
        flying_path = filepath + [x for x in image if x == 'frames_finalpass'][0] #找到包含飞行图像的文件夹，这个文件夹名字就是frames_finalpass
        flying_disp = filepath + [x for x in disp if x == 'disparity'][0] #找到包含飞行图像的视差文件的文件夹
        flying_dir = flying_path + '/TRAIN/' #飞行图像的训练集目录
        subdir = ['A', 'B', 'C'] #飞行图像的子目录

        for ss in subdir:
            flying = os.listdir(flying_dir + ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
                for im in imm_l:
                    if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                        all_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im) #添加左图

                    all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm') #添加左图对应的视差文件，在另一个文件夹中

                    if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im): #添加右图
                        all_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

        flying_dir = flying_path + '/TEST/' #飞行图像的测试集目录

        subdir = ['A', 'B', 'C']

        for ss in subdir:
            flying = os.listdir(flying_dir + ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
                for im in imm_l:
                    if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                        test_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                    test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                        test_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    if 2 in select:
        driving_dir = filepath + [x for x in image if 'driving' in x][0]
        driving_disp = filepath + [x for x in disp if 'driving' in x][0]

        driving_dir = driving_dir + '/frames_finalpass/'
        driving_disp = driving_disp + '/disparity/'

        subdir1 = ['35mm_focallength', '15mm_focallength'] #一级子目录
        subdir2 = ['scene_backwards', 'scene_forwards'] #二级子目录
        subdir3 = ['fast', 'slow'] #三级子目录

        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                    for im in imm_l:
                        if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                            all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                        all_left_disp.append(
                            driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                        if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                            all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)
        #基本上和上面一样的逻辑，添加左图、右图和视差文件

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

def dataloader_toy_SceneFlow(filepath):
    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_finalpass') > -1] #找到所有包含原始图片的文件夹
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]  #找到所有包含视差文件的文件夹

    flying_path = filepath + [x for x in image if x == 'frames_finalpass'][0] #找到包含飞行图像的文件夹，这个文件夹名字就是frames_finalpass
    flying_disp = filepath + [x for x in disp if x == 'disparity'][0] #找到包含飞行图像的视差文件的文件夹
    flying_dir = flying_path + '/TRAIN/' #飞行图像的训练集目录
    subdir = ['A', 'B', 'C'] #飞行图像的子目录

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)[:1]
        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    all_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im) #添加左图

                all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm') #添加左图对应的视差文件，在另一个文件夹中

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im): #添加右图
                    all_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    flying_dir = flying_path + '/TEST/' #飞行图像的测试集目录

    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)[:1]
        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    test_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                    test_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)  
    header = header.decode(encode_type['encoding'])
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode(encode_type['encoding']))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


def base_norm(normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])

def inception_color_preproccess(normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.Normalize(**normalize)
    ])

def pad_img(left_img,right_img,disp_L,th=576,tw=960):

    h = left_img.shape[1]
    w = left_img.shape[2]
    pad_w = tw - w if tw - w > 0 else 0
    pad_h = th - h if th - h > 0 else 0
    pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
    img_left_pad = pad_opr(left_img)
    img_right_pad = pad_opr(right_img)
    disp_L_pad = pad_opr(disp_L)

    return img_left_pad, img_right_pad, disp_L_pad

def crop_img(left_img,right_img,disp_L,th=256,tw=512):

    h, w = left_img.shape[-2], left_img.shape[-1]

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    left_img_crop = left_img[:,y1:(y1+th), x1:(x1+tw)]
    right_img_crop = right_img[:,y1:(y1+th), x1:(x1+tw)]

    dataL_crop = disp_L[y1:y1 + th, x1:x1 + tw]

    return left_img_crop, right_img_crop, dataL_crop

class myImageFloder_SceneFlow(data.Dataset):
    def __init__(self, left, right, left_disparity, training):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.training = training
        if self.training:
            self.transforms = inception_color_preproccess()
        else:
            self.transforms = base_norm()

    def __getitem__(self, index):

        left = self.left[index]
        right = self.right[index]
        _disp_L = self.disp_L[index]

        left_img = Image.open(left).convert('RGB')
        right_img = Image.open(right).convert('RGB')
        disp_L, scaleL = self.disparity_loader(_disp_L)
        disp_L = torch.tensor(disp_L.copy(), dtype=torch.float32)
        
        # imgL   [wigth:960,height:540] type:Image
        # dataL  [wigth:960,height:540]

        if self.training:
            #随机区域裁剪
            left_img = self.transforms(left_img)
            right_img = self.transforms(right_img)

            left_img,right_img,disp_L = crop_img(left_img,right_img,disp_L,th=256,tw=512)

            return left_img, right_img, disp_L

        else:
            left_img = self.transforms(left_img)
            right_img = self.transforms(right_img)
            left_img,right_img,disp_L = crop_img(left_img,right_img,disp_L,th=256,tw=512)

            return left_img, right_img, disp_L

    def __len__(self):
        return len(self.left)
    
    def disparity_loader(self,path):
        if '.png' in path:
            return Image.open(path)
        else:
            return readPFM(path)