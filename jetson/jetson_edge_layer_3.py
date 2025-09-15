import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
import pickle

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

class edge_layer(nn.Module):
    #在GPU上运行部署到edge上的部分
    #改成fpga能部署的stride=8
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, threshold=1.0, norm_layer=None,device='cpu'):
        super().__init__()

        self.threshold = threshold
        self.token_num = 3136
        #ViT 196 Swim 3136
        self.min_group_num = 10
        self.device = device

        self.time_counter_list = [time_counter() for x in range(1)]
    
    def forward(self, x):

        return x
    
    def get_step_time(self):
        counter_num = len(self.time_counter_list)
        temp = []
        for idx in range(counter_num):
            temp.append(self.time_counter_list[idx].avg_time())
        return temp
    
    def reset_time_counter(self):
        counter_num = len(self.time_counter_list)
        for idx in range(counter_num):
            self.time_counter_list[idx].reset()

    def cos_10(self,output):
        #优化cos_5
        #减少for循环
        #用布尔运算。尽量还原cos_5
        self.time_counter_list[0].start()

        outptut_norm = torch.mul(output,output) #[196,768]
        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j的相似度

        cos_sim_bool = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        #聚类

        rebuild_idx = torch.arange(self.token_num).to(self.device)

        # debug = {}
        while(1):
            sum_cos_sim = cos_sim_bool.sum(dim=1)
            target_group_idx = torch.argmax(sum_cos_sim)

            if sum_cos_sim[target_group_idx] <= self.min_group_num:
                break
            rebuild_idx[cos_sim_bool[target_group_idx]] = target_group_idx
            cos_sim_bool = cos_sim_bool & (~cos_sim_bool[target_group_idx].unsqueeze(0).t())

        centroid_idx = torch.arange(self.token_num).to(self.device)
        centroid_idx = (centroid_idx == rebuild_idx) #在rebuild_idx中被修改表示已被舍弃，舍弃为False

        self.time_counter_list[0].end()

        return output[centroid_idx],rebuild_idx
    
    def cos_9(self,output):
        #先norm，再计算逐步cos，判断跳过
        #token size [196,786]
        #会被替换两次

        self.time_counter_list[0].start()

        token_num = output.shape[-2]
        device = output.device

        outptut_norm = torch.mul(output,output) #[196,768]
        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        replaced = torch.zeros(token_num, dtype=torch.bool).to(device) #初始化为False，被替换过则同下标为True，用来在循环中跳过

        replaced_idx = torch.arange(token_num,dtype=torch.int16).to(device) #复原用的列表
        
        for idx in range(token_num):
            if replaced[idx]:
                continue
            
            cos_sim = torch.zeros(token_num).to(device)
            for idx_2 in range(idx+1,token_num):
            
                cos_sim[idx_2] = torch.dot(outptut_norm[idx],outptut_norm[idx_2]) #idx的token和后面所有token的相关性

            cos_sim_idx = cos_sim > self.threshold

            replaced = torch.logical_or(replaced,cos_sim_idx)
            replaced_idx[cos_sim_idx] = idx
        
        replaced_idx[replaced] += token_num #被替换过则大于token_num
        replaced = torch.logical_not(replaced) #被替换过则同下标为False，这样只在这里做一次取反就能避免在循环中使用xor

        self.time_counter_list[0].end()
        
        return output[replaced],replaced_idx
    
    def cos_5(self,output):
        #直接计算相似度
        #改进自cos_mvm,尽量减少判断
        #token size [196,786]
        #会被替换两次
        self.time_counter_list[0].start()
        token_num = output.shape[-2]

        outptut_norm = torch.mul(output,output) #[196,768]

        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j之间的相似度

        device = output.device

        cos_sim_idx = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        cos_sim_idx.fill_diagonal_(False)

        replaced = torch.zeros(token_num, dtype=torch.bool).to(device) #初始化为False，被替换过则同下标为True，用来在循环中跳过

        replaced_idx = torch.arange(token_num,dtype=torch.int16).to(device) #复原用的列表
        
        for idx in range(token_num):
            if replaced[idx]:
                continue

            this_sim_bool = cos_sim_idx[idx]
            replaced = torch.logical_or(replaced,this_sim_bool)
            replaced_idx[this_sim_bool] = idx
        
        replaced_idx[replaced] += token_num #被替换过则大于token_num
        replaced = torch.logical_not(replaced) #被替换过则同下标为False，这样只在这里做一次取反就能避免在循环中使用xor

        self.time_counter_list[0].end()
        return output[replaced],replaced_idx
    
    def np_cos_sim(self,A,B):
        
        temp_1 = np.dot(A, B)
        temp_2 = np.linalg.norm(A, axis=0) * np.linalg.norm (B, axis=0)
        cos = temp_1 / temp_2
        
        return cos
    
    def torch_cos_sim(self,A,B):
        
        temp_1 = torch.dot(A, B)
        temp_2 = torch.linalg.norm(A) * torch.linalg.norm (B)
        cos = temp_1 / temp_2

        return cos
        
    def cos_8(self,output):
        #for循环，内置函数计算，baseline

        self.time_counter_list[0].start()

        token_num = output.shape[-2]
        device = output.device

        cos_sim = torch.zeros((196,196)).to(device)

        for i in range(token_num):
            for j in range(token_num):
                
                cos_sim[i,j] = self.torch_cos_sim(output[i],output[j])

        cos_sim_idx = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        replaced = torch.zeros(token_num, dtype=torch.bool).to(device) #初始化为False，被替换过则同下标为True，用来在循环中跳过

        replaced_idx = torch.arange(token_num,dtype=torch.int16).to(device) #复原用的列表
        
        for idx in range(token_num):
            if replaced[idx]:
                continue
            
            cos_sim_idx[idx][idx] = False #自身改为False
            replaced = torch.logical_or(replaced,cos_sim_idx[idx])
            replaced_idx[cos_sim_idx[idx]] = idx
        
        replaced_idx[replaced] += token_num #被替换过则大于token_num
        replaced = torch.logical_not(replaced) #被替换过则同下标为False，这样只在这里做一次取反就能避免在循环中使用xor

        self.time_counter_list[0].end()

        return output[replaced],replaced_idx

    
def get_image_list(runTotal,image_dir):
    listimage=os.listdir(image_dir)
    img = []

    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(path)
    
    return img

def get_batch_img(idx,batch_size,img_list,device):
    img = []
    for batch in range(batch_size):
        with open(img_list[idx*batch_size+batch],'rb') as f:
            img.append(pickle.load(f).to(device))
    return img

def test(threshold,runTotal,batch_size,image_dir):

    device = 'cuda'
    img_list = get_image_list(runTotal*batch_size,image_dir)
    layer = edge_layer(threshold,device=device).to(device)

    with open('log.txt','a') as f:
        print('Threshold: {}'.format(threshold),file=f)

    # for i in range(runTotal):

    #     token = get_batch_img(i,batch_size,img_list,device)
        
    #     for t in token:
    #         layer.cos_5(t)
    #     with open('log.txt','a') as f:
    #         print('{} / {}'.format(i,runTotal),file=f)

    # with open('log.txt','a') as f:
    #     print('cos_5',file=f)
    #     print(layer.get_step_time()[0],file=f)

    for i in range(runTotal):

        token = get_batch_img(i,batch_size,img_list,device)
        
        for t in token:
            layer.cos_10(t)
        with open('log.txt','a') as f:
            print('{} / {}'.format(i,runTotal),file=f)

    with open('log.txt','a') as f:
        print('cos_10',file=f)
        print(layer.get_step_time()[0],file=f)

    # layer.reset_time_counter()

    # for i in range(runTotal):

    #     token = get_batch_img(i,batch_size,img_list,device)
        
    #     for t in token:
    #         layer.cos_9(t)
    #     with open('log.txt','a') as f:
    #         print('{} / {}'.format(i,runTotal),file=f)

    # with open('log.txt','a') as f:
    #     print('cos_9',file=f)
    #     print(layer.get_step_time()[0],file=f)
    
    # layer.reset_time_counter()

    # for i in range(runTotal):

    #     token = get_batch_img(i,batch_size,img_list,device)

    #     for t in token:
    #         layer.cos_8(t)
    #     with open('log.txt','a') as f:
    #         print('{} / {}'.format(i,runTotal),file=f)

    # with open('log.txt','a') as f:
    #     print('cos_8',file=f)
    #     print(layer.get_step_time()[0],file=f)

    
if __name__ == "__main__":
    #test all func time

    runTotal = 100
    batch_size = 10 #run times = runTotal*batch_size
    image_dir = 'test_in_swim'
    # 输入应该是3维tensor，每张图片一个文件

    with open('log.txt','w') as f:
        print('Start',file=f)
    for threshold in [0.99, 0.99, 0.98, 0.97, 0.96, 0.95]:
        test(threshold,runTotal,batch_size,image_dir)
