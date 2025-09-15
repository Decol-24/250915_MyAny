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
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, threshold=1.0, norm_layer=None):
        super().__init__()

        self.threshold = threshold

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
    
    def cos_mvm_9(self,output):
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
    
    def cos_mvm_5(self,output):
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
    
    
    def torch_cos_sim(self,A,B):
        
        temp_1 = torch.dot(A, B)
        temp_2 = torch.linalg.norm(A) * torch.linalg.norm (B)
        cos = temp_1 / temp_2
        return cos
        
    def cos_mvm_8(self,output):
        #for循环，内置函数计算，baseline

        self.time_counter_list[0].start()

        token_num = output.shape[-2]
        device = output.device

        cos_sim = np.zeros((196,196))

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

def test(threshold):
    device = 'cuda'
    runTotal = 10
    batch_size = 100 #run times = runTotal*batch_size
    image_dir = 'test_out_2'

    img_list = get_image_list(runTotal*batch_size,image_dir)
    layer = edge_layer(threshold).to(device)

    with open('log.txt','a') as f:
        print('Threshold: {}'.format(threshold),file=f)

    for i in range(runTotal):

        token = get_batch_img(i,batch_size,img_list,device)
        
        for t in token:
            layer.cos_mvm_5(t)
        with open('log.txt','a') as f:
            print('{} / {}'.format(i,runTotal),file=f)

    with open('log.txt','a') as f:
        print(layer.get_step_time()[0],file=f)

    
if __name__ == "__main__":
    #test func time

    with open('log.txt','w') as f:
        print('Start',file=f)
    for threshold in [0.85,0.9,0.95]:
        test(threshold)
