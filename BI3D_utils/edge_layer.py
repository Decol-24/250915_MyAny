import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_utils.common import to_2tuple,time_counter
import pickle

class edge_layer(nn.Module):
    #在GPU上运行
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, threshold=1.0, device="cpu", norm_layer=None, cos_mode=5):
        super().__init__()

        img_size = to_2tuple(img_size)
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.token_num = self.num_patches
        self.min_group_num = 10
        self.device = device

        self.threshold = threshold
        if self.threshold < 1.0:
            self.cos = True
        else:
            self.cos = False

        if patch_size == 16:
            #ViT
            #改成fpga能部署的stride=8
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=int(patch_size/2))
            self.down = True
        elif patch_size == 4 :
            #Swim Transformer
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.down = False

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.time_counter_list = [time_counter() for x in range(1)]

        self.set_cos_func(cos_mode)

    
    def set_cos_func(self,cos_mode):
        if cos_mode == 5:
            self.cos_func = self.cos_5
        elif cos_mode == 9:
            self.cos_func = self.cos_9
        elif cos_mode == 10:
            self.cos_func = self.cos_10
        elif cos_mode == 0:
            self.cos_func = self.cos_0
            
    
    def forward(self, x):

        B = x.shape[0]
        if self.cos:
            x = self.forward_cos(x)
        
        else:
            self.time_counter_list[0].start()
            x = self.forward_PatchEmbed(x)
            self.time_counter_list[0].end(B)
        return x
        
    def forward_PatchEmbed(self, x):

        x = self.proj(x)
        if self.down:
            x = x[:, :, ::2, ::2]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x
    
    def forward_cos(self, x):

        x = self.proj(x)
        if self.down:
            x = x[:, :, ::2, ::2]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        self.time_counter_list[0].start()
        x = self.cos_func(x)
        self.time_counter_list[0].end()

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
    
    def set_device(self, device):
        self.device = device

    def cos_10(self,output):
        #优化cos_5
        #减少for循环
        #用布尔运算。尽量还原cos_5
        output = output.squeeze()

        outptut_norm = torch.mul(output,output) #[196,768]
        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j的相似度

        cos_sim_bool = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        #聚类

        # cos_sim_bool_copy = cos_sim_bool.clone()

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

        return output[centroid_idx],rebuild_idx

    def cos_9(self,output):
        #优化cos_5
        #减少for循环
        #用布尔运算消去已分配项。只关注已分配项
        output = output.squeeze()

        outptut_norm = torch.mul(output,output) #[196,768]
        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j的相似度

        cos_sim_bool = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        #聚类

        cos_sim_bool_copy = cos_sim_bool.clone()

        rebuild_idx = torch.arange(self.token_num).to(self.device)

        # debug = {}

        while(1):
            sum_cos_sim = cos_sim_bool_copy.sum(dim=1)
            target_group_idx = torch.argmax(sum_cos_sim)

            if sum_cos_sim[target_group_idx] <= self.min_group_num:
                break
            rebuild_idx[cos_sim_bool_copy[target_group_idx]] = target_group_idx
            # debug[max_group_idx] = torch.nonzero(cos_sim_bool[max_group_idx]).squeeze(dim=1)
            cos_sim_bool_copy = cos_sim_bool_copy & (~cos_sim_bool_copy[target_group_idx])

        centroid_idx = torch.arange(self.token_num).to(self.device)
        centroid_idx = (centroid_idx == rebuild_idx) #在rebuild_idx中被修改表示已被舍弃，舍弃为False

        # self.cos_5_clustering(cos_sim_bool,self.token_num,self.device)

        return output[centroid_idx],rebuild_idx

    def cos_7(self,output):
        #优化cos_5
        #减少for循环
        #用布尔运算消去已分配项
        output = output.squeeze()

        outptut_norm = torch.mul(output,output) #[196,768]
        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j的相似度

        cos_sim_bool = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        #聚类

        non_alone_idx = torch.zeros(self.token_num, dtype=torch.bool).to(self.device) #非孤独项bool索引，初始化为False

        cos_sim_bool.fill_diagonal_(False) #自身为False

        non_alone_idx = torch.any(non_alone_idx | cos_sim_bool,dim=1) #结果中孤独项为False

        cos_sim_bool.fill_diagonal_(True)

        num = torch.nonzero(non_alone_idx).shape[0]
        if num == 0:
            return output,cos_sim_bool
        elif num == self.token_num:
            return output[0],cos_sim_bool

        cos_sim_bool_copy = cos_sim_bool.clone()

        centroid_idx = []
        rebuild_idx = torch.arange(self.token_num).to(self.device)

        while(1):
            sum_cos_sim = cos_sim_bool_copy.sum(dim=1)
            max_group_idx = torch.argmax(sum_cos_sim)
            if sum_cos_sim[max_group_idx] <= self.min_group_num:
                break
            centroid_idx.append(max_group_idx)
            rebuild_idx[cos_sim_bool_copy[max_group_idx]] = max_group_idx
            cos_sim_bool_copy = cos_sim_bool_copy & (~cos_sim_bool_copy[max_group_idx])

        cos_sim_bool_copy.fill_diagonal_(False)

        non_alone_idx_2 = torch.zeros(self.token_num, dtype=torch.bool).to(self.device)
        non_alone_idx_2 = torch.any(non_alone_idx_2 | cos_sim_bool_copy,dim=1) #没有被分配的非孤独项bool索引
        non_alone_num_idx_2 = torch.nonzero(non_alone_idx_2).squeeze(dim=1)

        alone_idx = torch.logical_not(non_alone_idx) #孤独项bool索引
        alone_num_idx = torch.nonzero(alone_idx).squeeze(dim=1)

        discard = [idx for idx in range(len(rebuild_idx)) if rebuild_idx[idx] !=  idx]

        if len(centroid_idx) == 0:
            centroid_idx,_ = torch.cat((alone_num_idx,non_alone_num_idx_2)).sort()
        else:
            centroid_idx = torch.stack(centroid_idx)
            centroid_idx,_ = torch.cat((alone_num_idx,non_alone_num_idx_2,centroid_idx)).sort()

        # self.cos_5_clustering(cos_sim_bool,self.self.token_num,device)

        return output[centroid_idx],centroid_idx,rebuild_idx
    
    def cos_8(self,output):
        #优化cos_5
        #减少for循环
        #单独取出非孤独项进行布尔运算
        output = output.squeeze()
        device = output.device

        outptut_norm = torch.mul(output,output) #[196,768]

        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j的相似度

        cos_sim_bool = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        #聚类

        self.time_counter_list[1].start()
        num_index = torch.arange(self.token_num,dtype=torch.int16).to(device) #数字索引

        non_alone_idx = torch.zeros(self.token_num, dtype=torch.bool).to(device) #非孤独项bool索引，初始化为False

        cos_sim_bool.fill_diagonal_(False) #自身为False

        non_alone_idx = torch.any(non_alone_idx | cos_sim_bool,dim=1) #结果中孤独项为False

        if torch.nonzero(non_alone_idx).shape[0] == 0:
            self.time_counter_list[1].end()
            return output,True

        alone_idx = torch.logical_not(non_alone_idx) #孤独项bool索引

        cos_sim_bool.fill_diagonal_(True)

        # non_alone = cos_sim_bool[non_alone_idx,:]

        cos_sim_bool_copy = cos_sim_bool.clone()[non_alone_idx] #仅有非孤独项

        centroid_idx = []

        self.time_counter_list[1].end()

        self.time_counter_list[2].start()
        while(1):
            sum_cos_sim = cos_sim_bool_copy.sum(dim=1)
            first_group_idx = torch.argmax(sum_cos_sim)
            if sum_cos_sim[first_group_idx] <= 30:# 这里需要手动设
                break
            centroid_idx.append(first_group_idx)
            cos_sim_bool_copy = cos_sim_bool_copy & (~cos_sim_bool_copy[first_group_idx])
        self.time_counter_list[2].end()

        return output,True

    def cos_6(self,output):
        #优化cos_5
        #减少for循环
        #找独特项
        token_num = output.shape[-2]
        output = output.squeeze()
        device = output.device

        outptut_norm = torch.mul(output,output) #[196,768]

        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j的相似度

        cos_sim_bool = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        #聚类

        centroid_index = torch.arange(token_num,dtype=torch.int16).to(device) #用来记录质心

        non_alone_idx = torch.zeros(token_num, dtype=torch.bool).to(device) #非孤独项索引，初始化为False

        cos_sim_bool.fill_diagonal_(False) #自身为False

        for idx in range(token_num):

            non_alone_idx = torch.logical_or(non_alone_idx,cos_sim_bool[idx]) #孤独项为False

        alone_idx = torch.logical_not(non_alone_idx) #孤独项索引

        cos_sim_bool.fill_diagonal_(True) #自身为True

        non_alone = cos_sim_bool[non_alone_idx,:]

        unique_elements = torch.unique(non_alone,dim=0)

        #打印

        print('New')
        for idx,_ in enumerate(unique_elements):
            matches = torch.all(cos_sim_bool == unique_elements[idx], dim=1)
            indices = torch.nonzero(matches)[0].item()
            assigned_idx = [i for i in range(len(unique_elements[idx])) if unique_elements[idx][i]==True ]
            if assigned_idx:
                print('{}: Assign: {}<--{}'.format(indices,indices,assigned_idx))
        
        self.cos_5_clustering(cos_sim_bool,token_num,device)

        return output,True
    
    def cos_5_clustering(self,cos_sim_idx,token_num,device):
        replaced = torch.zeros(token_num, dtype=torch.bool).to(device) #初始化为False，被替换过则同下标为True，用来在循环中跳过

        replaced_idx = torch.arange(token_num,dtype=torch.int16).to(device) #复原用的列表

        cos_sim_idx.fill_diagonal_(False)

        assigned_set = set()
        print("Old")
        for idx in range(token_num):
            if replaced[idx]:
                continue

            this_sim_bool = cos_sim_idx[idx]
            replaced = torch.logical_or(replaced,this_sim_bool)
            replaced_idx[this_sim_bool] = idx
            assigned_idx = [i for i in range(len(this_sim_bool)) if this_sim_bool[i]==True ]
            if assigned_idx:
                print('{}: Assign: {}<--{}'.format(idx,idx,assigned_idx))
        #     assigned_set.update(assigned_idx)
        # print(assigned_set)

    def cos_5(self,output):
        #直接计算相似度
        #改进自cos_mvm,尽量减少判断
        #token size [196,786]
        #会被替换两次
        token_num = output.shape[-2]
        output = output.squeeze()

        outptut_norm = torch.mul(output,output) #[196,768]

        outptut_norm = torch.sum(outptut_norm,dim=-1,keepdim=True) #[196,1]
        outptut_norm = torch.sqrt(outptut_norm) #[196,1]
        outptut_norm = torch.div(output,outptut_norm) #after norm [196,768]

        cos_sim = outptut_norm @ outptut_norm.transpose(-2, -1) #cos_sim [196,196] cos_sim[i][j]表示i和j之间的相似度

        device = output.device

        cos_sim_idx = cos_sim > self.threshold #[b,196,196] True表示符合要求的index，自身为True

        replaced = torch.zeros(token_num, dtype=torch.bool).to(device) #初始化为False，被替换过则同下标为True，用来在循环中跳过

        replaced_idx = torch.arange(token_num,dtype=torch.int16).to(device) #复原用的列表

        cos_sim_idx.fill_diagonal_(False)

        for idx in range(token_num):
            if replaced[idx]:
                continue

            this_sim_bool = cos_sim_idx[idx]
            replaced = torch.logical_or(replaced,this_sim_bool)
            replaced_idx[this_sim_bool] = idx
        
        replaced_idx[replaced] += token_num #被替换过则大于token_num
        replaced = torch.logical_not(replaced) #被替换过则同下标为False，这样只在这里做一次取反就能避免在循环中使用xor

        return output[replaced],replaced_idx
    
    def cos_0(self,output):
        return None,None


    
