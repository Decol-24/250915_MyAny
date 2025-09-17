import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import chardet
import re

with open("./plt/feats","rb") as f:
    data = pickle.load(f)

# for psv in range(7):
#     # 直接绘图
#     # cv2.imwrite("./plt/psv1_{}.png".format(psv),seg[psv]*255)

#     #密度图
#     fig = plt.figure(figsize=(12, 6))
#     plt.hist(data[psv,0].flatten(), bins=5)
#     # plt.show()

#     # 热力图
#     # data = (seg[psv]*255).astype(int)

#     # 循环数据并创建文本注释
#     # ha 水平对齐方式
#     # for i in range(540):
#     #     for j in range(960):
#     #         text = ax.text(j, i, '{:.1f}'.format(data[i, j]),
#     #                     ha="center", va="center", color="w")

#     # fig.tight_layout()
#     plt.savefig('./plt/hist1_{}.png'.format(psv))

# 看网络输出和gt
# gt = data[0].cpu().detach().numpy()
# et = data[1].cpu().detach().numpy()
# b_gt = data[2].cpu().detach().numpy()
# for B in range(32):
#     cv2.imwrite("./plt/gt_{}.png".format(B),gt[B,0])
#     cv2.imwrite("./plt/et_{}.png".format(B),et[B,0]*255)
#     cv2.imwrite("./plt/b_gt_{}.png".format(B),b_gt[B,0]*255)

# 代价体合并
# psv = data[0].cpu().detach().numpy()
# left = data[1].cpu().detach().numpy()
# right = data[2].cpu().detach().numpy()

# # for B in range(32):
# #     cv2.imwrite("./plt/psv_{}.png".format(B),psv[B,0].mean(0)*2000)
# #     cv2.imwrite("./plt/left_{}.png".format(B),left[B].mean(0)*2000)
# #     cv2.imwrite("./plt/right_{}.png".format(B),right[B].mean(0)*2000)

# a = psv[0,0]
# b = left[0]
# c = right[0]
# print(a = b)

# 看输入图
# def to_png(img):
#     img = (img*0.229+0.485)*255
#     img = np.clip(img,0,255).astype(int)
#     img = np.transpose(img, (1, 2, 0)) 
#     return img

# left = data[0].cpu().detach().numpy()
# right = data[1].cpu().detach().numpy()
# disp_l = data[2].cpu().detach().numpy()


# B = 0
# cv2.imwrite("./plt/left_{}.png".format(B),to_png(left))
# cv2.imwrite("./plt/right_{}.png".format(B),to_png(right))
# cv2.imwrite("./plt/disp_l_{}.png".format(B),disp_l)

# 看注意力指导图

def to_png(img):
    img = img*25500
    img = np.clip(img,0,255).astype(int)
    return img

data = data.mean(dim=1).cpu().numpy()
data = data[0]

for d in range(data.shape[0]):
    # cv2.imwrite("./plt/attention_{}.png".format(d),to_png(data[d]))
    fig = plt.figure(figsize=(24, 6))
    plt.hist(data[d].flatten(), bins=50)
    plt.savefig('./plt/hist1_{}.png'.format(d))
    plt.close()