import Any_utils.stereo_dataset as SD
import torch

def creat_SceneFlow(datapath,batch_size,**kwargs):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = SD.dataloader_SceneFlow(datapath)
    trainset = SD.myImageFloder_SceneFlow(all_left_img, all_right_img, all_left_disp, True)
    # trainset = SD.subdataset(trainset,1000)
    testset = SD.myImageFloder_SceneFlow(test_left_img, test_right_img, test_left_disp, False)
    # testset = SD.subdataset(testset,1000)
    TrainImgLoader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=16, drop_last=False, timeout=60)
    TestImgLoader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, timeout=60)
    return TrainImgLoader,TestImgLoader

def creat_toydataset(datapath,batch_size,**kwargs):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = SD.dataloader_SceneFlow(datapath)
    trainset = SD.myImageFloder_SceneFlow(all_left_img, all_right_img, all_left_disp, True)
    trainset = SD.sub_dataset(trainset,1000)
    testset = SD.myImageFloder_SceneFlow(test_left_img, test_right_img, test_left_disp, False)
    testset = SD.sub_dataset(testset,1000)
    TrainImgLoader = torch.utils.data.DataLoader(trainset,batch_size=1, shuffle=True, num_workers=16, drop_last=False, timeout=60)
    TestImgLoader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False, num_workers=4, drop_last=False, timeout=60)
    return TrainImgLoader,TestImgLoader

def creat_toy_SceneFlow(datapath,batch_size,**kwargs):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = SD.dataloader_toy_SceneFlow(datapath)
    trainset = SD.myImageFloder_SceneFlow(all_left_img, all_right_img, all_left_disp, True)
    testset = SD.myImageFloder_SceneFlow(test_left_img, test_right_img, test_left_disp, False)
    TrainImgLoader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=16, drop_last=False, timeout=60)
    TestImgLoader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, timeout=60)
    return TrainImgLoader,TestImgLoader

def creat_mid_SceneFlow(datapath,batch_size,**kwargs):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = SD.dataloader_SceneFlow(datapath,select=[1])
    trainset = SD.myImageFloder_SceneFlow(all_left_img, all_right_img, all_left_disp, True)
    testset = SD.myImageFloder_SceneFlow(test_left_img, test_right_img, test_left_disp, False)
    TrainImgLoader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=16, drop_last=False, timeout=60)
    TestImgLoader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, timeout=60)
    return TrainImgLoader,TestImgLoader