import torch
import torch.nn as nn

class Disp_binary(nn.Module):

    def __init__(self, start_disp=0, end_disp=192):
        super(Disp_binary, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.start_disp = start_disp
        self.end_disp = end_disp
        #gt_Disp是视差越远，值越大

    def forward(self,et_Disp,et_Disp_refine,gt_Disp,psv_disp):
        # [BatchSize, 1, Height, Width]
        B = et_Disp.shape[0]
        gt_Disp = gt_Disp.unsqueeze(dim=1)
        # the gtDisp must be (start_disp, end_disp), otherwise, we have to mask it out
        mask = (gt_Disp > self.start_disp) & (gt_Disp < self.end_disp)
        mask = mask.detach()
        gt_Disp = gt_Disp * mask
        et_Disp = et_Disp * mask
        et_Disp_refine = et_Disp_refine * mask

        # import pickle
        # with open('disp','wb') as f:
        #     pickle.dump((gt_Disp,et_Disp),f)

        binary_gt = self.disp_to_binary(gt_Disp,psv_disp)

        loss_1 = self.loss_fn(et_Disp.flatten(), binary_gt)
        loss_2 = self.loss_fn(et_Disp_refine.flatten(), binary_gt)
        loss = 0.4 * loss_1 + 0.6 * loss_2

        epe = torch.mean(torch.abs(binary_gt - et_Disp.flatten())).detach() * 255

        return loss,epe

    def disp_to_binary(self,gtDisp,psv_disp):
        return (gtDisp <= psv_disp).flatten().to(torch.float32)