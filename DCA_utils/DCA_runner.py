import os
import torch
from pytorch_utils.mixup import mixup_data,mixup_criterion
from pytorch_utils.grad_scale import dispatch_clip_grad
import torch.nn as nn
from DCA_utils.loss import model_loss,focal_loss

class my_runner(object):
    def __init__(self) -> None:
        pass

    def initialize_runner(self,setting):
        from pytorch_utils.common import get_logger

        self.setting = setting
        torch.backends.cudnn.benchmark = True
        self.logger = get_logger(os.path.join(self.setting.save_path,'log.log'))

        parser_attrs = vars(self.setting)
        for attr, value in parser_attrs.items():
            self.logger.info(f'{attr}: {value}')
        

    def set_model(self,model):
        self.model = model.to(self.setting.device)

    def train(self, train_loader, val_loader, scheduler, optimizer):

        best_loss = best_epe = 9999
        optimizer.zero_grad()
        optimizer.step()
        psv_disps = [10., 20., 30., 40., 50.]
        psv_disps = torch.Tensor(psv_disps).type(torch.LongTensor).to(self.setting.device)

        self.logger.info("Train start.")
        for ep in range(1, self.setting.train_EPOCHS + 1):

            self.logger.info("Epoches: [{}/{}] ============================".format(ep, self.setting.train_EPOCHS))
            scheduler.step(ep)

            self.train_one_epoch(ep,train_loader,optimizer,self.setting.device)

            val_loss,val_epe = self.val_onece(val_loader,self.setting.device)

            self.logger.info('val_loss: {:.3f} | val_epe: {:.3f}'.format(val_loss,val_epe))

            if val_loss >= best_loss:
                best_loss = val_loss
                best_epe = val_epe
                self.save("{}_{:.2f}".format(ep,val_loss))

        self.logger.info('Final loss is {:.3f} | Final acc is {:.3f} |'.format(best_loss,best_epe))
        self.logger.info('Train end.')
        return best_loss, best_epe

    def train_one_epoch(self,ep,train_loader,optimizer,device):
        train_loss = 0
        train_epe = 0
        self.model.train()

        for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):

            imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_true.to(device)
            mask = ((disp_true < 192) & (disp_true > 0)).byte().bool() # 得到一个布尔张量，标记出 0 < disp_true < 192 的像素
            mask.detach_()
            optimizer.zero_grad()
            cls_outputs, disp_outputs = self.model(imgL, imgR)
            # cls_outputs是 [pred0, pred_dca1, pred_dca2, pred1, pred2]
            # disp_outputs是 [pred_dca3, pred4]
            loss = focal_loss(cls_outputs, disp_true, self.setting.start_disp, self.setting.end_disp, self.setting.focal_coefficient, self.setting.sparse) + model_loss(disp_outputs, disp_true, mask)
            epe = torch.mean(torch.abs(disp_outputs[-1][:,mask] - disp_true[mask]))
            train_loss += loss.item()
            train_epe += epe.item()
            loss.backward()
            dispatch_clip_grad(self.model.parameters(), self.setting.grad_clip_value)
            optimizer.step()

            if batch_idx % 20 == 0:
                self.logger.info("step: [{}/{}] | loss: {:.3f} | epe: {:.3f}"
                                 .format(batch_idx+1,len(train_loader), train_loss / (batch_idx + 1), train_epe / (batch_idx + 1)))

    @torch.no_grad()
    def val_onece(self,val_loader,device):
        val_loss = val_epe = 0
        self.model.eval()

        for idx, (imgL, imgR, disp_true) in enumerate(val_loader):
            imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_true.to(device)
            mask = ((disp_true < 192) & (disp_true > 0)).byte().bool() # 得到一个布尔张量，标记出 0 < disp_true < 192 的像素
            mask.detach_()
            disp_outputs = self.model(imgL, imgR)
            epe = torch.mean(torch.abs(disp_outputs[:,mask] - disp_true[mask]))
            # val_loss += loss.item()
            val_epe += epe.item()
            val_idx = (idx + 1)

        # val_loss /= val_idx
        val_epe /= val_idx

        return val_loss,val_epe

    @torch.no_grad()
    def test(self,test_loader):
        correct = total = 0
        self.model.eval()

        for idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(self.setting.device), targets.to(self.setting.device)
            outputs = self.model(inputs)
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

        test_acc = correct / total *100.0

        return test_acc

    def save(self,name):
        check_point = {
                'model_state_dict': self.model.state_dict(),
                }

        path = os.path.join(self.setting.save_path,name)
        torch.save(check_point,path +'.pth')

    def load_pth(self,path):

        path = add_pth_suffix(path)
        if os.path.exists(path):

            check_point = torch.load(path,map_location=self.setting.device)

            missing_key, unexpected_key = self.model.load_state_dict(check_point['model_state_dict'],strict=False)
            self.logger.info("Pre-train parameter loaded.")
            if missing_key:
                self.logger.info("Missing key: {}".format(missing_key))
            if unexpected_key:
                self.logger.info("Unexpected key: {}".format(unexpected_key))
            self.model.to(self.setting.device)
        else:
            self.logger.info("Loaded faild.")
    

def add_pth_suffix(filename):
    # 检测字符串是否以 '.pth' 结尾
    if not filename.endswith('.pth'):
        # 如果不是，则添加 '.pth' 后缀
        filename += '.pth'
    return filename

