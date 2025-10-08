import os
import torch
from pytorch_utils.mixup import mixup_data,mixup_criterion
from pytorch_utils.grad_scale import dispatch_clip_grad
import torch.nn as nn
from Any_utils.loss import l1_loss
import gc;

class my_runner(object):
    def __init__(self,setting) -> None:
        self.setting = setting

        from pytorch_utils.common import get_logger
        self.logger = get_logger(os.path.join(self.setting.save_path,'log.log'))

        if self.setting != None:
            parser_attrs = vars(self.setting)
            for attr, value in parser_attrs.items():
                self.logger.info(f'{attr}: {value}')

        torch.backends.cudnn.benchmark = True

    def set_model(self,model):
        self.model = model.to(self.setting.device)

    def train(self, train_loader, val_loader, scheduler, optimizer):

        best_epe = self.setting.save_epe
        criterion = l1_loss(self.setting.start_disp, self.setting.end_disp, self.logger, self.setting.sparse)
        optimizer.zero_grad()
        optimizer.step()

        self.logger.info("Train start.")
        for ep in range(1, self.setting.train_EPOCHS + 1):

            self.logger.info("Epoches: [{}/{}] ============================".format(ep, self.setting.train_EPOCHS))
            scheduler.step(ep)

            self.train_one_epoch_amp(ep,train_loader,optimizer,criterion,self.setting.device)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            val_epe = self.val_onece(val_loader,self.setting.device)

            self.logger.info('val_epe: {:.3f}'.format(val_epe))

            if val_epe <= best_epe:
                best_epe = val_epe
                self.save("{}_{:.2f}".format(ep,val_epe))

        self.logger.info('Final epe is {:.3f} '.format(best_epe))
        self.logger.info('Train end.')
        return best_epe

    def train_one_epoch(self,ep,train_loader,optimizer,criterion,device):
        train_loss_1 = train_loss_2 = train_loss_3 = 0
        train_epe = 0
        self.model.train()
        idx = 0

        for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
            
            imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_true.to(device)
            mask = ((disp_true >= self.setting.start_disp) & (disp_true < self.setting.end_disp)).byte().bool() # 让超出范围的视差不影响loss
            mask.detach_()
            optimizer.zero_grad()
            preds = self.model(imgL, imgR)
            loss = criterion(preds, disp_true)
            epe = torch.mean(torch.abs(preds[-1][mask] - disp_true[mask]))
            train_loss_1 += loss[0].item()
            train_loss_2 += loss[1].item()
            train_loss_3 += loss[2].item()
            train_epe += epe.item()
            sum(loss).backward()
            dispatch_clip_grad(self.model.parameters(), self.setting.grad_clip_value)
            optimizer.step()
            idx += disp_true.shape[0]

            if batch_idx % 200 == 0:
                self.logger.info("step: [{}/{}] | loss_1: {:.3f} | loss_2: {:.3f} | loss_3: {:.3f} | epe: {:.3f}"
                                 .format(batch_idx, len(train_loader), train_loss_1 / (idx), train_loss_2 / (idx), train_loss_3 / (idx), train_epe / (idx)))

    def train_one_epoch_amp(self,ep,train_loader,optimizer,criterion,device):
        train_loss_1 = train_loss_2 = train_loss_3 = 0
        train_epe = 0
        self.model.train()
        scaler = torch.amp.GradScaler()

        for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
            
            imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_true.to(device)

            # 让超出范围的视差不影响loss
            mask = ((disp_true >= self.setting.start_disp) & (disp_true < self.setting.end_disp)).byte().bool() 
            mask.detach_()
            zero_mask = 0
            for m in mask:
                if m.sum() < 0:
                    zero_mask += 1
            valid_sample = disp_true.shape[0] - zero_mask

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = self.model(imgL, imgR)
                loss = criterion(preds, disp_true)

            H, W = disp_true[0].shape[-2], disp_true[0].shape[-1]
            epe = torch.sum(torch.abs(preds[-1][mask] - disp_true[mask])) / (H * W * valid_sample)

            train_loss_1 += (loss[0].item()) / valid_sample
            train_loss_2 += (loss[1].item()) / valid_sample
            train_loss_3 += (loss[2].item()) / valid_sample
            train_epe += epe.item()
            loss = sum(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            dispatch_clip_grad(self.model.parameters(), self.setting.grad_clip_value)
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 200 == 0:
                self.logger.info("step: [{}/{}] | loss_1: {:.3f} | loss_2: {:.3f} | loss_3: {:.3f} | epe: {:.3f}"
                                 .format(batch_idx+1, len(train_loader), train_loss_1 / (batch_idx+1), train_loss_2 / (batch_idx+1), train_loss_3 / (batch_idx+1), train_epe / (batch_idx+1)))
                

    @torch.no_grad()
    def val_onece(self,val_loader,device):
        val_epe = idx = 0
        self.model.eval()

        for batch_idx, (imgL, imgR, disp_true) in enumerate(val_loader):
            imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_true.to(device)
            mask = ((disp_true >= self.setting.start_disp) & (disp_true < self.setting.end_disp)).byte().bool() # 得到一个布尔张量，标记出 0 < disp_true < 192 的像素
            mask.detach_()
            zero_mask = 0
            for m in mask:
                if m.sum() < 0:
                    zero_mask += 1
            valid_sample = disp_true.shape[0] - zero_mask

            preds = self.model(imgL, imgR)

            H, W = disp_true[0].shape[-2], disp_true[0].shape[-1]
            epe = torch.sum(torch.abs(preds[-1][mask] - disp_true[mask])) / (H * W * valid_sample)
            val_epe += epe.item()

        val_epe /= batch_idx

        return val_epe

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

