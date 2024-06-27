import time
import torch
import torch.nn as nn


class SDAWarp(nn.Module):
    def __init__(self, backbone, num_classes, info, feature_dim=None, fc_name=None, brsda_kl_weight=8e-4, brsda_recon_weight=1) -> None:
        super().__init__()
        
        self.backbone = backbone
        # TODO 写得更加鲁棒一些
        self.backbone.linear = nn.Identity()    # ensure the backbone has no linear layer
        self.feature_dim = self.backbone.feature_dim    # ensure the backbone has a feature_dim attribute
        # TODO add isda version
        
        # 设置基础
        self.multi = info['multi']
        self.multi = info['mu']
        self.sigma = info['sigma']
        self.drop_rate = info['lambda']
        self.alpha = info['alpha']
        # 分类线性层网络
        self.linear = nn.Linear(self.feature_dim, num_classes)

        self.drop = nn.Dropout(self.drop_rate)
    
    def get_loss(self, outputs, targets, criterion, logger=None, writer=None, epoch=None, is_train=False, brsda_alpha=0.5, task=''):
        if task == 'multi-label, binary-class':
            targets = targets.float()
        else:
            targets = targets.squeeze().long()
        if not is_train:
            # print(outputs.shape)
            # print(targets.shape)
            return criterion(outputs, targets)
        
        (y_hat, y_hat_tilde), (a, a_tilde)= outputs

        loss_task = criterion(y_hat, targets)
        
        if task == 'multi-label, binary-class':
            loss_task_tilde = criterion(y_hat_tilde, targets.repeat(self.multi, 1))
        else:
            loss_task_tilde = criterion(y_hat_tilde, targets.repeat(self.multi, 1))
            
        loss = loss_task_tilde
        
        if self.use_ori:
            loss = loss * brsda_alpha + loss_task 
            
        # logging is optional
        if logger is not None and writer is not None and epoch is not None:
            if epoch % 10 == 0:
                logger.info(f'loss: {loss.item():.4f}, loss_task: {loss_task.item():.4f}, loss_task_tilde: {loss_task_tilde.item():.4f}, brsda_alpha:{brsda_alpha}')
            writer.add_scalar('loss_task', loss_task.item(), epoch)
            writer.add_scalar('loss_task_tilde', loss_task_tilde.item(), epoch)
            writer.add_scalar('loss', loss.item(), epoch)
            writer.add_scalar('brsda_alpha', brsda_alpha, epoch)
        
        return loss        
    
    def forward(self, x, is_train=False):
        """
        """
        a = self.backbone(x)
        y_hat = self.linear(a)
        
        if not is_train:
            return y_hat
        
        # generate m using Gaussian distribution, mean=mu, simga=sigma
        m = torch.normal(self.mu, self.sigma, size=(a.size(0), self.multi, self.feature_dim)).to(a.device)
        a_tilde = a.repeat(self.multi, 1) + self.drop(m)

        y_hat_tilde = self.linear(a_tilde)

        return (y_hat, y_hat_tilde), (a, a_tilde)
