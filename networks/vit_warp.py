import time
import torch
import torch.nn as nn


class VitWarp(nn.Module):
    def __init__(self, backbone, num_classes, info, feature_dim=None, fc_name=None, brsda_kl_weight=8e-4, brsda_recon_weight=1) -> None:
        super().__init__()
        
        self.backbone = backbone
        # TODO 写得更加鲁棒一些
        # self.backbone.linear = nn.Identity()    # ensure the backbone has no linear layer
        if 'swin' in str(type(self.backbone)).lower():
                
            self.feature_dim = self.backbone.head.in_features    # ensure the backbone has a feature_dim attribute
            self.backbone.head.fc = nn.Identity()
        else:
            self.feature_dim = self.backbone.embed_dim    # ensure the backbone has a feature_dim attribute
            self.backbone.head = nn.Identity()

        
        # 分类线性层网络
        self.linear = nn.Linear(self.feature_dim, num_classes)
        
        # TODO add time logging
    
    def get_loss(self, outputs, targets, criterion, logger, writer, epoch, task='', is_train=False, brsda_alpha=1.0):
        if task == 'multi-label, binary-class':
                targets = targets.float()
                loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        logger.info(f'loss: {loss.item():.4f}')
        writer.add_scalar('loss', loss.item(), epoch)
        return loss     
    
    def forward(self, x, is_train=False):
        """
            all signature accrording to the brsda paper on arxiv
            
            x: (batch_size, n_channels, height, width)
            a: (batch_size, feature_dim)
            y_hat: (batch_size, num_classes)
            y_hat_tilde: (batch_size, num_classes)
            a_tilde: (batch_size, feature_dim)
            a_hat: (batch_size, feature_dim)
            m: (batch_size, feature_dim)
            mu: (batch_size, feature_dim)
            logvar: (batch_size, feature_dim)
        """
        a = self.backbone(x)
        y_hat = self.linear(a)

        return y_hat
