import time
import torch
import torch.nn as nn


class BSDAWarp(nn.Module):
    def __init__(self, backbone, num_classes, info, feature_dim=None, fc_name=None, bsda_kl_weight=8e-4, bsda_recon_weight=1) -> None:
        super().__init__()
        
        self.backbone = backbone
        # TODO 写得更加鲁棒一些
        if 'swin' in str(type(self.backbone)).lower():
            self.feature_dim = self.backbone.head.in_features    # ensure the backbone has a feature_dim attribute
            self.backbone.head.fc = nn.Identity()
        elif 'trans' in type(backbone).__name__.lower():
            self.backbone.head = nn.Identity()
            self.feature_dim = self.backbone.embed_dim    # ensure the backbone has a feature_dim attribute
        else:
            self.backbone.linear = nn.Identity()    # ensure the backbone has no linear layer
            self.feature_dim = self.backbone.feature_dim    # ensure the backbone has a feature_dim attribute
        
        # 设置基础信息
        self.multi = info['bsda_multi']
        self.use_ori = info['bsda_use_ori']
        self.bsda_lambda = info['bsda_lambda']
        self.n_channels = info['n_channels']
        # self.train_eval = info['train_evaluator']
        # self.val_eval = info['val_evaluator']
        # self.test_eval = info['test_evaluator']
        
        # 分类线性层网络
        self.linear = nn.Linear(self.feature_dim, num_classes)
        self.bsda_layer = BSDALayer(self.feature_dim, self.bsda_lambda)
        
        # 设置loss的参数
        self.bsda_kl_weight = bsda_kl_weight
        self.bsda_recon_weight = bsda_recon_weight
        
    
    def get_loss(self, outputs, targets, criterion, logger=None, writer=None, epoch=None, is_train=False, bsda_alpha=0.5, task=''):
        if task == 'multi-label, binary-class':
            targets = targets.float()
        else:
            targets = targets.squeeze().long()
        if not is_train:
            # print(outputs.shape)
            # print(targets.shape)
            return criterion(outputs, targets)
        
        (y_hat, y_hat_tilde), (a, a_tilde, a_hat, m, mu, logvar)= outputs

        loss_task = criterion(y_hat, targets)
        
        if task == 'multi-label, binary-class':
            loss_task_tilde = criterion(y_hat_tilde, targets.repeat(self.multi, 1))
        else:
            loss_task_tilde = criterion(y_hat_tilde, targets.repeat(self.multi, ))
            
        loss_bsda_kl = self.bsda_layer.calc_kl_loss(mu, logvar)
        loss_bsda_recon = self.bsda_layer.calc_recon_loss(a, a_hat, self.multi)

        loss_bsda = loss_bsda_kl * self.bsda_kl_weight + loss_bsda_recon * self.bsda_recon_weight
        
        loss = loss_task_tilde + loss_bsda
        
        if self.use_ori:
            loss = loss * bsda_alpha + loss_task 
            
        # logging is optional
        if logger is not None and writer is not None and epoch is not None:
            if epoch % 10 == 0:
                logger.info(f'loss: {loss.item():.4f}, loss_task: {loss_task.item():.4f}, loss_task_tilde: {loss_task_tilde.item():.4f}, loss_bsda: {loss_bsda.item():.4f}, loss_bsda_kl: {loss_bsda_kl.item():.4f}, loss_bsda_recon: {loss_bsda_recon.item():.4f}, bsda_alpha:{bsda_alpha}')
            writer.add_scalar('loss_task', loss_task.item(), epoch)
            writer.add_scalar('loss_task_tilde', loss_task_tilde.item(), epoch)
            writer.add_scalar('loss_bsda_kl', loss_bsda_kl.item(), epoch)
            writer.add_scalar('loss_bsda_recon', loss_bsda_recon.item(), epoch)
            writer.add_scalar('loss_bsda', loss_bsda.item(), epoch)
            writer.add_scalar('loss', loss.item(), epoch)
            writer.add_scalar('bsda_alpha', bsda_alpha, epoch)
        
        return loss        
    
    def forward(self, x, is_train=False):
        """
            all signature accrording to the bsda paper on arxiv
            
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
        
        if not is_train:
            return y_hat
        
        m, mu, logvar, a_hat = self.bsda_layer(a, multi=self.multi)
        
        a_tilde = self.bsda_layer.calc_a_tilde(a, m, multi=self.multi)
        y_hat_tilde = self.linear(a_tilde)

        return (y_hat, y_hat_tilde), (a, a_tilde, a_hat, m, mu, logvar)


class BSDALayer(nn.Module):
    def __init__(self, feature_dim, bsda_lambda=0.8) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        self.bsda_lambda = bsda_lambda
        
        self.logvar = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        
        self.d = nn.Dropout(p=self.bsda_lambda)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
        )
    
    def modified_indicator_function(self, x):
        return torch.where(x >= 0, torch.sign(x), -torch.sign(x))

    def calc_a_tilde(self, a, m, multi=1):
        a = a.repeat(multi, 1)
        return a + self.d(m) * self.modified_indicator_function(a)

    def reparameterize(self, mu, logvar, multi=1):
        std = torch.exp(0.5 * logvar)
        std = std.repeat(multi, 1)
        eps = torch.randn_like(std, device=std.device)  # TODO test whether this is right
        mu = mu.repeat(multi, 1)
        return eps * std + mu
    
    def forward(self, a, multi=1):
        """
            a: (batch_size, feature_dim)
            m: (batch_size, feature_dim)
            mu: (batch_size, feature_dim)
            logvar: (batch_size, feature_dim)
        """
        x = self.encoder(a)
        
        logvar = self.logvar(x)
        mu = torch.zeros_like(logvar, device=logvar.device)
        
        m = self.reparameterize(mu, logvar, multi)
        a_hat = self.decoder(m)
        
        return m, mu, logvar, a_hat
    
    def calc_kl_loss(self, mu, logvar):
        # MARK mu is zeros
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    
    def calc_recon_loss(self, a, a_hat, multi=1):
        recon_loss = torch.mean((a.repeat(multi, 1) - a_hat) ** 2) * 0.5
        return recon_loss
    