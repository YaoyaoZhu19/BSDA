import torch
import torch.nn as nn

import torch
import torch.nn as nn
from networks.bsda_warp import BSDALayer
class ISDAWarp(nn.Module):
    def __init__(self, backbone, num_classes, lambda0=0.5, task='Multi-Class') -> None:
        super().__init__()
        
        self.backbone = backbone
        self.backbone.linear = nn.Identity()    # ensure the backbone has no linear layer
        self.feature_dim = self.backbone.feature_dim    # ensure the backbone has a feature_dim attribute
        self.isda_criterion = ISDALoss(self.feature_dim , num_classes).cuda()
        self.task = task
        if self.task == 'multi-label, binary-class':
            self.isdas = [ISDALoss(self.feature_dim , 2).cuda() for i in range(num_classes)]
        # 分类线性层网络
        self.linear = nn.Linear(self.feature_dim, num_classes)
        
    def get_loss(self, outputs, targets, criterion, logger=None, writer=None, epoch=None, is_train=False, brsda_alpha=0.5, task='', ratio=0):
        a, y_hat = outputs
        if self.task == 'multi-label, binary-class':
            # multi-label binary-classification
            loss_task = 0.
            for i in range(targets.shape[1]):
                loss_task = loss_task + self.isdas[i](a, y_hat[:, i], [list(self.linear.parameters())[0][i,].view(1, -1)], torch.autograd.Variable(targets[:, i]), ratio)
            loss_task = loss_task / targets.shape[1]
        else:
            target_var = torch.autograd.Variable(targets)
            loss_task = self.isda_criterion(a, y_hat, self.linear.parameters(), target_var, ratio)

        loss = loss_task
            
        # logging is optional
        if logger is not None and writer is not None and epoch is not None:
            logger.info(f'loss: {loss.item():.4f}, loss_task: {loss_task.item():.4f}')
            writer.add_scalar('loss_task', loss_task.item(), epoch)
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
        if not is_train:
            return y_hat
        return a, y_hat
    

class IS_BRS_DAWarp(nn.Module):
    def __init__(self, backbone, num_classes, info, feature_dim=None, fc_name=None, brsda_kl_weight=8e-4, brsda_recon_weight=1) -> None:
        super().__init__()
        
        self.backbone = backbone
        self.backbone.linear = nn.Identity()    # ensure the backbone has no linear layer
        self.feature_dim = self.backbone.feature_dim    # ensure the backbone has a feature_dim attribute

        self.isda_criterion = ISDALoss(self.feature_dim , num_classes).cuda()
        # 设置基础信息

        self.multi = info['brsda_multi']
        self.use_ori = info['brsda_use_ori']
        self.brsda_lambda = info['brsda_lambda']
        self.n_channels = info['n_channels']
        self.train_eval = info['train_evaluator']
        self.val_eval = info['val_evaluator']
        self.test_eval = info['test_evaluator']
        
        # 分类线性层网络
        self.linear = nn.Linear(self.feature_dim, num_classes)
        
        # 设置loss的参数
        self.brsda_kl_weight = brsda_kl_weight
        self.brsda_recon_weight = brsda_recon_weight
        self.brsda_layer = BRSDALayer(self.feature_dim, self.brsda_lambda)
        # TODO add time logging
    
    def get_loss(self, outputs, targets, criterion, logger=None, writer=None, epoch=None, is_train=False, brsda_alpha=0.5, task='', ratio=0.5):

        
        (y_hat, y_hat_tilde), (a, a_tilde, a_hat, m, mu, logvar)= outputs

        target_var = torch.autograd.Variable(targets)
        loss_task = self.isda_criterion(a, y_hat, self.linear.parameters(), target_var, ratio)
        if not is_train:
            return loss_task
        
        loss_task_tilde = criterion(y_hat_tilde, targets.repeat(self.multi, ))

        loss_brsda_kl = self.brsda_layer.calc_kl_loss(mu, logvar)
        loss_brsda_recon = self.brsda_layer.calc_recon_loss(a, a_hat, self.multi)
        
        loss_brsda = loss_brsda_kl * self.brsda_kl_weight + loss_brsda_recon * self.brsda_recon_weight
        
        loss = loss_task_tilde + loss_brsda
        
        if self.use_ori:
            loss = loss * brsda_alpha + loss_task 
            
        # logging is optional
        if logger is not None and writer is not None and epoch is not None:
            logger.info(f'loss: {loss.item():.4f}, loss_task: {loss_task.item():.4f}, loss_task_tilde: {loss_task_tilde.item():.4f}, loss_brsda: {loss_brsda.item():.4f}, loss_brsda_kl: {loss_brsda_kl.item():.4f}, loss_brsda_recon: {loss_brsda_recon.item():.4f}')
            writer.add_scalar('loss_task', loss_task.item(), epoch)
            writer.add_scalar('loss_task_tilde', loss_task_tilde.item(), epoch)
            writer.add_scalar('loss_brsda_kl', loss_brsda_kl.item(), epoch)
            writer.add_scalar('loss_brsda_recon', loss_brsda_recon.item(), epoch)
            writer.add_scalar('loss_brsda', loss_brsda.item(), epoch)
            writer.add_scalar('loss', loss.item(), epoch)
            writer.add_scalar('brsda_alpha', brsda_alpha, epoch)
        
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
        
        if not is_train:
            return self.linear(a)
        
        m, mu, logvar, a_hat = self.brsda_layer(a, multi=self.multi)
        
        a_tilde = self.brsda_layer.calc_a_tilde(a, m, multi=self.multi)
        y_hat_tilde = self.linear(a_tilde)

        return (y_hat, y_hat_tilde), (a, a_tilde, a_hat, m, mu, logvar)
    

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, linear_parms, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(linear_parms)[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)
        # if C == 2:
        sigma2 = ratio * \
             torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                 CV_temp.squeeze()),
                       (NxW_ij - NxW_kj).permute(0, 2, 1))
        # sigma2 = ratio * \
        #         torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                             CV_temp),
        #                 (NxW_ij - NxW_kj).permute(0, 2, 1))

        sigma2 = sigma2.mul(torch.eye(C).cuda()
                            .expand(N, C, C)).sum(2).view(N, C)

        if len(y.shape) == 1:
            aug_result = y+ 0.5 * sigma2[:, 0]
        else:
            aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, a, y_hat, linear_parms, target_x, ratio):

        self.estimator.update_CV(a.detach(), target_x.squeeze())

        isda_aug_y = self.isda_aug(linear_parms, a, y_hat, target_x, self.estimator.CoVariance.detach(), ratio)

        # target_x = target_x.float()
        # FOR multi task
        # if len(target_x.shape) == 1:
        #     target_x = target_x.unsqueeze(1).float()
        #     isda_aug_y = torch.sigmoid(isda_aug_y)
        loss = self.cross_entropy(isda_aug_y, target_x.squeeze())

        return loss
