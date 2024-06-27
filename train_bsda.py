"""
    训练模型
"""
import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from tensorboardX import SummaryWriter
from tqdm import trange
import logging
from utils.datasets import get_dataset_and_info
from networks import get_model
from utils.tools import AverageMeter
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from torchvision.models import efficientnet_b0, vit_b_16, swin_b
import torch.optim.lr_scheduler as lr_scheduler
# 构建网络参数
# 构建数据集
# 构建优化器
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(args):

    # 构建保存模型的路径
    last_prefix = f'{time.strftime("%y%m%d_%H%M%S")}_{args.model_name}_{args.aug}' + ('_isda' if args.isda else '') + ('_bsda' if args.bsda else '') + (f'_{args.bsda_lambda:.1f}' if args.bsda else '') + (f'_{args.bsda_multi:.1f}' if args.bsda else '')+ (f'_use_ori_a' if args.bsda_use_ori else '') + f'_seed_{args.seed}'
    save_path = os.path.join(
        args.output_root, args.data_flag, last_prefix, 
    )
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    # 构建logger
    logger = get_logger(save_path)

    logger.info('==> Creating dataset')
    (train_dataset, val_dataset, test_dataset), data_info = get_dataset_and_info(args.data_flag, args.aug, resize=args.resize)
    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,  batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    
    logger.info("==> Building model")
    data_info = {
        'n_classes': len(data_info['label']),
        'n_channels': data_info['n_channels'],
        'task': data_info['task'],
        'size': data_info['size'],
        'train_evaluator': data_info['train_evaluator'],
        'val_evaluator': data_info['val_evaluator'],
        'test_evaluator': data_info['test_evaluator'],
        'bsda_lambda': args.bsda_lambda, 
        'bsda_multi': args.bsda_multi, 
        'bsda_use_ori': args.bsda_use_ori,
        'isda_lambda': args.isda_lambda
    }
    model, model_info = get_model(args.model_name, args.isda, args.bsda, data_info)
    
    # TODO 是否需要进行恢复训练
    
    if data_info['task'] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    # 构建优化器以及学习率调度
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler()
    cosine_decay_epochs = 100
    warmup_epochs = 5
    
    scheduler_warmup = lr_scheduler.StepLR(optim, step_size=1, gamma=1.1)  # 学习率保持不变
    # 余弦退火阶段
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(optim, T_max=cosine_decay_epochs, eta_min=0)

    if args.bsda:
        logger.critical(f"Backbone Parameters:{sum([p.data.nelement() for p in model.backbone.parameters()])}, BSDA Parameters:{sum([p.data.nelement() for p in model.bsda_layer.parameters()])}")
    else:
        logger.critical(f"Backbone Parameters:{sum([p.data.nelement() for p in model.parameters()])}")
    model = model.to(device)
    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_' + log for log in logs]
    val_logs = ['val_' + log for log in logs]
    test_logs = ['test_' + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'TensorboardResults'))
    
    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)
    
    global epoch
    epoch = 0
    addictional_time = AverageMeter()
    elary_stop = 0
    for iter in range(model_info['epochs']):
        logger.info(f'Training Epoch: {epoch}')
        train_loss, train_time = train(model, train_loader, criterion, optim, logger, writer, task=data_info['task'], alpha=args.bsda_alpha, max_epoch=model_info['epochs'])
        addictional_time.update(train_time, 1)
        
        logger.info(f'Testing Epoch: {epoch}')
        train_metrics = test(model, data_info['train_evaluator'], train_loader_at_eval, criterion, device, logger, writer, split='train', task=data_info['task'])
        val_metrics = test(model, data_info['val_evaluator'], val_loader, criterion, device, logger, writer, split='val', task=data_info['task'])
        test_metrics = test(model, data_info['test_evaluator'], test_loader, criterion, device, logger, writer, split='test', task=data_info['task'])
        
        # 记录学习率
        writer.add_scalar('learning_rate', optim.param_groups[0]['lr'], epoch)  
        
        # 记录本轮次的结果
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]
        
        
        logger.critical(f'Epoch: {epoch},' + ','.join([f'{key}:{log_dict[key]:.3f}' for key in log_dict]))
        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
        
        # 保存最好的结果
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            
            logger.info(f'current best epoch: {best_epoch}, current best auc: {best_auc}, saving model to {os.path.join(save_path, "best_model.pth")}')
            
            state = {'net': best_model.state_dict()}
            torch.save(state, os.path.join(save_path, 'best_model.pth'))

            elary_stop = 0
        else:
            elary_stop += 1
            if elary_stop > 10:
                break
        
        # 调整学习率
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        # else:
        #     scheduler_cosine.step()
            
    # 保存最终结果
    state = {'net': best_model.state_dict()}
    torch.save(state, os.path.join(save_path, 'last_model.pth'))
    
    # 最终测试
    train_metrics = test(best_model, data_info['train_evaluator'], train_loader_at_eval, criterion, device, logger, writer, split='train', task=data_info['task'])
    val_metrics = test(best_model, data_info['val_evaluator'], val_loader, criterion, device, logger, writer, split='val', task=data_info['task'])
    test_metrics = test(best_model, data_info['test_evaluator'], test_loader, criterion, device, logger, writer, split='test', task=data_info['task'])
        
    # 记录日志
    # 记录运行时间
    logger.critical('total time: %.3f, avg time: %.3f' % (addictional_time.sum, addictional_time.ave))
    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])
    log = '%s\n' % (args.data_flag) + train_log + val_log + test_log
    logger.critical(log)
    
    # 写入最终结果到文件
    with open(os.path.join(save_path, f'{args.data_flag}_log.txt'), 'a') as f:
        f.write(log)
        f.write('total time: %.3f, avg time: %.3f\n' % (addictional_time.sum, addictional_time.ave))
        # f.write(f"Backbone Parameters:{sum([p.data.nelement() for p in model.backbone.parameters()])}, BSDA Parameters:{sum([p.data.nelement() for p in model.bsda_layer.parameters()])}")
        # # f.write(f'avg')
        if args.bsda:
            logger.critical(f"Backbone Parameters:{sum([p.data.nelement() for p in model.backbone.parameters()])}, BSDA Parameters:{sum([p.data.nelement() for p in model.bsda_layer.parameters()])}")
        else:
            logger.critical(f"Backbone Parameters:{sum([p.data.nelement() for p in model.parameters()])}")

    writer.close()
    
def train(model, data_loader, criterion, optim, logger, writer, task='', alpha=0.5, max_epoch=100):
    total_loss = []
    global epoch 
    model.train()
    ratio = min(alpha * (epoch / (max_epoch // 2)), alpha)
    batch_time = AverageMeter()
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        
        outputs = model(inputs, is_train=True)
        loss = model.get_loss(outputs, targets, criterion, logger, writer, epoch, is_train=True, bsda_alpha=ratio, task=task)
        
        loss.backward()
        optim.step()
        total_loss.append(loss.item())
        
        batch_time.update(time.time() - end)
        end = time.time()
        # print('Time: {batch_time.value:.3f} ({batch_time.ave:.3f})'.format(batch_time=batch_time))
    epoch += 1
    epoch_loss = np.mean(total_loss)
    return epoch_loss, batch_time.sum

def test(model, test_evaluator, data_loader, criterion, device, logger, writer, split='test', task=''):
    model.eval()
    total_loss = []
    y_scores = torch.tensor([]).to(device)
    y_s = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = model.get_loss(outputs, targets, criterion, logger, writer, epoch, task=task)
            if task == 'multi-label, binary-class':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            
            total_loss.append(loss.item())
            y_scores = torch.cat((y_scores, outputs), dim=0)
            y_s = torch.cat((y_s, targets), dim=0)
        
        y_s = y_s.detach().cpu().numpy()
        y_scores = y_scores.detach().cpu().numpy()
        if test_evaluator is None:
            # 自己计算auc and acc
            # report = classification_report(y_s, np.argmax(y_scores, axis=1), output_dict=True)
            # logger.critical(report)
            # auc = report['macro avg']['f1-score']
            # acc = report['accuracy']
            # 二分类的话更换参数
            if y_scores.shape[1] > 2:
                auc, acc = roc_auc_score(y_s, y_scores, multi_class='ovr', average='weighted'), accuracy_score(y_s, np.argmax(y_scores, axis=1))
            else:
                auc, acc = roc_auc_score(y_s, y_scores[:, 1]), accuracy_score(y_s, np.argmax(y_scores, axis=1))
        else:

            auc, acc = test_evaluator.evaluate(y_scores)
        test_loss = np.mean(total_loss)
    return test_loss, auc, acc

def get_logger(save_path: str):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(save_path, 'log.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 添加FileHandler到Logger对象
    logger.addHandler(file_handler)
    return logger

def set_seed(seed:int =3047):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 如果使用CuDNN，可能还需要设置以下两行以确保结果的一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='Run training process for Bayesian Random Semantic Data Augmentation')
    parser.add_argument('--data_flag', type=str, default='breastmnist', help='name of dataset')
    parser.add_argument('--output_root', type=str, default='./output', help='output path')
    parser.add_argument('--model_name', type=str, default='resnet18', help='name of model', 
                        # choices=['resnet18', 'resnet50', 'resnext', 'wrn28', 'vit', 'swin', 'efficientnet', 'densenet']
                        choices=['resnet18', 'resnet50', 'efficientnet', 'densenet', 'vit-t', 'vit-b', 'vit-s', 'swin-t', 'swin-b', 'swin-s']
                        )
    parser.add_argument('--download', action='store_true', default=True, help='whether to download dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--aug', type=str, default='none', help='type of augmentation', 
                        # choices=['none', 'bsa', 'cutmix', 'mixup', 'cutmixup', 'cutout', 'cutmixout', 'mixupout', 'cutmixupout']
                        choices=['none', 'augmix', 'ra', 'aa', 'trivial', 'med', '']
                        )
    parser.add_argument('--isda', action='store_true', default=False, help='whether to use ISDA')
    parser.add_argument('--isda_lambda', type=float, default=0.5, help='alpha for ISDA')
    parser.add_argument('--bsda', action='store_true', default=True, help='whether to use BSDA')
    parser.add_argument('--bsda_lambda', type=float, default=0.8, help='dropout probability for bayesian random semantic data augmentation')
    parser.add_argument('--bsda_multi', type=int, default=10, help='number of augmentation for BSDA')
    parser.add_argument('--bsda_alpha', type=float, default=0.5, help='alpha for BSDA')
    parser.add_argument('--bsda_use_ori', action='store_true', help='use original feature')
    parser.add_argument('--num_workers', type=int, default=0, choices=[0, 1, 2, 4, 6, 8], help='number of workers for data loader')
    parser.add_argument('--resize', help='resize images of size 28x28 to 224x224', action="store_true")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()   # 获取参数
    set_seed(args.seed)          # 设置随机种子
    main(args)
