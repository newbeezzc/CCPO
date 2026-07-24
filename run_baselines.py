#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline对比脚本

比较Meta-Schedule策略与传统学习率调度方法的性能差异

支持的Baseline:
1. 固定学习率 (Fixed LR)
2. StepLR衰减
3. CosineAnnealing
4. ReduceLROnPlateau (使用源域损失作为监控指标)

使用方法:
    python run_baselines.py --data_root /path/to/OfficeHome \
                            --tasks Ar-Cl Ar-Pr \
                            --baselines fixed step cosine
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
from datetime import datetime
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 假设这些模块存在
from module.network import ImageClassifier
from module import backbone as BackboneNetwork
from module.Loss import AdversarialLoss_PDD_Double, MI
from utils.data_list import ImageList
from utils.transforms import ResizeImage
from utils.util import ContinuousDataloader

from configs.default_config import get_task_paths, get_num_classes
from utils.training_utils import set_seed, setup_device, AverageMeter


def parse_baseline_args():
    """解析baseline参数"""
    parser = argparse.ArgumentParser(description='Run baseline learning rate schedulers')
    
    # 数据相关
    parser.add_argument('--dset', type=str, default='officehome',
                        choices=['office31', 'officehome'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--tasks', type=str, nargs='+', required=True,
                        help='任务列表，如 Ar-Cl Ar-Pr')
    
    # 模型相关
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    
    # 训练相关
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--iters_per_epoch', type=int, default=500)
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    
    # UDA损失权重
    parser.add_argument('--pdd_tradeoff', type=float, default=1.0)
    parser.add_argument('--entropy_tradeoff', type=float, default=0.1)
    parser.add_argument('--MI_tradeoff', type=float, default=0.1)
    
    # Baseline相关
    parser.add_argument('--baselines', type=str, nargs='+', 
                        default=['fixed', 'step', 'cosine'],
                        choices=['fixed', 'step', 'cosine', 'plateau'],
                        help='要运行的baseline方法')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='每个配置运行次数')
    
    # StepLR参数
    parser.add_argument('--step_size', type=int, default=10,
                        help='StepLR的step_size')
    parser.add_argument('--step_gamma', type=float, default=0.1,
                        help='StepLR的gamma')
    
    # 其他
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./outputs/baselines')
    
    return parser.parse_args()


class BaselineTrainer:
    """Baseline训练器"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_classes = get_num_classes(args.dset)
        
        # 数据变换
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ])
        self.val_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def create_model(self):
        """创建模型"""
        backbone = BackboneNetwork.__dict__[self.args.arch](pretrained=True)
        model = ImageClassifier(
            backbone, self.num_classes, self.args.bottleneck_dim, args=self.args
        )
        return model.to(self.device)
    
    def load_data(self, source_path, target_path):
        """加载数据"""
        data_root = os.path.dirname(source_path)
        
        train_source = ImageList(data_root, open(source_path).readlines(), 
                                 transform=self.train_transform)
        train_target = ImageList(data_root, open(target_path).readlines(),
                                 transform=self.train_transform)
        val_target = ImageList(data_root, open(target_path).readlines(),
                               transform=self.val_transform)
        
        train_source_loader = DataLoader(train_source, batch_size=self.args.batch_size,
                                        shuffle=True, num_workers=self.args.workers, 
                                        drop_last=True, pin_memory=True)
        train_target_loader = DataLoader(train_target, batch_size=self.args.batch_size,
                                        shuffle=True, num_workers=self.args.workers,
                                        drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_target, batch_size=self.args.batch_size,
                               shuffle=False, num_workers=self.args.workers,
                               pin_memory=True)
        
        return (ContinuousDataloader(train_source_loader),
                ContinuousDataloader(train_target_loader),
                val_loader)
    
    def create_scheduler(self, optimizer, scheduler_type):
        """创建学习率调度器"""
        total_iters = self.args.epochs * self.args.iters_per_epoch
        
        if scheduler_type == 'fixed':
            return None
        elif scheduler_type == 'step':
            return StepLR(optimizer, step_size=self.args.step_size * self.args.iters_per_epoch,
                         gamma=self.args.step_gamma)
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=total_iters)
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                    patience=5, verbose=True)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @torch.no_grad()
    def evaluate(self, model, val_loader):
        """评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return 100.0 * correct / total
    
    def train_one_step(self, model, optimizer, pdd_adv, source_iter, target_iter, 
                       current_iter, total_iters):
        """训练一步"""
        model.train()
        pdd_adv.train()
        
        rho = current_iter / total_iters
        
        x_s, labels_s = next(source_iter)
        x_t, labels_t = next(target_iter)
        
        x_s = x_s.to(self.device)
        x_t = x_t.to(self.device)
        labels_s = labels_s.to(self.device)
        labels_t = labels_t.to(self.device)
        
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        
        cls_loss = F.cross_entropy(y_s, labels_s)
        loss_pdd, loss_entropy = pdd_adv(f, torch.cat([labels_s, labels_t], dim=0),
                                         self.args, rho)
        MI_item1, MI_item2 = MI(y_t)
        MI_loss = MI_item1 - MI_item2
        
        total_loss = (cls_loss 
                      - self.args.pdd_tradeoff * loss_pdd
                      - self.args.MI_tradeoff * MI_loss
                      - self.args.entropy_tradeoff * loss_entropy)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), cls_loss.item()
    
    def train(self, task_name, scheduler_type):
        """完整训练流程"""
        # 获取数据路径
        source_path, target_path = get_task_paths(self.args, task_name)
        
        # 加载数据
        source_iter, target_iter, val_loader = self.load_data(source_path, target_path)
        
        # 创建模型
        model = self.create_model()
        
        # 创建优化器
        optimizer = SGD(model.get_parameters(), lr=self.args.init_lr,
                       momentum=self.args.momentum, weight_decay=self.args.weight_decay,
                       nesterov=True)
        
        # 创建调度器
        scheduler = self.create_scheduler(optimizer, scheduler_type)
        
        # 创建对抗损失
        pdd_adv = AdversarialLoss_PDD_Double(model.head, model.head_aux).to(self.device)
        
        # 训练
        total_iters = self.args.epochs * self.args.iters_per_epoch
        best_acc = 0.0
        lr_history = []
        acc_history = []
        
        current_iter = 0
        for epoch in range(self.args.epochs):
            for i in range(self.args.iters_per_epoch):
                loss, cls_loss = self.train_one_step(
                    model, optimizer, pdd_adv, source_iter, target_iter,
                    current_iter, total_iters
                )
                
                # 更新学习率
                if scheduler is not None:
                    if scheduler_type == 'plateau':
                        pass  # 在epoch结束时更新
                    else:
                        scheduler.step()
                
                current_iter += 1
                
                # 记录学习率
                if current_iter % 50 == 0:
                    lr_history.append(optimizer.param_groups[0]['lr'])
            
            # 评估
            acc = self.evaluate(model, val_loader)
            acc_history.append(acc)
            best_acc = max(best_acc, acc)
            
            # ReduceLROnPlateau使用验证损失更新
            if scheduler_type == 'plateau' and scheduler is not None:
                scheduler.step(loss)
            
            print(f"  Epoch {epoch+1}/{self.args.epochs}: Acc={acc:.2f}%, Best={best_acc:.2f}%")
        
        return {
            'best_accuracy': best_acc,
            'final_accuracy': acc,
            'lr_history': lr_history,
            'acc_history': acc_history
        }


def main():
    args = parse_baseline_args()
    device, _ = setup_device(args.gpu_id)
    
    print("=" * 60)
    print("Baseline Learning Rate Schedulers Comparison")
    print("=" * 60)
    print(f"Tasks: {args.tasks}")
    print(f"Baselines: {args.baselines}")
    print(f"Runs per configuration: {args.num_runs}")
    print("=" * 60)
    
    trainer = BaselineTrainer(args, device)
    
    all_results = {}
    
    for baseline in args.baselines:
        print(f"\n{'=' * 60}")
        print(f"Running baseline: {baseline.upper()}")
        print(f"{'=' * 60}")
        
        all_results[baseline] = {}
        
        for task in args.tasks:
            print(f"\nTask: {task}")
            task_results = []
            
            for run in range(args.num_runs):
                print(f"  Run {run+1}/{args.num_runs}")
                set_seed(args.seed + run)
                
                result = trainer.train(task, baseline)
                task_results.append(result)
                print(f"    Best: {result['best_accuracy']:.2f}%")
            
            # 计算统计量
            best_accs = [r['best_accuracy'] for r in task_results]
            all_results[baseline][task] = {
                'mean': np.mean(best_accs),
                'std': np.std(best_accs),
                'max': np.max(best_accs),
                'min': np.min(best_accs),
                'all_runs': task_results
            }
    
    # 打印总结
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    header = f"{'Task':<12}"
    for baseline in args.baselines:
        header += f" | {baseline:^20}"
    print(header)
    print("-" * 80)
    
    for task in args.tasks:
        row = f"{task:<12}"
        for baseline in args.baselines:
            r = all_results[baseline][task]
            row += f" | {r['mean']:>6.2f}% ± {r['std']:>5.2f}%"
        print(row)
    
    # 总体平均
    print("-" * 80)
    row = f"{'Average':<12}"
    for baseline in args.baselines:
        means = [all_results[baseline][t]['mean'] for t in args.tasks]
        row += f" | {np.mean(means):>6.2f}% ± {np.std(means):>5.2f}%"
    print(row)
    print("=" * 80)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(args.output_dir, f'baseline_results_{timestamp}.json')
    
    # 简化结果用于保存
    save_results = {}
    for baseline in args.baselines:
        save_results[baseline] = {}
        for task in args.tasks:
            r = all_results[baseline][task]
            save_results[baseline][task] = {
                'mean': r['mean'],
                'std': r['std'],
                'max': r['max'],
                'min': r['min']
            }
    
    with open(result_file, 'w') as f:
        json.dump({
            'args': {k: str(v) for k, v in vars(args).items()},
            'results': save_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")


if __name__ == '__main__':
    main()
