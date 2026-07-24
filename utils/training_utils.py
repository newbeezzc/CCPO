# -*- coding: utf-8 -*-
"""
训练辅助工具函数
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 注意：下面这行会降低性能，但保证确定性
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_device(gpu_id: str) -> Tuple[torch.device, List[int]]:
    """
    设置计算设备
    
    Args:
        gpu_id: GPU ID字符串，如 "0" 或 "0,1,2"
    
    Returns:
        device: 主设备
        gpu_ids: GPU ID列表
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    if torch.cuda.is_available():
        gpu_ids = [int(i) for i in gpu_id.split(',')]
        device = torch.device('cuda:0')
        print(f"Using GPU(s): {gpu_id}")
    else:
        gpu_ids = []
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device, gpu_ids


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, filepath: str):
    """保存检查点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, map_location: str = 'cpu') -> dict:
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsLogger:
    """训练指标记录器"""
    
    def __init__(self, log_dir: str, exp_name: str):
        self.log_dir = log_dir
        self.exp_name = exp_name
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'{exp_name}_{timestamp}.json')
        
        # 存储所有指标
        self.metrics = {
            'config': {},
            'episodes': [],
            'evaluations': []
        }
        
    def log_config(self, config: dict):
        """记录配置"""
        self.metrics['config'] = config
        self._save()
        
    def log_episode(self, episode: int, metrics: dict):
        """记录单个episode的指标"""
        entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics['episodes'].append(entry)
        self._save()
        
    def log_evaluation(self, episode: int, eval_results: dict):
        """记录评估结果"""
        entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **eval_results
        }
        self.metrics['evaluations'].append(entry)
        self._save()
        
    def _save(self):
        """保存到文件"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算广义优势估计 (Generalized Advantage Estimation, GAE)
    
    Args:
        rewards: [T] 奖励序列
        values: [T] 状态价值序列
        dones: [T] 终止标志序列
        gamma: 折扣因子
        gae_lambda: GAE参数
    
    Returns:
        advantages: [T] 优势值
        returns: [T] 回报值
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    
    # 反向计算GAE
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0  # 假设episode结束后value为0
        else:
            next_value = values[t + 1]
        
        # TD误差
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE累积
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # 计算回报
    advantages = advantages.to(values.device)
    # print(advantages.device, values.device)
    returns = advantages + values
    
    return advantages, returns


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    """软更新目标网络"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    """硬更新目标网络"""
    target.load_state_dict(source.state_dict())


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: 允许的最大无改进epoch数
            min_delta: 最小改进阈值
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def print_training_info(episode: int, metrics: dict, prefix: str = ''):
    """格式化打印训练信息"""
    info_str = f"{prefix}Episode {episode:5d}"
    for key, value in metrics.items():
        if isinstance(value, float):
            info_str += f" | {key}: {value:.4f}"
        else:
            info_str += f" | {key}: {value}"
    print(info_str)
