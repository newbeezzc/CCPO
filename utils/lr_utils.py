# -*- coding: utf-8 -*-
"""
学习率调整工具函数
"""

import torch
import numpy as np
from typing import Optional


def apply_lr_action(current_lr: float, action: float, scale: float = 0.5,
                    lr_min: float = 1e-6, lr_max: float = 0.1) -> float:
    """
    应用连续动作调整学习率
    
    采用乘法调整：lr' = lr × exp(action × scale)
    在对数空间上是均匀的：log(lr') = log(lr) + action × scale
    
    Args:
        current_lr: 当前学习率
        action: 连续动作值 ∈ [-1, 1]
        scale: 缩放系数
        lr_min: 学习率下界
        lr_max: 学习率上界
    
    Returns:
        new_lr: 调整后的学习率
    """
    # 乘法调整
    multiplier = np.exp(action * scale)
    new_lr = current_lr * multiplier
    
    # 裁剪到合法范围
    new_lr = np.clip(new_lr, lr_min, lr_max)
    
    return float(new_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    设置optimizer的学习率
    
    Args:
        optimizer: PyTorch优化器
        lr: 新的学习率
    """
    for param_group in optimizer.param_groups:
        if 'lr_mult' not in param_group:
            param_group['lr_mult'] = 1.
        param_group['lr'] = lr * param_group['lr_mult']


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    获取optimizer当前的学习率
    
    Args:
        optimizer: PyTorch优化器
    
    Returns:
        当前学习率
    """
    return optimizer.param_groups[0]['lr']

def update_warmup_lr(current_lr, iter_num) -> float:
    return current_lr * (1 + 0.001 * iter_num) ** (-0.75)


class ActionDiscretizer:
    """
    将连续动作离散化（用于对比实验）
    """
    
    # 离散动作定义（与文档一致）
    DISCRETE_ACTIONS = {
        0: 0.5,    # 大幅降低
        1: 0.9,    # 小幅降低
        2: 1.0,    # 保持不变
        3: 1.1,    # 小幅提升
        4: 2.0     # 大幅提升
    }
    
    @classmethod
    def continuous_to_discrete(cls, action: float) -> int:
        """
        将连续动作映射到最近的离散动作
        
        Args:
            action: 连续动作 ∈ [-1, 1]
        
        Returns:
            离散动作索引 ∈ {0, 1, 2, 3, 4}
        """
        # 计算对应的乘数
        multiplier = np.exp(action * 0.5)  # 假设scale=0.5
        
        # 找到最接近的离散动作
        min_dist = float('inf')
        best_action = 2  # 默认保持不变
        
        for idx, mult in cls.DISCRETE_ACTIONS.items():
            dist = abs(multiplier - mult)
            if dist < min_dist:
                min_dist = dist
                best_action = idx
                
        return best_action
    
    @classmethod
    def discrete_to_multiplier(cls, action: int) -> float:
        """
        将离散动作转换为学习率乘数
        
        Args:
            action: 离散动作索引
        
        Returns:
            学习率乘数
        """
        return cls.DISCRETE_ACTIONS.get(action, 1.0)


def compute_lr_stats(lr_history: list) -> dict:
    """
    计算学习率调整的统计信息
    
    Args:
        lr_history: 学习率历史列表
    
    Returns:
        统计信息字典
    """
    lr_array = np.array(lr_history)
    
    # 计算调整次数统计
    changes = np.diff(lr_array)
    increases = np.sum(changes > 0)
    decreases = np.sum(changes < 0)
    unchanged = np.sum(changes == 0)
    
    return {
        'initial_lr': lr_array[0],
        'final_lr': lr_array[-1],
        'min_lr': lr_array.min(),
        'max_lr': lr_array.max(),
        'mean_lr': lr_array.mean(),
        'std_lr': lr_array.std(),
        'num_increases': increases,
        'num_decreases': decreases,
        'num_unchanged': unchanged,
        'total_adjustments': len(lr_array) - 1
    }
