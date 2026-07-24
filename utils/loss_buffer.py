# -*- coding: utf-8 -*-
"""
损失序列缓冲区：管理滑动窗口的损失序列
"""

import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional


class LossBuffer:
    """
    损失序列缓冲区，维护最近W步的多通道损失值
    
    用于构建RL状态中的损失曲线窗口 X_t ∈ R^{W×d}
    """
    
    def __init__(self, window_size: int, num_channels: int, device: str = 'cuda'):
        """
        Args:
            window_size: 窗口大小 W
            num_channels: 损失通道数 d
            device: 设备
        """
        self.window_size = window_size
        self.num_channels = num_channels
        self.device = device
        
        # 使用deque实现高效的滑动窗口
        self.buffer = deque(maxlen=window_size)
        
        # 用于归一化的统计量（running statistics）
        self.running_mean = torch.zeros(num_channels, device=device)
        self.running_var = torch.ones(num_channels, device=device)
        self.count = 0
        self.momentum = 0.01  # 指数移动平均的动量
        
    def reset(self):
        """重置缓冲区"""
        self.buffer.clear()
        self.running_mean = torch.zeros(self.num_channels, device=self.device)
        self.running_var = torch.ones(self.num_channels, device=self.device)
        self.count = 0
        
    def push(self, losses: torch.Tensor):
        """
        添加一步的损失值
        
        Args:
            losses: [d] 当前步的多通道损失值
        """
        if not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses, device=self.device, dtype=torch.float32)
        else:
            losses = losses.to(self.device).float()
            
        # 更新running statistics
        self.count += 1
        delta = losses - self.running_mean
        self.running_mean += self.momentum * delta
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (delta ** 2)
        
        self.buffer.append(losses)
        
    def is_ready(self) -> bool:
        """检查缓冲区是否已填满"""
        return len(self.buffer) >= self.window_size
    
    def get_window(self, normalize: bool = True) -> torch.Tensor:
        """
        获取当前的损失窗口
        
        Args:
            normalize: 是否进行归一化
            
        Returns:
            window: [W, d] 损失曲线窗口
        """
        if not self.is_ready():
            # 如果缓冲区未满，用零填充前面的部分
            padding_size = self.window_size - len(self.buffer)
            if len(self.buffer) > 0:
                first_loss = self.buffer[0]
                padding = [first_loss.clone() for _ in range(padding_size)]
                window_list = padding + list(self.buffer)
            else:
                window_list = [torch.zeros(self.num_channels, device=self.device) 
                              for _ in range(self.window_size)]
        else:
            window_list = list(self.buffer)
            
        window = torch.stack(window_list, dim=0)  # [W, d]
        
        if normalize:
            # 使用running statistics进行归一化
            window = (window - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
            
        return window
    
    def get_context_features(self) -> torch.Tensor:
        """
        计算序列统计特征（用于上下文融合层）
        
        Returns:
            context: [4 * d] 包含 mean, std, trend, last 四种统计量
        """
        window = self.get_window(normalize=False)  # [W, d]
        
        # 1. 均值
        mean = window.mean(dim=0)  # [d]
        
        # 2. 标准差
        std = window.std(dim=0)  # [d]
        
        # 3. 趋势（线性回归斜率的近似：最后一半均值 - 前一半均值）
        half = self.window_size // 2
        trend = window[half:].mean(dim=0) - window[:half].mean(dim=0)  # [d]
        
        # 4. 最后一个值
        last = window[-1]  # [d]
        
        # 拼接
        context = torch.cat([mean, std, trend, last], dim=0)  # [4*d]
        
        return context
    
    def get_state_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取状态的损失相关组件
        
        Returns:
            window: [W, d] 归一化后的损失窗口
            context: [4*d] 上下文特征
        """
        window = self.get_window(normalize=True)
        context = self.get_context_features()
        return window, context


class TrajectoryBuffer:
    """
    轨迹缓冲区，存储一个episode的所有transitions

    支持离散动作空间：存储动作索引和对应的学习率值
    """

    def __init__(self):
        self.states = []
        self.action_indices = []  # 离散动作索引
        self.action_lrs = []  # 对应的学习率值
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def push(self, state: dict, action_idx: int, action_lr: float,
             log_prob: torch.Tensor, reward: float, done: bool):
        """
        添加一个transition

        Args:
            state: 状态字典
            action_idx: 离散动作索引
            action_lr: 对应的学习率值
            log_prob: 动作对数概率
            reward: 奖励
            done: 是否结束
        """
        self.states.append(state)
        self.action_indices.append(action_idx)
        self.action_lrs.append(action_lr)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.action_indices.clear()
        self.action_lrs.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)

    def get_batch(self) -> dict:
        """
        获取批量数据用于训练

        Returns:
            batch: 包含所有数据的字典
        """
        return {
            'states': self.states,
            'action_indices': torch.tensor(self.action_indices, dtype=torch.long),  # 离散索引用long类型
            'action_lrs': torch.tensor(self.action_lrs, dtype=torch.float32),
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.float32)
        }
