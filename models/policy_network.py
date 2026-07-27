# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional
import math

from .evaluator import EvaluatorModel, ContextFusionLayer


def make_action_set_log_symmetric(K=150, alpha=3.0):
    """
    在对数域对称、0附近密集的动作空间

    关键洞察：
      action ∈ [-0.5, 1.0] 对应乘法因子 (1+action) ∈ [0.5, 2.0]
      log(0.5) = -log(2),  log(2.0) = +log(2)
      → 在对数域天然对称于0

    策略：在对数域[-log2, +log2]均匀采样后做sinh变换增加中心密度，
          再映射回action空间
    """
    log_low = np.log(0.5)  # = -log(2) ≈ -0.693
    log_high = np.log(2.0)  # = +log(2) ≈ +0.693

    # 在[-1, 1]均匀采样，sinh变换后缩放到[log_low, log_high]
    u = np.linspace(-1, 1, K)
    log_factors = log_high * np.sinh(alpha * u) / np.sinh(alpha)

    # 映射回 action = (1+action的乘法因子) - 1 = exp(log_factor) - 1
    action_set = np.exp(log_factors) - 1

    return action_set  # 范围精确落在 [-0.5, 1.0]，关于乘法意义对称

class PolicyConfig:
    """策略模型配置（需与一阶段保持一致）"""

    def __init__(self,
                 seq_len: int = 10,  # 窗口大小 W
                 enc_in: int = 4,  # 损失通道数 d
                 embed_size: int = 128,  # 嵌入维度
                 hidden_size: int = 256,  # 隐藏层维度
                 channel_independence: str = '0',
                 task_name: str = 'classification',
                 pred_len: int = 1):
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channel_independence = channel_independence
        self.task_name = task_name
        self.pred_len = pred_len


class PolicyNetwork(nn.Module):
    """
    损失曲线评价模型 g_φ: (X, η, p, ctx) → ŷ
    【修改】输出从连续值改为离散动作概率，在预定义的调整幅度集合中选择
    """

    # ========== 预定义的离散动作集合 ==========
    # 学习率调整幅度候选集，单位为倍率
    # 可根据实际需求自行增删
    ACTION_SET = make_action_set_log_symmetric(K=100, alpha=6.0)

    # ======================================================

    def __init__(self, configs):
        super(PolicyNetwork, self).__init__()
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.embed_size = 128
        self.hidden_size = 256
        self.feature_size = configs.enc_in  # num_features (损失类型数)
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        # 新增：上下文特征维度 = 4 * num_features (mean, std, trend, last)
        self.context_dim = 4 * self.feature_size

        # ========== 新增：动作数量 & 注册动作集合为 buffer ==========
        self.num_actions = len(self.ACTION_SET)
        self.register_buffer(
            'action_values',
            torch.tensor(self.ACTION_SET, dtype=torch.float32)  # [num_actions]
        )
        # ====================================================================

        # ============ (1) 时序编码器: FreTS ============
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.temporal_proj = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size * self.feature_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.embed_size)
        )

        # ============ (2) 上下文融合层（修改） ============
        self.context_fusion = ContextFusionLayer(
            d_model=self.embed_size,
            context_dim=self.context_dim,  # 新增参数
            n_heads=4,
            dropout=0.1
        )

        # ============ (3) 评价预测头 ============
        # ========== [修改3] 输出维度从 1 改为 num_actions，输出各动作的 logits ==========
        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_actions),  # 【修改】1 → num_actions
            # 原来: nn.Linear(self.hidden_size, 1)
        )
        # ================================================================================

    def tokenEmb(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    def MLP_temporal(self, x, B, N, L):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    def MLP_channel(self, x, B, N, L):
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        return x

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + rb
        )
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + ib
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def encode(self, x_enc):
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)
        bias = x
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        z = self.temporal_proj(x.reshape(B, -1))
        return z

    def forward(self, x_enc, progress, lr, context_features):
        """
        Args:
            x_enc: [B, T, N] 损失曲线窗口
            progress: [B] 训练进度
            lr: [B，2] 学习率，（与初始学习率的比值 / 基线值）
            context_features: [B, context_dim] 序列统计特征（新增）

        Returns:
            logits:       [B, num_actions]  各动作的原始得分（用于训练，配合 CrossEntropyLoss）
            probs:        [B, num_actions]  各动作的概率分布（Softmax）
            action_idx:   [B]              概率最大的动作索引
            action_value: [B]              对应的离散调整幅度值（来自 ACTION_SET）

        【修改说明】
            原始 forward 直接返回 y_hat: [B, 1] 连续值。
            现在经过 prediction_head 得到 logits: [B, num_actions]，
            再用 Softmax 得到概率，argmax 得到离散动作，并映射回 ACTION_SET 中的具体数值。
        """
        progress = torch.zeros_like(progress)
        lr = torch.zeros_like(lr)
        context_features = torch.zeros_like(context_features)

        # (1) 时序编码器
        z = self.encode(x_enc)

        # (2) 上下文融合层（传入新特征）
        h = self.context_fusion(z, lr, progress, context_features)

        # (3) 评价预测头 → logits
        logits = self.prediction_head(h)  # [B, num_actions]

        # ========== 从 logits 得到概率、离散动作索引、动作值 ==========
        probs = F.softmax(logits, dim=-1)  # [B, num_actions]
        action_idx = torch.argmax(probs, dim=-1)  # [B]  贪心选择概率最大的动作
        action_value = self.action_values[action_idx]  # [B]  映射到 ACTION_SET 中的数值
        # ========================================================================
        return logits, probs, action_idx, action_value

    # ========== 新增：随机采样方法（用于强化学习探索 / 训练时采样） ==========
    def sample_action(self, x_enc, progress, lr, context_features, deterministic: bool = False, mutation_rate: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        按概率分布随机采样一个动作（用于 RL 训练中的探索）。
        mutation_rate: 均匀随机探索的概率（偏好学习阶段建议 0.05~0.15，随训练退火）
        #TODO：随机采样，而不是通过策略模型采样
        Returns:
            action_idx:   [B]  采样得到的动作索引
            action_value: [B]  对应的离散调整幅度值
            log_prob:     [B]  该动作的 log 概率（用于 policy gradient）
        """
        logits, probs, greedy_idx, greedy_value = self.forward(x_enc, progress, lr, context_features)
        bs = probs.shape[0]
        n_actions = probs.shape[-1]
        if deterministic:
            # 贪心选择概率最大的动作
            action_idx = greedy_idx  # [B]
        elif mutation_rate > 0:
            # ε-突变：每个样本独立决定是否随机探索
            use_random = torch.rand(bs, device=probs.device) < mutation_rate  # [B] bool
            random_idx = torch.randint(0, n_actions, (bs,), device=probs.device)
            policy_dist = torch.distributions.Categorical(probs=probs)
            policy_idx = policy_dist.sample()
            action_idx = torch.where(use_random, random_idx, policy_idx)  # [B]
        else:
            # 按概率分布随机采样动作
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = dist.sample()  # [B]

        log_prob = torch.log(probs[range(bs), action_idx] + 1e-8)
        action_value = self.action_values[action_idx]  # [B]
        return action_idx, action_value, log_prob

    def evaluate_action(self, x_enc, progress, lr, context_features, action_idx) -> torch.Tensor:
        """
        评估当前状态下各动作的得分（logits），用于训练时计算损失。

        Returns:
            log probs: [B, num_actions] 各动作的 log 概率
        """
        logits, probs, _, _ = self.forward(x_enc, progress, lr, context_features)
        log_probs = torch.log(probs[range(len(action_idx)), action_idx] + 1e-8)  # [B] 选择对应动作的 log 概率
        return log_probs


def load_policy(checkpoint_path: str,
                config: Optional[PolicyConfig] = None,
                device: str = 'cuda',
                freeze: bool = True) -> PolicyNetwork:
    """
    加载预训练的评价模型

    Args:
        checkpoint_path: checkpoint文件路径
        config: 模型配置，如果为None则使用默认配置
        device: 设备
        freeze: 是否冻结参数

    Returns:
        model: 加载好的评价模型
    """
    if config is None:
        config = PolicyConfig()

    model = PolicyNetwork(config)

    # 加载checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Loaded policy checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random initialization")

    model = model.to(device)

    # 冻结参数
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        print("Evaluator parameters frozen")

    return model


class ActorCritic:
    pass