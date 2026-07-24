# -*- coding: utf-8 -*-
"""
一阶段评价模型加载器

加载预训练的损失曲线评价模型 g_φ，用于为二阶段RL提供奖励信号
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import os
import sys

# 添加一阶段模型的路径（根据实际情况调整）
# sys.path.append('/path/to/stage1')


class EvaluatorModelConfig:
    """评价模型配置（需与一阶段保持一致）"""
    
    def __init__(self, 
                 seq_len: int = 10,           # 窗口大小 W
                 enc_in: int = 4,             # 损失通道数 d
                 embed_size: int = 128,       # 嵌入维度
                 hidden_size: int = 256,      # 隐藏层维度
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


class ContextFusionLayer(nn.Module):
    """
    上下文融合层：将时序特征 z 与学习率 η、训练进度 p、序列统计特征融合
    （从一阶段FreTS.py复制，保持一致）
    """

    def __init__(self, d_model: int, context_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # 上下文嵌入
        self.lr_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.progress_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # 序列统计特征嵌入
        self.context_embed = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, lr: torch.Tensor, 
                progress: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, D] 时序特征向量
            lr: [B] 学习率
            progress: [B] 训练进度
            context_features: [B, context_dim] 序列统计特征
        Returns:
            h: [B, D] 融合后的综合状态表示
        """
        # 对数变换学习率
        lr_log = torch.log(lr + 1e-8).unsqueeze(-1)  # [B, 1]
        progress = progress.unsqueeze(-1)  # [B, 1]

        # 嵌入上下文
        lr_emb = self.lr_embed(lr_log)  # [B, D]
        p_emb = self.progress_embed(progress)  # [B, D]
        ctx_emb = self.context_embed(context_features)  # [B, D]

        # 交叉注意力 + 残差
        # z_seq = z.unsqueeze(1)  # [B, 1, D]
        # context = torch.stack([lr_emb, p_emb, ctx_emb], dim=1)  # [B, 3, D] 改为3个token
        # h, _ = self.cross_attn(z_seq, context, context)  # [B, 1, D]
        # h = self.norm1(h + z_seq)

        # 相加
        h = z + lr_emb + p_emb + ctx_emb  # [B, D]
        h = self.norm1(h)

        # FFN + 残差
        h = self.norm2(h + self.ffn(h))

        return h  # .squeeze(1)  # [B, D]


class EvaluatorModel(nn.Module):
    """
    损失曲线评价模型 g_φ: (X, η, p, ctx) → ŷ
    （与一阶段FreTS.py中的Model类保持一致）
    """

    def __init__(self, configs: EvaluatorModelConfig):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name in ['classification', 'anomaly_detection', 'imputation']:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.feature_size = configs.enc_in
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        # 上下文特征维度 = 4 * num_features
        self.context_dim = 4 * self.feature_size

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

        # ============ (2) 上下文融合层 ============
        self.context_fusion = ContextFusionLayer(
            d_model=self.embed_size,
            context_dim=self.context_dim,
            n_heads=4,
            dropout=0.1
        )

        # ============ (3) 评价预测头 ============
        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def tokenEmb(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    def MLP_temporal(self, x: torch.Tensor, B: int, N: int, L: int) -> torch.Tensor:
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    def MLP_channel(self, x: torch.Tensor, B: int, N: int, L: int) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        return x

    def FreMLP(self, B: int, nd: int, dimension: int, x: torch.Tensor,
               r: torch.Tensor, i: torch.Tensor, 
               rb: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) -
            torch.einsum('bijd,dd->bijd', x.imag, i) + rb
        )
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) +
            torch.einsum('bijd,dd->bijd', x.real, i) + ib
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def encode(self, x_enc: torch.Tensor) -> torch.Tensor:
        """时序编码器：将损失曲线编码为特征向量"""
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)
        bias = x
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        z = self.temporal_proj(x.reshape(B, -1))
        return z

    def forward(self, x_enc: torch.Tensor, progress: torch.Tensor,
                lr: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_enc: [B, T, N] 损失曲线窗口
            progress: [B] 训练进度
            lr: [B] 学习率
            context_features: [B, context_dim] 序列统计特征
        
        Returns:
            y_hat: [B, 1] 预测的训练质量评分
        """
        # (1) 时序编码器
        z = self.encode(x_enc)

        # (2) 上下文融合层
        h = self.context_fusion(z, lr, progress, context_features)

        # (3) 评价预测头
        y_hat = self.prediction_head(h)

        return y_hat
    
    def get_encoder_output(self, x_enc: torch.Tensor) -> torch.Tensor:
        """仅获取时序编码器的输出，用于策略网络复用"""
        return self.encode(x_enc)


def load_evaluator(checkpoint_path: str, 
                   config: Optional[EvaluatorModelConfig] = None,
                   device: str = 'cuda',
                   freeze: bool = True) -> EvaluatorModel:
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
        config = EvaluatorModelConfig()
    
    model = EvaluatorModel(config)
    
    # 加载checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded evaluator checkpoint from {checkpoint_path}")
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


class EvaluatorWrapper:
    """
    评价模型的封装类，提供便捷的接口用于RL奖励计算
    """
    
    def __init__(self, model: EvaluatorModel, device: str = 'cuda'):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def evaluate_state(self, loss_window: torch.Tensor, lr: float, 
                       progress: float, context_features: torch.Tensor) -> float:
        """
        评估单个状态的质量
        
        Args:
            loss_window: [W, d] 损失曲线窗口
            lr: 当前学习率
            progress: 训练进度
            context_features: [4*d] 上下文特征
        
        Returns:
            score: 状态质量评分
        """
        # 添加batch维度
        x = loss_window.unsqueeze(0).to(self.device)  # [1, W, d]
        lr_t = torch.tensor([lr], device=self.device)
        progress_t = torch.tensor([progress], device=self.device)
        ctx = context_features.unsqueeze(0).to(self.device)  # [1, 4*d]
        
        # 前向传播
        score = self.model(x, progress_t, lr_t, ctx)
        
        return score.item()
    
    @torch.no_grad()
    def compute_reward(self, prev_state: dict, curr_state: dict,
                       acc_change: Optional[float] = None,
                       alpha: float = 1.0, beta: float = 0.0,
                       threshold: float = 0.02) -> float:
        """
        计算状态转移的奖励
        
        Args:
            prev_state: 前一状态 {'loss_window', 'lr', 'progress', 'context'}
            curr_state: 当前状态
            acc_change: 准确率变化（可选，用于周期性校准）
            alpha: 评价模型奖励权重
            beta: 准确率奖励权重
            threshold: 离散化阈值
        
        Returns:
            reward: 奖励值
        """
        # 计算评价模型给出的状态改善评分
        prev_score = self.evaluate_state(
            prev_state['loss_window'],
            prev_state['lr'],
            prev_state['progress'],
            prev_state['context']
        )
        curr_score = self.evaluate_state(
            curr_state['loss_window'],
            curr_state['lr'],
            curr_state['progress'],
            curr_state['context']
        )
        
        delta_g = curr_score - prev_score
        # delta_g = curr_score
        
        # 混合奖励
        reward = alpha * delta_g
        if acc_change is not None and beta > 0:
            reward += beta * acc_change

        # 阈值离散化逻辑
        if reward > threshold:
            reward = 1.0  # 显著改进
        elif reward < -threshold:
            reward = -1.0  # 显著恶化
        else:
            reward = 0.0  # 平台期或微小波动
            
        return reward
    
    def get_encoder(self) -> nn.Module:
        """获取时序编码器模块，用于策略网络复用"""
        return self.model
