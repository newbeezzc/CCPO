# -*- coding: utf-8 -*-

"""
CCPO 学习率调度器 (CCPO Schedule)

加载 CCPO 元训练的策略网络（PolicyNetwork），封装为标准 PyTorch _LRScheduler，
供模型再训练时调用，根据训练动态（loss / accuracy / grad_norm 窗口）自适应调整学习率。

与 UBA/vision/schedulers_official.py 中的 UBA / Rex 等公式驱动调度器不同，
CCPOSchedule 是数据驱动的：它观测每个 batch 的训练统计量，用预训练的神经网络
决策下一步的学习率调整幅度。

参考：
  - CCPO 策略网络: models/policy_network.py  (PolicyNetwork, PolicyConfig)
  - 损失缓冲区:     utils/loss_buffer.py      (LossBuffer)
  - UBA 调度器参考: D:/study/博士/时序×元学习/code/UBA/vision/schedulers_official.py
"""
import os
import sys
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional

# Ensure CCPO modules are importable from this file's directory
_CCPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _CCPO_ROOT not in sys.path:
    sys.path.insert(0, _CCPO_ROOT)

from models.policy_network import PolicyNetwork, PolicyConfig
from utils.loss_buffer import LossBuffer


class CCPOSchedule(_LRScheduler):
    """
    CCPO 策略驱动的学习率调度器。

    继承 _LRScheduler，遵循 PyTorch 标准调度器接口，同时新增 update_stats()
    方法供训练循环在每 batch 后喂入 raw loss。

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        被包装的优化器（所有 param_group 使用相同的 LR）。
    checkpoint_path : str
        CCPO 训练好的策略 checkpoint 路径（best_model.pth 或 final_model.pth）。
        兼容 DPOAgent.save() 格式和纯 state_dict 格式。
    warmup_steps : int
        预热步数。在此期间 LR 固定为 init_lr，收集初始损失序列填充观测窗口。
    adjust_interval : int
        策略干预间隔（每 N 步查询一次策略网络）。
    total_steps : int
        训练总步数（epochs * steps_per_epoch），用于计算 progress = step / total_steps。
    init_lr : float
        初始学习率（同时也是预热阶段的学习率）。
    lr_min : float
        学习率下界（clip 保护）。
    lr_max : float
        学习率上界（clip 保护）。
    loss_window : int
        损失窗口大小 W（需与 CCPO 训练时的 --loss_window 一致，默认 100）。
    loss_channels : int
        损失通道数 d（默认 1：仅 raw CE loss）。
    loss_norm : str
        loss 归一化模式: 'window' 窗口内z-score | 'initial' 除以warmup结束时的loss（默认）。
    encoder_type : str
        时序编码器类型: 'frets' 频域MLP | 'gru' 2层GRU（需与训练时一致，默认 'frets'）。
    embed_dim : int
        策略网络嵌入维度（需与 CCPO 训练时一致，默认 128）。
    hidden_dim : int
        策略网络隐藏层维度（需与 CCPO 训练时一致，默认 256）。
    device : str
        策略网络运行设备（cuda / cpu）。
    last_epoch : int
        _LRScheduler 参数，默认 -1（从第 0 步开始）。
    verbose : bool
        是否打印 LR 变化日志。
    """

    def __init__(
        self,
        optimizer,
        checkpoint_path: str,
        warmup_steps: int = 100,
        adjust_interval: int = 100,
        total_steps: int = 7000,
        init_lr: float = 1e-3,
        lr_min: float = 1e-5,
        lr_max: float = 1e-2,
        loss_window: int = 100,
        loss_channels: int = 1,
        loss_norm: str = "initial",
        encoder_type: str = "frets",
        embed_dim: int = 128,
        hidden_dim: int = 256,
        device: str = "cuda",
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        # ---- save user parameters ----
        self.warmup_steps = max(1, int(warmup_steps))
        self.adjust_interval = max(1, int(adjust_interval))
        self.total_steps = max(1, int(total_steps))
        self.init_lr = float(init_lr)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.loss_window = int(loss_window)
        self.loss_channels = int(loss_channels)
        self.loss_norm = str(loss_norm)
        self.encoder_type = str(encoder_type)
        self.device = device
        self._verbose = verbose

        # ---- load CCPO policy network (frozen) ----
        self._load_policy(checkpoint_path, embed_dim, hidden_dim)

        # ---- internal loss buffer (single-channel: raw CE loss) ----
        self.loss_buffer = LossBuffer(
            window_size=self.loss_window,
            num_channels=self.loss_channels,
            normalization=self.loss_norm,
            device="cpu",
        )
        self._warmup_ref_set = False  # 是否已在 warmup 结束后设定了 ref_loss

        # ---- internal state ----
        self._current_lr: float = self.init_lr

        # ---- init parent (compatible with PyTorch 1.x / 2.x) ----
        try:
            super().__init__(optimizer, last_epoch, verbose)
        except TypeError:
            super().__init__(optimizer, last_epoch)

        print(f"[CCPOSchedule] Initialized:")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  warmup={self.warmup_steps}, adjust={self.adjust_interval}, "
              f"total={self.total_steps}")
        print(f"  init_lr={self.init_lr:.2e}, lr_range=[{self.lr_min:.2e}, "
              f"{self.lr_max:.2e}]")
        print(f"  loss_norm={self.loss_norm}, loss_channels={self.loss_channels}, "
              f"loss_window={self.loss_window}")
        print(f"  encoder_type={self.encoder_type}")
        print(f"  device={self.device}  verbose={self._verbose}")

    # ==================================================================
    # Policy loading
    # ==================================================================

    def _load_policy(self, checkpoint_path: str, embed_dim: int, hidden_dim: int):
        """Load CCPO-trained policy network (freeze + eval mode)."""
        config = PolicyConfig(
            seq_len=self.loss_window,
            enc_in=self.loss_channels,
            embed_size=embed_dim,
            hidden_size=hidden_dim,
            encoder_type=self.encoder_type,
        )
        self.policy = PolicyNetwork(config).to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"CCPO checkpoint not found: {checkpoint_path}\n"
                f"Please check the path to a trained CCPO model "
                f"(e.g. outputs/full_run/best_model.pth)."
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Compatible with two checkpoint formats:
        #   (a) DPOAgent.save() -> {"policy": state_dict, "ref": ..., "optimizer": ...}
        #   (b) torch.save(model.state_dict(), ...) -> raw state_dict
        if isinstance(ckpt, dict) and "policy" in ckpt:
            state_dict = ckpt["policy"]
            extra_keys = [k for k in ckpt if k != "policy"]
            print(f"[CCPOSchedule] Loaded from DPOAgent checkpoint"
                  f"{' (also has: ' + ', '.join(extra_keys) + ')' if extra_keys else ''}")
        else:
            state_dict = ckpt
            print(f"[CCPOSchedule] Loaded from raw state_dict checkpoint")

        missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[CCPOSchedule] Warning: missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"[CCPOSchedule] Warning: unexpected keys in checkpoint: {unexpected}")

        # freeze + eval
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"[CCPOSchedule] Policy loaded & frozen ({n_params:,} parameters)")

    # ==================================================================
    # Core API: update_stats() -- feed training stats per batch
    # ==================================================================

    def update_stats(self, raw_loss: float):
        """
        Call after every batch to push training statistics into the internal LossBuffer.

        **Must be called BEFORE scheduler.step()** so the policy observes the latest
        training dynamics when making its next decision.

        Parameters
        ----------
        raw_loss : float
            Raw CE loss of the current batch.
        """
        self.loss_buffer.push(
            torch.tensor([float(raw_loss)], dtype=torch.float32))

    # ==================================================================
    # _LRScheduler entry: get_lr()
    # ==================================================================

    def get_lr(self):
        """Compute the LR for each param_group at the current step.

        Warmup phase (last_epoch < warmup_steps): return fixed init_lr.
        Scheduling phase: query policy every adjust_interval steps; reuse cached LR otherwise.
        """
        t = self.last_epoch

        # ---- warmup: constant init_lr, no policy query ----
        if t < self.warmup_steps:
            return [self._current_lr for _ in self.base_lrs]

        # ---- warmup → scheduling transition: auto-set ref_loss for 'initial' mode ----
        if self.loss_norm == 'initial' and not self._warmup_ref_set:
            self._warmup_ref_set = True
            if self.loss_buffer.is_ready():
                self.loss_buffer.set_ref_loss()
                ref = self.loss_buffer.ref_loss.item()
                print(f"[CCPOSchedule] Warmup complete, ref_loss = {ref:.4f}")

        # ---- scheduling: check if this step triggers a policy query ----
        effective_step = t - self.warmup_steps
        if effective_step % self.adjust_interval == 0:
            self._query_policy()

        return [self._current_lr for _ in self.base_lrs]

    # ==================================================================
    # Policy query (internal)
    # ==================================================================

    @torch.no_grad()
    def _query_policy(self):
        """Query the policy network to determine the LR for the next segment.

        Policy inputs:
          - loss_window [W, d]: normalized loss curve window
          - lr: current learning rate (log-encoded)
          - progress: step / total_steps
          - context [4d]: sequence statistics (mean, std, trend, last)

        Policy output:
          - action_value: multiplicative adjustment offset (1+action_value in [0.5, 2.0])

        LR update:
          new_lr = clip(current_lr * (1 + action_value), lr_min, lr_max)
        """
        if not self.loss_buffer.is_ready():
            if self._verbose:
                print(f"[CCPOSchedule] step={self.last_epoch:5d}  "
                      f"loss buffer not ready (need {self.loss_window}, "
                      f"have {len(self.loss_buffer.buffer)}), skipping query")
            return

        window, context = self.loss_buffer.get_state_components()
        progress = min(self.last_epoch / self.total_steps, 1.0)

        x = window.unsqueeze(0).to(self.device)
        lr_t = torch.tensor([self._current_lr], device=self.device)
        prog_t = torch.tensor([progress], device=self.device)
        ctx = context.unsqueeze(0).to(self.device)

        # policy forward -> greedy action (deterministic)
        _, probs, action_idx, action_value = self.policy(x, prog_t, lr_t, ctx)

        # multiplicative adjustment
        factor = 1.0 + action_value.item()
        old_lr = self._current_lr
        self._current_lr = float(np.clip(old_lr * factor, self.lr_min, self.lr_max))

        if self._verbose:
            a_idx = action_idx.item()
            a_val = action_value.item()
            top5_vals, top5_idxs = torch.topk(probs[0], min(5, len(probs[0])))
            top5_str = "  ".join(
                f"a{i:3d}({self.policy.action_values[i].item():+.4f})={p:.3f}"
                for i, p in zip(top5_idxs.tolist(), top5_vals.tolist())
            )
            print(f"[CCPOSchedule] step={self.last_epoch:5d}  "
                  f"p={progress:.3f}  "
                  f"chosen: a{a_idx:3d}({a_val:+.4f})  "
                  f"lr: {old_lr:.6f} -> {self._current_lr:.6f}  "
                  f"top5: {top5_str}")

    # ==================================================================
    # Utility methods
    # ==================================================================

    def reset(self):
        """Reset internal state (LossBuffer / LR) for a new training run.

        Note: does NOT reload policy weights; only clears runtime statistics.
        """
        self.loss_buffer.reset()
        self._warmup_ref_set = False
        self._current_lr = self.init_lr
        if self._verbose:
            print("[CCPOSchedule] State reset (buffer/LR cleared)")

    def current_lr(self) -> float:
        """Return the currently active learning rate."""
        return self._current_lr

    @property
    def action_set(self):
        """Return the policy network's discrete action set (multiplicative offset values).

        The actual multiplier for action_values[i] is 1 + action_values[i], in [0.5, 2.0].
        """
        return self.policy.action_values.detach().cpu().numpy()

    # Serialization not implemented (would need to serialize LossBuffer/EMA).
    # Following schedulers_official.py convention: raise NotImplementedError.
    def _get_closed_form_lr(self):
        raise NotImplementedError(
            "CCPOSchedule is data-driven; no closed-form LR formula exists.")

    def state_dict(self):
        raise NotImplementedError(
            "CCPOSchedule.state_dict() is not implemented. "
            "To resume training, re-create the scheduler and re-run warmup.")

    def load_state_dict(self):
        raise NotImplementedError(
            "CCPOSchedule.load_state_dict() is not implemented. "
            "To resume training, re-create the scheduler.")


def create_ccpo_schedule(
    optimizer,
    checkpoint_path: str,
    total_steps: int,
    warmup_steps: int = 100,
    adjust_interval: int = 100,
    init_lr: float = 1e-3,
    lr_min: float = 1e-5,
    lr_max: float = 1e-2,
    loss_window: int = 100,
    loss_channels: int = 1,
    loss_norm: str = "initial",
    encoder_type: str = "frets",
    embed_dim: int = 128,
    hidden_dim: int = 256,
    device: str = "cuda",
    verbose: bool = False,
) -> CCPOSchedule:
    """
    便捷工厂函数：一步创建 CCPOSchedule。

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        PyTorch 优化器实例。
    checkpoint_path : str
        CCPO 训练好的策略 checkpoint 路径。
    total_steps : int
        总训练步数。
    warmup_steps : int
        预热步数（收集初始损失序列）。
    adjust_interval : int
        策略干预间隔（每 N 步查询一次策略）。
    init_lr : float
        初始学习率。
    lr_min, lr_max : float
        学习率裁剪范围。
    loss_window : int
        损失窗口大小 W。
    loss_channels : int
        损失通道数 d（默认 1：仅 raw loss）。
    loss_norm : str
        归一化模式 'window' | 'initial'（默认 'initial'）。
    encoder_type : str
        编码器类型 'frets' | 'gru'（默认 'frets'）。
    embed_dim, hidden_dim : int
        策略网络架构参数。
    device : str
        cuda 或 cpu。
    verbose : bool
        是否打印 LR 变化日志。

    Returns
    -------
    CCPOSchedule
        配置好的调度器实例。
    """
    return CCPOSchedule(
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        warmup_steps=warmup_steps,
        adjust_interval=adjust_interval,
        total_steps=total_steps,
        init_lr=init_lr,
        lr_min=lr_min,
        lr_max=lr_max,
        loss_window=loss_window,
        loss_channels=loss_channels,
        loss_norm=loss_norm,
        encoder_type=encoder_type,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        device=device,
        verbose=verbose,
    )
