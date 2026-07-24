# -*- coding: utf-8 -*-
"""
偏好学习 Agent（原 A2C → DPO）

用于学习率调度策略的偏好学习算法实现。
核心变化：将 A2C 的即时奖励信号替换为从同一 episode 内相邻干预步骤
构建偏好对，并使用带动态权重的 DPO 损失训练策略网络。
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
from collections import deque
from typing import Dict, List, Tuple, Optional

from torch import Tensor

from utils.util import _estimate_k_hat_online

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_network import PolicyConfig, load_policy
from utils.loss_buffer import TrajectoryBuffer
from utils.training_utils import compute_gae, save_checkpoint, load_checkpoint


# ==============================================================================
# [★ 修改点2] 新增 PreferenceBuffer：从轨迹中构建偏好对
# ==============================================================================

class PreferenceBuffer:
    """
    偏好对缓冲区

    收集 episode 内每个干预步的 (state, action_idx, reward, raw_delta_acc, progress)，
    并在 episode 结束后构建偏好对。

    支持的偏好信号（signal）:
      - 'return': 折现前瞻累计 Δacc（return-to-go），对单步评估噪声更鲁棒（推荐/默认）
      - 'smooth': 旧 'new' 方法——接下来最多 3 步 raw_Δacc 的均值
      - 'reward': 旧 'old' 方法——time-decayed reward 直接比较

    偏好对构造:
      - signal='return'/'smooth' 时按 progress 分桶，仅桶内比较（消除 "后期天然好" 混淆）
      - signal='reward' 时保留原滑动窗口相邻比较逻辑

    跨 episode 复用:
      - 内部用 deque(maxlen=keep_episodes) 保存最近若干 episode 的偏好对，
        使每次 DPO 更新拥有更大、更稳定的 batch。keep_episodes=1 时等价于原「每 episode 清空」行为。
    """

    def __init__(self, epsilon: float = 0.02, method: str = 'new', n_buckets: int = 5,
                 signal: str = 'return', horizon: int = 8, gamma: float = 0.9,
                 keep_episodes: int = 1):
        """
        Args:
            epsilon: 偏好阈值，|Δsignal| > epsilon 才构建偏好对
            method: 'old' | 'new'（向后兼容；signal 未显式指定时据此推断）
            n_buckets: progress 分桶数
            signal: 'return' | 'smooth' | 'reward'
            horizon: return-to-go 的前瞻步数 H
            gamma: return-to-go 折扣因子
            keep_episodes: 偏好对跨 episode 复用窗口（1=原行为）
        """
        self.epsilon = epsilon
        self.method = method
        self.n_buckets = n_buckets
        self.signal = signal
        self.horizon = int(horizon)  # <=0 表示折现到 episode 末尾（terminal）
        self.gamma = gamma

        # 当前 episode 的干预步记录
        self._steps: List[Dict] = []

        # 跨 episode 的偏好对（每个元素是一个 episode 产生的 pair list）
        self._episode_pairs: deque = deque(maxlen=max(1, int(keep_episodes)))

    def push_step(self, state: Dict, action_idx: int, log_prob, reward: float,
                  raw_delta_acc: float = 0.0, progress: float = 0.0):
        """记录一个干预步。"""
        self._steps.append({
            'state': state,
            'action_idx': action_idx,
            'log_prob': log_prob,
            'reward': reward,
            'raw_delta_acc': raw_delta_acc,
            'progress': progress,
        })

    # --------------------------------------------------------------------------
    # 偏好信号计算
    # --------------------------------------------------------------------------
    def _compute_signals(self) -> np.ndarray:
        """
        为每个干预步计算标量偏好信号 score[i]。

        - 'return': G_i = Σ_{h=0}^{H-1} γ^h · raw_Δacc_{i+h}（折现前瞻累计准确率增益）
        - 'smooth': 接下来最多 3 步 raw_Δacc 的均值（旧 new 方法）
        - 'reward': 本步 time-decayed reward（旧 old 方法）
        """
        steps = self._steps
        n = len(steps)
        scores = np.zeros(n, dtype=np.float32)

        if self.signal == 'reward':
            for i in range(n):
                scores[i] = float(steps[i]['reward'])
            return scores

        if self.signal == 'smooth':
            for i in range(n):
                window = [steps[j]['raw_delta_acc'] for j in range(i, min(i + 3, n))
                          if steps[j]['raw_delta_acc'] is not None]
                scores[i] = float(np.mean(window)) if window else 0.0
            return scores

        # 'return' / 'terminal': 折现前瞻累计 Δacc（return-to-go）
        # horizon<=0 → 折现到 episode 末尾（terminal，最贴合"最终准确率"目标，抗短视）
        raw = np.array([(s['raw_delta_acc'] if s['raw_delta_acc'] is not None else 0.0)
                        for s in steps], dtype=np.float32)
        H = self.horizon if self.horizon > 0 else n
        for i in range(n):
            g = 0.0
            disc = 1.0
            for h in range(H):
                if i + h >= n:
                    break
                g += disc * raw[i + h]
                disc *= self.gamma
            scores[i] = g
        return scores

    def build_pairs(self):
        """构建本 episode 的偏好对，并压入跨-episode 队列。"""
        if self.signal == 'reward':
            pairs = self._build_pairs_reward()
        else:
            pairs = self._build_pairs_bucketed()
        self._episode_pairs.append(pairs)

    def _build_pairs_reward(self) -> List[Dict]:
        """【旧 old 方法】滑动窗口内相邻比较 time-decayed reward。"""
        steps = self._steps
        pairs: List[Dict] = []
        for i in range(len(steps) - 1):
            s1 = steps[i]
            window_size = int(np.min([i + 20, len(steps) - 1]))
            for s2 in steps[i + 1:window_size]:
                diff = s2['reward'] - s1['reward']
                if abs(diff) > self.epsilon:
                    if diff > 0:
                        winner, loser = s2, s1
                    else:
                        winner, loser = s1, s2
                    if winner['reward'] < 0:
                        continue
                    pairs.append(self._make_pair(winner, loser, abs(diff)))
        return pairs

    def _build_pairs_bucketed(self) -> List[Dict]:
        """
        【改进方法】progress 分桶 + 标量信号（return/smooth）桶内两两比较。
        仅当 |score_w - score_l| > epsilon 才成对，margin 记录用于损失加权。
        """
        steps = self._steps
        n = len(steps)
        pairs: List[Dict] = []
        if n < 2:
            return pairs

        scores = self._compute_signals()

        # 按 progress 等距分桶
        prog_vals = np.array([s['progress'] for s in steps], dtype=np.float32)
        p_min, p_max = prog_vals.min(), prog_vals.max()
        if p_max - p_min < 1e-6:
            buckets = np.zeros(n, dtype=np.int32)
        else:
            bucket_width = (p_max - p_min) / self.n_buckets
            buckets = np.clip(((prog_vals - p_min) / bucket_width).astype(np.int32),
                              0, self.n_buckets - 1)

        for b in range(self.n_buckets):
            idxs = [i for i in range(n) if buckets[i] == b]
            if len(idxs) < 2:
                continue
            for ii in range(len(idxs)):
                for jj in range(ii + 1, len(idxs)):
                    i, j = idxs[ii], idxs[jj]
                    diff = float(scores[j] - scores[i])
                    if abs(diff) <= self.epsilon:
                        continue
                    if diff > 0:
                        winner, loser = steps[j], steps[i]
                    else:
                        winner, loser = steps[i], steps[j]
                    pairs.append(self._make_pair(winner, loser, abs(diff)))
        return pairs

    @staticmethod
    def _make_pair(winner: Dict, loser: Dict, margin: float) -> Dict:
        return {
            's_w': winner['state'],
            'a_w': winner['action_idx'],
            'lp_w': winner['log_prob'],
            's_l': loser['state'],
            'a_l': loser['action_idx'],
            'lp_l': loser['log_prob'],
            'reward_diff': margin,
        }

    def get_pairs(self) -> List[Dict]:
        """返回跨-episode 窗口内累计的所有偏好对（展平）。"""
        out: List[Dict] = []
        for ep_pairs in self._episode_pairs:
            out.extend(ep_pairs)
        return out

    def clear_episode(self):
        """清空本 episode 的步记录（保留已构建偏好对）。"""
        self._steps = []

    def clear_all(self):
        """清空所有数据。"""
        self._steps = []
        self._episode_pairs.clear()

    def __len__(self):
        return sum(len(p) for p in self._episode_pairs)


# ==============================================================================
# A2CAgent → DPOAgent
# ==============================================================================

class DPOAgent:
    """
    偏好学习 Agent（兼容原 A2CAgent 接口）

    主要变化：
    1. 新增参考策略 pi_ref（冻结副本，定期同步）
    2. 新增 PreferenceBuffer，替换原 TrajectoryBuffer 用于策略更新
    3. update() 使用 DPO 损失替换 A2C 损失
    """

    def __init__(self, args, device: str = 'cuda'):
        """
        Args:
            args: 配置参数
            device: 计算设备
        """
        self.args = args
        self.device = device
        self.max_grad_norm = args.max_grad_norm

        # [★ 修改点2/3] DPO 超参数
        self.dpo_beta = args.dpo_beta  # 温度 β
        self.pref_epsilon = args.pref_epsilon  # 偏好阈值 ε
        self.ref_update_interval = args.ref_update_interval  # 参考策略更新间隔（episodes）

        # 加载一阶段评价模型（用于编码器初始化，非奖励）
        policy_config = PolicyConfig(
            seq_len=args.loss_window,
            enc_in=args.loss_channels,
            embed_size=args.embed_dim,
            hidden_size=args.hidden_dim
        )
        self.policy = load_policy(
            args.evaluator_ckpt,
            config=policy_config,
            device=device,
            freeze=args.freeze_encoder
        )

        # [★ 修改点2] 创建参考策略 π_ref（初始化为当前策略的冻结副本）
        self.ref = copy.deepcopy(self.policy).to(device)
        self._freeze_ref_policy()
        print("Reference policy initialized (frozen copy of current policy)")

        self.optimizer = optim.Adam(params=self.policy.parameters(), lr=args.policy_lr)

        # 偏好对缓冲区（替代原 TrajectoryBuffer 用于 DPO 更新）
        # signal 未显式给出时，据旧 pref_method 推断：old→reward, new→return
        _method = getattr(args, 'pref_method', 'new')
        _signal = getattr(args, 'pref_signal', None)
        if _signal is None:
            _signal = 'reward' if _method == 'old' else 'return'
        self.preference_buffer = PreferenceBuffer(
            epsilon=self.pref_epsilon,
            method=_method,
            n_buckets=getattr(args, 'n_pref_buckets', 5),
            signal=_signal,
            horizon=getattr(args, 'pref_horizon', 8),
            gamma=getattr(args, 'pref_gamma', 0.9),
            keep_episodes=getattr(args, 'pref_buffer_episodes', 1),
        )

        # 改进版 DPO 更新超参
        self.n_dpo_epochs = int(getattr(args, 'n_dpo_epochs', 1))
        self.dpo_minibatch = int(getattr(args, 'dpo_minibatch', 0))
        self.pref_weighting = bool(getattr(args, 'pref_weighting', 0))

        # [★] 变异率策略参数
        self.mutation_strategy = getattr(args, 'mutation_strategy', 'decay')
        self.mutation_rate_end = getattr(args, 'mutation_rate_end', 0.05)
        self.mutation_rate_decay = getattr(args, 'mutation_rate_decay', 0.95)

        # 保留原轨迹缓冲区（仅用于收集 episode 内的步序列供 build_pairs 调用）
        self.trajectory_buffer = TrajectoryBuffer()

        # 训练统计
        self.total_steps = 0
        self.total_episodes = 0

    # --------------------------------------------------------------------------
    # 参考策略管理
    # --------------------------------------------------------------------------

    def _freeze_ref_policy(self):
        """冻结参考策略的所有参数"""
        for param in self.ref.parameters():
            param.requires_grad = False
        self.ref.eval()

    def update_reference_policy(self):
        """
        [★ 修改点2] 将当前策略 π_θ 同步到参考策略 π_ref。

        调用时机：在 A2CTrainer.train() 中每隔 ref_update_interval 个 episode 调用一次。
        """
        self.ref.load_state_dict(
            copy.deepcopy(self.policy.state_dict())
        )
        self._freeze_ref_policy()
        print(f"[Episode {self.total_episodes}] Reference policy updated.")

    # --------------------------------------------------------------------------
    # 动作选择（接口不变）
    # --------------------------------------------------------------------------

    def select_action(self, state: Dict, deterministic: bool = False, mutation_rate: Optional[float] = None
                      ) -> tuple[int | float | bool, float, Tensor]:
        """
        根据当前状态选择动作

        Args:
            state: 状态字典 {'loss_window', 'lr', 'progress', 'context'}
            deterministic: 是否使用确定性策略
            mutation_rate: 可选的突变率

        Returns:
            action_idx: 动作索引 (int)
            action_lr: 对应的学习率值 (float)
            log_prob: 对数概率
            value: 状态价值估计
        """
        with torch.no_grad():
            loss_window = state['loss_window'].unsqueeze(0).to(self.device)
            lr = torch.tensor([state['lr']], device=self.device)
            progress = torch.tensor([state['progress']], device=self.device)
            context = state['context'].unsqueeze(0).to(self.device)

            action_idx, action_value, log_prob = self.policy.sample_action(
                loss_window, progress, lr, context, deterministic, mutation_rate
            )
            action_value = action_value.item()  # * self.args.adjust_interval
            action_lr = (1 + action_value) * state['lr']  # 从动作值映射回学习率调整后的值

        return action_idx.item(), action_lr, log_prob.squeeze()

    # --------------------------------------------------------------------------
    # Transition 存储
    # --------------------------------------------------------------------------

    def store_transition(self, state: Dict, action_idx: int, action_lr: float,
                         log_prob: torch.Tensor, reward: float, done: bool,
                         extra_info: Optional[Dict] = None):
        """
        存储一个 transition。

        同时向 preference_buffer 推送当前干预步信息，
        用于后续构建偏好对。extra_info 中可携带 raw_delta_acc 等字段。
        """
        # 原 trajectory_buffer（可保留用于其他统计）
        self.trajectory_buffer.push(state, action_idx, action_lr, log_prob, reward, done)

        # 推送到偏好对缓冲区（携带 raw_delta_acc 和 progress 供新构造方法使用）
        raw_delta_acc = extra_info.get('raw_delta_acc', 0.0) if extra_info else 0.0
        progress = state.get('progress', 0.0)
        self.preference_buffer.push_step(state, action_idx, log_prob, reward,
                                         raw_delta_acc=raw_delta_acc,
                                         progress=progress)

        self.total_steps += 1

    def finalize_episode(self):
        """
        [★ 修改点2] episode 结束后调用：
        1. 构建本 episode 产生的偏好对
        2. 清空步记录（保留偏好对）
        """
        self.preference_buffer.build_pairs()
        self.preference_buffer.clear_episode()

    # --------------------------------------------------------------------------
    # [★ 修改点3] DPO 策略更新
    # --------------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        使用积累的偏好对按 DPO 损失更新策略网络。

        损失公式（标准 DPO，可选 margin 加权）：
            L(θ) = - E_{(w,l)} [ ω · log σ( β · (Δlogratio_w - Δlogratio_l) ) ]
        其中 Δlogratio = log π_θ(a|s) - log π_ref(a|s)，ω 为归一化后的 |margin| 权重。

        与原实现的关键差异：
        1. 【修复】evaluate_action 的入参顺序修正为 (x_enc, progress, lr, ctx, a)——
           原实现把 lr 和 progress 传反，导致训练时的输入编码与 select_action 不一致，
           preference_logit 恒≈0、DPO loss 卡在 ln2。
        2. 多轮（n_dpo_epochs）小批量（dpo_minibatch）迭代，显著增加有效梯度步数。
        3. π_ref 固定，其 log-prob 只计算一次并缓存，多轮复用。

        Returns:
            losses: 损失字典
        """
        pairs = self.preference_buffer.get_pairs()
        if len(pairs) == 0:
            print("Warning: No preference pairs to update on.")
            self.total_episodes += 1
            return {}

        B = len(pairs)

        # ---- 一次性堆叠所有偏好对的状态张量（放到 device） ----
        def _stack_states(state_list):
            return (
                torch.stack([s['loss_window'] for s in state_list]).to(self.device),   # [B, W, d]
                torch.tensor([s['lr'] for s in state_list], device=self.device),        # [B]
                torch.tensor([s['progress'] for s in state_list], device=self.device),  # [B]
                torch.stack([s['context'] for s in state_list]).to(self.device),        # [B, 4d]
            )

        lw_w, lr_w, prog_w, ctx_w = _stack_states([p['s_w'] for p in pairs])
        lw_l, lr_l, prog_l, ctx_l = _stack_states([p['s_l'] for p in pairs])
        a_w = torch.tensor([p['a_w'] for p in pairs], dtype=torch.long, device=self.device)  # [B]
        a_l = torch.tensor([p['a_l'] for p in pairs], dtype=torch.long, device=self.device)  # [B]
        reward_diffs = torch.tensor([p['reward_diff'] for p in pairs],
                                    dtype=torch.float32, device=self.device)  # [B]

        # margin 归一化权重（均值=1，保持整体损失尺度稳定）
        if self.pref_weighting:
            weights = reward_diffs / (reward_diffs.mean() + 1e-8)
            weights = weights.clamp(max=5.0)  # 防止极端 margin 主导梯度
        else:
            weights = torch.ones_like(reward_diffs)

        # ---- 【★ 修复入参顺序】参考策略 log-prob 只算一次并缓存 ----
        # evaluate_action(x_enc, progress, lr, context_features, action_idx)
        with torch.no_grad():
            log_p_ref_w = self.ref.evaluate_action(lw_w, prog_w, lr_w, ctx_w, a_w)  # [B]
            log_p_ref_l = self.ref.evaluate_action(lw_l, prog_l, lr_l, ctx_l, a_l)  # [B]

        mb = self.dpo_minibatch if self.dpo_minibatch and self.dpo_minibatch > 0 else B
        n_epochs = max(1, self.n_dpo_epochs)

        last_loss = 0.0
        last_pref_logit = 0.0
        last_lr_w = 0.0
        last_lr_l = 0.0
        n_grad_steps = 0

        for _ in range(n_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, mb):
                idx = perm[start:start + mb]

                # 当前策略 π_θ 的 log-prob（正确入参顺序）
                lp_theta_w = self.policy.evaluate_action(
                    lw_w[idx], prog_w[idx], lr_w[idx], ctx_w[idx], a_w[idx])
                lp_theta_l = self.policy.evaluate_action(
                    lw_l[idx], prog_l[idx], lr_l[idx], ctx_l[idx], a_l[idx])

                log_ratio_w = lp_theta_w - log_p_ref_w[idx]
                log_ratio_l = lp_theta_l - log_p_ref_l[idx]

                preference_logit = self.dpo_beta * (log_ratio_w - log_ratio_l)  # [mb]
                dpo_loss_per_pair = -F.logsigmoid(preference_logit) * weights[idx]
                dpo_loss = dpo_loss_per_pair.mean()

                self.optimizer.zero_grad()
                dpo_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                n_grad_steps += 1
                last_loss = dpo_loss.item()
                with torch.no_grad():
                    last_pref_logit = preference_logit.mean().item()
                    last_lr_w = log_ratio_w.mean().item()
                    last_lr_l = log_ratio_l.mean().item()

        self.total_episodes += 1
        # 注意：不再 clear_all()——偏好对由 deque(maxlen=pref_buffer_episodes) 自动淘汰，
        # 以支持跨 episode 复用。keep_episodes=1 时等价于原「每轮只用当前 episode」行为。
        self.trajectory_buffer.clear()

        return {
            'dpo_loss': last_loss,
            'total_loss': last_loss,
            'num_pairs': B,
            'n_grad_steps': n_grad_steps,
            'mean_reward_diff': reward_diffs.mean().item(),
            'mean_log_ratio_w': last_lr_w,
            'mean_log_ratio_l': last_lr_l,
            'mean_preference_logit': last_pref_logit,
        }

    # --------------------------------------------------------------------------
    # [P2] BC 暖启：用人工调度的 (state, action_idx) 监督策略
    # --------------------------------------------------------------------------

    def bc_update(self, states: List[Dict], target_idxs: List[int],
                  n_epochs: int = 8, minibatch: int = 256, bc_lr: float = 1e-3) -> Dict[str, float]:
        """
        行为克隆：对收集到的 (state, 人工调度对应的离散动作 index) 做交叉熵监督，
        让策略初始就 ≈ 人工调度（起点 = baseline 水平）。

        Args:
            states: 状态字典列表（每个含 loss_window/lr/progress/context）
            target_idxs: 每个状态对应的目标离散动作索引（人工调度映射到 ACTION_SET 的最近点）
            n_epochs: 监督训练轮数
            minibatch: minibatch 大小（<=0 全量）
            bc_lr: BC 阶段学习率（独立于 DPO 的 policy_lr）
        Returns:
            统计字典
        """
        B = len(states)
        if B == 0:
            print("Warning: BC has no data to train on.")
            return {}

        lw = torch.stack([s['loss_window'] for s in states]).to(self.device)      # [B, W, d]
        lr = torch.tensor([s['lr'] for s in states], device=self.device)          # [B]
        prog = torch.tensor([s['progress'] for s in states], device=self.device)  # [B]
        ctx = torch.stack([s['context'] for s in states]).to(self.device)         # [B, 4d]
        tgt = torch.tensor(target_idxs, dtype=torch.long, device=self.device)     # [B]

        bc_opt = optim.Adam(self.policy.parameters(), lr=bc_lr)
        mb = minibatch if minibatch and minibatch > 0 else B
        n_epochs = max(1, int(n_epochs))

        self.policy.train()
        last_loss, last_acc = 0.0, 0.0
        for ep in range(n_epochs):
            perm = torch.randperm(B, device=self.device)
            ep_loss, ep_correct = 0.0, 0
            n_mb = 0
            for start in range(0, B, mb):
                idx = perm[start:start + mb]
                # forward(x_enc, progress, lr, context_features) → logits[B, num_actions]
                logits, _, _, _ = self.policy(lw[idx], prog[idx], lr[idx], ctx[idx])
                loss = F.cross_entropy(logits, tgt[idx])

                bc_opt.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                bc_opt.step()

                ep_loss += loss.item()
                ep_correct += (logits.argmax(-1) == tgt[idx]).sum().item()
                n_mb += 1
            last_loss = ep_loss / max(1, n_mb)
            last_acc = ep_correct / B
            print(f"  [BC] epoch {ep + 1}/{n_epochs}  ce_loss={last_loss:.4f}  top1={last_acc * 100:.1f}%")

        return {'bc_ce_loss': last_loss, 'bc_top1': last_acc, 'bc_num_samples': B}

    # --------------------------------------------------------------------------
    # 保存 / 加载
    # --------------------------------------------------------------------------

    def save(self, filepath: str, extra_info: Optional[Dict] = None):
        state = {
            'policy': self.policy.state_dict(),
            'ref': self.ref.state_dict(),  # 保存 ref
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes
        }
        if extra_info:
            state.update(extra_info)
        save_checkpoint(state, filepath)

    def load(self, filepath: str):
        checkpoint = load_checkpoint(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        if 'ref' in checkpoint:
            self.ref.load_state_dict(checkpoint['ref'])
            self._freeze_ref_policy()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        print(f"Loaded agent from {filepath}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Total episodes: {self.total_episodes}")

    def train_mode(self):
        """设置为训练模式"""
        self.policy.train()
        # ref policy 始终保持 eval
        self.ref.eval()

    def eval_mode(self):
        """设置为评估模式"""
        self.policy.eval()


# ==============================================================================
# A2CTrainer — 训练循环
# ==============================================================================

class DPOTrainer:
    """
    偏好学习训练器（兼容原 A2CTrainer 接口）

    主要变化：
    1. collect_episode 结束后调用 agent.finalize_episode() 构建偏好对
    2. train() 中每隔 ref_update_interval 个 episode 更新参考策略
    3. 日志字段适配 DPO 损失输出
    """

    def __init__(self, agent: DPOAgent, train_env, test_env=None, args=None):
        self.agent = agent
        self.train_env = train_env
        self.test_env = test_env
        self.args = args

        self.best_reward = float('-inf')
        self.best_accuracy = 0.0

    def collect_episode(self, env, task_name: str = None, deterministic: bool = False) -> Dict:
        """
        收集一个完整的 episode。

        episode 结束后调用 agent.finalize_episode()，触发偏好对构建。
        支持两种变异率策略：'decay'（逐步衰减）和 'constant'（保持不变）。
        """
        if self.args.adapt_k:
            k_hat = _estimate_k_hat_online(env.get_acc_history(), self.args.reward_time_decay)
        else:
            k_hat = self.args.reward_time_decay
        state = env.reset(task_name, k_hat=k_hat)
        done = False

        episode_reward = 0.0
        episode_length = 0

        # ---- 变异率策略 ----
        mutation_strategy = self.agent.mutation_strategy
        if mutation_strategy == 'decay':
            mutation_rate = self.args.mutation_rate           # 起始值（如 0.3）
            mutation_rate_end = self.agent.mutation_rate_end   # 终点（如 0.05）
            mutation_rate_decay = self.agent.mutation_rate_decay  # 衰减因子（如 0.95）
        elif mutation_strategy == 'constant':
            mutation_rate = self.args.mutation_rate
            mutation_rate_end = self.args.mutation_rate
            mutation_rate_decay = 1.0

        while not done:
            # action_idx, action_lr, log_prob = self.agent.select_action(state, deterministic, mutation_rate)
            action_lr = 0.001
            next_state, reward, done, info = env.step(action_lr, deterministic)

            # if not deterministic:
            #     self.agent.store_transition(
            #         state, action_idx, action_lr, log_prob, reward, done,
            #         extra_info=info  # 携带 raw_delta_acc 等字段
            #     )

            episode_reward += reward
            episode_length += 1
            state = next_state
            if mutation_strategy == 'decay':
                mutation_rate = max(mutation_rate * mutation_rate_decay, mutation_rate_end)

        # episode 结束：构建偏好对
        if not deterministic:
            self.agent.finalize_episode()

        episode_info = env.get_episode_info()
        episode_info['episode_reward'] = episode_reward
        episode_info['episode_length'] = episode_length
        # 不再将偏好对存入 episode_info，避免 tensor 状态序列化导致 JSON 日志保存卡死

        print(f'Task {task_name} completed: Final Acc={episode_info.get("final_accuracy", 0):.2f}%')

        return episode_info

    # --------------------------------------------------------------------------
    # [P2] BC 暖启：采集人工调度 rollout 数据 → 监督策略 → 同步 π_ref
    # --------------------------------------------------------------------------

    def _collect_bc_rollout(self, task_name: str, schedule: str) -> Tuple[List[Dict], List[int]]:
        """
        在单个 train 任务上，用人工调度脚本 deterministic 跑一条完整 episode，
        逐干预步记录 (state, 目标离散动作 index)。

        目标动作 index 的映射：给定当前已实现 lr 与调度在该进度点期望的 lr，
        期望乘法因子 f = lr_target / lr_now，取 ACTION_SET 中最接近 (f-1) 的索引。
        用该离散动作实际驱动环境（保证状态落在策略可表达的流形上，且量化误差逐步自我纠偏）。
        """
        from eval_baselines import lr_from_schedule

        action_set = self.agent.policy.action_values.detach().cpu().numpy()  # [num_actions]
        init_lr = self.args.init_lr

        state = env_reset = self.train_env.reset(task_name)
        inner = self.train_env.env  # 底层 UDAEnv
        total = inner.total_steps

        states: List[Dict] = []
        target_idxs: List[int] = []
        done = False
        while not done:
            lr_now = float(state['lr'])
            p = inner.current_step / total
            lr_target = lr_from_schedule(schedule, p, init_lr)
            # 期望乘法因子 → action 值 (f-1) → 最近离散索引
            factor = lr_target / max(lr_now, 1e-12)
            desired_action_val = factor - 1.0
            idx = int(np.argmin(np.abs(action_set - desired_action_val)))
            realized_val = float(action_set[idx])
            realized_lr = (1.0 + realized_val) * lr_now

            # 记录 (当前 state, 目标动作 index)
            states.append(state)
            target_idxs.append(idx)

            next_state, _, done, _ = self.train_env.step(realized_lr, deterministic=True)
            state = next_state

        return states, target_idxs

    def bc_warmstart(self) -> Dict[str, float]:
        """
        P2 核心：BC 暖启 + 锚定。
        1. 遍历 train_tasks 采集 bc_rollouts 条人工调度 rollout。
        2. 交叉熵监督策略（agent.bc_update）。
        3. 若 bc_set_ref，则把暖启后的策略同步进 π_ref（DPO 的 KL 锚点变为人工调度）。
        """
        schedule = getattr(self.args, 'bc_schedule', 'inv')
        n_rollouts = int(getattr(self.args, 'bc_rollouts', len(self.train_env.task_list)))
        print("\n" + "=" * 60)
        print(f"[P2] BC Warmstart: imitating '{schedule}' schedule, {n_rollouts} rollouts")
        print("=" * 60)

        tasks = self.train_env.task_list
        all_states: List[Dict] = []
        all_targets: List[int] = []
        for i in range(n_rollouts):
            task = tasks[i % len(tasks)]
            print(f"  [BC] collecting rollout {i + 1}/{n_rollouts} on task {task} ...")
            s, t = self._collect_bc_rollout(task, schedule)
            all_states.extend(s)
            all_targets.extend(t)
        print(f"  [BC] collected {len(all_states)} (state, action) pairs")

        info = self.agent.bc_update(
            all_states, all_targets,
            n_epochs=int(getattr(self.args, 'bc_epochs', 8)),
            minibatch=int(getattr(self.args, 'bc_minibatch', 256)),
            bc_lr=float(getattr(self.args, 'bc_lr', 1e-3)),
        )

        if int(getattr(self.args, 'bc_set_ref', 1)) == 1:
            self.agent.update_reference_policy()
            print("[P2] Reference policy set to BC-warmstarted policy (KL anchor = hand-designed schedule).")

        print("=" * 60 + "\n")
        return info

    def train(self, num_episodes: int,
              eval_interval: int = 50,
              save_interval: int = 100,
              log_interval: int = 10) -> List[Dict]:
        """
        训练主循环。

        [★ 修改点2] 每隔 ref_update_interval 个 episode 更新参考策略。
        """
        training_history = []
        ref_update_interval = getattr(self.args, 'ref_update_interval', 20)

        for episode in range(1, num_episodes + 1):
            print(f"\n=== Starting Episode {episode}/{num_episodes} ===")
            self.agent.train_mode()

            # 收集 episode（内部调用 finalize_episode 构建偏好对）
            episode_info = self.collect_episode(self.train_env, deterministic=False)

            # 使用 DPO 损失更新策略
            update_info = self.agent.update()

            # [★ 修改点2] 定期更新参考策略
            if episode % ref_update_interval == 0:
                self.agent.update_reference_policy()

            # 合并信息
            episode_info.update(update_info)
            episode_info['episode'] = episode
            training_history.append(episode_info)

            if episode % log_interval == 0:
                self._print_log(episode, episode_info)

            # ★ 启用训练中评估：跟踪泛化性能，保存最佳模型
            if self.test_env is not None and episode % eval_interval == 0:
                eval_results = self.evaluate()
                episode_info['eval_results'] = eval_results

                mean_acc = np.mean([r['best_accuracy'] for r in eval_results])
                if mean_acc > self.best_accuracy:
                    self.best_accuracy = mean_acc
                    self._save_best_model(episode)

            if episode % save_interval == 0:
                self._save_checkpoint(episode)

        return training_history

    def evaluate(self, num_episodes_per_task: int = 1, env=None) -> List[Dict]:
        """在测试任务上评估策略。

        Args:
            num_episodes_per_task: 每个任务重复评估的 episode 数
            env: 评估用环境；默认 self.test_env（同分布 seed 任务）。
                 传入 final_eval_env 可在跨上下文任务（跨数据集/架构）上评估。
        """
        env = env if env is not None else self.test_env
        self.agent.eval_mode()
        eval_results = []

        for task in env.task_list:
            for _ in range(num_episodes_per_task):
                episode_info = self.collect_episode(
                    env,
                    task_name=task,
                    deterministic=True
                )
                episode_info['task'] = task
                eval_results.append(episode_info)

        self._print_eval_results(eval_results)
        return eval_results

    def _print_log(self, episode: int, info: Dict):
        """[★ 修改点3] 日志字段适配 DPO 损失输出"""
        log_str = f"Episode {episode:5d}"
        log_str += f" | Task: {info.get('task', 'N/A'):>8s}"
        log_str += f" | Reward: {info.get('episode_reward', 0):.4f}"
        log_str += f" | Acc: {info.get('final_accuracy', 0):.2f}%"
        log_str += f" | #Pairs: {info.get('num_pairs', 0)}"
        log_str += f" | DPO_Loss: {info.get('dpo_loss', 0):.4f}"
        log_str += f" | MeanDiff: {info.get('mean_reward_diff', 0):.4f}"
        print(log_str)

    def _print_eval_results(self, results: List[Dict]):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("-" * 60)

        task_results = {}
        for r in results:
            task = r['task']
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(r['best_accuracy'])

        # 列宽自适应（跨上下文任务名如 'cifar100_resnet18_s100' 较长）
        name_w = max([len(str(t)) for t in task_results] + [len('Average')])
        for task, accs in task_results.items():
            mean_acc = np.mean(accs)
            std_acc = np.std(accs) if len(accs) > 1 else 0
            print(f"  {task:>{name_w}s}: {mean_acc:.2f}% ± {std_acc:.2f}%")

        all_accs = [r['best_accuracy'] for r in results]
        print("-" * 60)
        print(f"  {'Average':>{name_w}s}: {np.mean(all_accs):.2f}% ± {np.std(all_accs):.2f}%")
        print("=" * 60 + "\n")

    def _save_checkpoint(self, episode: int):
        filepath = os.path.join(
            self.args.output_dir,
            f"checkpoint_episode_{episode}.pth"
        )
        self.agent.save(filepath, {'episode': episode})

    def _save_best_model(self, episode: int):
        filepath = os.path.join(self.args.output_dir, "best_model.pth")
        self.agent.save(filepath, {
            'episode': episode,
            'best_accuracy': self.best_accuracy
        })
        print(f"New best model saved! Accuracy: {self.best_accuracy:.2f}%")
