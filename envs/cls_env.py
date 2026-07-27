# -*- coding: utf-8 -*-
"""
图像分类训练环境（替代 uda_env.py 的 UDA 内循环）

将标准的图像分类训练（CIFAR-10 等）封装为 Gym 风格的强化学习环境。
CCPO 策略网络通过观测训练动态（多通道损失曲线）调整学习率，以优化最终模型质量。

与 UDAEnv 的差异：
  - 内循环从 "对抗式域自适应" 换成 "标准交叉熵图像分类"
  - 去掉所有 UDA 逻辑（对抗损失 / MMD / 伪标签 / 混淆矩阵）
  - "任务" 由随机种子 + 训练数据子集划分定义（同一 CIFAR-10 的不同 seed/split），
    为元学习提供任务多样性
  - loss_window 为单通道：仅 raw batch 交叉熵损失
CCPO 的奖励（时间感知 Δacc）、DPO、偏好缓冲区等逻辑保持不变。

接口与 UDAEnv / MultiTaskUDAEnv 完全一致（reset / step / get_episode_info /
get_acc_history / .env / .total_steps / .current_step / task_list / num_tasks），
因此 agents / trainer / eval_baselines 无需改动即可复用。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss_buffer import LossBuffer
from utils.lr_utils import set_optimizer_lr
from utils.util import ContinuousDataloader
from utils.vision_utils import (
    build_datasets, get_model, in_channels_for, DATASET_META,
)
from configs.default_config import parse_task_spec


def parse_task_seed(task_name: str) -> int:
    """从任务名解析随机种子。

    支持: 's0' / 's12' / 'cifar10_resnet18_s3' → 末尾整数即 seed。
    无法解析时回退到对任务名做哈希。
    """
    if task_name is None:
        return 0
    tail = str(task_name).rsplit('_', 1)[-1]
    if tail.startswith('s') and tail[1:].isdigit():
        return int(tail[1:])
    if tail.isdigit():
        return int(tail)
    return abs(hash(task_name)) % (2 ** 31)


class ClsEnv:
    """
    图像分类训练环境

    状态空间:
        - loss_window: [W, d] 最近 W 步的多通道训练统计
        - lr: 当前学习率
        - progress: 训练进度 t / T_max
        - context: [4*d] 损失序列统计特征

    动作空间:
        - 离散动作索引 a ∈ {0, ..., K-1}，由策略映射为乘法学习率调整

    奖励:
        - 时间感知 Δacc（与 UDAEnv 一致）:
            r_t = Δacc_t × exp(k × (p_t - 1))
    """

    def __init__(self, args, flag: str = 'train', device: str = 'cuda',
                 epochs_by_dataset: Optional[Dict[str, int]] = None,
                 gpu_ids: Optional[List[int]] = None):
        self.args = args
        self.device = device
        self.flag = flag  # 'train' 或 'test'
        self.gpu_ids = gpu_ids or []  # 多卡 DataParallel 用
        # 逐数据集 epochs 覆盖（跨上下文最终评估用；未列出的数据集回退到 self.epochs）
        self.epochs_by_dataset = dict(epochs_by_dataset or {})

        # ---- 环境参数 ----
        self.window_size = args.loss_window          # W
        self.num_channels = args.loss_channels        # d（分类内循环：仅 raw CE loss）
        self.loss_norm = getattr(args, 'loss_norm', 'initial')  # 归一化模式
        self.adjust_interval = args.adjust_interval   # 每 N 步干预一次
        self.warmup_steps = args.warmup_steps
        self.lr_min = args.lr_min
        self.lr_max = args.lr_max
        self.reward_time_decay = args.reward_time_decay

        # ---- 分类任务超参 ----
        # 默认 dataset/arch 来自 args；实际使用的 dataset/arch 可由任务名逐任务覆盖
        # （跨上下文评估：famnist_cnn / cifar100_resnet18 等），见 reset()。
        self.default_dataset = getattr(args, 'dataset', 'cifar10')
        self.default_arch = getattr(args, 'arch', 'resnet18')
        self.dataset = self.default_dataset      # 当前激活的 dataset（reset 中更新）
        self.arch = self.default_arch            # 当前激活的 arch（reset 中更新）
        self.optimizer_name = getattr(args, 'optimizer', 'adamw')
        self.base_epochs = int(getattr(args, 'epochs', 0) or getattr(args, 'uda_epochs', 20))
        self.epochs = self.base_epochs   # 当前激活数据集的 epochs（reset 中按覆盖表更新）
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.weight_decay = args.weight_decay
        self.momentum = getattr(args, 'momentum', 0.9)
        self.init_lr = args.init_lr
        self.train_subset_ratio = float(getattr(args, 'train_subset_ratio', 0.9))
        self.train_val_ratio = float(getattr(args, 'train_val_ratio', 0.2))
        self.data_dir = getattr(args, 'data_dir', './data')

        # 元信息按默认 dataset 先填一份（廉价查表，不读盘），供 reset 前的自省使用
        meta = DATASET_META[self.dataset]
        self.num_classes = meta['num_classes']
        self.cifar_stem = meta['cifar_stem']
        self.in_channels = in_channels_for(self.dataset, self.arch)

        # ---- 底层数据集按 (dataset, arch) 缓存，惰性构建（首个用到的 reset 触发）----
        # 好处：train_env 只建 cifar10 一次并复用；final_eval_env 只建它实际用到的
        # (famnist/cifar100) 组合，不会浪费内存/时间去建用不到的 cifar10。
        self._ds_cache = {}   # (dataset, arch) -> (base_train_ds, base_test_ds)
        self.base_train_ds = None
        self.base_test_ds = None

        # ---- 运行时状态（reset 中初始化）----
        self.model = None
        self.optimizer = None
        self.train_iter = None
        self.val_loader = None
        self.loss_buffer = None

        self.total_steps = 0
        self.steps_per_epoch = 0
        self.current_step = 0
        self.current_lr = 0.0
        self.prev_state = None
        self.prev_acc = None

        self.lr_history = []
        self.loss_history = []
        self.acc_history = []
        self.reward_history = []
        self.task_name = None

    # ------------------------------------------------------------------
    # 数据 / 模型 / 优化器构建
    # ------------------------------------------------------------------
    def _build_loaders(self, seed: int):
        """按任务 seed 构建 train 子集 loader 与 val loader。"""
        rng = np.random.RandomState(seed)

        # 任务特定的训练子集（seed 决定 split → 任务多样性）
        n_train = len(self.base_train_ds)
        subset_size = max(self.batch_size, int(n_train * self.train_subset_ratio))
        subset_size = min(subset_size, n_train)
        train_idx = rng.choice(n_train, size=subset_size, replace=False)
        train_subset = Subset(self.base_train_ds, train_idx.tolist())

        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True)

        # 验证集：train flag 用 test 的子集（降低单步评估开销），test flag 用完整 test
        if self.flag == 'train':
            n_test = len(self.base_test_ds)
            vr = min(max(self.train_val_ratio, 0.01), 1.0)
            v_size = max(1, int(n_test * vr))
            v_idx = np.linspace(0, n_test - 1, v_size).astype(int)
            v_idx = sorted(set(v_idx.tolist()))
            val_dataset = Subset(self.base_test_ds, v_idx)
        else:
            val_dataset = self.base_test_ds

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)

        self.train_iter = ContinuousDataloader(train_loader)
        self.val_loader = val_loader
        self.steps_per_epoch = len(train_loader)

    def _activate_spec(self, dataset: str, arch: str):
        """切换当前激活的 (dataset, arch)：更新元信息并（惰性）准备底层数据集。

        跨上下文评估时同一个 env 会被不同任务反复 reset，可能涉及不同
        dataset/arch 组合；此处按 (dataset, arch) 缓存底层数据集，避免重复读盘。
        """
        self.dataset = dataset
        self.arch = arch
        meta = DATASET_META[dataset]
        self.num_classes = meta['num_classes']
        self.cifar_stem = meta['cifar_stem']
        self.in_channels = in_channels_for(dataset, arch)
        # 按数据集覆盖 epochs（跨上下文评估：famnist=20, cifar100=60 …），否则用基准值
        self.epochs = int(self.epochs_by_dataset.get(dataset, self.base_epochs))

        key = (dataset, arch)
        if key not in self._ds_cache:
            download = bool(getattr(self.args, 'download_data', False))
            tr, te, _ = build_datasets(dataset, self.data_dir, arch=arch, download=download)
            self._ds_cache[key] = (tr, te)
        self.base_train_ds, self.base_test_ds = self._ds_cache[key]

    def _create_model(self):
        model = get_model(self.arch, self.num_classes,
                          cifar_stem=self.cifar_stem,
                          in_channels=self.in_channels).to(self.device)
        # 多卡并行：DataParallel 透明包装，forward/backward/optimizer 自动分发
        if len(self.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.gpu_ids)
        return model

    def _create_optimizer(self, model, lr: float):
        if self.optimizer_name == 'adamw':
            return AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            return SGD(model.parameters(), lr=lr, momentum=self.momentum,
                       weight_decay=self.weight_decay, nesterov=True)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    # ------------------------------------------------------------------
    # reset / warmup
    # ------------------------------------------------------------------
    def reset(self, task_name: str, source_path: str = None,
              target_path: str = None, **kwargs) -> Dict:
        """重置环境，开始新 episode。

        Args:
            task_name: 任务名。可只编码 seed（'s0'，用 env 默认 dataset/arch），
                也可编码完整上下文 'dataset_arch_seed'（如 'cifar100_resnet18_s3'、
                'fashion_mnist_cnn_s100'）以支持跨数据集/架构评估。
            source_path/target_path: 兼容 UDAEnv 签名，分类任务忽略
        """
        self.task_name = task_name
        self.reward_time_decay = kwargs.get('k_hat', self.args.reward_time_decay)

        # 解析任务名 → (dataset, arch, seed)，并激活对应上下文（惰性构建数据集）
        dataset, arch, seed = parse_task_spec(
            task_name, self.default_dataset, self.default_arch)
        self._activate_spec(dataset, arch)

        print(f"Resetting ClsEnv for task {task_name} (dataset={self.dataset}, "
              f"arch={self.arch}, seed={seed})")

        # 数据 / 模型 / 优化器
        self._build_loaders(seed)
        self.total_steps = max(1, self.epochs * self.steps_per_epoch)

        torch.manual_seed(seed)
        self.model = self._create_model()
        self.current_lr = self.init_lr
        self.optimizer = self._create_optimizer(self.model, self.current_lr)
        set_optimizer_lr(self.optimizer, self.current_lr)

        # 损失缓冲区
        self.loss_buffer = LossBuffer(
            window_size=self.window_size,
            num_channels=self.num_channels,
            normalization=self.loss_norm,
            device='cpu',
        )

        # 计数器 / 历史
        self.current_step = 0
        self.prev_state = None
        self.prev_acc = None
        self.lr_history = [self.current_lr]
        self.loss_history = []
        self.acc_history = []
        self.reward_history = []

        # 预热
        self._warmup()

        return self._get_state()

    def _warmup(self):
        """预热阶段：固定初始 lr 跑若干步，收集初始损失序列。"""
        print(f"Warmup: running {self.warmup_steps} steps to fill loss window...")
        set_optimizer_lr(self.optimizer, self.init_lr)
        self.current_lr = self.init_lr
        for _ in range(self.warmup_steps):
            losses = self._train_one_step()
            self.loss_buffer.push(losses)
            self.current_step += 1
            self.loss_history.append(losses.cpu().numpy())

        # 'initial' 模式：用 warmup 窗口均值作为归一化分母
        if self.loss_norm == 'initial':
            self.loss_buffer.set_ref_loss()
            ref = self.loss_buffer.ref_loss.cpu().numpy()
            print(f"Warmup ref_loss: {ref}")

        self.prev_acc = self._evaluate()
        print(f"Warmup completed. Initial accuracy: {self.prev_acc:.2f}%")
        self.acc_history.append(self.prev_acc)

    # ------------------------------------------------------------------
    # 训练 / 评估
    # ------------------------------------------------------------------
    def _train_one_step(self) -> torch.Tensor:
        """执行一步分类训练，返回 [1] 单通道统计量（raw CE loss）。"""
        self.model.train()
        x, labels = next(self.train_iter)
        x = x.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        logits = self.model(x)
        loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return torch.tensor([float(loss.item())], device=self.device, dtype=torch.float32)

    def _train_n_steps(self, n: int) -> List[torch.Tensor]:
        losses_list = []
        for _ in range(n):
            if self.current_step >= self.total_steps:
                break
            losses = self._train_one_step()
            self.loss_buffer.push(losses)
            self.current_step += 1
            losses_list.append(losses)
            self.loss_history.append(losses.cpu().numpy())
        return losses_list

    @torch.no_grad()
    def _evaluate(self) -> float:
        """在 val_loader 上评估准确率 (0-100)。"""
        self.model.eval()
        correct, total = 0, 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            preds = self.model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / max(1, total)

    def _get_state(self) -> Dict:
        loss_window, context_features = self.loss_buffer.get_state_components()
        progress = self.current_step / self.total_steps
        return {
            'loss_window': loss_window,   # [W, d]
            'lr': self.current_lr,
            'progress': progress,
            'context': context_features,  # [4*d]
        }

    # ------------------------------------------------------------------
    # step / reward
    # ------------------------------------------------------------------
    def step(self, action_lr: float, deterministic: bool) -> Tuple[Dict, float, bool, Dict]:
        """执行一步环境交互（接收策略输出的学习率值）。"""
        self.prev_state = self._get_state()

        new_lr = float(np.clip(action_lr, self.lr_min, self.lr_max))
        self.current_lr = new_lr
        set_optimizer_lr(self.optimizer, new_lr)
        self.lr_history.append(new_lr)

        if self.current_step % max(1, 5 * self.adjust_interval) == 0:
            print('\t Step {}: New LR {:.6f}'.format(self.current_step, new_lr))

        self._train_n_steps(self.adjust_interval)

        next_state = self._get_state()

        if not deterministic:
            reward, current_acc, raw_delta_acc = self._compute_reward(next_state)
            print("step {}, acc: {:.2f}%, reward: {:.4f}".format(
                self.current_step, current_acc, reward))
        else:
            # 评估阶段：仅在每个 epoch 边界评估一次，减少开销
            if self.steps_per_epoch > 0 and self.current_step % self.steps_per_epoch < self.adjust_interval:
                current_acc = self._evaluate()
            else:
                current_acc = self.prev_acc if self.prev_acc is not None else self._evaluate()
            reward = 0.0
            raw_delta_acc = 0.0

        self.acc_history.append(current_acc)
        self.reward_history.append(reward)

        done = self.current_step >= self.total_steps

        info = {
            'task': self.task_name,
            'step': self.current_step,
            'lr': self.current_lr,
            'new_lr': new_lr,
            'accuracy': current_acc,
            'prev_acc': self.prev_acc,
            'reward': reward,
            'raw_delta_acc': raw_delta_acc,
        }
        if done:
            info['final_accuracy'] = current_acc

        self.prev_acc = current_acc
        return next_state, reward, done, info

    def _compute_reward(self, curr_state: Dict) -> Tuple[float, float, float]:
        """时间感知奖励: r_t = Δacc_t × exp(k × (p_t - 1))（与 UDAEnv 一致）。"""
        current_acc = self._evaluate()
        delta_acc = current_acc - self.prev_acc
        progress = curr_state['progress']
        time_weight = np.exp(self.reward_time_decay * (progress - 1.0))
        reward = delta_acc * time_weight
        #return reward, current_acc, delta_acc
        return delta_acc, current_acc, delta_acc

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def get_final_accuracy(self) -> float:
        return self._evaluate()

    def get_best_accuracy(self) -> float:
        return max(self.acc_history) if self.acc_history else 0.0

    def get_episode_info(self) -> Dict:
        return {
            'task': self.task_name,
            'total_steps': self.current_step,
            'final_accuracy': self.get_final_accuracy(),
            'initial_lr': self.lr_history[0] if self.lr_history else None,
            'final_lr': self.lr_history[-1] if self.lr_history else None,
            'lr_history': self.lr_history.copy(),
            'acc_history': self.acc_history.copy(),
            'reward_history': self.reward_history.copy(),
            'best_accuracy': self.get_best_accuracy(),
            'num_adjustments': len(self.lr_history) - 1,
        }

    def get_acc_history(self) -> List[float]:
        return self.acc_history.copy()


class MultiTaskClsEnv:
    """多任务图像分类环境（管理多个 seed/split 任务，支持任务采样）。"""

    def __init__(self, args, task_list: List[str] = None, flag: str = 'train',
                 device: str = 'cuda', epochs_by_dataset: Optional[Dict[str, int]] = None,
                 gpu_ids: Optional[List[int]] = None):
        self.args = args
        self.task_list = task_list or []
        self.device = device
        self.env = ClsEnv(args, flag, device, epochs_by_dataset=epochs_by_dataset,
                          gpu_ids=gpu_ids)
        self.current_task = None

    def sample_task(self) -> str:
        return np.random.choice(self.task_list)

    def reset(self, task_name: Optional[str] = None, **kwargs) -> Dict:
        if task_name is None:
            task_name = self.sample_task()
        self.current_task = task_name
        return self.env.reset(task_name, **kwargs)

    def step(self, action_lr: float, deterministic: bool) -> Tuple[Dict, float, bool, Dict]:
        return self.env.step(action_lr, deterministic)

    def get_episode_info(self) -> Dict:
        return self.env.get_episode_info()

    def get_acc_history(self) -> List[float]:
        return self.env.get_acc_history()

    @property
    def num_tasks(self) -> int:
        return len(self.task_list)
