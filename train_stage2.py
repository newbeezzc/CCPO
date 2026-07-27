#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二阶段主训练脚本（偏好学习版本）

Meta-Schedule Stage 2: 基于偏好学习（DPO）的学习率调度策略训练

使用方法:
    python train_stage2.py --evaluator_ckpt /path/to/stage1/checkpoint.pth \
                           --data_root /path/to/OfficeHome \
                           --output_dir ./outputs/stage2

新增超参数（相比 A2C 版本）:
    --dpo_beta           DPO 温度系数 β（默认 0.1）
    --pref_epsilon       偏好对选取阈值 ε（默认 0.02）
    --ref_update_interval  参考策略更新间隔（默认 20 episodes）
    --reward_time_decay  时间折现系数 k（默认 3.0）
"""

import os
import sys
import time
import random
from copy import deepcopy

import numpy as np
import torch

from agents.dpo_agent import DPOAgent, DPOTrainer
from configs.default_config import parse_args, get_task_paths, parse_epochs_map
from envs.cls_env import ClsEnv, MultiTaskClsEnv
from envs.cls_env import MultiTaskClsEnv as MultiTaskUDAEnv  # 向后兼容别名
from agents.a2c_agent import A2CAgent, A2CTrainer
from utils.training_utils import (
    set_seed, setup_device, MetricsLogger,
    print_training_info, count_parameters
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # ==================== 解析参数 ====================
    args = parse_args()

    # ==================== 设置设备和随机种子 ====================
    device, gpu_ids = setup_device(args.gpu_id)
    set_seed(args.seed)

    # ==================== 创建输出目录 ====================
    os.makedirs(args.output_dir, exist_ok=True)

    # ==================== 初始化日志 ====================
    logger = MetricsLogger(args.output_dir, args.exp_name)
    logger.log_config(vars(args))

    print("=" * 60)
    print("Meta-Schedule Stage 2: Preference Learning (DPO) LR Scheduling")
    print("=" * 60)
    print(f"Device: {device}  (GPUs: {gpu_ids if gpu_ids else 'single'})")
    print(f"Output directory: {args.output_dir}")
    print(f"Training tasks: {args.train_tasks}")
    print(f"Test tasks (periodic eval / best-model): {args.test_tasks}")
    print(f"Final-eval tasks (cross-context): {args.final_eval_tasks}")
    print(f"DPO beta: {args.dpo_beta}")
    print(f"Preference epsilon: {args.pref_epsilon}")
    print(f"Ref update interval: {args.ref_update_interval} episodes")
    print(f"Reward time decay k: {args.reward_time_decay}")
    print("=" * 60)

    # ==================== 创建环境 ====================
    print("\nCreating classification environments...")

    train_env = MultiTaskUDAEnv(
        args=args,
        task_list=args.train_tasks,
        flag='train',
        device=str(device),
        gpu_ids=gpu_ids,
    )
    print(f"Training environment created with {train_env.num_tasks} tasks")

    test_args = deepcopy(args)
    # test_args.uda_epochs = 40
    # test_args.iters_per_epoch = 500
    test_env = MultiTaskUDAEnv(
        args=test_args,
        task_list=args.test_tasks,
        flag='test',
        device=str(device),
        gpu_ids=gpu_ids,
    )
    print(f"Test environment created with {test_env.num_tasks} tasks")

    # 跨上下文最终评估环境：任务名编码 dataset_arch_seed，按任务切换数据集/架构，
    # 用于评估在 CIFAR-10/ResNet18 上元训练得到的策略向未见数据集/架构的迁移能力。
    # 与元训练不同：(1) 每个数据集跑完整 epochs（--final_eval_epochs，如 famnist=20/cifar100=60）；
    #              (2) 默认用全量训练集（--final_eval_subset_ratio=1.0）得到可比 benchmark 数字。
    final_epochs_map = parse_epochs_map(args.final_eval_epochs)
    final_args = deepcopy(args)
    final_args.train_subset_ratio = float(args.final_eval_subset_ratio)
    final_eval_env = MultiTaskUDAEnv(
        args=final_args,
        task_list=args.final_eval_tasks,
        flag='test',
        device=str(device),
        epochs_by_dataset=final_epochs_map,
        gpu_ids=gpu_ids,
    )
    print(f"Final-eval environment created with {final_eval_env.num_tasks} "
          f"cross-context tasks: {args.final_eval_tasks}")
    print(f"  per-dataset epochs: {final_epochs_map} (others -> --epochs={args.epochs})")
    print(f"  train subset ratio: {final_args.train_subset_ratio}")

    # ==================== 创建 Agent（DPO 版本） ====================
    print("\nCreating Preference Learning Agent (DPO)...")
    agent = DPOAgent(args, device=str(device))

    total_params = count_parameters(agent.policy)
    print(f"DPO network parameters: {total_params:,}")

    # ==================== 创建训练器 ====================
    trainer = DPOTrainer(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        args=args
    )

    if args.is_training == 1:
        # ==================== [P2] BC 暖启（可选） ====================
        if int(getattr(args, 'bc_warmstart', 0)) == 1:
            trainer.bc_warmstart()
            # 暖启后先评估一次：应 ≈ 人工调度 baseline 分数（验证 BC 成功）
            print("\n[P2] Post-warmstart evaluation (before any DPO update):")
            warm_eval = trainer.evaluate(num_episodes_per_task=1)
            warm_acc = np.mean([r['best_accuracy'] for r in warm_eval])
            print(f"[P2] Warmstart eval mean best_acc = {warm_acc:.2f}%")
            trainer.best_accuracy = max(trainer.best_accuracy, warm_acc)

        # ==================== 开始训练 ====================
        print("\n" + "=" * 60)
        print("Starting Preference Learning training...")
        print("=" * 60 + "\n")

        start_time = time.time()

        training_history = trainer.train(
            num_episodes=args.num_episodes,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            log_interval=args.log_interval
        )

        total_time = time.time() - start_time

        # ==================== 保存最终模型 ====================
        final_model_path = os.path.join(args.output_dir, "final_model.pth")
        agent.save(final_model_path, {'training_history': training_history})
        for episode_info in training_history:
            logger.log_episode(episode_info['episode'], episode_info)
    else:
        print("\n" + "=" * 60)
        print("\nTraining is disabled. Skipping to evaluation...")
        print("\n" + "=" * 60)
        print(f"Loading trained policy from {args.checkpoint}")
        agent.load(args.checkpoint)
        total_time = 0.0  # 评估模式下不计算训练时间



    # ==================== 最终评估（跨上下文泛化）====================
    print("\n" + "=" * 60)
    print("Final Evaluation (cross-context transfer)")
    print(f"Tasks: {args.final_eval_tasks}")
    print("=" * 60)

    final_eval_results = trainer.evaluate(num_episodes_per_task=1, env=final_eval_env)

    logger.log_evaluation(args.num_episodes, {
        'final_results': final_eval_results,
        'total_training_time': total_time
    })

    # ==================== 打印总结 ====================
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Total training time: {total_time / 3600:.2f} hours")
    print(f"Best test accuracy: {trainer.best_accuracy:.2f}%")
    print(f"Final model saved to: {final_model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
