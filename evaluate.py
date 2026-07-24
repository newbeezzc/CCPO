#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略评估脚本

加载训练好的学习率调度策略，在指定任务上评估其性能

使用方法:
    python evaluate.py --checkpoint /path/to/checkpoint.pth \
                       --evaluator_ckpt /path/to/stage1/checkpoint.pth \
                       --data_root /path/to/OfficeHome \
                       --tasks Pr-Rw Rw-Ar Rw-Cl Rw-Pr
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
from datetime import datetime

from agents.dpo_agent import DPOAgent

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default_config import get_default_config, get_task_paths
from models.evaluator import load_evaluator, EvaluatorWrapper, EvaluatorModelConfig
from envs.cls_env import MultiTaskClsEnv as MultiTaskUDAEnv
from agents.a2c_agent import A2CAgent
from utils.training_utils import set_seed, setup_device


def parse_eval_args():
    """解析评估参数"""
    parser = get_default_config()

    # 评估特定参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的策略checkpoint路径')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='要评估的任务列表，默认使用test_tasks')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='每个任务运行次数')
    parser.add_argument('--save_trajectories', action='store_true',
                        help='是否保存学习率调整轨迹')

    args = parser.parse_args()

    # 如果没有指定任务，使用test_tasks
    if args.tasks is None:
        args.tasks = args.test_tasks

    return args


def evaluate_policy(agent, env, task_name, num_runs=3, save_trajectories=False):
    """
    评估策略在单个任务上的性能
    
    Args:
        agent: A2C Agent
        env: UDA环境
        task_name: 任务名称
        num_runs: 运行次数
        save_trajectories: 是否保存轨迹
    
    Returns:
        results: 评估结果字典
    """
    agent.eval_mode()

    all_accuracies = []
    all_rewards = []
    trajectories = []

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end=" ")

        state = env.reset(task_name)
        done = False
        episode_reward = 0.0

        while not done:
            _, action_lr, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action_lr, deterministic=True)
            episode_reward += reward

        # 获取episode信息
        episode_info = env.get_episode_info()

        all_accuracies.append(episode_info['best_accuracy'])
        all_rewards.append(episode_reward)

        if save_trajectories:
            trajectories.append({
                'run': run,
                'lr_history': episode_info['lr_history'],
                'acc_history': episode_info['acc_history'],
                'final_accuracy': episode_info['final_accuracy'],
                'best_accuracy': episode_info['best_accuracy']
            })

        print(f"Accuracy: {episode_info['final_accuracy']:.2f}%")

    results = {
        'task': task_name,
        'mean_accuracy': np.mean(all_accuracies),
        'std_accuracy': np.std(all_accuracies),
        'max_accuracy': np.max(all_accuracies),
        'min_accuracy': np.min(all_accuracies),
        'mean_reward': np.mean(all_rewards),
        'all_accuracies': all_accuracies,
        'num_runs': num_runs
    }

    if save_trajectories:
        results['trajectories'] = trajectories

    return results


def main():
    # 解析参数
    args = parse_eval_args()

    # 设置设备和随机种子
    device, gpu_ids = setup_device(args.gpu_id)
    set_seed(args.seed)

    print("=" * 60)
    print("Meta-Schedule Policy Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {args.tasks}")
    print(f"Number of runs per task: {args.num_runs}")
    print("=" * 60)

    # 创建环境
    print("\nCreating evaluation environment...")
    eval_env = MultiTaskUDAEnv(
        args=args,
        task_list=args.tasks,
        device=str(device)
    )

    # 创建Agent并加载checkpoint
    print("\nLoading trained policy...")
    agent = DPOAgent(args, device=str(device))
    agent.load(args.checkpoint)

    # 评估每个任务
    print("\n" + "=" * 60)
    print("Starting Evaluation")
    print("=" * 60)

    all_results = []

    for task in args.tasks:
        print(f"\nEvaluating on task: {task}")
        results = evaluate_policy(
            agent, eval_env, task,
            num_runs=args.num_runs,
            save_trajectories=args.save_trajectories
        )
        all_results.append(results)

    # 打印总结
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"{'Task':<12} {'Mean':>8} {'Std':>8} {'Max':>8} {'Min':>8}")
    print("-" * 60)

    total_mean = []
    for r in all_results:
        print(f"{r['task']:<12} {r['mean_accuracy']:>7.2f}% {r['std_accuracy']:>7.2f}% "
              f"{r['max_accuracy']:>7.2f}% {r['min_accuracy']:>7.2f}%")
        total_mean.append(r['mean_accuracy'])

    print("-" * 60)
    print(f"{'Average':<12} {np.mean(total_mean):>7.2f}% {np.std(total_mean):>7.2f}%")
    print("=" * 60)

    # 保存结果
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(args.output_dir, f'eval_results_{timestamp}.json')

        # 移除不可序列化的内容
        save_results = []
        for r in all_results:
            save_r = {k: v for k, v in r.items() if k != 'trajectories'}
            if 'trajectories' in r:
                # 只保存轨迹的关键信息
                save_r['trajectories'] = [
                    {
                        'run': t['run'],
                        'final_accuracy': t['final_accuracy'],
                        'num_lr_adjustments': len(t['lr_history']) - 1,
                        'acc_history': t['acc_history']
                    }
                    for t in r['trajectories']
                ]
            save_results.append(save_r)

        with open(result_file, 'w') as f:
            json.dump({
                'args': {k: str(v) for k, v in vars(args).items()},
                'results': save_results,
                'summary': {
                    'mean_accuracy': np.mean(total_mean),
                    'std_accuracy': np.std(total_mean)
                }
            }, f, indent=2)

        print(f"\nResults saved to: {result_file}")


if __name__ == '__main__':
    main()
