#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 学习率调度评估 —— 与策略评估协议完全一致

关键点：直接复用 train_stage2 的 MultiTaskUDAEnv(flag='test')，用「脚本化学习率调度」
驱动 env.step(...)，从而保证 UDA 模型 / 损失 / 数据 / 评估节奏与 DPO 策略评估 100% 相同，
得到可直接对比的 baseline 数字（这是 run_baselines.py 无法提供的——它用了不同的
epochs / tradeoff / 评估协议）。

支持的调度:
    fixed   : 恒定 init_lr（== "LR 不变" 对照，回答"方法是否优于常数 LR"）
    cosine  : init_lr · 0.5·(1+cos(π·p))
    inv     : init_lr · (1+10·p)^(-0.75)   (DANN/常见 UDA 退火)
    step    : init_lr · gamma^floor(p / step_frac)
    linear  : init_lr · (1 - p)

用法（与 run_script 同一入口风格，通过 train_stage2 的默认参数体系）:
    python eval_baselines.py --schedule fixed
    python eval_baselines.py --schedule cosine --gpu_id 0
    python eval_baselines.py --schedule all      # 依次跑全部调度

结果打印每个 test task 的 best/final acc 及平均，并保存 JSON 到 outputs/OfficeHome/baseline_eval/。
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default_config import get_default_config
from envs.cls_env import MultiTaskClsEnv as MultiTaskUDAEnv
from utils.training_utils import set_seed, setup_device


def lr_from_schedule(name: str, p: float, init_lr: float,
                     step_frac: float = 0.33, gamma: float = 0.1) -> float:
    """给定进度 p∈[0,1] 返回该调度的学习率。"""
    if name == 'fixed':
        return init_lr
    if name == 'cosine':
        return init_lr * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    if name == 'inv':
        return init_lr * (1.0 + 10.0 * p) ** (-0.75)
    if name == 'linear':
        return init_lr * max(0.0, 1.0 - p)
    if name == 'step':
        k = int(p / step_frac)
        return init_lr * (gamma ** k)
    raise ValueError(f'unknown schedule {name}')


def run_schedule(env: MultiTaskUDAEnv, task: str, schedule: str,
                 init_lr: float, step_frac: float, gamma: float) -> dict:
    """在单个 task 上用脚本化调度跑完整 episode（deterministic，协议同策略评估）。"""
    env.reset(task)
    done = False
    inner = env.env  # 底层 UDAEnv
    total = inner.total_steps
    while not done:
        p = inner.current_step / total
        lr = lr_from_schedule(schedule, p, init_lr, step_frac, gamma)
        _, _, done, _ = env.step(lr, deterministic=True)
    info = env.get_episode_info()
    return {
        'task': task,
        'best_accuracy': info.get('best_accuracy', 0.0),
        'final_accuracy': info.get('final_accuracy', 0.0),
        'acc_history': info.get('acc_history', []),
    }


def main():
    parser = get_default_config()
    # 追加 baseline 专用参数
    parser.add_argument('--schedule', type=str, default='all',
                        help="fixed|cosine|inv|step|linear|all")
    parser.add_argument('--step_frac', type=float, default=0.33,
                        help='step 调度每隔多少比例进度衰减一次')
    parser.add_argument('--step_gamma', type=float, default=0.1,
                        help='step 调度衰减因子')
    # evaluator_ckpt 是 required，但 baseline 不用策略；给个占位默认
    args = parser.parse_args(_inject_dummy_ckpt())

    device, _ = setup_device(args.gpu_id)
    set_seed(args.seed)

    schedules = ['fixed', 'cosine', 'inv', 'step', 'linear'] if args.schedule == 'all' \
        else [args.schedule]

    epochs = int(getattr(args, 'epochs', 0) or getattr(args, 'uda_epochs', 20))
    print("=" * 70)
    print("Baseline LR-Schedule Evaluation (protocol == policy eval)")
    print(f"  test tasks : {args.test_tasks}")
    print(f"  dataset={args.dataset} arch={args.arch} optimizer={args.optimizer} "
          f"epochs={epochs}")
    print(f"  adjust={args.adjust_interval} init_lr={args.init_lr}")
    print(f"  schedules  : {schedules}")
    print("=" * 70)

    test_env = MultiTaskUDAEnv(args=args, task_list=args.test_tasks,
                               flag='test', device=str(device))

    all_results = {}
    for sched in schedules:
        print(f"\n{'─'*60}\n### Schedule: {sched}\n{'─'*60}")
        per_task = []
        for task in args.test_tasks:
            r = run_schedule(test_env, task, sched, args.init_lr,
                             args.step_frac, args.step_gamma)
            per_task.append(r)
            print(f"  {task:>8s}: best={r['best_accuracy']:.2f}%  final={r['final_accuracy']:.2f}%")
        avg_best = float(np.mean([r['best_accuracy'] for r in per_task]))
        avg_final = float(np.mean([r['final_accuracy'] for r in per_task]))
        print(f"  {'AVG':>8s}: best={avg_best:.2f}%  final={avg_final:.2f}%")
        all_results[sched] = {'per_task': per_task,
                              'avg_best': avg_best, 'avg_final': avg_final}

    # 汇总表
    print("\n" + "=" * 70)
    print("SUMMARY (avg over test tasks)")
    print("-" * 70)
    print(f"  {'schedule':<10} {'avg_best':>10} {'avg_final':>10}")
    for sched, r in sorted(all_results.items(), key=lambda kv: -kv[1]['avg_best']):
        print(f"  {sched:<10} {r['avg_best']:>9.2f}% {r['avg_final']:>9.2f}%")
    print("=" * 70)

    out_dir = os.path.join('outputs', str(getattr(args, 'dataset', 'cifar10')), 'baseline_eval')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'baseline_eval_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump({'config': {k: (v if _jsonable(v) else str(v))
                              for k, v in vars(args).items()},
                   'results': all_results}, f, indent=2)
    print(f"\nSaved: {out_path}")


def _inject_dummy_ckpt():
    """分类任务下 --evaluator_ckpt 已可选、--data_dir 有默认值，无需再注入占位路径。
    保留此函数仅为兼容旧调用点（直接透传命令行参数）。"""
    return sys.argv[1:]


def _jsonable(v):
    return isinstance(v, (int, float, str, bool, list, dict, type(None)))


if __name__ == '__main__':
    main()
