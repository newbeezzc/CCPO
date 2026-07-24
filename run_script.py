#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Schedule Stage 2 实验运行脚本（DPO 偏好学习版本）

每次调用运行且仅运行 1 个实验（train_stage2.py 自带最终评估）。
通过 --preset 使用预设配置，也可直接覆盖任意参数。

用法:
    # 查看所有预设
    python run_script.py --list

    # 查看建议的多终端运行方案
    python run_script.py --plan

    # 使用预设运行一个实验
    python run_script.py --preset pref_new
    python run_script.py --preset mut_decay_0.3
    python run_script.py --preset rtd_3.0

    # 在预设基础上覆盖参数
    python run_script.py --preset pref_new --gpu_id 1 --num_episodes 200

    # 完全自定义（不使用预设）
    python run_script.py --exp_name my_test --pref_method old --dpo_beta 0.5

    # 预览命令（不实际执行）
    python run_script.py --preset pref_new --dry_run
"""

import argparse
import os
import subprocess
import sys


# ==============================================================================
# 公共默认参数
# ==============================================================================
DEFAULTS = dict(
    dataset_name   = 'cifar10',
    dataset        = 'cifar10',
    arch           = 'resnet18',
    optimizer      = 'adamw',
    epochs         = 20,
    data_dir       = './data',
    gpu_id         = '0',
    num_episodes   = 100,
    adjust_interval = 100,
    warmup_steps   = 100,
    eval_interval  = 500,
    loss_window    = 100,
    policy_lr      = 0.001,
    rtd            = 1.0,
    adapt_k        = True,
    dpo_beta       = 0.1,
    pref_epsilon   = 0.05,
    ref_update_interval = 500,
    pref_method    = 'old',
    n_pref_buckets = 5,
    mutation_rate      = 1.0,
    mutation_strategy  = 'decay',
    mutation_rate_end  = 0.05,
    mutation_rate_decay = 0.95,
)


# ==============================================================================
# 预设配置 — 每个预设只定义「相比 DEFAULTS 不同的参数」
# ==============================================================================
PRESETS = {}

# ---- Round 1: pref_method 对比 ----
PRESETS['pref_new'] = dict(
    desc = '偏好对构造: progress分桶 (新方法)',
    pref_method = 'new',
)
PRESETS['pref_old'] = dict(
    desc = '偏好对构造: time-decayed reward直接比较 (旧方法)',
    pref_method = 'old',
)

# ---- Round 1: mutation 对比 ----
for mr in [0.1, 0.3, 0.5, 0.8, 1.0]:
    PRESETS[f'mut_decay_{mr}'] = dict(
        desc = f'变异率: decay从{mr}衰减到0.05',
        mutation_strategy = 'decay',
        mutation_rate = mr,
    )
for mr in [0.05, 0.1, 0.2, 0.3, 0.5]:
    PRESETS[f'mut_const_{mr}'] = dict(
        desc = f'变异率: 恒定{mr}',
        mutation_strategy = 'constant',
        mutation_rate = mr,
    )

# ---- Round 1: rtd 对比 ----
for rtd in [0.5, 1.0, 2.0, 3.0, 5.0]:
    PRESETS[f'rtd_{rtd}'] = dict(
        desc = f'reward_time_decay = {rtd}',
        rtd = rtd,
    )

# ---- Round 2: dpo_beta 对比 ----
for beta in [0.01, 0.05, 0.1, 0.5, 1.0]:
    PRESETS[f'beta_{beta}'] = dict(
        desc = f'DPO beta = {beta}',
        dpo_beta = beta,
    )

PRESETS['default'] = dict(desc='全部使用默认参数')


# ==============================================================================
# 构建命令行
# ==============================================================================
def build_cmd(params):
    p = {**DEFAULTS, **params}
    exp_name = p['exp_name']
    output_dir = os.path.join('outputs', p['dataset_name'], exp_name)

    return [
        sys.executable, 'train_stage2.py',
        '--dataset', str(p['dataset']),
        '--arch', str(p['arch']),
        '--optimizer', str(p['optimizer']),
        '--epochs', str(p['epochs']),
        '--data_dir', str(p['data_dir']),
        '--output_dir', output_dir,
        '--exp_name', exp_name,
        '--gpu_id', str(p['gpu_id']),
        '--num_episodes', str(p['num_episodes']),
        '--adjust_interval', str(p['adjust_interval']),
        '--warmup_steps', str(p['warmup_steps']),
        '--eval_interval', str(p['eval_interval']),
        '--loss_window', str(p['loss_window']),
        '--policy_lr', str(p['policy_lr']),
        '--reward_time_decay', str(p['rtd']),
        '--adapt_k', str(p['adapt_k']),
        '--dpo_beta', str(p['dpo_beta']),
        '--pref_epsilon', str(p['pref_epsilon']),
        '--ref_update_interval', str(p['ref_update_interval']),
        '--pref_method', str(p['pref_method']),
        '--n_pref_buckets', str(p['n_pref_buckets']),
        '--mutation_rate', str(p['mutation_rate']),
        '--mutation_strategy', str(p['mutation_strategy']),
        '--mutation_rate_end', str(p['mutation_rate_end']),
        '--mutation_rate_decay', str(p['mutation_rate_decay']),
    ]


# ==============================================================================
# 列出预设
# ==============================================================================
def list_presets():
    print("可用的预设配置 (--preset <name>):\n")
    for name, cfg in PRESETS.items():
        desc = cfg.get('desc', '')
        overrides = {k: v for k, v in cfg.items() if k != 'desc' and v != DEFAULTS.get(k)}
        pad = ' ' * max(0, 28 - len(name))
        print(f"  {name}{pad}{desc}")
        if overrides:
            print(f"  {' ' * 30}覆盖: {overrides}")
        print()
    print(f"共 {len(PRESETS)} 个预设\n")


# ==============================================================================
# 建议的多终端运行方案
# ==============================================================================
def print_terminal_plan():
    plans = {
        '终端 1 (4 实验)': [
            'pref_new', 'pref_old', 'mut_decay_0.1', 'rtd_0.5',
        ],
        '终端 2 (4 实验)': [
            'mut_decay_0.5', 'mut_const_0.05', 'mut_const_0.1', 'mut_const_0.2',
        ],
        '终端 3 (5 实验)': [
            'rtd_1.0', 'rtd_2.0', 'rtd_3.0', 'rtd_5.0', 'mut_decay_0.3',
        ],
    }
    print("建议 3 终端运行方案 (依次执行):\n")
    for term, presets in plans.items():
        print(f"  # ── {term} ──")
        for name in presets:
            print(f"  python run_script.py --preset {name}")
        print()
    print(f"共 {sum(len(v) for v in plans.values())} 个实验\n")


# ==============================================================================
# 主入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Meta-Schedule Stage 2 实验运行脚本 — 每次只跑 1 个实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_script.py --list                    # 查看所有预设
  python run_script.py --plan                    # 查看多终端运行方案
  python run_script.py --preset pref_new         # 运行预设实验
  python run_script.py --preset rtd_3.0 --dry_run  # 预览不执行
  python run_script.py --exp_name my_test --dpo_beta 0.5  # 自定义实验
        """,
    )

    # ---- 运行模式 ----
    g = parser.add_argument_group('运行模式')
    g.add_argument('--preset', type=str, default=None,
                   help='预设名称 (使用 --list 查看全部)')
    g.add_argument('--exp_name', type=str, default='default',
                   help='实验名称 (不使用预设时必填)')
    g.add_argument('--dry_run', action='store_true',
                   help='仅打印命令不执行')
    g.add_argument('--list', action='store_true',
                   help='列出所有预设配置')
    g.add_argument('--plan', action='store_true',
                   help='打印建议的多终端运行方案')

    # ---- 数据集/设备/模型 (可覆盖默认值) ----
    g = parser.add_argument_group('数据集/设备/模型')
    g.add_argument('--dataset_name', type=str, default=None, help='输出目录分组名')
    g.add_argument('--dataset', type=str, default=None, choices=['cifar10', 'cifar100', 'fashion_mnist'])
    g.add_argument('--arch', type=str, default=None, choices=['resnet18', 'resnet34', 'resnet50', 'cnn'])
    g.add_argument('--optimizer', type=str, default=None, choices=['adamw', 'sgd'])
    g.add_argument('--epochs', type=int, default=None)
    g.add_argument('--data_dir', type=str, default=None)
    g.add_argument('--gpu_id', type=str, default=None)

    # ---- 训练流程 (可覆盖默认值) ----
    g = parser.add_argument_group('训练流程')
    g.add_argument('--num_episodes', type=int, default=None)
    g.add_argument('--adjust_interval', type=int, default=None)
    g.add_argument('--warmup_steps', type=int, default=None)
    g.add_argument('--eval_interval', type=int, default=None)
    g.add_argument('--loss_window', type=int, default=None)

    # ---- 优化器 (可覆盖默认值) ----
    g = parser.add_argument_group('优化器')
    g.add_argument('--policy_lr', type=float, default=None)

    # ---- 奖励 (可覆盖默认值) ----
    g = parser.add_argument_group('奖励')
    g.add_argument('--rtd', type=float, default=None, help='reward_time_decay')
    g.add_argument('--adapt_k', type=bool, default=None)

    # ---- DPO (可覆盖默认值) ----
    g = parser.add_argument_group('DPO')
    g.add_argument('--dpo_beta', type=float, default=None)
    g.add_argument('--pref_epsilon', type=float, default=None)
    g.add_argument('--ref_update_interval', type=int, default=None)
    g.add_argument('--pref_method', type=str, default=None, choices=['old', 'new'])
    g.add_argument('--n_pref_buckets', type=int, default=None)

    # ---- 变异率 (可覆盖默认值) ----
    g = parser.add_argument_group('变异率')
    g.add_argument('--mutation_rate', type=float, default=None)
    g.add_argument('--mutation_strategy', type=str, default=None, choices=['decay', 'constant'])
    g.add_argument('--mutation_rate_end', type=float, default=None)
    g.add_argument('--mutation_rate_decay', type=float, default=None)

    args = parser.parse_args()

    # ---- 特殊模式 ----
    if args.list:
        list_presets()
        return
    if args.plan:
        print_terminal_plan()
        return

    # ---- 构建 params ----
    params = DEFAULTS.copy()

    if args.preset:
        if args.preset not in PRESETS:
            print(f"错误: 未知预设 '{args.preset}'。用 --list 查看全部。")
            sys.exit(1)
        preset = PRESETS[args.preset]
        params['exp_name'] = args.preset
        for k, v in preset.items():
            if k != 'desc':
                params[k] = v

    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    if args.exp_name != 'default':
        params['exp_name'] = args.exp_name

    # ---- 运行 ----
    cmd = build_cmd(params)
    if args.dry_run:
        print(f"[DRY RUN] $ {' '.join(cmd)}")
        return

    print(f"\n{'─'*70}")
    print(f"Experiment: {params['exp_name']}")
    print(f"$ {' '.join(cmd)}")
    print(f"{'─'*70}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n实验失败 (exit code: {result.returncode})")
        sys.exit(1)


if __name__ == '__main__':
    main()
