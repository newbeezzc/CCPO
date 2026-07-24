#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Schedule Stage 2 — 改进版实验运行脚本

相比 run_script.py 的核心改进（针对"DPO 不学习、结果停在 ~74%"的诊断）:
  1. 【修复】DPO update() 中 lr / progress 入参顺序颠倒的 bug（无条件生效）
  2. 每次更新多轮小批量迭代 (n_dpo_epochs) + 跨 episode 偏好对复用 (pref_buffer_episodes)
     → 有效梯度步数从 ~100 提升 1~2 个数量级
  3. 偏好信号改为 return-to-go（折现前瞻累计 Δacc, pref_signal=return）
     + margin 加权 (pref_weighting) → 压制单步评估噪声
  4. 增大 DPO 温度 β 默认到 0.3；训练验证子集比例可调 (train_val_ratio) 进一步降噪

用法与 run_script.py 完全一致:
    python run_improved.py --list                 # 查看所有预设
    python run_improved.py --plan                 # 查看建议运行顺序
    python run_improved.py --preset improved      # 跑主推荐配置（先跑这个）
    python run_improved.py --preset improved --dry_run          # 预览命令
    python run_improved.py --preset improved --num_episodes 150 # 覆盖参数
    python run_improved.py --exp_name my_test --dpo_beta 0.5    # 完全自定义

建议先跑 `improved`，与旧 `pref_new` 对比；若提升明显，再按 --plan 跑消融。
"""

import argparse
import os
import subprocess
import sys


# ==============================================================================
# 公共默认参数（改进版）— 与 configs/default_config.py 的新默认保持一致
# ==============================================================================
DEFAULTS = dict(
    dataset_name    = 'cifar10',
    dataset         = 'cifar10',
    arch            = 'resnet18',
    optimizer       = 'adamw',
    epochs          = 20,
    init_lr         = 0.001,
    batch_size      = 128,
    weight_decay    = 0.0001,
    data_dir        = './data',
    gpu_id          = '0',
    num_episodes    = 100,
    adjust_interval = 100,
    warmup_steps    = 100,
    eval_interval   = 20,        # 更频繁地跟踪泛化 & 保存 best（原 500 > num_episodes，从不触发）
    loss_window     = 100,
    policy_lr       = 0.001,
    rtd             = 1.0,
    adapt_k         = True,
    dpo_beta        = 0.3,       # ↑ 原 0.1，放大偏好 logit
    pref_epsilon    = 0.3,       # return-to-go 尺度更大，阈值相应调大
    ref_update_interval = 500,
    pref_method     = 'new',
    n_pref_buckets  = 5,
    mutation_rate       = 0.5,
    mutation_strategy   = 'decay',
    mutation_rate_end   = 0.05,
    mutation_rate_decay = 0.97,
    # ---- 新增改进项 ----
    n_dpo_epochs        = 8,
    dpo_minibatch       = 256,
    pref_buffer_episodes = 3,
    pref_horizon        = 8,
    pref_gamma          = 0.9,
    pref_signal         = 'return',
    pref_weighting      = 1,
    train_val_ratio     = 0.2,   # ↑ 原 0.1，降低单步 Δacc 评估噪声
    # ---- P2: BC 暖启 + 锚定 DPO ----
    bc_warmstart        = 0,     # 默认关闭；P2 预设里打开
    bc_schedule         = 'inv',
    bc_rollouts         = 8,
    bc_epochs           = 8,
    bc_lr               = 0.001,
    bc_minibatch        = 256,
    bc_set_ref          = 1,
)


# ==============================================================================
# 预设配置 — 每个预设只定义「相比 DEFAULTS 不同的参数」
# ==============================================================================
PRESETS = {}

# ---- 主推荐配置：所有改进一起开 ----
PRESETS['improved'] = dict(
    desc = '★主推荐：修复bug + 多轮更新 + 跨episode复用 + return-to-go + margin加权',
)

# ---- 消融：逐个关掉一个改进项，验证各自贡献 ----
PRESETS['abl_single_epoch'] = dict(
    desc = '消融：关多轮更新（每次仅1轮，暴露"梯度步数不足"影响）',
    n_dpo_epochs = 1,
    pref_buffer_episodes = 1,
)
PRESETS['abl_no_replay'] = dict(
    desc = '消融：关跨episode复用（仅用当前episode偏好对）',
    pref_buffer_episodes = 1,
)
PRESETS['abl_signal_smooth'] = dict(
    desc = '消融：偏好信号用旧的3步平滑Δacc（对比return-to-go）',
    pref_signal = 'smooth',
    pref_epsilon = 0.05,
)
PRESETS['abl_no_weight'] = dict(
    desc = '消融：关 margin 加权（标准无权重 DPO）',
    pref_weighting = 0,
)
PRESETS['abl_bugfix_only'] = dict(
    desc = '消融：仅修复入参bug，其余回退旧配置（隔离"bug修复"单独贡献）',
    n_dpo_epochs = 1,
    pref_buffer_episodes = 1,
    pref_signal = 'smooth',
    pref_weighting = 0,
    dpo_beta = 0.1,
    pref_epsilon = 0.05,
    train_val_ratio = 0.1,
    eval_interval = 20,
)

# ---- ★ 抗短视：full-horizon return-to-go（折现到 episode 末尾，最贴合"最终准确率"）----
# 诊断发现 improved 的短视 return(H=8) 会促成"过度早衰减"，反而略低于 ~74%。
# terminal 信号让偏好直接对齐最终收益，是当前最值得优先尝试的方法级改动。
PRESETS['imp_terminal'] = dict(
    desc = '★优先试：full-horizon return-to-go(H到末尾)+γ0.99，抗过度早衰减',
    pref_horizon = 0,      # 0 = 折现到 episode 末尾
    pref_gamma = 0.99,
    pref_epsilon = 0.5,    # terminal 信号尺度更大，阈值相应调大
)
PRESETS['imp_terminal_g95'] = dict(
    desc = 'full-horizon + γ0.95（比 0.99 更看重近端）',
    pref_horizon = 0, pref_gamma = 0.95, pref_epsilon = 0.4,
)

# ---- 关键超参微调（在 improved 基础上扫描） ----
for beta in [0.1, 0.5, 1.0]:
    PRESETS[f'imp_beta_{beta}'] = dict(desc=f'improved + dpo_beta={beta}', dpo_beta=beta)
for ne in [4, 12, 16]:
    PRESETS[f'imp_epochs_{ne}'] = dict(desc=f'improved + n_dpo_epochs={ne}', n_dpo_epochs=ne)
for h in [4, 12, 16]:
    PRESETS[f'imp_horizon_{h}'] = dict(desc=f'improved + pref_horizon={h}', pref_horizon=h)
for vr in [0.1, 0.3]:
    PRESETS[f'imp_valratio_{vr}'] = dict(desc=f'improved + train_val_ratio={vr}', train_val_ratio=vr)

# ---- ★★ P2 核心：BC 暖启 + 锚定 DPO（"把人工调度写进目标"的便宜版）----
# 先用 inv 衰减调度行为克隆策略（起点=baseline 水平），再把策略拷进 π_ref，
# 让 DPO 在 baseline 之上做局部精修。这是当前最有可能稳定破 75% 的方向。
PRESETS['bc_warmstart_inv'] = dict(
    desc = '★★P2核心：inv调度BC暖启 + π_ref锚定 + 终局return DPO精修',
    bc_warmstart = 1,
    bc_schedule = 'inv',
    bc_set_ref = 1,
    # 精修阶段用终局信号，避免把策略推回盲目早衰减
    pref_horizon = 0,
    pref_gamma = 0.99,
    pref_epsilon = 0.5,
    train_val_ratio = 0.3,   # 降低偏好标签噪声
)
PRESETS['bc_warmstart_cosine'] = dict(
    desc = 'P2变体：cosine调度BC暖启（对比 inv）',
    bc_warmstart = 1, bc_schedule = 'cosine', bc_set_ref = 1,
    pref_horizon = 0, pref_gamma = 0.99, pref_epsilon = 0.5, train_val_ratio = 0.3,
)
PRESETS['bc_only_noref'] = dict(
    desc = '消融：BC暖启但不锚定π_ref（暴露"KL锚点=baseline"的作用）',
    bc_warmstart = 1, bc_schedule = 'inv', bc_set_ref = 0,
    pref_horizon = 0, pref_gamma = 0.99, pref_epsilon = 0.5, train_val_ratio = 0.3,
)
PRESETS['bc_pure'] = dict(
    desc = '诊断：纯BC暖启后立即评估（num_episodes=0 等价，只看BC能否复现baseline）',
    bc_warmstart = 1, bc_schedule = 'inv', bc_set_ref = 1, num_episodes = 1,
)

PRESETS['default'] = dict(desc='全部使用改进版默认参数（等同 improved）')


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
        '--init_lr', str(p['init_lr']),
        '--batch_size', str(p['batch_size']),
        '--weight_decay', str(p['weight_decay']),
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
        # ---- 改进项 ----
        '--n_dpo_epochs', str(p['n_dpo_epochs']),
        '--dpo_minibatch', str(p['dpo_minibatch']),
        '--pref_buffer_episodes', str(p['pref_buffer_episodes']),
        '--pref_horizon', str(p['pref_horizon']),
        '--pref_gamma', str(p['pref_gamma']),
        '--pref_signal', str(p['pref_signal']),
        '--pref_weighting', str(p['pref_weighting']),
        '--train_val_ratio', str(p['train_val_ratio']),
        # ---- P2: BC 暖启 ----
        '--bc_warmstart', str(p['bc_warmstart']),
        '--bc_schedule', str(p['bc_schedule']),
        '--bc_rollouts', str(p['bc_rollouts']),
        '--bc_epochs', str(p['bc_epochs']),
        '--bc_lr', str(p['bc_lr']),
        '--bc_minibatch', str(p['bc_minibatch']),
        '--bc_set_ref', str(p['bc_set_ref']),
    ]


# ==============================================================================
# 列出预设
# ==============================================================================
def list_presets():
    print("可用的预设配置 (--preset <name>):\n")
    for name, cfg in PRESETS.items():
        desc = cfg.get('desc', '')
        overrides = {k: v for k, v in cfg.items() if k != 'desc' and v != DEFAULTS.get(k)}
        pad = ' ' * max(0, 24 - len(name))
        print(f"  {name}{pad}{desc}")
        if overrides:
            print(f"  {' ' * 26}覆盖: {overrides}")
        print()
    print(f"共 {len(PRESETS)} 个预设\n")


# ==============================================================================
# 建议的运行顺序
# ==============================================================================
def print_terminal_plan():
    print("建议运行顺序:\n")
    print("  # ── 第 0 步：先测 baseline（决定性！回答『方法是否优于常数LR』）──")
    print("  python eval_baselines.py --schedule all")
    print("     → 得到 fixed/cosine/inv/step 在同一评估协议下的 avg_best，作为对比基准。\n")

    plans = {
        '第 1 步 (★★P2 核心：BC 暖启 + 锚定 DPO，最可能稳定破 75%)': [
            'bc_pure',             # 先验证 BC 能复现 baseline（快，num_episodes=1）
            'bc_warmstart_inv',    # 主推荐：暖启 + 终局 return DPO 精修
        ],
        '第 2 步 (P2 消融：确认"暖启"与"π_ref锚定"各自贡献)': [
            'bc_only_noref',       # 暖启但不锚定 π_ref
            'bc_warmstart_cosine', # 换 cosine 暖启对比
        ],
        '第 3 步 (无暖启的方法级改动，作对照)': [
            'imp_terminal', 'imp_terminal_g95',
        ],
    }
    for stage, presets in plans.items():
        print(f"  # ── {stage} ──")
        for name in presets:
            print(f"  python run_improved.py --preset {name}")
        print()
    print("说明: 先跑第 0 步 baseline —— 若 fixed/cosine 已达 ~73%，说明当前 UDA 配置下")
    print("      LR 调度空间很窄；此时提升重点应转向『稳定优于最佳baseline』而非绝对分数。")
    print("      然后跑第 1 步 imp_terminal，与 baseline 及旧 improved(~72%) 对比。\n")


# ==============================================================================
# 主入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Meta-Schedule Stage 2 改进版运行脚本 — 每次只跑 1 个实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_improved.py --list
  python run_improved.py --plan
  python run_improved.py --preset improved
  python run_improved.py --preset improved --dry_run
  python run_improved.py --exp_name my_test --n_dpo_epochs 12
        """,
    )

    g = parser.add_argument_group('运行模式')
    g.add_argument('--preset', type=str, default=None, help='预设名称 (使用 --list 查看全部)')
    g.add_argument('--exp_name', type=str, default='improved', help='实验名称 (不使用预设时填)')
    g.add_argument('--dry_run', action='store_true', help='仅打印命令不执行')
    g.add_argument('--list', action='store_true', help='列出所有预设配置')
    g.add_argument('--plan', action='store_true', help='打印建议的运行顺序')

    g = parser.add_argument_group('数据集/设备/模型')
    g.add_argument('--dataset_name', type=str, default=None, help='输出目录分组名')
    g.add_argument('--dataset', type=str, default=None, choices=['cifar10', 'cifar100', 'fashion_mnist'])
    g.add_argument('--arch', type=str, default=None, choices=['resnet18', 'resnet34', 'resnet50', 'cnn'])
    g.add_argument('--optimizer', type=str, default=None, choices=['adamw', 'sgd'])
    g.add_argument('--epochs', type=int, default=None)
    g.add_argument('--init_lr', type=float, default=None)
    g.add_argument('--batch_size', type=int, default=None)
    g.add_argument('--weight_decay', type=float, default=None)
    g.add_argument('--data_dir', type=str, default=None)
    g.add_argument('--gpu_id', type=str, default=None)

    g = parser.add_argument_group('训练流程')
    g.add_argument('--num_episodes', type=int, default=None)
    g.add_argument('--adjust_interval', type=int, default=None)
    g.add_argument('--warmup_steps', type=int, default=None)
    g.add_argument('--eval_interval', type=int, default=None)
    g.add_argument('--loss_window', type=int, default=None)

    g = parser.add_argument_group('优化器')
    g.add_argument('--policy_lr', type=float, default=None)

    g = parser.add_argument_group('奖励')
    g.add_argument('--rtd', type=float, default=None, help='reward_time_decay')
    g.add_argument('--adapt_k', type=bool, default=None)
    g.add_argument('--train_val_ratio', type=float, default=None)

    g = parser.add_argument_group('DPO')
    g.add_argument('--dpo_beta', type=float, default=None)
    g.add_argument('--pref_epsilon', type=float, default=None)
    g.add_argument('--ref_update_interval', type=int, default=None)
    g.add_argument('--pref_method', type=str, default=None, choices=['old', 'new'])
    g.add_argument('--n_pref_buckets', type=int, default=None)

    g = parser.add_argument_group('DPO 改进项')
    g.add_argument('--n_dpo_epochs', type=int, default=None)
    g.add_argument('--dpo_minibatch', type=int, default=None)
    g.add_argument('--pref_buffer_episodes', type=int, default=None)
    g.add_argument('--pref_horizon', type=int, default=None)
    g.add_argument('--pref_gamma', type=float, default=None)
    g.add_argument('--pref_signal', type=str, default=None, choices=['return', 'smooth', 'reward'])
    g.add_argument('--pref_weighting', type=int, default=None, choices=[0, 1])

    g = parser.add_argument_group('变异率')
    g.add_argument('--mutation_rate', type=float, default=None)
    g.add_argument('--mutation_strategy', type=str, default=None, choices=['decay', 'constant'])
    g.add_argument('--mutation_rate_end', type=float, default=None)
    g.add_argument('--mutation_rate_decay', type=float, default=None)

    g = parser.add_argument_group('P2: BC 暖启')
    g.add_argument('--bc_warmstart', type=int, default=None, choices=[0, 1])
    g.add_argument('--bc_schedule', type=str, default=None,
                   choices=['inv', 'cosine', 'step', 'linear', 'fixed'])
    g.add_argument('--bc_rollouts', type=int, default=None)
    g.add_argument('--bc_epochs', type=int, default=None)
    g.add_argument('--bc_lr', type=float, default=None)
    g.add_argument('--bc_minibatch', type=int, default=None)
    g.add_argument('--bc_set_ref', type=int, default=None, choices=[0, 1])

    args = parser.parse_args()

    if args.list:
        list_presets()
        return
    if args.plan:
        print_terminal_plan()
        return

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

    if args.exp_name != 'improved':
        params['exp_name'] = args.exp_name
    elif 'exp_name' not in params:
        params['exp_name'] = 'improved'

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
