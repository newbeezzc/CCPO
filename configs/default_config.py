# -*- coding: utf-8 -*-
"""
CCPO 二阶段训练默认超参数配置（图像分类内循环版）

相比 UDA 版：
  - 内循环任务从 "域自适应" 换成 "图像分类"（默认 CIFAR-10）
  - "任务" 由随机种子/数据划分定义（train_tasks=['s0'..], test_tasks=['s100'..]）
  - 新增 --dataset/--data_dir/--optimizer/--epochs/--train_subset_ratio 等分类参数
  - --evaluator_ckpt 变为可选（分类任务无一阶段评价模型，策略编码器随机初始化 + BC 暖启）
  - DPO / BC / 变异率 / 奖励等 CCPO 核心超参保持不变
"""

import argparse


def get_default_config():
    parser = argparse.ArgumentParser(
        description='CCPO Stage 2: Cross-Context Preference Optimization for LR Scheduling')

    # ==================== 通用 ====================
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU设备ID，多卡用逗号分隔如"0,1,2"')

    # ==================== 分类任务 / 模型 ====================
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'fashion_mnist'],
                        help='内循环数据集')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集根目录（含 cifar-10-batches-py 等）')
    parser.add_argument('--download_data', action='store_true', default=False,
                        help='数据缺失时是否联网下载')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'cnn'],
                        help='模型架构')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'], help='内循环优化器')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')

    # 兼容旧 UDA 参数名（部分脚本仍引用；分类内循环忽略语义无关项）
    parser.add_argument('--dset', type=str, default='cifar10', help='(兼容) 数据集别名')
    parser.add_argument('--data_root', type=str, default='./data', help='(兼容) 数据根目录别名')

    # 内循环训练超参
    parser.add_argument('--epochs', type=int, default=20,
                        help='单个分类任务的训练 epochs（每个 episode 一次完整训练）')
    parser.add_argument('--uda_epochs', type=int, default=20,
                        help='(兼容旧脚本) 等价于 --epochs；两者取非零者，epochs 优先')
    parser.add_argument('--iters_per_epoch', type=int, default=500,
                        help='(兼容) 仅用于旧脚本占位；分类内循环按真实 loader 长度计步')
    parser.add_argument('--train_subset_ratio', type=float, default=0.9,
                        help='每个任务按 seed 采样的训练子集占比（制造任务多样性）')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='内循环初始学习率（建议 = 该配置最优 Cosine LR）')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='学习率下界')
    parser.add_argument('--lr_max', type=float, default=1e-2, help='学习率上界')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')

    # ==================== 学习率调整 ====================
    parser.add_argument('--adjust_interval', type=int, default=100,
                        help='学习率调整间隔（每 N 步干预一次）')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='预热步数，收集初始损失序列')

    # ==================== 策略 / 评价模型编码器 ====================
    parser.add_argument('--evaluator_ckpt', type=str, default='',
                        help='一阶段评价模型 checkpoint（分类任务可留空，编码器随机初始化）')
    parser.add_argument('--loss_window', type=int, default=100,
                        help='损失曲线窗口大小 W')
    parser.add_argument('--loss_channels', type=int, default=1,
                        help='损失通道数 d（分类内循环：仅 raw CE loss）')
    parser.add_argument('--loss_norm', type=str, default='initial',
                        choices=['window', 'initial'],
                        help='loss 归一化模式: window=窗口内z-score, initial=除以warmup结束时的loss')
    parser.add_argument('--encoder_type', type=str, default='frets',
                        choices=['frets', 'gru'],
                        help='时序编码器类型: frets=频域MLP, gru=2层GRU')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='是否冻结复用的时序编码器')
    parser.add_argument('--hidden_dim', type=int, default=256, help='策略网络隐藏层维度')
    parser.add_argument('--embed_dim', type=int, default=128, help='时序编码器嵌入维度')

    # ==================== DPO 算法 ====================
    parser.add_argument('--policy_lr', type=float, default=0.001, help='策略网络学习率')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵正则化系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪阈值')
    parser.add_argument('--dpo_beta', type=float, default=0.3, help='DPO温度 β')
    parser.add_argument('--pref_epsilon', type=float, default=0.05, help='偏好阈值 ε')
    parser.add_argument('--ref_update_interval', type=int, default=500, help='参考策略更新间隔')

    parser.add_argument('--n_dpo_epochs', type=int, default=8,
                        help='每次策略更新对偏好对集合迭代的梯度轮数')
    parser.add_argument('--dpo_minibatch', type=int, default=256,
                        help='DPO 更新的 minibatch 大小，<=0 表示全量')
    parser.add_argument('--pref_buffer_episodes', type=int, default=3,
                        help='偏好对跨 episode 复用窗口（1=原行为）')
    parser.add_argument('--pref_horizon', type=int, default=8,
                        help='return-to-go 前瞻步数 H（signal=return 时）')
    parser.add_argument('--pref_gamma', type=float, default=0.9,
                        help='return-to-go 折扣因子 γ')
    parser.add_argument('--pref_signal', type=str, default='return',
                        choices=['return', 'smooth', 'reward'],
                        help='偏好信号来源')
    parser.add_argument('--pref_weighting', type=int, default=1,
                        help='是否用 |margin| 作为 DPO 损失权重（1=开启）')
    parser.add_argument('--mutation_rate', type=float, default=0.3, help='变异率')
    parser.add_argument('--mutation_strategy', type=str, default='decay',
                        choices=['decay', 'constant'], help='变异率策略')
    parser.add_argument('--mutation_rate_end', type=float, default=0.05, help='变异率衰减终点')
    parser.add_argument('--mutation_rate_decay', type=float, default=0.95, help='变异率衰减因子')
    parser.add_argument('--pref_method', type=str, default='new', choices=['old', 'new'],
                        help='偏好对构造方法')
    parser.add_argument('--n_pref_buckets', type=int, default=5, help='progress 分桶数')
    parser.add_argument('--adapt_k', type=bool, default=True, help='是否自适应调整 rtd')

    # ==================== 奖励 ====================
    parser.add_argument('--reward_time_decay', type=float, default=3.0, help='时间衰减系数 k')
    parser.add_argument('--train_val_ratio', type=float, default=0.2,
                        help='训练阶段验证子集占比（增大可降低单步 Δacc 噪声）')

    # ==================== BC 暖启 ====================
    parser.add_argument('--bc_warmstart', type=int, default=0, help='是否 BC 暖启（1=开启）')
    parser.add_argument('--bc_schedule', type=str, default='inv',
                        choices=['inv', 'cosine', 'step', 'linear', 'fixed'],
                        help='BC 暖启模仿的人工调度')
    parser.add_argument('--bc_rollouts', type=int, default=8, help='BC 采集 rollout 条数')
    parser.add_argument('--bc_epochs', type=int, default=8, help='BC 监督训练轮数')
    parser.add_argument('--bc_lr', type=float, default=1e-3, help='BC 暖启学习率')
    parser.add_argument('--bc_minibatch', type=int, default=256, help='BC minibatch（<=0 全量）')
    parser.add_argument('--bc_set_ref', type=int, default=1, help='暖启后是否同步进 π_ref')

    # ==================== 训练流程 ====================
    parser.add_argument('--num_episodes', type=int, default=100, help='总训练 episodes 数')
    parser.add_argument('--eval_interval', type=int, default=20, help='策略评估间隔（episode）')
    parser.add_argument('--save_interval', type=int, default=50, help='模型保存间隔')
    parser.add_argument('--log_interval', type=int, default=1, help='日志打印间隔')
    parser.add_argument('--is_training', type=int, default=1, help='1=训练，否则仅评估')
    parser.add_argument('--checkpoint', type=str, default='outputs/stage2/final_model.pth',
                        help='训练好的策略 checkpoint 路径（评估模式加载）')

    # ==================== 任务划分（seed/split 定义）====================
    parser.add_argument('--train_tasks', type=str, nargs='+',
                        default=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7'],
                        help='元训练任务列表（每个是一个数据 split seed）')
    parser.add_argument('--test_tasks', type=str, nargs='+',
                        default=['s100', 's101', 's102', 's103'],
                        help='元测试（泛化）任务列表；训练中周期性评估 + best-model 选择用（同分布 CIFAR-10 未见 seed）')
    parser.add_argument('--final_eval_tasks', type=str, nargs='+',
                        default=['fashion_mnist_cnn_s100',
                                 'fashion_mnist_resnet18_s100',
                                 'cifar100_resnet18_s100'],
                        help='最终评估（跨上下文泛化）任务列表：任务名编码 dataset_arch_seed，'
                             '在未见过的数据集/架构上评估策略迁移能力')
    parser.add_argument('--final_eval_epochs', type=str, nargs='+',
                        default=['fashion_mnist:20', 'cifar100:60'],
                        help='最终评估各数据集的完整训练 epochs（"dataset:epochs" 对，'
                             '未列出的数据集回退到 --epochs）')
    parser.add_argument('--final_eval_subset_ratio', type=float, default=1.0,
                        help='最终评估任务的训练数据占比（默认 1.0=全量，得到可比的 benchmark 数字；'
                             '与元训练用于制造任务多样性的 --train_subset_ratio 区分）')

    # ==================== 输出 ====================
    parser.add_argument('--output_dir', type=str, default='./outputs/stage2', help='输出目录')
    parser.add_argument('--exp_name', type=str, default='ccpo', help='实验名称')

    return parser


def parse_args():
    return get_default_config().parse_args()


def parse_task_seed(task_name: str) -> int:
    """从任务名解析随机种子（'s0'/'s12'/'cifar10_resnet18_s3' → 末尾整数）。"""
    if task_name is None:
        return 0
    tail = str(task_name).rsplit('_', 1)[-1]
    if tail.startswith('s') and tail[1:].isdigit():
        return int(tail[1:])
    if tail.isdigit():
        return int(tail)
    return abs(hash(task_name)) % (2 ** 31)


# 数据集别名 → 规范名；架构白名单（供 parse_task_spec 使用）
_DATASET_ALIASES = {
    'cifar10': 'cifar10',
    'cifar100': 'cifar100',
    'fashion_mnist': 'fashion_mnist',
    'fashionmnist': 'fashion_mnist',
    'famnist': 'fashion_mnist',
    'fmnist': 'fashion_mnist',
}
_KNOWN_ARCHS = {'cnn', 'resnet18', 'resnet34', 'resnet50'}


def parse_task_spec(task_name, default_dataset='cifar10', default_arch='resnet18'):
    """从任务名解析 (dataset, arch, seed)，用于跨上下文（跨数据集/架构）评估。

    支持格式（下划线分隔，各段可缺省，缺省用默认值）：
        's0'                          → (default_dataset, default_arch, 0)
        's100'                        → (default_dataset, default_arch, 100)
        'cifar100_resnet18_s3'        → ('cifar100', 'resnet18', 3)
        'fashion_mnist_cnn_s100'      → ('fashion_mnist', 'cnn', 100)
        'famnist_resnet18'            → ('fashion_mnist', 'resnet18', 0)   # 别名，seed 缺省=0

    解析顺序：先剥离末尾 seed（'sN' 或纯数字），再剥离末尾架构（若在白名单），
    余下部分按别名表匹配数据集；任一段无法识别则回退到对应默认值。
    """
    if task_name is None:
        return default_dataset, default_arch, 0

    dataset = None
    arch = None
    seed = None
    tokens = str(task_name).strip().split('_')

    # 末尾 seed：'sN' 或纯数字
    if tokens:
        last = tokens[-1]
        if last[:1] == 's' and last[1:].isdigit():
            seed = int(last[1:]); tokens = tokens[:-1]
        elif last.isdigit():
            seed = int(last); tokens = tokens[:-1]

    # 末尾架构
    if tokens and tokens[-1] in _KNOWN_ARCHS:
        arch = tokens[-1]; tokens = tokens[:-1]

    # 余下部分匹配数据集别名
    rem = '_'.join(tokens)
    if rem in _DATASET_ALIASES:
        dataset = _DATASET_ALIASES[rem]

    if dataset is None:
        dataset = default_dataset
    if arch is None:
        arch = default_arch
    if seed is None:
        seed = 0  # 未显式给出 seed 时用确定值 0（保证可复现）
    return dataset, arch, seed


def parse_epochs_map(pairs):
    """把 ['fashion_mnist:20', 'cifar100:60'] 解析为 {'fashion_mnist':20, 'cifar100':60}。"""
    out = {}
    if not pairs:
        return out
    for item in pairs:
        s = str(item).strip()
        if not s or ':' not in s:
            continue
        k, v = s.rsplit(':', 1)
        try:
            out[k.strip()] = int(v.strip())
        except ValueError:
            continue
    return out


# 兼容旧调用：分类任务不再使用 source/target 文件路径，返回任务名占位
def get_task_paths(args, task_name):
    """(兼容 UDA 接口) 分类任务无源/目标域路径，返回 (task_name, task_name)。"""
    return task_name, task_name


def get_num_classes(dset):
    """获取数据集类别数。"""
    return {'cifar10': 10, 'cifar100': 100, 'fashion_mnist': 10}.get(dset, 10)

