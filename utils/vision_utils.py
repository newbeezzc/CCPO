# -*- coding: utf-8 -*-
"""
视觉分类内循环的模型与数据工具

从 UBA/vision/train_vision.py 移植（SimpleCNN / get_model / 数据构建），
供 envs/cls_env.py 构建 CIFAR-10（及可扩展的 CIFAR-100 / Fashion-MNIST）内循环使用。

设计要点（对齐实验计划 Exp-1）：
  - CIFAR / Fashion-MNIST 使用 3x3 conv stem（cifar_stem=True），避免标准 ResNet
    的 7x7+maxpool 在小图上过度下采样。
  - Normalize 常量严格对齐实验计划：
        CIFAR-10  (0.4914,0.4822,0.4465)/(0.2470,0.2435,0.2616)
        CIFAR-100 (0.5071,0.4867,0.4408)/(0.2675,0.2565,0.2761)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# ── 数据集元信息 ────────────────────────────────────────────────────────────
DATASET_META = {
    'cifar10': {
        'num_classes': 10,
        'norm_mean': (0.4914, 0.4822, 0.4465),
        'norm_std': (0.2470, 0.2435, 0.2616),
        'cifar_stem': True,
        'in_channels': 3,
    },
    'cifar100': {
        'num_classes': 100,
        'norm_mean': (0.5071, 0.4867, 0.4408),
        'norm_std': (0.2675, 0.2565, 0.2761),
        'cifar_stem': True,
        'in_channels': 3,
    },
    'fashion_mnist': {
        'num_classes': 10,
        'norm_mean': (0.2860, 0.2860, 0.2860),
        'norm_std': (0.3530, 0.3530, 0.3530),
        'cifar_stem': True,
        'in_channels': 3,
    },
}


# ── Simple CNN（LODE 论文 Appendix A.1；Fashion-MNIST 用）─────────────────────
# 注意：以下 _LocalSimpleCNN / _local_get_model 是"本地副本"，仅在无法从
# UBA/vision 导入权威定义时作为回退使用（见文件末尾的 UBA 桥接）。
class _LocalSimpleCNN(nn.Module):
    """2-conv CNN（对齐 LODE / 实验计划 V1）。默认 28x28 灰度输入。"""

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def _local_get_model(arch: str, num_classes: int, cifar_stem: bool = False, in_channels: int = 3):
    """按架构名构建模型（本地回退实现）。ResNet 在小图上用 3x3 stem。"""
    if arch == "cnn":
        return _LocalSimpleCNN(num_classes=num_classes, in_channels=in_channels)

    resnet_map = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
    }
    if arch not in resnet_map:
        raise ValueError(f"Unknown architecture: {arch}")
    model = resnet_map[arch](num_classes=num_classes)
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def build_transforms(dataset: str, arch: str = "resnet18"):
    """返回 (train_transform, test_transform)。"""
    meta = DATASET_META[dataset]
    norm = transforms.Normalize(meta['norm_mean'], meta['norm_std'])

    if dataset in ("cifar10", "cifar100"):
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm])
        test_tf = transforms.Compose([transforms.ToTensor(), norm])
    elif dataset == "fashion_mnist":
        if arch == "cnn":
            # 28x28 灰度（1 通道）
            gray_norm = transforms.Normalize((0.2860,), (0.3530,))
            train_tf = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), gray_norm])
            test_tf = transforms.Compose([transforms.ToTensor(), gray_norm])
        else:
            # ResNet：灰度 -> 32x32 3 通道
            train_tf = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), norm])
            test_tf = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(), norm])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_tf, test_tf


def build_datasets(dataset: str, data_dir: str, arch: str = "resnet18",
                   download: bool = False):
    """返回 (train_dataset, test_dataset, num_classes)。"""
    train_tf, test_tf = build_transforms(dataset, arch)
    if dataset == "cifar10":
        ds_cls = torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        ds_cls = torchvision.datasets.CIFAR100
    elif dataset == "fashion_mnist":
        ds_cls = torchvision.datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_ds = ds_cls(root=data_dir, train=True, download=download, transform=train_tf)
    test_ds = ds_cls(root=data_dir, train=False, download=download, transform=test_tf)
    return train_ds, test_ds, DATASET_META[dataset]['num_classes']


def in_channels_for(dataset: str, arch: str) -> int:
    """CNN + Fashion-MNIST 用 1 通道灰度；其余 3 通道。"""
    if arch == "cnn" and dataset == "fashion_mnist":
        return 1
    return 3


# ── UBA/vision 桥接 ─────────────────────────────────────────────────────────
# 模型定义（get_model / SimpleCNN）优先从源项目 UBA/vision/train_vision.py 直接
# 导入，这样若后续在 vision 项目里修改网络结构，CCPO 无需手动同步即可自动跟随。
# 若导入失败（源项目被移动/缺依赖等），回退到上面的本地副本，保证 CCPO 仍可独立运行。
#
# 数据构建（build_datasets / build_transforms）刻意保留在本地：UBA 的
# get_dataloaders 返回的是 DataLoader（不兼容本项目按 Subset 采样的用法）、
# 强制 download=True、且 CIFAR-10 归一化常量与实验计划不同。fashion_mnist /
# cifar100 的 transform 已与 UBA 对齐，故直接跨集评估不会有一致性损失。
#
# 覆盖路径可用环境变量 CCPO_UBA_VISION_DIR 指定。
def _load_uba_models():
    import os as _os
    import sys as _sys

    override = _os.environ.get("CCPO_UBA_VISION_DIR", "")
    candidates = []
    if override:
        candidates.append(override)
    _here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # .../CCPO
    _code_root = _os.path.dirname(_here)                                    # .../code
    candidates.append(_os.path.join(_code_root, "UBA", "vision"))

    for uba_dir in candidates:
        tv_path = _os.path.join(uba_dir, "train_vision.py")
        if not _os.path.isfile(tv_path):
            continue
        try:
            if uba_dir not in _sys.path:
                _sys.path.insert(0, uba_dir)
            import train_vision as _tv  # noqa: F401
            if hasattr(_tv, "get_model") and hasattr(_tv, "SimpleCNN"):
                return _tv.get_model, _tv.SimpleCNN, tv_path
        except Exception:
            continue
    return None, None, None


_uba_get_model, _uba_SimpleCNN, _UBA_SRC = _load_uba_models()

if _uba_get_model is not None:
    # 直接复用源项目定义（自动同步 vision 的网络结构改动）
    get_model = _uba_get_model
    SimpleCNN = _uba_SimpleCNN
    VISION_MODEL_SOURCE = _UBA_SRC
else:
    # 回退到本地副本
    get_model = _local_get_model
    SimpleCNN = _LocalSimpleCNN
    VISION_MODEL_SOURCE = "local (utils/vision_utils.py)"
