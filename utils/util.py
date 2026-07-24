import sys

import numpy as np
import torch
import os
import shutil
from torch.utils.data.dataloader import DataLoader

class ContinuousDataloader:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def avg(list):
    return sum(list) / len(list) if len(list) > 0 else 0

def _estimate_k_hat_online(acc_records: list, fallback_k) -> float:
    """
    用上一个 episode 内已累积的 (progress, acc) 序列滚动估计收敛率 α_hat。

    方法：对 log(Δacc_t) ~ p_t 做最小二乘线性回归，k_hat = -slope。
    数据点不足或质量差时回退到固定 k0。

    Args:
        acc_records: List of acc，长度随 episode 推进增长

    Returns:
        k_hat: 本 step 使用的收敛率估计值
    """
    if len(acc_records) < 3:
        return fallback_k
    length = len(acc_records)
    prog_list = np.array([(i + 1) / length for i in range(length)])  # 进度列表
    acc_list = np.array(acc_records)

    # 构建相邻差分序列
    delta_accs = np.diff(acc_list)  # 长度 n-1
    prog_mid = 0.5 * (prog_list[:-1] + prog_list[1:])  # 对应中点进度

    # 只保留 Δacc > 0 的有效点
    mask = delta_accs > 1e-6
    if mask.sum() < 3:
        return fallback_k

    p_valid = prog_mid[mask]
    ld_valid = np.log(delta_accs[mask])

    # OLS：log(Δacc) = intercept + slope * p，k_hat = -slope
    p_mean = p_valid.mean()
    ld_mean = ld_valid.mean()
    slope = (np.sum((p_valid - p_mean) * (ld_valid - ld_mean))
             / (np.sum((p_valid - p_mean) ** 2) + 1e-12))
    k_hat = float(np.clip(-slope, 0.1, fallback_k * 5.0))
    return k_hat