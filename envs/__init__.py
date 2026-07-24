# -*- coding: utf-8 -*-
"""
环境模块（图像分类内循环）
"""

from .cls_env import ClsEnv, MultiTaskClsEnv

# 向后兼容别名：旧代码仍以 UDAEnv / MultiTaskUDAEnv 引用
UDAEnv = ClsEnv
MultiTaskUDAEnv = MultiTaskClsEnv

__all__ = ['ClsEnv', 'MultiTaskClsEnv', 'UDAEnv', 'MultiTaskUDAEnv']
