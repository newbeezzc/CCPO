# -*- coding: utf-8 -*-
"""
配置模块
"""

from .default_config import (
    get_default_config,
    parse_args,
    get_task_paths,
    get_num_classes,
    parse_task_seed,
    parse_task_spec,
    parse_epochs_map,
)

__all__ = [
    'get_default_config', 'parse_args', 'get_task_paths', 'get_num_classes',
    'parse_task_seed', 'parse_task_spec', 'parse_epochs_map',
]
