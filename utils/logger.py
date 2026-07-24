import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime



class TrainingLogger:
    def __init__(self, save_path, e_keys=None, b_key=None):
        self.save_path = save_path
        # 初始化训练历史记录器
        if e_keys is None:
            self.training_history = {
                'epoch': [],
                'train_loss': []
            }
        else:
            self.training_history = {key: [] for key in e_keys}

        # 用于存储每个batch的详细信息
        if b_key is None:
            self.batch_history = {
                'epoch': [],
                'batch_idx': [],
                'train_loss': []
            }
        else:
            self.batch_history = {key: [] for key in b_key}

    def get_metrics(self, h_type='epoch', return_type='train'):
        if h_type == 'epoch':
            metrics = self.training_history.keys()
        elif h_type == 'batch':
            metrics = self.batch_history.keys()
        else:
            raise ValueError("h_type must be 'epoch' or 'batch'")

        if return_type == 'train':
            metrics = [m for m in metrics if m.startswith('train_')]
        elif return_type == 'test':
            metrics = [m for m in metrics if m.startswith('test_')]
        elif return_type == 'val':
            metrics = [m for m in metrics if m.startswith('val_')]
        else:
            raise ValueError("return_type must be 'train' or 'test'")
        return metrics

    def save_training_history(self, setting: str, suffix: str = ''):
        """保存训练历史数据"""
        # 创建保存目录
        # history_path = os.path.join(self.args.checkpoints, setting, 'training_history')
        history_path = os.path.join(self.save_path, setting)
        if not os.path.exists(history_path):
            os.makedirs(history_path)

        # 保存epoch级别的数据
        epoch_df = pd.DataFrame(self.training_history)
        epoch_name = f'epoch_history{suffix}.csv'
        epoch_csv_path = os.path.join(history_path, epoch_name)
        epoch_df.to_csv(epoch_csv_path, index=False)
        print(f"Epoch history saved to {epoch_csv_path}")

        # 保存batch级别的数据
        batch_df = pd.DataFrame(self.batch_history)
        batch_name = f'batch_history{suffix}.csv'
        batch_csv_path = os.path.join(history_path, batch_name)
        batch_df.to_csv(batch_csv_path, index=False)
        print(f"Batch history saved to {batch_csv_path}")

    def batch_update(self, epoch, batch_idx, train_loss, **kwargs):
        self.batch_history['epoch'].append(epoch)
        self.batch_history['batch_idx'].append(batch_idx)
        self.batch_history['train_loss'].append(train_loss)

        for k, v in kwargs.items():
            if k in self.batch_history:
                self.batch_history[k].append(v)
            else:
                self.batch_history[k] = [v]


    def epoch_update(self, epoch, train_loss, **kwargs):
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)

        for k, v in kwargs.items():
            if k in self.training_history:
                self.training_history[k].append(v)
            else:
                self.training_history[k] = [v]

    def load_from(self, setting: str, suffix: str = ''):
        """加载训练历史数据"""
        history_path = os.path.join(self.save_path, setting)
        epoch_csv_path = os.path.join(history_path, f'epoch_history{suffix}.csv')
        batch_csv_path = os.path.join(history_path, f'batch_history{suffix}.csv')

        if os.path.exists(epoch_csv_path):
            epoch_df = pd.read_csv(epoch_csv_path)
            self.training_history = epoch_df.to_dict(orient='list')
            print(f"Epoch history loaded from {epoch_csv_path}")
        else:
            print(f"No epoch history found at {epoch_csv_path}")

        if os.path.exists(batch_csv_path):
            batch_df = pd.read_csv(batch_csv_path)
            self.batch_history = batch_df.to_dict(orient='list')
            print(f"Batch history loaded from {batch_csv_path}")
        else:
            print(f"No batch history found at {batch_csv_path}")