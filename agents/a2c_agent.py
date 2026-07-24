# -*- coding: utf-8 -*-
"""
A2C (Advantage Actor-Critic) Agent

用于学习率调度策略的强化学习算法实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_network import ActorCritic
from models.evaluator import EvaluatorModel, load_evaluator, EvaluatorModelConfig
from utils.loss_buffer import TrajectoryBuffer
from utils.training_utils import compute_gae, save_checkpoint, load_checkpoint


class A2CAgent:
    """
    A2C Agent for Learning Rate Scheduling
    
    使用Advantage Actor-Critic算法训练学习率调度策略
    """

    def __init__(self, args, device: str = 'cuda'):
        """
        Args:
            args: 配置参数
            device: 计算设备
        """
        self.args = args
        self.device = device

        # 算法超参数
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.max_grad_norm = args.max_grad_norm

        # 加载一阶段评价模型
        evaluator_config = EvaluatorModelConfig(
            seq_len=args.loss_window,
            enc_in=args.loss_channels,
            embed_size=args.embed_dim,
            hidden_size=args.hidden_dim
        )
        self.evaluator = load_evaluator(
            args.evaluator_ckpt,
            config=evaluator_config,
            device=device,
            freeze=args.freeze_encoder
        )

        # 创建Actor-Critic网络（带CRF约束的离散动作）
        self.actor_critic = ActorCritic(
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            context_dim=4 * args.loss_channels,
            num_discrete_lrs=args.num_discrete_lrs,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            lambda_crf=args.lambda_crf,
            evaluator=self.evaluator,
            freeze_encoder=args.freeze_encoder,
            share_encoder=True
        ).to(device)

        # 创建优化器（包含可学习转移矩阵）
        actor_params = list(self.actor_critic.actor.policy_net.parameters()) + \
                       list(self.actor_critic.actor.emission_layer.parameters()) + \
                       [self.actor_critic.actor.transition_matrix]  # 添加转移矩阵

        critic_params = list(self.actor_critic.critic.value_net.parameters())

        self.optimizer = optim.Adam([
            {'params': actor_params, 'lr': args.policy_lr},
            {'params': critic_params, 'lr': args.value_lr}
        ])

        # 轨迹缓冲区
        self.trajectory_buffer = TrajectoryBuffer()

        # 训练统计
        self.total_steps = 0
        self.total_episodes = 0

    def select_action(self, state: Dict, deterministic: bool = False
                      ) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        """
        根据当前状态选择动作

        Args:
            state: 状态字典 {'loss_window', 'lr', 'progress', 'context'}
            deterministic: 是否使用确定性策略

        Returns:
            action_idx: 动作索引 (int)
            action_lr: 对应的学习率值 (float)
            log_prob: 对数概率
            value: 状态价值估计
        """
        with torch.no_grad():
            # 准备输入
            loss_window = state['loss_window'].unsqueeze(0).to(self.device)  # [1, W, d]
            lr = torch.tensor([state['lr']], device=self.device)
            progress = torch.tensor([state['progress']], device=self.device)
            context = state['context'].unsqueeze(0).to(self.device)  # [1, 4*d]

            # 获取动作索引、学习率值、对数概率、价值
            action_idx, action_lr, log_prob, value = self.actor_critic.act(
                loss_window, lr, progress, context, deterministic
            )

        return action_idx.item(), float(action_lr.item()), log_prob.squeeze(), value.squeeze()

    def store_transition(self, state: Dict, action_idx: int, action_lr: float,
                         log_prob: torch.Tensor, reward: float,
                         value: torch.Tensor, done: bool):
        """存储一个transition"""
        self.trajectory_buffer.push(state, action_idx, action_lr, log_prob, reward, value, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """
        使用收集的轨迹更新策略

        Returns:
            losses: 损失字典
        """
        if len(self.trajectory_buffer) == 0:
            return {}

        # 获取批量数据
        batch = self.trajectory_buffer.get_batch()

        # 计算GAE和回报
        advantages, returns = compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones'],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # 归一化优势值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 将数据移到设备上
        action_indices = batch['action_indices'].to(self.device)  # [B] 离散动作索引
        old_log_probs = batch['log_probs'].to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # 准备状态批量
        loss_windows = torch.stack([s['loss_window'] for s in batch['states']]).to(self.device)
        lrs = torch.tensor([s['lr'] for s in batch['states']], device=self.device)
        progresses = torch.tensor([s['progress'] for s in batch['states']], device=self.device)
        contexts = torch.stack([s['context'] for s in batch['states']]).to(self.device)

        # 前向传播（传入动作索引）
        log_probs, values, entropy = self.actor_critic.evaluate(
            loss_windows, lrs, progresses, contexts, action_indices
        )
        values = values.squeeze(-1)

        # 计算损失
        # 策略损失
        policy_loss = -(log_probs * advantages).mean()

        # 价值损失
        value_loss = nn.functional.mse_loss(values, returns)

        # 熵损失（鼓励探索）
        # 注意：CRF约束会降低熵，可能需要提高entropy_coef
        entropy_loss = -entropy.mean()

        # 计算CRF转移矩阵正则化损失
        transition_reg_loss = self.actor_critic.actor.compute_transition_regularization()

        # 总损失（添加正则化项）
        total_loss = (policy_loss
                      + self.value_coef * value_loss
                      + self.entropy_coef * entropy_loss
                      + transition_reg_loss)  # 新增

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.max_grad_norm
            )

        self.optimizer.step()

        # 清空缓冲区
        self.trajectory_buffer.clear()
        self.total_episodes += 1

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'transition_reg_loss': transition_reg_loss.item(),  # 新增
            'total_loss': total_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item(),
            'mean_entropy': entropy.mean().item()
        }

    def save(self, filepath: str, extra_info: Optional[Dict] = None):
        """
        保存Agent
        
        Args:
            filepath: 保存路径
            extra_info: 额外信息
        """
        state = {
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes
        }
        if extra_info:
            state.update(extra_info)

        save_checkpoint(state, filepath)

    def load(self, filepath: str):
        """
        加载Agent
        
        Args:
            filepath: checkpoint路径
        """
        checkpoint = load_checkpoint(filepath, map_location=self.device)

        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)

        print(f"Loaded agent from {filepath}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Total episodes: {self.total_episodes}")

    def train_mode(self):
        """设置为训练模式"""
        self.actor_critic.train()
        # 但编码器保持eval模式（如果冻结）
        if self.args.freeze_encoder:
            self.evaluator.eval()

    def eval_mode(self):
        """设置为评估模式"""
        self.actor_critic.eval()


class A2CTrainer:
    """
    A2C训练器
    
    管理整个训练流程，包括episode收集、策略更新、评估等
    """

    def __init__(self, agent: A2CAgent, train_env, test_env=None, args=None):
        """
        Args:
            agent: A2C Agent
            train_env: 训练环境 (MultiTaskUDAEnv)
            test_env: 测试环境 (可选)(MultiTaskUDAEnv)
            args: 配置参数
        """
        self.agent = agent
        self.train_env = train_env
        self.test_env = test_env
        self.args = args

        # 最佳模型记录
        self.best_reward = float('-inf')
        self.best_accuracy = 0.0

    def collect_episode(self, env, task_name: str = None, deterministic: bool = False) -> Dict:
        """
        收集一个完整的episode
        """
        state = env.reset(task_name)
        done = False

        episode_reward = 0.0
        episode_length = 0

        while not done:
            # 选择动作（返回索引和学习率值）
            action_idx, action_lr, log_prob, value = self.agent.select_action(state, deterministic)

            # 执行动作（传入学习率值）
            next_state, reward, done, info = env.step(action_lr)

            # 存储transition（仅在训练时）
            if not deterministic:
                self.agent.store_transition(
                    state, action_idx, action_lr, log_prob, reward, value, done
                )

            # 更新统计
            episode_reward += reward
            episode_length += 1

            state = next_state

        # 获取episode信息
        episode_info = env.get_episode_info()
        episode_info['episode_reward'] = episode_reward
        episode_info['episode_length'] = episode_length

        return episode_info

    def train(self, num_episodes: int,
              eval_interval: int = 50,
              save_interval: int = 100,
              log_interval: int = 10) -> List[Dict]:
        """
        训练主循环
        
        Args:
            num_episodes: 总训练episode数
            eval_interval: 评估间隔
            save_interval: 保存间隔
            log_interval: 日志间隔
        
        Returns:
            training_history: 训练历史
        """
        training_history = []

        for episode in range(1, num_episodes + 1):
            print(f"\n=== Starting Episode {episode}/{num_episodes} ===")
            self.agent.train_mode()

            # 收集episode
            episode_info = self.collect_episode(self.train_env, deterministic=False)

            # 更新策略
            update_info = self.agent.update()

            # 合并信息
            episode_info.update(update_info)
            episode_info['episode'] = episode
            training_history.append(episode_info)

            # 打印日志
            if episode % log_interval == 0:
                self._print_log(episode, episode_info)

            # 评估
            if self.test_env is not None and episode % eval_interval == 0:
                eval_results = self.evaluate()
                episode_info['eval_results'] = eval_results

                # 更新最佳模型
                mean_acc = np.mean([r['final_accuracy'] for r in eval_results])
                if mean_acc > self.best_accuracy:
                    self.best_accuracy = mean_acc
                    self._save_best_model(episode)

            # 保存检查点
            if episode % save_interval == 0:
                self._save_checkpoint(episode)

        return training_history

    def evaluate(self, num_episodes_per_task: int = 1) -> List[Dict]:
        """
        在测试任务上评估策略
        
        Args:
            num_episodes_per_task: 每个任务评估的episode数
        
        Returns:
            eval_results: 评估结果列表
        """
        self.agent.eval_mode()
        eval_results = []

        for task in self.test_env.task_list:
            for _ in range(num_episodes_per_task):
                episode_info = self.collect_episode(
                    self.test_env,
                    task_name=task,
                    deterministic=True
                )
                episode_info['task'] = task
                eval_results.append(episode_info)

        # 打印评估结果
        self._print_eval_results(eval_results)

        return eval_results

    def _print_log(self, episode: int, info: Dict):
        """打印训练日志"""
        log_str = f"Episode {episode:5d}"
        log_str += f" | Task: {info.get('task', 'N/A'):>8s}"
        log_str += f" | Reward: {info.get('episode_reward', 0):.4f}"
        log_str += f" | Acc: {info.get('final_accuracy', 0):.2f}%"
        log_str += f" | P_Loss: {info.get('policy_loss', 0):.4f}"
        log_str += f" | V_Loss: {info.get('value_loss', 0):.4f}"
        print(log_str)

    def _print_eval_results(self, results: List[Dict]):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("-" * 60)

        # 按任务分组
        task_results = {}
        for r in results:
            task = r['task']
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(r['final_accuracy'])

        for task, accs in task_results.items():
            mean_acc = np.mean(accs)
            std_acc = np.std(accs) if len(accs) > 1 else 0
            print(f"  {task:>10s}: {mean_acc:.2f}% ± {std_acc:.2f}%")

        # 总体统计
        all_accs = [r['final_accuracy'] for r in results]
        print("-" * 60)
        print(f"  {'Average':>10s}: {np.mean(all_accs):.2f}% ± {np.std(all_accs):.2f}%")
        print("=" * 60 + "\n")

    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        filepath = os.path.join(
            self.args.output_dir,
            f"checkpoint_episode_{episode}.pth"
        )
        self.agent.save(filepath, {'episode': episode})

    def _save_best_model(self, episode: int):
        """保存最佳模型"""
        filepath = os.path.join(self.args.output_dir, "best_model.pth")
        self.agent.save(filepath, {
            'episode': episode,
            'best_accuracy': self.best_accuracy
        })
        print(f"New best model saved! Accuracy: {self.best_accuracy:.2f}%")
