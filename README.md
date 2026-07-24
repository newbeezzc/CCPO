# Meta-Schedule: 基于元学习的UDA学习率自动调整

本项目实现了论文中提出的二阶段学习率调度方法：**"损失曲线评价模型预训练 + 强化学习策略优化"**。

## 项目结构

```
meta_schedule/
├── configs/
│   └── default_config.py        # 超参数配置
├── envs/
│   └── uda_env.py               # UDA训练环境（Gym风格）
├── models/
│   ├── evaluator.py             # 一阶段评价模型加载器
│   └── policy_network.py        # Actor-Critic策略网络
├── agents/
│   └── a2c_agent.py             # A2C算法实现
├── utils/
│   ├── loss_buffer.py           # 损失序列缓冲区
│   ├── lr_utils.py              # 学习率调整工具
│   └── training_utils.py        # 训练辅助工具
├── train_stage2.py              # 二阶段主训练脚本
├── evaluate.py                  # 策略评估脚本
├── run_baselines.py             # Baseline对比脚本
└── README.md
```

## 依赖要求

```bash
pip install torch torchvision numpy
```

此外需要你原有的UDA项目依赖：
- `module.network` (ImageClassifier)
- `module.backbone` (ResNet等)
- `module.Loss` (AdversarialLoss_PDD_Double, MI)
- `utils.data_list` (ImageList)
- `utils.transforms` (ResizeImage)
- `utils.util` (ContinuousDataloader)

## 使用方法

### 1. 二阶段训练

```bash
python train_stage2.py \
    --evaluator_ckpt /path/to/stage1/checkpoint.pth \
    --data_root /path/to/OfficeHome \
    --output_dir ./outputs/stage2 \
    --gpu_id 0 \
    --num_episodes 1000 \
    --uda_epochs 30 \
    --adjust_interval 50
```

### 2. 策略评估

```bash
python evaluate.py \
    --checkpoint ./outputs/stage2/best_model.pth \
    --evaluator_ckpt /path/to/stage1/checkpoint.pth \
    --data_root /path/to/OfficeHome \
    --tasks Pr-Rw Rw-Ar Rw-Cl Rw-Pr \
    --num_runs 3
```

### 3. Baseline对比

```bash
python run_baselines.py \
    --data_root /path/to/OfficeHome \
    --tasks Ar-Cl Ar-Pr Cl-Ar Cl-Pr \
    --baselines fixed step cosine \
    --num_runs 3
```

## 核心参数说明

### 环境参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--loss_window` | 10 | 损失曲线窗口大小 W |
| `--loss_channels` | 4 | 损失通道数 d |
| `--adjust_interval` | 50 | 学习率调整间隔（每N步调整一次）|
| `--warmup_steps` | 100 | 预热步数 |
| `--action_scale` | 0.5 | 动作缩放系数 |

### A2C算法参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--gamma` | 0.99 | 折扣因子 |
| `--gae_lambda` | 0.95 | GAE参数 |
| `--policy_lr` | 3e-4 | 策略网络学习率 |
| `--value_lr` | 1e-3 | 价值网络学习率 |
| `--entropy_coef` | 0.01 | 熵正则化系数 |

### 奖励设计参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--reward_alpha` | 1.0 | 评价模型奖励权重 |
| `--reward_beta` | 0.0 | 真实准确率奖励权重 |
| `--acc_eval_interval` | 500 | 准确率评估间隔 |

## 任务划分

默认的任务划分（OfficeHome）：

**训练任务**（8个）:
- Ar-Cl, Ar-Pr, Ar-Rw
- Cl-Ar, Cl-Pr, Cl-Rw  
- Pr-Ar, Pr-Cl

**测试任务**（4个）:
- Pr-Rw, Rw-Ar, Rw-Cl, Rw-Pr

## 代码设计说明

### 1. UDAEnv (uda_env.py)

将UDA训练封装为Gym风格的RL环境：

```python
env = UDAEnv(args, evaluator_wrapper, device)
state = env.reset(task_name, source_path, target_path)

while not done:
    action = agent.select_action(state)  # a ∈ [-1, 1]
    state, reward, done, info = env.step(action)
```

**状态空间**:
- `loss_window`: [W, d] 最近W步的多通道损失
- `lr`: 当前学习率
- `progress`: 训练进度 t/T_max
- `context`: [4*d] 损失序列统计特征

**动作空间**: 连续 a ∈ [-1, 1]，映射为 `lr' = lr × exp(a × scale)`

### 2. PolicyNetwork (policy_network.py)

复用一阶段评价模型的时序编码器：

```python
actor_critic = ActorCritic(
    evaluator=stage1_evaluator,  # 复用编码器
    freeze_encoder=True          # 冻结编码器参数
)
```

### 3. A2CAgent (a2c_agent.py)

实现A2C算法，支持GAE优势估计：

```python
agent = A2CAgent(args, device)
action, log_prob, value = agent.select_action(state)
agent.store_transition(state, action, log_prob, reward, value, done)
losses = agent.update()  # Episode结束后更新
```

## 扩展建议

1. **多GPU支持**: 当前代码支持多GPU，通过 `--gpu_id 0,1,2` 指定
2. **添加新数据集**: 修改 `configs/default_config.py` 中的域名映射
3. **切换RL算法**: 可以基于 `a2c_agent.py` 实现PPO等算法
4. **调整奖励设计**: 修改 `UDAEnv._compute_reward()` 方法

## 注意事项

1. 确保一阶段评价模型的配置（W, d, embed_dim等）与二阶段一致
2. 训练时建议先在小规模设置下验证代码正确性
3. 二阶段训练计算量较大，建议使用多GPU或减少 `uda_epochs`

## 引用的外部代码

本项目的以下部分参考了开源实现：

- GAE计算: 参考 [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- A2C算法框架: 基于标准Actor-Critic实现
- 评价模型架构: 复用你提供的一阶段FreTS.py

## License

MIT License
