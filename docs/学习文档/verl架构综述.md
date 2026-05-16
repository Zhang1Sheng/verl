# verl 架构综述 —— 模块、组件与执行流程

> 本文从整体视角梳理 verl 的架构设计、核心模块及其关系，帮助理解系统全貌，为深入研读子模块奠定基础。

---

## 目录

1. [仓库总览](#1-仓库总览)
2. [核心架构：HybridFlow 编程模型](#2-核心架构hybridflow-编程模型)
3. [模块分层架构](#3-模块分层架构)
4. [执行入口：Main Entry Points](#4-执行入口main-entry-points)
5. [配置系统：Hydra + OmegaConf](#5-配置系统hydra--omegaconf)
6. [分布式训练引擎](#6-分布式训练引擎)
7. [推理引擎（Rollout）](#7-推理引擎rollout)
8. [Worker 系统与 ActorRolloutRefWorker](#8-worker-系统与-actorrolloutrefworker)
9. [数据协议：DataProto](#9-数据协议dataproto)
10. [Single Controller 与 Ray 集成](#10-single-controller-与-ray-集成)
11. [RL 算法流程](#11-rl-算法流程)
12. [Checkpoint 引擎](#12-checkpoint-引擎)
13. [Reward 系统](#13-reward-系统)
14. [实用工具层](#14-实用工具层)
15. [实验性功能](#15-实验性功能)
16. [总结：各模块关系图](#16-总结各模块关系图)

---

## 1. 仓库总览

```
verl/
├── verl/                          # 核心库
│   ├── trainer/                   # 训练入口 + 算法实现
│   │   ├── main_ppo.py            # PPO/GRPO 主入口
│   │   ├── main_ppo_sync.py       # 同步 PPO 入口（TransferQueue 版）
│   │   ├── main_eval.py           # 离线评估入口
│   │   ├── main_generation_server.py  # 推理服务入口
│   │   ├── sft_trainer.py         # SFT 训练器
│   │   ├── config/                # Hydra 配置文件树
│   │   ├── ppo/                   # PPO 核心逻辑
│   │   │   ├── ray_trainer.py     # RayPPOTrainer（主训练器）
│   │   │   ├── core_algos.py      # RL 核心算法（GAE, GRPO...）
│   │   │   ├── reward.py          # Reward 函数
│   │   │   └── ...
│   │   └── config/                # Hydra YAML 配置
│   │
│   ├── workers/                   # Worker 实现
│   │   ├── engine_workers.py      # ActorRolloutRefWorker + TrainingWorker
│   │   ├── engine/                # 模型训练引擎后端
│   │   │   ├── base.py            # BaseEngine 抽象基类
│   │   │   ├── fsdp/              # FSDP 引擎
│   │   │   ├── fsdp2/             # FSDP2 引擎
│   │   │   ├── megatron/          # Megatron-LM 引擎
│   │   │   ├── veomni/            # VeOmni 引擎
│   │   │   ├── torchtitan/        # TorchTitan 引擎
│   │   │   ├── automodel/         # AutoModel 引擎
│   │   │   └── mindspeed/         # MindSpeed 引擎（Ascend NPU）
│   │   ├── rollout/               # 推理引擎后端
│   │   │   ├── base.py            # BaseRollout 抽象基类
│   │   │   ├── vllm_rollout/      # vLLM 推理
│   │   │   ├── sglang_rollout/    # SGLang 推理
│   │   │   ├── trtllm_rollout/    # TensorRT-LLM 推理
│   │   │   ├── hf_rollout.py      # HuggingFace 推理
│   │   │   ├── llm_server.py      # LLM Server 管理器
│   │   │   └── replica.py         # Server Replica 抽象
│   │   ├── reward_manager/        # Reward 管理器
│   │   └── config/                # Worker 配置 dataclass
│   │
│   ├── single_controller/         # 分布式控制器（核心抽象层）
│   │   ├── base/
│   │   │   ├── worker.py          # Worker 基类（Ray remote actor 封装）
│   │   │   ├── worker_group.py    # WorkerGroup（多 Worker 编排）
│   │   │   ├── decorator.py       # @register 调度装饰器
│   │   │   └── __init__.py
│   │   └── ray/
│   │       └── base.py            # RayWorkerGroup（Ray 实现）
│   │
│   ├── protocol.py                # DataProto（统一数据协议）
│   ├── base_config.py             # BaseConfig（配置基类）
│   ├── checkpoint_engine/         # Checkpoint 引擎插件
│   ├── experimental/              # 实验性功能
│   │   ├── agent_loop/            # 多轮 Agent 循环
│   │   ├── reward_loop/           # Reward 循环管理
│   │   ├── teacher_loop/          # Teacher 模型循环
│   │   ├── fully_async_policy/    # 完全异步策略
│   │   ├── one_step_off_policy/   # 一步 off-policy
│   │   └── separation/            # 分离式架构
│   ├── models/                    # 模型定义
│   ├── tools/                     # Tool 集成
│   ├── third_party/               # 第三方适配
│   │
│   └── utils/                     # 工具库
│       ├── config.py              # OmegaConf 工具
│       ├── dataset/               # 数据集
│       ├── profiler/              # 性能分析
│       ├── checkpoint/            # Checkpoint 管理
│       ├── fs.py                  # 文件系统工具
│       ├── reward_score/          # Reward 评分
│       ├── tracking.py            # 实验追踪（wandb/tensorboard）
│       ├── vllm/                  # vLLM 工具
│       ├── sglang/                # SGLang 工具
│       └── ...
│
├── examples/                      # 示例脚本（按算法分目录）
│   ├── grpo_trainer/              # GRPO 训练示例（~30 个脚本）
│   ├── ppo_trainer/               # PPO 训练示例
│   ├── sft/                       # SFT 示例
│   ├── data_preprocess/           # 数据预处理
│   └── ...
│
├── recipe/                        # 算法配方（submodule）
│
├── scripts/                       # 辅助脚本
│
├── docs/                          # 文档
├── tests/                         # 测试
├── docker/                        # Docker 配置
└── pyproject.toml                 # 项目配置
```

---

## 2. 核心架构：HybridFlow 编程模型

verl 的核心理念是 **HybridFlow**（EuroSys 2025 论文），其本质是一种 **混合控制器编程模型**。

### 2.1 三层抽象

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: 算法层 (Algorithm)                                 │
│  定义 RL 数据流（PPO/GRPO/DAPO/PRIME ...）                   │
│  trainer/ppo/ray_trainer.py                                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 分布式控制器 (Single Controller)                    │
│  解耦计算与数据依赖，提供 Worker 编排                          │
│  single_controller/ (base + ray)                             │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 模块化 API (Modular APIs)                          │
│  模型训练引擎 (FSDP/Megatron/VeOmni)                         │
│  + 推理引擎 (vLLM/SGLang/TRT-LLM)                            │
│  + Reward 模型                                              │
│  workers/engine/* + workers/rollout/*                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键设计目标

| 目标 | 实现方式 |
|------|----------|
| **灵活表达 RL 数据流** | 用几行 Python 代码定义 GRPO/PPO 等复杂数据流 |
| **解耦计算与数据依赖** | `@register(dispatch_mode=...)` 注解声明调度语义 |
| **设备映射灵活** | `ResourcePoolManager` 将角色映射到不同 GPU 组 |
| **无缝集成现有 LLM 框架** | Engine + Rollout 插件化架构 |

---

## 3. 模块分层架构

```
用户接口层
   ┌──────────────────────────────────────────────────────────────┐
   │  Shell 脚本 (examples/*/*.sh)                                │
   │  Hydra CLI 参数覆盖                                          │
   └──────────────────────┬───────────────────────────────────────┘
                          │
   ┌──────────────────────▼───────────────────────────────────────┐
   │  入口点 (Entry Points)                                       │
   │  verl/trainer/main_ppo.py     (PPO/GRPO 系列)                │
   │  verl/trainer/main_ppo_sync.py (同步 PPO, TransferQueue)     │
   │  verl/trainer/main_eval.py     (离线评估)                     │
   │  verl/trainer/main_generation_server.py (推理服务)           │
   │  verl/trainer/sft_trainer.py   (SFT 微调)                    │
   └──────────────────────┬───────────────────────────────────────┘
                          │
   ┌──────────────────────▼───────────────────────────────────────┐
   │  Trainer 层 (训练器)                                         │
   │  RayPPOTrainer (ppo/ray_trainer.py)                          │
   │   ├── 资源管理: ResourcePoolManager                          │
   │   ├── 训练循环: fit() → rollout → compute → update           │
   │   ├── RL 算法: core_algos.py (GAE/GRPO/REINFORCE++)          │
   │   └── 数据流编排                                              │
   └──────────────────────┬───────────────────────────────────────┘
                          │  通过 Singleton Controller 调用
   ┌──────────────────────▼───────────────────────────────────────┐
   │  Worker 层 (分布式工作进程)                                   │
   │  ActorRolloutRefWorker                                       │
   │   ├── Actor Model: TrainingWorker → Engine (FSDP/Megatron)  │
   │   ├── Ref Model:  TrainingWorker → Engine                    │
   │   └── Rollout:    BaseRollout → vLLM/SGLang/TRT-LLM         │
   │                                                              │
   │  TrainingWorker (Critic)                                     │
   │   └── Critic Model: Engine (FSDP/Megatron)                   │
   │                                                              │
   │  RewardLoop Worker / RewardModel                             │
   │                                                              │
   │  Teacher Model Worker (蒸馏)                                 │
   └──────────────────────┬───────────────────────────────────────┘
                          │  使用 Ray Remote Actor
   ┌──────────────────────▼───────────────────────────────────────┐
   │  Single Controller 层                                        │
   │  RayWorkerGroup:  Ray remote actor 的多进程编排               │
   │  @register(dispatch_mode): 自动数据分发 + 收集                │
   │  ResourcePool:  GPU 资源池管理                                │
   └──────────────────────────────────────────────────────────────┘
```

---

## 4. 执行入口：Main Entry Points

verl 提供了 5 个主要入口点（都在 `verl/trainer/` 下）：

| 入口 | 功能 | 启动方式 | 适用算法 |
|------|------|----------|----------|
| `main_ppo.py` | PPO 系列 RL 训练（异步） | `python3 -m verl.trainer.main_ppo` | PPO, GRPO, RLOO, ReMax, DAPO... |
| `main_ppo_sync.py` | 同步 PPO（TransferQueue） | `python3 -m verl.trainer.main_ppo_sync` | 同步 PPO |
| `main_eval.py` | 离线评估生成结果 | `python3 -m verl.trainer.main_eval` | 评估 |
| `main_generation_server.py` | 启动推理服务 | `python3 -m verl.trainer.main_generation_server` | 推理 |
| `sft_trainer.py` | 监督微调 | `torchrun` | SFT |

### 4.1 PPO 入口工作流（`main_ppy.py`）

```python
@hydra.main(config_path="config", config_name="ppo_trainer")
def main(config):
    auto_set_device(config)
    run_ppo(config)          # 初始化 Ray → 启动 TaskRunner → RayPPOTrainer

class TaskRunner:
    def run(self, config):
        # 1. 注册 Worker 角色（Actor/Rollout/Ref/Critic/Reward）
        self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # 2. 初始化资源池
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # 3. 创建数据集
        train_dataset = create_rl_dataset(...)

        # 4. 创建 RayPPOTrainer
        trainer = RayPPOTrainer(
            config=config,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ...
        )
        trainer.init_workers()    # 启动所有 Worker
        trainer.fit()             # 开始训练循环
```

---

## 5. 配置系统：Hydra + OmegaConf

详见 [verl参数传递机制.md](verl参数传递机制.md)，此处仅概述：

- **主 YAML**: `verl/trainer/config/ppo_trainer.yaml`
- **引擎切换**: `model_engine=dp|megatron|veomni|torchtitan`
- **子配置目录**: `actor/`, `ref/`, `rollout/`, `model/`, `data/`, `critic/`, `reward/`, `engine/`, `optim/`
- **配置消费**: 三种方式——`config.key` 属性访问、`OmegaConf.to_container()` 转 dict、`omega_conf_to_dataclass()` 转 dataclass

---

## 6. 分布式训练引擎

`verl/workers/engine/` 是模型训练的抽象层，定义 `BaseEngine` 接口，支持多种后端：

```
BaseEngine (verl/workers/engine/base.py)
├── FSDP Engine        (fsdp/)    # PyTorch FSDP (fully sharded data parallel)
├── FSDP2 Engine       (fsdp2/)   # 新版 FSDP（推荐）
├── Megatron Engine    (megatron/)# NVIDIA Megatron-LM（支持 TP/PP/EP/CP）
├── VeOmni Engine      (veomni/)  # ByteDance VeOmni
├── TorchTitan Engine  (torchtitan/)# PyTorch TorchTitan
├── AutoModel Engine   (automodel/)# HuggingFace AutoModel
└── MindSpeed Engine   (mindspeed/)# Ascend NPU MindSpeed
```

### 6.1 BaseEngine 核心接口

```python
class BaseEngine:
    def initialize(self)           # 初始化模型、优化器、LR scheduler
    def train_mode(self, **kwargs)  # 切换到训练模式
    def eval_mode(self, **kwargs)   # 切换到评估模式
    def train_batch(self, data, loss_function)  # 训练一个 batch
    def infer_batch(self, data, loss_function)  # 推理一个 batch
    def save_checkpoint(...)
    def load_checkpoint(...)
    def get_data_parallel_rank()
    def get_data_parallel_group()
    @property
    def is_param_offload_enabled(self)
```

### 6.2 Engine 注册机制

通过 `EngineRegistry` 实现插件化注册：

```python
# Engine 通过 EngineRegistry 按 backend 名称查找实现类
self.engine: BaseEngine = EngineRegistry.new(
    model_type=self.config.model_type,
    backend=self.engine_config.strategy,  # "fsdp" / "megatron" / "veomni"
    model_config=self.model_config,
    engine_config=self.engine_config,
    optimizer_config=self.optimizer_config,
)
```

---

## 7. 推理引擎（Rollout）

`verl/workers/rollout/` 是推理/生成的抽象层：

```
BaseRollout (verl/workers/rollout/base.py)
├── vLLM Rollout   (vllm_rollout/)   # 基于 vLLM
├── SGLang Rollout (sglang_rollout/) # 基于 SGLang
├── TRT-LLM Rollout(trtllm_rollout/)# 基于 TensorRT-LLM
└── HF Rollout     (hf_rollout.py)   # 基于 HuggingFace transformers
```

### 7.1 Rollout 功能

```python
class BaseRollout:
    def generate_sequences(self, prompts: DataProto) -> DataProto
    #        └─ 接收 prompt，生成 response + log_probs
    def compute_log_prob(self, data: DataProto) -> DataProto
    #        └─ 计算给定序列的 log_probs
```

### 7.2 LLM Server 管理器

`llm_server.py` 提供了推理服务器的生命周期管理：

```
LLMServerManager
├── create()          # 创建 rollout 推理服务
├── get_client()      # 获取服务客户端
├── update_weights()  # 从训练引擎同步权重到推理引擎（HybridEngine）
└── 管理多个 Replica 实例
```

---

## 8. Worker 系统与 ActorRolloutRefWorker

### 8.1 Worker 类型

| Worker | 类 | 功能 |
|--------|-----|------|
| **ActorRolloutRefWorker** | `engine_workers.py` | 混合 Worker，内聚 Actor + Rollout + Ref 三种角色 |
| **TrainingWorker** | `engine_workers.py` | 通用训练 Worker（用于 Critic 或其他纯训练模型） |
| **RewardLoop Workers** | `experimental/reward_loop/` | Reward 模型 Worker |

### 8.2 ActorRolloutRefWorker —— 核心混合 Worker

这是 verl 最核心的 Worker 类，在同一进程中管理三种角色：

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self, config, role, distillation_config):
        # role 可以是: "actor" | "rollout" | "ref" |
        #            "actor_rollout" | "actor_rollout_ref"
        self.actor: TrainingWorker = None      # Actor 模型
        self.ref: TrainingWorker = None        # Ref 模型
        self.rollout: BaseRollout = None       # Rollout 引擎

    def init_model(self):
        # 1. 初始化 Ref Model（如果 role 包含 "ref"）
        ref_config = omega_conf_to_dataclass(self.config.ref)
        self.ref = TrainingWorker(config=ref_training_config)
        self.ref.reset()

        # 2. 初始化 Actor Model（如果 role 包含 "actor"）
        actor_config = omega_conf_to_dataclass(self.config.actor)
        self.actor = TrainingWorker(config=actor_training_config)
        self.actor.reset()
        self.actor.set_loss_fn(self.loss_fn)

        # 3. 初始化 Rollout（如果 role 包含 "rollout"）
        rollout_config = omega_conf_to_dataclass(self.config.rollout)
        rollout_cls = get_rollout_class(rollout_config.name)  # vllm/sglang/trtllm
        self.rollout = rollout_cls(config=rollout_config, ...)
```

### 8.3 角色映射与 GPU 分配

```
ResourcePoolManager
    │
    ├── "global_pool"        [所有 GPU，如 8 GPUs]
    │     ├── ActorRolloutRef  → 使用全部 GPU（TP+PP 划分）
    │     ├── Critic            → 与 Actor 共享 GPU（colocated）
    │     └── Ref Model         → 与 Actor 共享 GPU
    │
    └── "reward_pool"        [独立 GPU 组，如 2 GPUs]
          └── Reward Model     → 在独立 GPU 上运行
```

---

## 9. 数据协议：DataProto

`verl/protocol.py` 定义了 verl 内部各模块之间的统一数据交换协议。

### 9.1 DataProto 结构

```python
@dataclass
class DataProto:
    batch: TensorDict                # PyTorch Tensor 数据（可 GPU 驻留）
    non_tensor_batch: dict           # NumPy 非张量数据（uid, data_source 等）
    meta_info: dict                  # 元信息（metrics, 配置等）
```

- **`batch`**: 使用 `torch.TensorDict`，包含 `input_ids`, `attention_mask`, `responses`, `old_log_probs`, `advantages`, `values` 等
- **`non_tensor_batch`**: `numpy.ndarray`，包含 `uid`, `data_source`, `reward_model` 等非张量数据
- **`meta_info`**: 通用字典，携带 metrics、extra_info 等

### 9.2 核心能力

```python
class DataProto:
    @classmethod
    def from_dict(cls, tensors, non_tensors, meta_info)  # 从 dict 创建
    def chunk(self, chunks) -> list[DataProto]            # 沿 batch dim 切分
    def repeat(self, repeat_times) -> DataProto           # 重复数据
    def concat(data: list[DataProto]) -> DataProto        # 拼接多个 DataProto
    def select(self, batch_keys, ...) -> DataProto        # 选择子集
    def to(self, device) -> DataProto                     # 移到指定设备
    def make_iterator(self, mini_batch_size, epochs)      # 创建迭代器（用于 PPO minibatch）
    def get_data_info(self) -> str                        # 打印数据摘要
```

### 9.3 DataProtoFuture —— 异步数据引用

```python
@dataclass
class DataProtoFuture:
    collect_fn: Callable               # 从多个 future 收集数据的函数
    futures: list[ray.ObjectRef]       # Ray 远程对象引用列表
    dispatch_fn: Callable              # 分发函数

    def get(self):                     # 实际获取数据（触发 ray.get）
```

DataProtoFuture 使得 **driver 进程不需要实际拉取数据**，只在 final 消费时才执行 `ray.get`，实现异步流水线。

---

## 10. Single Controller 与 Ray 集成

`verl/single_controller/` 是 verl 的分布式控制核心，提供了 Worker 远程调用、数据分发和收集的抽象。

### 10.1 三层结构

```
single_controller/
├── base/
│   ├── worker.py           # Worker 基类（封装 Ray remote actor）
│   ├── worker_group.py     # WorkerGroup（多 Worker 编排 + @register 注解引擎）
│   └── decorator.py        # @register 调度装饰器
│
└── ray/
    └── base.py             # RayWorkerGroup（Ray 驱动实现）
```

### 10.2 @register 装饰器 —— 调度语义

这是 verl 最核心的抽象之一，通过在方法上标注注解，声明该方法在分布式环境下的调度方式：

```python
from verl.single_controller.base.decorator import Dispatch, register

# 已定义的调度模式：
Dispatch.ONE_TO_ALL     # 1 个 Worker 执行，结果广播给所有 Worker
Dispatch.ALL_TO_ALL     # 每个 Worker 独立执行，结果收集合并
Dispatch.RANK_ZERO      # 仅在 rank 0 上执行
Dispatch.DP_COMPUTE     # 数据并行计算（切分数据 → 各自计算 → 收集结果）
Dispatch.DP_COMPUTE_PROTO  # DP_COMPUTE + DataProto 分发
Dispatch.DP_COMPUTE_PROTO_WITH_FUNC  # DP_COMPUTE_PROTO + function
Dispatch.DP_COMPUTE_METRIC  # 仅收集 metrics
Dispatch.DIRECT_ROLLOUT_METHOD  # vLLM 直接分发
```

### 10.3 执行模式（Execute）

```python
Execute.ALL        # 所有 rank 都执行
Execute.RANK_ZERO  # 仅 rank 0 执行
```

### 10.4 WorkerGroup 远程调用机制

```python
class WorkerGroup:
    # 通过 __getattr__ 将方法调用动态分发到所有 Worker
    # 每个被 @register 注解的方法会生成对应的 Functor

    def _init_dispatch_info(self):
        # 根据 @register(dispatch_mode=...) 自动选择：
        # - dispatch_fn:  如何将参数分发给各个 Worker
        # - collect_fn:   如何从各个 Worker 收集结果
        # - execute_fn:   每个 Worker 如何执行

    def __getattr__(self, method_name):
        # 返回一个 Functor 对象
        # 调用时自动执行: dispatch → execute → collect
```

```python
# 例如：
actor_rollout_wg.generate_sequences(prompts)
# 实际流程：
# 1. dispatch: prompts.chunk(world_size) → 分片给每个 Worker
# 2. execute:  每个 Worker 的 generate_sequences() 远程执行
# 3. collect:  DataProto.concat() 收集所有结果
```

### 10.5 ResourcePool —— GPU 资源管理

```python
class ResourcePoolManager:
    def __init__(self, resource_pool_spec, mapping):
        # resource_pool_spec = {
        #     "global_pool": [8, 8],     # 2 nodes × 8 GPUs
        #     "reward_pool": [2, 2],     # 2 nodes × 2 GPUs
        # }
        # mapping = {
        #     Role.ActorRollout: "global_pool",
        #     Role.Critic: "global_pool",
        #     Role.RewardModel: "reward_pool",
        # }

    def create_resource_pool(self):
        # 为每个 pool 创建 Ray PlacementGroup
        # 确保 GPU 亲和性调度

    def get_resource_pool(self, role):
        # 根据 role 返回对应的 ResourcePool
```

---

## 11. RL 算法流程

以一个典型的 GRPO 训练步骤为例，展示各模块如何协作：

### 11.1 单步训练流程（RayPPOTrainer.fit）

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 从 DataLoader 取一个 batch 的 prompt                         │
│     train_dataloader → batch: TensorDict                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  2. Rollout: ActorRolloutRefWorker.generate_sequences()          │
│     • 通过 vLLM/SGLang 为每个 prompt 生成 n 个 response          │
│     • 返回 DataProto: {prompts, responses, old_log_probs, ...}  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  3. Compute Ref Log Prob: ref.compute_log_prob()                │
│     • 用 Ref Policy 计算 response 的 log_probs（用于 KL 惩罚）    │
│     • 返回 DataProto: {ref_log_prob, ...}                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  4. Reward: reward_loop_manager / reward_fn                     │
│     • 函数式 Reward 或模型 Reward                                │
│     • 计算每个 response 的 reward score                          │
│     • 返回 DataProto: {token_level_scores, ...}                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  5. Advantage: compute_advantage()                              │
│     • GRPO: 按 uid 分组，归一化 group reward 作为 advantage      │
│     • GAE: 用 critic values 计算 GAE                            │
│     • 返回 DataProto: {advantages, returns, ...}                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  6. Actor Update: actor.train_mini_batch()                      │
│     • 多个 epoch 的 minibatch 训练                               │
│     • PPO loss = clip(ratio * advantage) + kl_loss + entropy    │
│     • 返回 metrics                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  7. Critic Update: critic.train_mini_batch()                    │
│     • 仅在 PPO 模式下（GRPO 不需要 Critic）                      │
│     • Value loss = MSE(values, returns)                         │
│     • 返回 metrics                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  8. Sync weights to Rollout (HybridEngine)                      │
│     • 将 Actor 更新后的权重同步到 Rollout 推理引擎               │
│     • 参考：3D-HybridEngine 论文                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 支持的 RL 算法

| 算法 | `algorithm.adv_estimator` | 需要 Critic | 需要 Ref |
|------|--------------------------|-------------|----------|
| **PPO** | `gae` | ✅ | 可选 |
| **GRPO** | `grpo` | ❌ | ✅ |
| **RLOO** | `rloo` | ❌ | 可选 |
| **ReMax** | `remax` | ❌ | 可选 |
| **REINFORCE++** | `reinforce_plus_plus` | ❌ | 可选 |
| **DAPO** | `grpo` + 定制配置 | ❌ | ✅ |
| **PRIME** | 定制 | ✅ | ✅ |
| **DrGRPO** | `grpo` + DrGRPO 配置 | ❌ | ✅ |
| **GDPO** | `gdpo` | ❌ | 可选 |

### 11.3 RayPPOTrainer 的 fit() 核心循环

```python
class RayPPOTrainer:
    def fit(self):
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # 1. rollout
                prompts = self._build_prompts(batch_dict)
                self.actor_rollout_wg.generate_sequences(prompts)

                # 2. compute ref log_prob
                if self.use_reference_policy:
                    ref_data = self.ref_policy_wg.compute_log_prob(data)

                # 3. compute reward
                data = self.reward_loop_manager.compute_reward(data)

                # 4. compute advantage
                data = compute_advantage(data, adv_estimator, gamma, lam)

                # 5. update actor (multiple PPO epochs, mini-batches)
                for mini_batch in data.make_iterator(mini_batch_size, ppo_epochs):
                    metrics = self.actor_rollout_wg.train_mini_batch(mini_batch)

                # 6. update critic (if PPO)
                if self.use_critic:
                    for mini_batch in data.make_iterator(...):
                        metrics = self.critic_wg.train_mini_batch(mini_batch)

                # 7. sync weights to rollout engine
                self.actor_rollout_wg.update_weights()

                # 8. checkpoint & validation
                if should_save:
                    self._save_checkpoint()
                if should_val:
                    self._validate()
```

---

## 12. Checkpoint 引擎

`verl/checkpoint_engine/` 提供了可插拔的 checkpoint 存储后端：

```
CheckpointEngine (base.py)
├── NCCL CheckpointEngine   (nccl_checkpoint_engine.py)   # 基于 NCCL
├── HCCL CheckpointEngine   (hccl_checkpoint_engine.py)   # 基于 HCCL（Ascend NPU）
├── Mooncake CheckpointEngine (mooncake_checkpoint_engine.py) # 基于 Mooncake
├── Kimi CheckpointEngine   (kimi_checkpoint_engine.py)   # 基于 Kimi 格式
└── NIXL CheckpointEngine   (nixl_checkpoint_engine.py)   # 基于 NIXL
```

CheckpointEngineManager 统管所有引擎的 checkpoint：

```python
class CheckpointEngineManager:
    def __init__(self, config, trainer, replicas):
        # trainer: ActorRolloutRefWorker（AdamW 状态）
        # replicas: rollout 引擎的 replica 列表

    def sleep_replicas(self):
        # 暂停 rollout 推理引擎（等待权重同步）

    def update_weights(self):
        # 从训练引擎更新权重到推理引擎
```

---

## 13. Reward 系统

### 13.1 Reward 类型

| 类型 | 机制 | 配置 |
|------|------|------|
| **函数式 Reward** | Python 函数计算（如 math 验证） | `reward.reward_manager.name=dapo` |
| **模型 Reward** | 独立 Reward Model 推理 | `reward.reward_model.enable=True` |
| **Rule-based** | 规则判断（如格式检查） | 自定义 `reward_fn` |
| **PRM** | Process Reward Model | 支持 |

### 13.2 Reward 流程

```
RewardLoopManager (experimental/reward_loop/)
├── 函数式 Reward (function_reward):
│   Actor 生成 response → Python reward_fn(data_source, response, ground_truth) → score
│
└── 模型 Reward (model_reward):
│   Actor 生成 response → Reward Model 推理 → score
│   （Reward Model 可在独立 GPU pool 上运行）
│
└── 混合模式:
│   function_reward + model_reward 可同时启用
```

---

## 14. 实用工具层

`verl/utils/` 包含丰富的工具模块：

| 模块 | 功能 |
|------|------|
| `config.py` | OmegaConf ↔ dataclass 互转和验证 |
| `dataset/` | RL 数据集实现（RLDataset, collate_fn） |
| `profiler/` | 性能分析（nsys/npu/torch/torch_memory） |
| `checkpoint/` | Checkpoint 发现、管理、分布式存储 |
| `fs.py` | 文件系统工具（HDFS 读取、本地缓存） |
| `tracking.py` | 实验追踪（wandb / swanlab / tensorboard / mlflow）|
| `reward_score/` | 预置 Reward 评分函数 |
| `tensordict_utils.py` | TensorDict 工具函数 |
| `torch_functional.py` | torch 功能函数（masked_mean, allgather 等）|
| `seqlen_balancing.py` | 序列长度平衡（提高吞吐）|
| `distributed.py` | 分布式初始化工具 |
| `device.py` | 设备检测（cuda/npu）|
| `vllm/` | vLLM 集成工具 |
| `sglang/` | SGLang 集成工具 |
| `trtllm/` | TensorRT-LLM 集成工具 |
| `megatron/` | Megatron 工具 |
| `kernel/` | 自定义 CUDA kernel |
| `qat/` | 量化感知训练工具 |
| `modelopt/` | NVIDIA ModelOpt 集成 |

---

## 15. 实验性功能

`verl/experimental/` 存放正在开发或尚未合入主库的功能：

| 模块 | 功能 | 状态 |
|------|------|------|
| `agent_loop/` | 多轮 Agent 循环（工具调用、多轮交互） | 开发中 |
| `reward_loop/` | Reward 循环管理 | 开发中 |
| `teacher_loop/` | Teacher 模型循环（蒸馏） | 开发中 |
| `fully_async_policy/` | 完全异步策略更新 | 实验性 |
| `one_step_off_policy/` | 一步 off-policy 更新 | 实验性 |
| `separation/` | 分离式架构（训练和推理在不同节点） | 研究 |

此外，`recipe/`（作为 submodule 管理）包含完整的算法配方实现，如 DAPO、PRIME、DrGRPO、ReTool 等。

---

## 16. 总结：各模块关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       用户接口                                            │
│   Shell 脚本 (examples/*/*.sh)  →  Hydra CLI (key=value 覆盖)           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      入口点 (Entry Points)                               │
│                                                                         │
│   main_ppo.py  main_ppo_sync.py  sft_trainer.py  main_eval.py          │
│                                                                         │
│   ┌─ @hydra.main ────────────────────────────────────────────────────┐  │
│   │  config = OmegaConf.merge(YAML_defaults, CLI_overrides)          │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  config 对象
┌────────────────────────────────▼────────────────────────────────────────┐
│                      Trainer 层                                          │
│                                                                         │
│   RayPPOTrainer (ppo/ray_trainer.py)                                    │
│     ├── fit() → 训练循环                                                 │
│     ├── core_algos.py → RL 算法（GAE, GRPO, REINFORCE++）               │
│     ├── reward.py → Reward 计算                                          │
│     └── 管理 DataLoader、Checkpoint、WandB 日志                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  @register 远程调用
┌────────────────────────────────▼────────────────────────────────────────┐
│                    Single Controller 层                                   │
│                                                                         │
│   RayWorkerGroup                                                        │
│     ├── dispatch: 数据分片 → 各 Worker                                   │
│     ├── execute: 远程执行 @register 方法                                  │
│     └── collect: 结果合并 → DataProto                                   │
│                                                                         │
│   ResourcePoolManager                                                   │
│     └── 角色 → GPU 资源池的映射管理                                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────────┐
        │                        │                            │
┌───────▼────────┐   ┌───────────▼────────┐   ┌──────────────▼──────────┐
│ ActorRolloutRef  │   │ TrainingWorker    │   │ RewardLoop Workers      │
│ (engine_worker) │   │ (Critic, etc.)    │   │ (reward_loop/)          │
│                 │   │                   │   │                         │
│ ┌─ Actor ─────┐ │   │ ┌─ Engine ──────┐ │   │ ┌─ Reward Model ──────┐ │
│ │ FSDP        │ │   │ │ FSDP/Megatron  │ │   │ │ vLLM/SGLang        │ │
│ │ Megatron    │ │   │ │ VeOmni/...     │ │   │ │ Python reward_fn   │ │
│ │ VeOmni      │ │   │ └───────────────┘ │   │ └─────────────────────┘ │
│ └─────────────┘ │   └───────────────────┘   └──────────────────────────┘
│ ┌─ Rollout ───┐ │
│ │ vLLM        │ │
│ │ SGLang      │ │
│ │ TRT-LLM     │ │
│ └─────────────┘ │
│ ┌─ Ref ───────┐ │
│ │ Engine      │ │
│ └─────────────┘ │
└─────────────────┘
```

### 关键关系总结

| 关系 | 描述 |
|------|------|
| **入口 → Trainer** | 入口点解析配置并创建 Trainer 实例 |
| **Trainer → Single Controller** | Trainer 通过 `@register` 方法远程调用 Worker |
| **Single Controller → Worker** | `WorkerGroup.__getattr__` 自动 dispatch + collect |
| **Worker → Engine** | `TrainingWorker` 封装 `BaseEngine`（FSDP/Megatron/...）|
| **Worker → Rollout** | `ActorRolloutRefWorker` 封装 `BaseRollout`（vLLM/SGLang/...）|
| **Worker → Worker** | 同一进程内 ActorRolloutRefWorker 管理 Actor + Rollout + Ref 三者 |
| **数据流** | 所有模块间通过 `DataProto` 交换数据（TensorDict + numpy + meta）|
| **配置流** | 所有模块共享同一个经过 Hydra 合并的 `OmegaConf.DictConfig` |

---

> **下一步学习建议**：
>
> 1. **入门**：看 1-2 个示例脚本（如 `examples/grpo_trainer/run_qwen3_8b_fsdp.sh`），跟踪参数到代码的路径
> 2. **核心训练循环**：阅读 `ppo/ray_trainer.py` 的 `fit()` 方法
> 3. **分布式抽象**：阅读 `single_controller/base/decorator.py` 的 `@register` 和 dispatch 机制
> 4. **Worker 实现**：阅读 `engine_workers.py` 的 `ActorRolloutRefWorker.init_model()`
> 5. **RL 算法**：阅读 `ppo/core_algos.py` 中的 `compute_grpo_outcome_advantage()` 或 `compute_gae_advantage_return()`
> 6. **数据协议**：阅读 `protocol.py` 的 `DataProto` 完整实现
> 7. **具体引擎**：选一个后端深入，如 `workers/engine/fsdp/` 或 `workers/engine/megatron/`
> 8. **具体 Rollout**：选一个后端深入，如 `workers/rollout/vllm_rollout/` 或 `workers/rollout/sglang_rollout/`
