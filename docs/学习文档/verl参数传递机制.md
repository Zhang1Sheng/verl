# verl 参数传递机制 —— 从 Shell 脚本到 Python 程序的完整链路

> 本文档通过分析 verl 仓库的源码，完整梳理了训练脚本中的参数如何层层传递，最终被 Python 程序消费的全过程。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [Shell 脚本层](#2-shell-脚本层)
3. [CLI 参数层](#3-cli-参数层)
4. [Hydra 配置加载层](#4-hydra-配置加载层)
5. [OmegaConf DictConfig 对象](#5-omegaconf-dictconfig-对象)
6. [程序内部消费配置](#6-程序内部消费配置)
7. [特殊语法与模式](#7-特殊语法与模式)
8. [端到端示例](#8-端到端示例)
9. [常见问题](#9-常见问题)

---

## 1. 整体架构概览

verl 使用 **Hydra + OmegaConf** 作为配置系统。参数从用户输入到程序消费，共经过 5 层：

```
Shell 脚本 (.sh)
    │
    │  ① Shell 变量 → 构建 key=value 参数数组
    │
    ▼
CLI 参数
    │
    │  ② python3 -m verl.trainer.main_ppo key=value key=value ...
    │
    ▼
@hydra.main() 入口
    │
    │  ③ 加载多层 YAML 默认值树（ppo_trainer.yaml → defaults → 子 YAML）
    │  ④ 合并 CLI 覆盖
    │  ⑤ 解析 ${interpolation}
    │
    ▼
OmegaConf.DictConfig 对象
    │
    │  ⑥ 程序通过三种方式消费：
    │     - 直接属性访问：config.trainer.nnodes
    │     - 转 dict：OmegaConf.to_container(config)
    │     - 转类型化 dataclass：omega_conf_to_dataclass(config.critic)
    │
    ▼
RayPPOTrainer / PPOTrainer / SFTTrainer / 各 Worker
```

---

## 2. Shell 脚本层

所有示例脚本位于 `examples/` 目录下，按算法分目录：

```
examples/
  ppo_trainer/        # PPO 算法示例
  grpo_trainer/       # GRPO 算法示例
  rloo_trainer/       # RLOO 算法示例
  remax_trainer/      # ReMax 算法示例
  ...
```

### 2.1 脚本的统一结构

每个 shell 脚本遵循相同的三段式结构（以 `examples/grpo_trainer/run_qwen3_8b_fsdp.sh` 为例）：

#### ① 用户可调节区（顶部）

```bash
#!/usr/bin/env bash
# GRPO | Qwen3-8B | FSDP training | NVIDIA GPUs or Ascend NPUs

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MACHINE=${MACHINE:-}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-}

train_batch_size=${TRAIN_BATCH_SIZE:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-}
rollout_n=${ROLLOUT_N:-5}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_grpo_${INFER_BACKEND}_fsdp}
########################### end user-adjustable ###########################
```

> **关键模式**：`小写变量名=${大写环境变量名:-默认值}`
>
> - 大写 = 用户可覆盖的 API（如 `ACTOR_LR=5e-7`）
> - 小写 = 内部使用（如 `actor_lr`）
> - `${VAR:-default}` = Bash 语法，如果环境变量未设置则使用默认值

#### ② 派生默认值区（中间）

根据 DEVICE、MACHINE 等条件自动调整参数：

```bash
########################### derived defaults ###########################
case "${DEVICE}" in
    gpu)
        actor_param_offload=False
        actor_optimizer_offload=False
        rollout_tp=${rollout_tp:-2}
        rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.6}

        case "${MACHINE}" in
            gb200)
                NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
                EXTRA+=(
                    actor_rollout_ref.rollout.enforce_eager=True
                    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
                )
                ;;
            *)
                NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
                ;;
        esac
        ;;
    npu)
        actor_param_offload=True
        actor_optimizer_offload=True
        rollout_tp=${rollout_tp:-4}
        EXTRA+=(
            actor_rollout_ref.actor.use_torch_compile=False
            actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
        )
        ;;
esac
```

#### ③ 参数数组 + 启动（底部）

```bash
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files="['$HOME/data/gsm8k/train.parquet', '$HOME/data/math/train.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.n=${rollout_n}
)

TRAINER=(
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.total_epochs=${total_epochs}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
```

### 2.2 关键设计原则

1. **所有可调参数通过大写环境变量暴露**，用户无需修改脚本即可覆盖：`ACTOR_LR=5e-7 NNODES=2 bash run_qwen3_8b_fsdp.sh`
2. **设备/后端特定的优化参数自动派生**（如 GPU vs NPU 有不同的 offload 策略）
3. **`"$@"` 透传**：用户可以在命令行追加额外参数
4. **按逻辑分组**：`DATA`、`ACTOR`、`ROLLOUT`、`REF`、`CRITIC`、`TRAINER`、`REWARD`、`EXTRA`

---

## 3. CLI 参数层

Shell 脚本最终展开为一串 `key=value` 参数传给 Python：

```bash
# 展开后等价于：
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['/home/user/data/gsm8k/train.parquet', '/home/user/data/math/train.parquet']" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=5 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.project_name=verl_grpo_gsm8k_math
```

> 这不是传统 `--arg value` 风格，而是 **Hydra/OmegaConf 原生的覆盖语法**——点号路径直接映射到 YAML 的嵌套层级。

---

## 4. Hydra 配置加载层

### 4.1 入口点

所有 PPO 系列训练的主入口在 `verl/trainer/main_ppy.py`：

```python
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """config: Hydra configuration dictionary containing training parameters."""
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)
```

`@hydra.main` 的参数：
- `config_path="config"` → 找 `verl/trainer/config/` 目录
- `config_name="ppo_trainer"` → 加载 `config/ppo_trainer.yaml`

### 4.2 入口 YAML 的 defaults 链

`verl/trainer/config/ppo_trainer.yaml` 开头：

```yaml
defaults:
  - model_engine: dp

  # 语法：<子配置名>@<挂载路径>: <文件名>
  # 含义：将 config/actor/dp_actor.yaml 加载后挂到 actor_rollout_ref.actor 路径下
  - actor@actor_rollout_ref.actor: ${model_engine}_actor    # → dp_actor.yaml
  - data@data: legacy_data                                   # → data/legacy_data.yaml
  - ref@actor_rollout_ref.ref: ${model_engine}_ref           # → dp_ref.yaml
  - rollout@actor_rollout_ref.rollout: rollout               # → rollout/rollout.yaml
  - model@actor_rollout_ref.model: hf_model                  # → model/hf_model.yaml
  - critic@critic: ${model_engine}_critic                    # → dp_critic.yaml
  - model@critic.model: hf_model                             # → model/hf_model.yaml
  - legacy_reward_impl
  - reward@reward: reward                                    # → reward/reward.yaml
  - algorithm@algorithm.rollout_correction: rollout_correction
  - distillation@distillation: distillation

  - _self_    # 当前 YAML 文件中的字段覆盖上面所有 defaults 的同名字段
```

**模型引擎切换**：`model_engine` 默认为 `dp`，如果传 `model_engine=megatron`，则所有 `${model_engine}_actor` 变成 `megatron_actor`，全树切换到 Megatron 配置。

### 4.3 递归加载的完整配置树

```
ppo_trainer.yaml
├── model_engine: dp
├── actor_rollout_ref/
│   ├── actor/               ← config/actor/dp_actor.yaml
│   │   ├── optim/           ← config/optim/fsdp.yaml
│   │   ├── fsdp_config/     ← config/engine/fsdp.yaml
│   │   ├── policy_loss/     ← actor.yaml 自带
│   │   ├── checkpoint/
│   │   ├── profiler/
│   │   ├── router_replay/
│   │   └── qat/
│   ├── ref/                 ← config/ref/dp_ref.yaml
│   ├── rollout/             ← config/rollout/rollout.yaml
│   └── model/               ← config/model/hf_model.yaml
├── data/                    ← config/data/legacy_data.yaml
├── critic/                  ← config/critic/dp_critic.yaml
│   └── model/               ← config/model/hf_model.yaml
├── algorithm/
├── trainer/
├── reward/
├── global_profiler/
├── transfer_queue/
└── ray_kwargs/
```

### 4.4 YAML 内插值（Interpolation）

YAML 中可以引用其他字段的值，语法 `${路径}` 或 `${oc.select:路径,默认值}`：

```yaml
# config/actor/actor.yaml
rollout_n: ${oc.select:actor_rollout_ref.rollout.n,1}
use_remove_padding: ${oc.select:actor_rollout_ref.model.use_remove_padding,false}
use_torch_compile: ${oc.select:actor_rollout_ref.model.use_torch_compile,true}
use_fused_kernels: ${oc.select:actor_rollout_ref.model.use_fused_kernels,false}

# ppo_trainer.yaml
default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}

# config/actor/dp_actor.yaml
# 引用全局 profiler 配置
save_path: ${oc.select:global_profiler.save_path,null}
```

`oc.select:path,default` 的语义：
- 尝试解析 `path` 对应的配置值
- 如果不存在或为 `null`，则使用 `default`
- 这是 OmegaConf 的 `oc` resolver

### 4.5 `_target_` 字段

许多配置节包含 `_target_` 字段，用于指定该配置应该被实例化为哪个 Python 类：

```yaml
# config/actor/dp_actor.yaml
_target_: verl.workers.config.FSDPActorConfig

# config/actor/actor.yaml
_target_: verl.workers.config.ActorConfig   # 基类配置
```

```yaml
# ppo_trainer.yaml
algorithm:
  _target_: verl.trainer.config.AlgoConfig
  kl_ctrl:
    _target_: verl.trainer.config.KLControlConfig
```

`_target_` 由 `omega_conf_to_dataclass()` 函数（`verl/utils/config.py`）消费：

```python
def omega_conf_to_dataclass(config, dataclass_type=None):
    if dataclass_type is None:
        # 从 _target_ 字段自动识别类型
        assert "_target_" in config
        from hydra.utils import instantiate
        return instantiate(config, _convert_="partial")
    # 或显式指定类型
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    return OmegaConf.to_object(cfg_merged)
```

---

## 5. OmegaConf DictConfig 对象

Hydra 加载和合并所有配置后，最终输出一个 `OmegaConf.DictConfig` 对象。

### 5.1 DictConfig 的特性

```python
# 它是类似嵌套字典的对象，但支持属性访问
config.trainer.nnodes           # → 8
config.trainer.get("nnodes")    # → 8（安全的 get）
config.data.train_batch_size    # → 1024

# 也可以像字典一样访问
config["trainer"]["nnodes"]

# 转普通 dict
OmegaConf.to_container(config, resolve=True)
# resolve=True 会解析所有 ${interpolation}

# 合并两个 config
OmegaConf.merge(base_cfg, override_cfg)

# 冻结/解冻
OmegaConf.set_struct(config, True)   # 禁止访问未定义字段
```

### 5.2 自定义 BaseConfig

verl 还封装了一个 `BaseConfig` 类（`verl/base_config.py`），继承自 `collections.abc.Mapping`：

```python
@dataclass
class BaseConfig(collections.abc.Mapping):
    _mutable_fields = set()
    _target_: str = ""

    def __setattr__(self, name, value):
        if name in self.__dict__ and name not in self._mutable_fields:
            raise FrozenInstanceError(f"Field '{name}' is frozen")
        super().__setattr__(name, value)

    def __getitem__(self, key):
        return getattr(self, key)

    # 支持 dict 风格的 get()
    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default
```

所有配置 dataclass（`AlgoConfig`、`KLControlConfig`、`ActorConfig`、`CriticConfig` 等）都继承自此 BaseConfig。

---

## 6. 程序内部消费配置

程序通过三种方式使用配置：

### 6.1 直接属性访问

```python
# main_ppo.py
if config.transfer_queue.enable:
    ...

ray_init_kwargs = config.ray_kwargs.get("ray_init", {})

if config.global_profiler.tool == "nsys":
    nsight_options = OmegaConf.to_container(
        config.global_profiler.global_tool_config.nsys.controller_nsight_options
    )

# TaskRunner.run()
n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
local_path = copy_to_local(config.actor_rollout_ref.model.path, ...)
```

### 6.2 转为 Python 原生容器

```python
# 传给 Ray 时需转为普通 dict
ray.init(**OmegaConf.to_container(ray_init_kwargs))

# 打印完整配置
pprint(OmegaConf.to_container(config, resolve=True))

# 用于 NSight profiling
nsight_options = OmegaConf.to_container(
    config.global_profiler.global_tool_config.nsys.controller_nsight_options
)
```

### 6.3 转为类型化 dataclass

```python
# verl/utils/config.py
from omegaconf import DictConfig, OmegaConf

def omega_conf_to_dataclass(config, dataclass_type=None):
    if not config:
        return dataclass_type() if dataclass_type else None
    if not isinstance(config, (DictConfig, dict)):
        return config
    if dataclass_type is None:
        # 通过 _target_ 自动实例化
        from hydra.utils import instantiate
        return instantiate(config, _convert_="partial")
    # 显式指定类型时，merge 后转对象
    cfg = OmegaConf.create(config)
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    return OmegaConf.to_object(cfg_merged)
```

使用示例：

```python
# 自动推断类型（需要 _target_）
algo_config = omega_conf_to_dataclass(config.algorithm)
# → 根据 config.algorithm._target_ = "verl.trainer.config.AlgoConfig"
# → 返回 AlgoConfig 实例

# 显式指定类型
critic_cfg: CriticConfig = omega_conf_to_dataclass(config.critic)
# → 返回 CriticConfig 实例，类型安全，IDE 可补全

# 用于验证
actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
actor_config.validate(n_gpus, ...)
```

---

## 7. 特殊语法与模式

### 7.1 `+` 前缀：添加不存在的字段

YAML 中没有预定义的字段，需要在 CLI 动态添加时使用 `+`：

```bash
# 脚本中：
EXTRA+=(
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
    +ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}
    +data.apply_chat_template_kwargs.reasoning_effort=${REASONING_EFFORT}
)
```

OmegaConf 中，`+` 表示 "如果不存在则新建"。

### 7.2 `~` 前缀：删除字段

```yaml
# Hydra CLI
~actor_rollout_ref.actor.some_obsolete_field
```

### 7.3 `model_engine` 配置开关

这是 verl 最核心的配置切换机制。`ppo_trainer.yaml` 中：

```yaml
defaults:
  - model_engine: dp   # ← 这是一键切换开关
  - actor@actor_rollout_ref.actor: ${model_engine}_actor
  - ref@actor_rollout_ref.ref: ${model_engine}_ref
  - critic@critic: ${model_engine}_critic
```

传 `model_engine=megatron` 时：

| 插值 | 解析结果 | 加载的文件 |
|------|----------|-----------|
| `${model_engine}_actor` | `megatron_actor` | `config/actor/megatron_actor.yaml` |
| `${model_engine}_ref` | `megatron_ref` | `config/ref/megatron_ref.yaml` |
| `${model_engine}_critic` | `megatron_critic` | `config/critic/megatron_critic.yaml` |

支持的引擎值：`dp`（FSDP）、`megatron`、`veomni`、`torchtitan`。

### 7.4 `@` 挂载语法

Hydra 的 `@` 语法将子配置挂载到指定路径：

```yaml
- actor@actor_rollout_ref.actor: dp_actor
# 等价于：加载 config/actor/dp_actor.yaml，将其内容放到 config.actor_rollout_ref.actor 下

- model@critic.model: hf_model
# 等价于：加载 config/model/hf_model.yaml，将其内容放到 config.critic.model 下
```

### 7.5 多配置文件覆盖顺序

```
CLI 覆盖（最高优先级） ↑
    │
_Self_（当前 YAML 的显式字段）
    │
每个 defaults 列表项（按出现顺序，后出现的优先级高）
    │
defaults 本身的最低优先级
```

---

## 8. 端到端示例

跟踪 `ACTOR_LR` 参数从用户输入到模型优化器的完整路径：

### Step 1: 用户在命令行设置

```bash
ACTOR_LR=5e-7 ROLLOUT_N=8 bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
```

### Step 2: Shell 脚本接收

```bash
actor_lr=${ACTOR_LR:-1e-6}   # → actor_lr=5e-7
```

### Step 3: 构建参数数组

```bash
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}   # → "actor_rollout_ref.actor.optim.lr=5e-7"
)
```

### Step 4: 传给 Python

```bash
python3 -m verl.trainer.main_ppo \
    "${ACTOR[@]}" \    # → "actor_rollout_ref.actor.optim.lr=5e-7"
    "$@"
```

### Step 5: Hydra 加载和合并

1. 加载 `ppo_trainer.yaml` → defaults → `config/actor/dp_actor.yaml` → `config/optim/fsdp.yaml`
2. `optim/lr` 默认值 `1e-6`
3. CLI 覆盖 `actor_rollout_ref.actor.optim.lr=5e-7`
4. 最终 `config.actor_rollout_ref.actor.optim.lr == 5e-7`

### Step 6: 程序使用

```python
# RayPPOTrainer 初始化优化器时
actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
# → actor_config.optim.lr == 5e-7

# 传给 torch.optim.AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=actor_config.optim.lr)
```

---

## 9. 常见问题

### Q: 如何在命令行快速覆盖参数？

```bash
# 直接在脚本命令后追加
bash run_qwen3_8b_fsdp.sh actor_rollout_ref.actor.optim.lr=1e-5

# 或使用环境变量（如果脚本定义了大写接口）
ACTOR_LR=1e-5 TOTAL_EPOCHS=30 bash run_qwen3_8b_fsdp.sh

# 组合使用
MODEL_PATH=Qwen/Qwen3-14B \
NNODES=2 NGPUS_PER_NODE=8 \
ROLLOUT_N=8 TRAIN_BATCH_SIZE=2048 \
bash run_qwen3_8b_fsdp.sh
```

### Q: 如何切换训练后端？

```bash
# 从 FSDP (dp) 切换到 Megatron
bash run_qwen3_8b_megatron.sh

# 或在任何脚本后追加
bash run_qwen3_8b_fsdp.sh model_engine=megatron
```

### Q: 如何添加 YAML 中没有的字段？

使用 `+` 前缀：

```bash
# 添加一个不存在的字段
python3 -m verl.trainer.main_ppo \
    "+new_section.new_field=some_value"
```

### Q: 配置文件的完整目录在哪？

```
verl/trainer/config/
├── ppo_trainer.yaml          # 主入口 YAML
├── _generated_*.yaml         # 自动生成的完整配置
├── actor/                    # Actor 相关配置
│   ├── actor.yaml            # Actor 基类配置
│   ├── dp_actor.yaml         # FSDP Actor 配置
│   ├── megatron_actor.yaml   # Megatron Actor 配置
│   └── ...
├── ref/                      # Reference 模型配置
├── rollout/                  # Rollout 配置
├── model/                    # 模型配置
├── data/                     # 数据配置
├── critic/                   # Critic 配置
├── reward/                   # Reward 配置
├── algorithm/                # 算法配置
├── engine/                   # 引擎配置（FSDP/Megatron 等）
├── optim/                    # 优化器配置
├── distillation/             # 蒸馏配置
└── profiler/                 # Profiler 配置
```

### Q: 用 GRPO 和 PPO 的核心参数区别是什么？

```bash
# GRPO（不需要 critic）
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001

# PPO（需要 critic）
algorithm.adv_estimator=gae
critic.model.path=...
critic.optim.lr=1e-5
```

---

> **总结**：verl 的参数传递系统基于 **Hydra + OmegaConf**，核心思想是**多层 YAML 默认值 + CLI key=value 覆盖**。Shell 脚本层提供用户友好的环境变量接口，Hydra 层负责配置的组合和合并，OmegaConf 提供灵活的运行时访问方式。这套设计使得：
>
> 1. 用户无需修改代码即可调整几乎所有参数
> 2. 不同训练后端（FSDP/Megatron/VeOmni）通过一个 `model_engine` 参数一键切换
> 3. 配置可以跨字段引用，避免重复
> 4. 参数类型安全（通过 dataclass 转换）
