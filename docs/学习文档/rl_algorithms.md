# PPO / GRPO / DPO / DAPO 算法详解

> 本文档围绕 verl 代码库，详细解释 PPO、GRPO、DPO、DAPO 四种算法的技术核心、
> 损失函数、优势估计、以及在 verl 中的具体实现模块。
> 基于 verl 代码库 (verl-project/verl) 分析。

---

## 目录

1. [概述：四种算法的关系](#1-概述四种算法的关系)
2. [PPO (Proximal Policy Optimization)](#2-ppo-proximal-policy-optimization)
3. [GRPO (Group Relative Policy Optimization)](#3-grpo-group-relative-policy-optimization)
4. [DPO (Direct Preference Optimization)](#4-dpo-direct-preference-optimization)
5. [DAPO (Decoupled-clip Policy Optimization)](#5-dapo-decoupled-clip-policy-optimization)
6. [算法对比总结](#6-算法对比总结)
7. [涉及模块清单](#7-涉及模块清单)

---

## 1. 概述：四种算法的关系

```
             PPO（Critic + GAE + Clip）
            /         |           \
      GRPO（无Critic） DPO（离线偏好） DAPO（无Critic+Clip优化）
      组归一化优势       对比学习        Token级过滤
```

| 算法 | 是否需要 Critic (Value Model) | 是否需要 Ref Policy | 优势估计方式 | 典型场景 |
|---|---|---|---|---|
| **PPO** | ✅ 是 | ✅ 是 | GAE (广义优势估计) | 标准的在线 RLHF |
| **GRPO** | ❌ 否 | ✅ 是 | 组内归一化 | 数学推理 (DeepSeek-R1) |
| **DPO** | ❌ 否 | ✅ 是 (冻结) | 偏好对直接对比 | 离线偏好对齐 |
| **DAPO** | ❌ 否 | ❌ 否 (KL=0) | 组内归一化 + 过滤 | 纯 RL 数学推理 |

---

## 2. PPO (Proximal Policy Optimization)

### 2.1 算法原理

PPO 是 RLHF 中最广泛使用的在线策略优化算法。verl 的实现基于 HuggingFace TRL 库，核心思想是在策略更新时加入 **clip 约束**，防止单步更新过大导致训练崩溃。

### 2.2 总体流程 (verl 实现)

```
1. Rollout: Actor 生成 responses → 获得 rewards
2. Ref Model: 计算 ref_log_prob（KL 约束）
3. Critic Model: 计算 values（状态价值）
4. GAE: 计算 advantages 和 returns（优势估计）
5. Policy Update: Actor 更新（clip 后的策略梯度）
6. Value Update: Critic 更新（MSE + clip）
```

### 2.3 优势估计 — GAE (Generalized Advantage Estimation)

文件：`verl/trainer/ppo/core_algos.py`，`compute_gae_advantage_return()`

```python
# 注册为 "gae"
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam):
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam             # GAE 核心递推
            nextvalues = values[:, t] * mask + (1 - mask) * nextvalues  # EOS 后截断
            lastgaelam = lastgaelam_ * mask + (1 - mask) * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)  # 白化
```

**核心公式**（按时间逆向递推）：

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)       # TD-error
A_t^{GAE(γ,λ)} = δ_t + (γλ)δ_{t+1} + ...   # GAE 递推
```

verl 中 `gamma`（折扣因子）和 `lam`（GAE λ 参数）通过算法配置传入。

### 2.4 Policy Loss — 带 Clip 的策略梯度

文件：`verl/trainer/ppo/core_algos.py`，`compute_policy_loss_vanilla()`（注册为 `"vanilla"`）

```python
ratio = exp(log_prob - old_log_prob)           # importance sampling ratio
pg_losses1 = -advantages * ratio                # 无 clip 的目标
pg_losses2 = -advantages * clip(ratio, 1-ε_low, 1+ε_high)  # clip 后的目标
pg_loss = max(pg_losses1, pg_losses2)          # PPO 核心：取最大值（更保守）
```

**核心公式**：

```
L^{CLIP}(θ) = -E_t[ min( r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t ) ]
```

其中 `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`。

verl 额外支持 **Dual-Clip PPO**（见论文 https://arxiv.org/pdf/1912.09729）：

```python
pg_losses3 = -advantages * clip_ratio_c         # 第二层 clip
pg_loss = where(advantages < 0, min(pg_losses2, pg_losses3), pg_losses1)
# 对负优势额外 clip，防止过大的负更新
```

### 2.5 Value Loss

文件：`verl/workers/utils/losses.py`，`value_loss()`

```python
vf_loss, vf_clipfrac = compute_value_loss(
    vpreds=vpreds, values=values, returns=returns,
    response_mask=response_mask,
    cliprange_value=config.cliprange_value,  # 对 value 也做 clip
)
```

Value 也使用了 clip 技巧：`clipped_v = values + clip(vpreds - values, -ε_v, ε_v)`，取 MSE 的较大值。

### 2.6 KL 惩罚

verl 支持两种 KL 控制（文件 `verl/trainer/ppo/core_algos.py`）：

| 类型 | 控制器 | 行为 |
|---|---|---|
| `fixed` | `FixedKLController` | 固定 KL 系数 `kl_coef` |
| `adaptive` | `AdaptiveKLController` | 根据当前 KL 动态调整系数（论文 https://arxiv.org/pdf/1909.08593.pdf） |

KL 惩罚有两种应用方式：
1. **`use_kl_in_reward=True`**：KL 作为 reward 的惩罚项，在 advantage 计算前扣除
2. **`use_kl_loss=True`**：KL 在 loss 中作为额外的正则项加入 policy loss

### 2.7 配置参数

| 参数 | 说明 |
|---|---|
| `actor.clip_ratio` (默认 0.2) | PPO clip ε |
| `actor.clip_ratio_low` / `clip_ratio_high` | 非对称 clip 范围 |
| `critic.cliprange_value` | Value clip ε |
| `algorithm.kl_ctrl.type` | `"fixed"` 或 `"adaptive"` |
| `algorithm.kl_ctrl.kl_coef` | KL 惩罚系数 |
| `algorithm.use_kl_in_reward` | 是否在 reward 中应用 KL |
| `algorithm.gamma` | GAE 折扣因子 |
| `algorithm.lam` | GAE λ 参数 |

---

## 3. GRPO (Group Relative Policy Optimization)

### 3.1 算法原理

GRPO 由 DeepSeek-R1 论文提出（https://arxiv.org/abs/2501.12948），**完全不需要 Critic 模型**。它对每个 prompt 生成一组 responses，用**组内相对分数**作为优势。

```
                   ┌── Response 1 → Score 1
Prompt → Group n  ── Response 2 → Score 2  →  优势 = (Score_i - μ_group) / σ_group
                   └── Response n → Score n
```

### 3.2 优势估计 — 组内归一化

文件：`verl/trainer/ppo/core_algos.py`，`compute_grpo_outcome_advantage()`（注册为 `"grpo"`）

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)
    
    # 按 prompt index 分组
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    # 计算每组 mean 和 std
    for idx in id2score:
        id2mean[idx] = mean(scores_tensor)
        id2std[idx] = std(scores_tensor)
    
    # 组内归一化
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + ε)
```

**核心公式**：

```
对于 prompt p 的第 i 个 response：
  A_i = (r_i - μ_g) / σ_g

其中：
  μ_g = mean({r_1, r_2, ..., r_n})   — 当前组内平均分
  σ_g = std({r_1, r_2, ..., r_n})    — 当前组内标准差
```

关键变体（verl 也支持）：
- **`norm_adv_by_std_in_grpo=False`**：仅减均值，不除标准差（Dr.GRPO 方式，见 https://arxiv.org/abs/2503.20783）
- **GRPO Vectorized**（`compute_grpo_vectorized_outcome_advantage`）：向量化版本，使用 `group_mean_std` 一步计算

### 3.3 GRPO 与 PPO 的关键区别

| 维度 | PPO | GRPO |
|---|---|---|
| Critic 模型 | ✅ 需要 | ❌ 不需要 |
| 优势函数 | GAE: 基于 Value 的 TD-error | 组内分数归一化 |
| Reward 信号 | 需要**密集**（每 token 奖励）或稀疏 | 仅需**稀疏**（结果信号） |
| Rollout n | 通常 n=1 | 通常 n≥8（每组多个响应） |
| 内存占用 | 3 个模型（Actor+Ref+Critic） | 2 个模型（Actor+Ref） |

### 3.4 GRPO 变体

| 变体名 | 注册名 | 特点 |
|---|---|---|
| GRPO | `grpo` | 原始组内归一化 |
| GRPO Vectorized | `grpo_vectorized` | 向量化实现，更高效 |
| GRPO Pass@k | `grpo_passk` | 只有组内最高分有非零优势：r_max - r_second_max |
| REINFORCE++ | `reinforce_plus_plus` | 无分组，reward-to-go + whitening |
| REINFORCE++ Baseline | `reinforce_plus_plus_baseline` | 组均值做 baseline |
| RLOO | `rloo` | Leave-One-Out 优势估计 |
| OPO | `opo` | 基于长度的 baseline: B = Σ(len_i * score_i) / Σ(len_i) |
| GDPO | `gdpo` | 按 reward 维度分别做 GRPO，再加权求和 |

### 3.5 DAPO 中的 GRPO 配置（DAPO 使用 GRPO 作为 adv_estimator）

DAPO 论文使用 `adv_estimator=grpo`，并通过额外参数进行增强：

```bash
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
```

---

## 4. DPO (Direct Preference Optimization)

### 4.1 算法原理

DPO（https://arxiv.org/abs/2305.18290）**不进行在线 RL 训练**，而是直接使用偏好数据对 (chosen/rejected) 优化策略。它推导出了 RLHF 优化目标的闭式解，避免了显式的 reward 模型。

### 4.2 verl 中的 DPO 支持

verl 通过以下方式支持 DPO 类算法：

1. **`ppo_trainer` 中的 `loss_mode` 配置**
2. **GSPO (Generalized SPO)** — 注册为 `"gspo"` 的 policy loss
3. **实验性 DPO** — 在 `reward_loop` 中实现的 `gdpo` (Group DPO)
4. **SFT 训练器** — 支持偏好数据的 SFT 式训练

#### DPO 损失函数数学形式

```
L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)}[ log σ( β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

其中：
- `y_w` — chosen（偏好胜出的回答）
- `y_l` — rejected（落败的回答）
- `β` — 控制对参考模型的偏离程度
- `σ` — sigmoid 函数

#### verl 中的 GSPO Loss

文件：`verl/trainer/ppo/core_algos.py`，`compute_policy_loss_gspo()`（注册为 `"gspo"`）

```python
@register_policy_loss("gspo")
def compute_policy_loss_gspo(old_log_prob, log_prob, advantages, ...):
    # GSPO 使用偏好对比的 log-ratio 作为训练信号
    # 它是对 DPO 损失的泛化——允许混合在线和离线偏好学习
    ...
```

### 4.3 配置文件中的 DPO 相关字段

```python
# verl/trainer/config/algorithm.py
@dataclass
class AlgoConfig(BaseConfig):
    adv_estimator: str = "gae"  # 切换为 "gdpo" 可启用 Group DPO
    ...
```

### 4.4 GDPO — Group Reward-Decoupled Policy Optimization

文件：`verl/trainer/ppo/core_algos.py`，`compute_gdpo_outcome_advantage()`（注册为 `"gdpo"`）

GDPO 是对 GRPO 的扩展：将多个 reward 维度分别做组内归一化，再加权聚合，防止单一 reward 信号主导。

```python
# 对每个 reward 维度 k，独立做 GRPO 归一化
for i in range(num_scores):
    normalized_score, _ = compute_grpo_outcome_advantage(
        token_level_rewards=score_list[i], ...)
    new_advantage += weights[i] * normalized_score

# 最终再做一次 batch 级 whiten
advantages = masked_whiten(new_advantage, response_mask)
```

---

## 5. DAPO (Decoupled-clip Policy Optimization)

### 5.1 算法原理

DAPO（https://arxiv.org/abs/2503.14460，SIA 团队提出）是一个**去除了 KL 惩罚的纯 RL 算法**，通过三处关键改进在 GRPO 基础上显著提升：

1. **Clip-Higher**：不对称 clip — `clip_ratio_low=0.2`，`clip_ratio_high=0.28`
2. **Dynamic Sampling**：基于 `filter_ratio` 过滤低 token 级优势的样本
3. **Overlong Buffer**：对过长响应施加惩罚但不截断
4. **无 KL 惩罚**：`kl_coef=0`，`use_kl_loss=False`

### 5.2 配置详解

来自 DAPO 的示例配置：

```bash
# DAPO 核心参数
adv_estimator=grpo                    # 使用 GRPO 作为优势估计
use_kl_in_reward=False                # ❌ 不使用 KL 惩罚
kl_coef=0.0                           # KL 系数为 0
use_kl_loss=False                     # ❌ 不使用 KL loss
kl_loss_coef=0.0                      # KL loss 系数为 0

# 不对称 clip（DAPO 的创新点）
clip_ratio_low=0.2                    # 下界 clip
clip_ratio_high=0.28                  # 上界 clip（比下界更大）

# Overlong Buffer
enable_overlong_buffer=True           # 启用过长缓冲
overlong_buffer_len=4096              # 缓冲长度
overlong_penalty_factor=1.0           # 惩罚因子
```

### 5.3 Clip-Higher 机制

DAPO 发现**对称 clip** 在 GRPO 中会限制对高优势样本的学习。因此使用 **非对称 clip**：

```
标准 PPO:  clip(ratio, 0.8, 1.2)     # 对称
DAPO:      clip(ratio, 0.8, 1.28)    # 非对称 — 允许更大的正向更新
```

这意味着当 `advantages > 0`（好样本）时，更新幅度可以更大，加速对优质行为的模仿。

### 5.4 Dynamic Sampling (Filter Ratio)

DAPO 通过 `filter_ratio` 过滤掉低质量的 token 级更新。这一逻辑在 verl 的 **reward manager** 中实现：

文件：`verl/workers/reward_manager/dapo.py`

```python
class DAPORewardManager(AbstractRewardManager):
    # 支持 overlong_buffer 惩罚：
    # 如果 response 超过 max_resp_len - overlong_buffer_len，
    # 对超出部分施加线性惩罚
    if self.overlong_buffer_cfg.enable:
        overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0)
        reward += overlong_reward
```

### 5.5 完整训练脚本分析

```bash
# 关键参数
rollout_mode="async"                  # 异步 rollout
n_resp_per_prompt=16                  # 每组 16 个响应（GRPO 分组）
train_prompt_bsz=0                    # 动态 batch
gen_prompt_bsz=1
partial_rollout=True                  # 部分 rollout 加速
staleness_threshold=0.1               # 陈旧性控制
```

### 5.6 DAPO 算法总结

| 组件 | 传统 PPO | DAPO |
|---|---|---|
| Critic | ✅ 需要 | ❌ 不需要 |
| Ref Policy | ✅ 需要 | ✅ 需要（但 KL=0） |
| KL Penalty | ✅ 有 | ❌ 无 |
| Clip | 对称 | 非对称 (0.2/0.28) |
| 优势估计 | GAE | GRPO（组归一化） |
| 过长生成长度 | 截断 | 惩罚缓冲 |
| Temperature | 通常 <1 | 1.0（鼓励探索） |

---

## 6. 算法对比总结

### 6.1 核心公式对比

| 算法 | 策略梯度 | 优势估计 |
|---|---|---|
| **PPO** | `L = -E[min(rÂ, clip(r)Â)]` | `A = GAE(reward, value, γ, λ)` |
| **GRPO** | `L = -E[min(rÂ, clip(r)Â)]` | `A_i = (r_i - μ_g) / σ_g` |
| **DPO** | `L = -E[log σ(β*(Δ_chosen - Δ_rejected))]` | 无（直接用偏好对比）|
| **DAPO** | `L = -E[min(rÂ, clip_higher(r)Â)]` | `A_i = (r_i - μ_g) / σ_g` + 过滤 |

### 6.2 模型需求

```
                Actor    Ref    Critic    Reward Model
PPO              ✅      ✅       ✅          ✅
GRPO             ✅      ✅       ❌          ✅
DPO              ✅      ✅(冻结)  ❌          ❌(使用偏好数据)
DAPO             ✅      ✅(KL=0) ❌          ✅
```

### 6.3 适用场景

| 场景 | 推荐算法 | 原因 |
|---|---|---|
| 通用 RLHF | PPO | 稳定，有 Critic 提供密集信号 |
| 数学推理 | GRPO / DAPO | 结果稀疏，Critic 难学，组内对比更有效 |
| 离线偏好对齐 | DPO | 无需在线 rollout，更简单 |
| 高吞吐推理 RL | DAPO | 无 KL、无 Critic，更高效 |

---

## 7. 涉及模块清单

### 7.1 核心算法

| 文件 | 关键类/函数 | 职责 |
|---|---|---|
| `verl/trainer/ppo/core_algos.py` | `compute_gae_advantage_return()` | GAE 优势估计 |
| | `compute_grpo_outcome_advantage()` | GRPO 组内归一化优势 |
| | `compute_policy_loss_vanilla()` | PPO Clip 策略损失 |
| | `compute_policy_loss_clip_cov()` | Clip-Cov 策略损失 |
| | `compute_policy_loss_gspo()` | GSPO 偏好对齐损失 |
| | `compute_value_loss()` | Value 网络损失 |
| | `kl_penalty()` | KL 散度惩罚计算 |
| | `AdaptiveKLController` | 自适应 KL 控制器 |
| | `FixedKLController` | 固定 KL 控制器 |
| | `AdvantageEstimator` | 所有优势估计器的枚举注册 |
| | `register_policy_loss()` | 策略损失注册器 |
| | `register_adv_est()` | 优势估计器注册器 |
| | `agg_loss()` | 损失聚合（支持多种 agg_mode） |

### 7.2 Loss 函数封装

| 文件 | 函数 | 职责 |
|---|---|---|
| `verl/workers/utils/losses.py` | `ppo_loss()` | PPO loss 封装（组装 policy loss + entropy + KL）|
| | `value_loss()` | Value loss 封装 |
| | `sft_loss()` | SFT loss |

### 7.3 训练器

| 文件 | 类/函数 | 职责 |
|---|---|---|
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer` | PPO/GRPO 分布式训练主循环 |
| | `apply_kl_penalty()` | 在 reward 上应用 KL 惩罚 |
| `verl/trainer/main_ppo_sync.py` | `main_ppo` | PPO 同步训练入口 |
| `verl/trainer/main_ppo_sync.py` | 配置组装 | 设置 actor/critic/ref 引擎参数 |

### 7.4 Reward Manager

| 文件 | 类 | 职责 |
|---|---|---|
| `verl/workers/reward_manager/dapo.py` | `DAPORewardManager` | DAPO 奖励（含 overlong buffer 惩罚）|
| `verl/experimental/reward_loop/reward_manager/dapo.py` | `DAPORewardManager` | 异步版的 DAPO reward manager |
| `verl/trainer/ppo/reward.py` | `extract_reward()` | 从 reward manager 提取奖励 |

### 7.5 算法配置

| 文件 | 类 | 职责 |
|---|---|---|
| `verl/trainer/config/algorithm.py` | `AlgoConfig` | 算法配置（adv_estimator, kl_ctrl 等）|
| | `FilterGroupsConfig` | DAPO 组过滤配置 |
| | `KLControlConfig` | KL 控制策略 |
| | `RolloutCorrectionConfig` | 离线校正配置（IS, RS）|
| `verl/workers/config/actor.py` | `ActorConfig` | Actor 参数（clip_ratio, entropy_coeff 等）|
| `verl/workers/config/critic.py` | `CriticConfig` | Critic 参数（cliprange_value 等）|

### 7.6 实用工具

| 文件 | 函数 | 职责 |
|---|---|---|
| `verl/trainer/ppo/metric_utils.py` | `compute_data_metrics()` | 数据级指标（KL散度，序列长度等）|
| | `compute_throughout_metrics()` | 吞吐量指标 |
| `verl/trainer/ppo/padding_utils.py` |  | 填充对齐工具 |
| `verl/trainer/ppo/rollout_corr_helper.py` |  | 离线校正（IS/RS）辅助函数 |
| `verl/utils/torch_functional.py` | `masked_mean()`, `masked_whiten()` | 掩码统计工具 |

---

## 8. PPO 训练器完整数据流

```
PPOTrainer.fit()
  ├─ 初始化: Actor, Ref, Critic, Reward Model（4 个引擎）
  ├─ 训练循环:
  │   ├─ rollout() → Actor 生成 responses
  │   ├─ compute_log_prob() → Ref & Actor 计算 old_log_probs
  │   ├─ compute_rm_score() → Reward Model 计算分数
  │   ├─ apply_kl_penalty() → 在 reward 中扣除 KL（可选）
  │   ├─ compute_advantage() → GAE / GRPO 等
  │   ├─ update_actor() → PPO policy loss
  │   │   ├─ ppo_loss() → policy_loss + entropy - KL
  │   │   └─ 反向传播
  │   └─ update_critic() → value loss (如果使用 Critic)
  └─ checkpoint 保存
```

对于 GRPO / DAPO，流程类似但不需要 Critic 模型和 value loss。
