# Dynamic Batch Size (use_dynamic_bsz) 技术详解

> 本文档分析 verl 中 Dynamic Batch Size 的技术核心、所涉及的技术细节和相关模块。
> 基于 verl 代码库 (verl-project/verl) 分析。

---

## 1. 背景与动机

### 1.1 问题

在大规模 RL 训练（PPO / GRPO）中，每条序列的长度差异巨大（例如 reasoning trace 可以相差 10 倍以上）。传统的**固定样本数** micro-batch 切分方式会导致严重的算力浪费：

| 序列 A (128 tokens) | 序列 B (4096 tokens) | 序列 C (64 tokens) | 序列 D (8192 tokens) |
|---|---|---|---|

如果按固定 `micro_batch_size = 2` 切分：

- Micro-batch 1: A+B → 4224 tokens → 计算负载适中
- Micro-batch 2: C+D → 8256 tokens → 计算负载极高

GPU 利用率严重不均衡，**最慢的 micro-batch 决定整体速度**。

### 1.2 解决方案：按 Token 数切分

Dynamic Batch Size 不再按样本数切分，而是按**总 token 数**切分 —— 每个 micro-batch 约包含相同的 token 数，从而保证计算负载均衡：

```
Micro-batch 1: A(128) + C(64) + ... → ~4096 tokens
Micro-batch 2: B(4096)              → ~4096 tokens
Micro-batch 3: D(2048) + ...        → ~4096 tokens
```

---

## 2. 核心公式与数学模型

### 2.1 计算负载估计

文件：`verl/utils/seqlen_balancing.py`，函数 `calculate_workload()`

```python
def calculate_workload(seqlen_list):
    return 24576 * seqlen_list + seqlen_list**2
```

该公式是对 Transformer 层 FLOPs 的近似：

```
FLOPs ≈ 12 * hidden_size² * seqlen + 2 * hidden_size * seqlen²
```

常数 `24576` 是基于 7B 模型 (hidden_size=4096) 校准的。当 `hidden_size=4096` 时：

```
12 * 4096² = 12 * 16777216 = 201326592
2 * 4096    = 8192
```

简化后相对比值为 `24576 * seqlen + seqlen²`。这个值不是精确的 FLOPs 计数，而是用于**相对比较**的工作量估计值。

### 2.2 核心限制条件

```
max_token_len_per_gpu >= max_seq_len  # 每个 micro-batch 的 token 上限
batch_size % force_group_size == 0    # 分组约束
num_micro_batches <= num_groups       # micro-batch 数不能超过样本/组数
```

Micro-batch 数量的计算公式：

```python
num_micro_batches = min(num_groups, ceil(total_seqlen / max_token_len))
```

即：按 token 容量算出的 micro-batch 数，但不超过样本数（避免出现空的 micro-batch）。

---

## 3. 分区算法：Karmarkar-Karp 最大差异法

文件：`verl/utils/seqlen_balancing.py`，函数 `karmarkar_karp()`

### 3.1 算法原理

Karmarkar-Karp 算法（又称 Largest Differencing Method, LDM）是一种**近似最优的多路数分区**算法：

1. **初始化**：将每个待分区项目作为一个独立集合
2. **迭代合并**：每次取出总和最大的两个集合，将它们合并（大集合减小集合），再将结果放回优先队列
3. **重复**直到只剩 k 个集合

这个算法的优势是**接近最优解**且时间复杂度为 O(n log n)。

### 3.2 两种模式

| 模式 | `equal_size=True` | `equal_size=False` |
|---|---|---|
| 用途 | 每个 partition 必须有相同数量的样本（用于 rollout.n） | 只需平衡负载，partition 大小可以不同 |
| 初始状态 | 每 k 个排序后的样本为一组初始化 | 每个样本独立作为一个集合 |
| 限制 | `len(seqlen_list) % k_partitions == 0` | 无 |

### 3.3 Greedy 备选方案

文件：`greedy_partition()`

作为 Karmarkar-Karp 的轻量替代，按序将每个项目分配给当前总和最小的 partition。当 `equal_size=True` 时，给每个项目加一个足够大的 bias，让算法先均匀分配样本数量再平衡负载。

---

## 4. 微批次排序与流水线气泡优化

文件：`verl/utils/seqlen_balancing.py`，`rearrange_micro_batches()` 末尾

```python
if use_dynamic_bsz_balance:
    micro_bsz_idx.sort(key=lambda p: sum(workloads[idx] for idx in p), reverse=True)
    micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]
```

### 4.1 排序策略

1. **按工作量降序排列所有 micro-batch**：最重的 micro-batch 排最前
2. **重-轻交错排列**：`[::2][::-1]` 取所有偶数位（即最重的几个）反转 → 放在最前；`[1::2]` 取所有奇数位（较轻的）跟在后面

最终排列模式：**最重, 次重, ..., 中间, 最轻, 次轻, ...**

### 4.2 为什么这样排列

在 Pipeline Parallelism 中，存在 warm-up 和 cool-down 阶段：
- **Warm-up**: 前几个 micro-batch 只有部分 PP stage 在工作
- **Steady state**: 所有 PP stage 满负荷
- **Cool-down**: 后几个 micro-batch 只有部分 PP stage 在工作

把**高负载的 micro-batch 放在两端**（warm-up / cool-down 阶段），**低负载的放在中间**（steady state），可以**最小化流水线气泡**。

![流水线气泡说明]
```
Warm-up → [重] [中] [轻] [轻] [中] [重] ← Cool-down
气泡少:   ██░░░░██████████████████░░░░██
```

---

## 5. 数据并行同步

文件：`verl/utils/seqlen_balancing.py`，`rearrange_micro_batches()`

```python
if dist.is_initialized() and same_micro_num_in_dp and dp_group is not None:
    num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
    dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
    num_micro_batches = num_micro_batches.cpu().item()
```

由于不同 DP rank 上的数据序列长度分布不同，每个 rank 算出的 micro-batch 数量可能不同。通过 `all_reduce(MAX)` 保证所有 DP rank 使用**相同的 micro-batch 数**，避免某些 rank 提前完成导致的同步等待。

---

## 6. Pipeline Parallelism 适配

文件：`verl/utils/seqlen_balancing.py`，`rearrange_micro_batches()`

### 6.1 micro-batch 数量对齐

```python
if num_batches_divided_by is not None:
    num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)
```

`num_batches_divided_by` 通常设为 Virtual Pipeline Parallelism (VPP) 的大小。Micro-batch 数量必须是 VPP 大小的倍数，以确保流水线各阶段负载均衡。

### 6.2 最小 micro-batch 数

```python
if min_num_micro_batch is not None:
    num_micro_batches = max(min_num_micro_batch, num_micro_batches)
```

PP 需要有足够多的 micro-batch 来填充流水线。`min_num_micro_batch` 确保不会出现 micro-batch 太少导致流水线空闲的情况。

---

## 7. force_group_size —— Reward Model 分组约束

文件：`verl/utils/seqlen_balancing.py`，`rearrange_micro_batches()`

```python
# When force_group_size > 1, aggregate workloads by groups
if force_group_size > 1:
    workloads_per_sample = calculate_workload(seq_len_effective)
    workloads_per_sample_grouped = workloads_per_sample.view(num_groups, force_group_size)
    group_workloads = workloads_per_sample_grouped.sum(dim=1).cpu().tolist()
    micro_bsz_group_idx = get_seqlen_balanced_partitions(group_workloads, num_micro_batches, equal_size=False)
```

某些场景需要将特定样本始终放在同一个 micro-batch 中（例如 Reward Model 的 pair 对比、preference learning 中的 chosen/rejected 对）。通过 `force_group_size` 对连续样本分组，先以组为单位做分区，再将组展开回样本。

---

## 8. 反向恢复

文件：`verl/utils/seqlen_balancing.py`，`restore_dynamic_batch()`

由于 micro-batch 中的数据顺序在分区时被打乱，forward 计算完成后需要将输出恢复回原始顺序：

```python
def restore_dynamic_batch(data, batch_idx_list):
    indices = list(chain.from_iterable(batch_idx_list))
    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
    if data.is_nested:
        data_lst = data.unbind()
        tensors = [data_lst[i] for i in revert_indices]
        reverted_data = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
    else:
        reverted_data = data[revert_indices]
    return reverted_data
```

`get_reverse_idx()` 构建索引的逆映射：如果原始索引映射为 `[2, 0, 1]`，则逆映射为 `[1, 2, 0]`。

---

## 9. 涉及模块与文件

### 9.1 核心算法

| 文件 | 关键函数 | 职责 |
|---|---|---|
| `verl/utils/seqlen_balancing.py` | `calculate_workload()` | 估算序列计算负载 |
| | `karmarkar_karp()` | Karmarkar-Karp 分区算法 |
| | `get_seqlen_balanced_partitions()` | 获取均衡分区 |
| | `rearrange_micro_batches()` | **核心**：按 token 数切分 batch |
| | `restore_dynamic_batch()` | 恢复原始顺序 |
| | `prepare_dynamic_batch()` | DataProto 包装版切分 |
| | `log_seqlen_unbalance()` | 记录均衡度指标 |

### 9.2 训练引擎集成

| 文件 | 关键函数 | 职责 |
|---|---|---|
| `verl/workers/engine/utils.py` | `prepare_micro_batches()` | **入口**：根据 `use_dynamic_bsz` 分发到不同切分策略 |
| | `postprocess_batch_func()` | 后处理：恢复顺序、聚合 loss 和 metrics |
| `verl/workers/engine/fsdp/transformer_impl.py` | `train_batch()` | FSDP 后端训练循环 |
| `verl/workers/engine/megatron/transformer_impl.py` | `train_batch()` | Megatron 后端训练循环 |
| `verl/workers/engine/automodel/transformer_impl.py` | `train_batch()` | AutoModel 后端训练循环 |
| `verl/workers/engine/torchtitan/transformer_impl.py` | `train_batch()` | TorchTitan 后端训练循环 |
| `verl/workers/engine/veomni/transformer_impl.py` | `train_batch()` | VeOmni 后端训练循环 |

所有训练后端都通过 `verl/workers/engine/utils.py` 中的 `prepare_micro_batches()` 和 `postprocess_batch_func()` 统一集成 dynamic batch size 逻辑。

### 9.3 Worker 层

| 文件 | 关键方法 | 职责 |
|---|---|---|
| `verl/workers/engine_workers.py` | `train_batch()` | 注入 `use_dynamic_bsz` 和 `max_token_len_per_gpu` 到 data |

### 9.4 配置定义

| 文件 | 关键字段 | 职责 |
|---|---|---|
| `verl/workers/config/engine.py` | `EngineConfig.use_dynamic_bsz` (default: True) | 总开关 |
| | `EngineConfig.max_token_len_per_gpu` | 每个 GPU 每 micro-batch 的最大 token 数 |
| | `EngineConfig.micro_batch_size_per_gpu` | 不使用 dynamic bsz 时的回退配置 |
| `verl/workers/config/actor.py` | `ActorConfig.use_dynamic_bsz` | Actor 独立的 dynamic bsz 配置 |
| `verl/workers/config/critic.py` | `CriticConfig.use_dynamic_bsz` | Critic 独立的 dynamic bsz 配置 |
| `verl/workers/config/rollout.py` | `RolloutConfig.log_prob_use_dynamic_bsz` | Log prob 推理时的 dynamic bsz |

### 9.5 配置验证

| 文件 | 功能 |
|---|---|
| `verl/utils/config.py` | 当 `use_dynamic_bsz=False` 时进行传统 batch size 整除数校验 |
| `verl/workers/config/actor.py` | Actor 配置验证：校验 micro_batch_size 互斥性 |
| `verl/workers/config/critic.py` | Critic 配置验证：同理 |

### 9.6 训练器

| 文件 | 功能 |
|---|---|
| `verl/trainer/sft_trainer.py` | SFT 训练中使用 dynamic bsz |
| `verl/trainer/sft_trainer_ray.py` | SFT Ray 训练中使用 dynamic bsz |
| `verl/trainer/main_ppo_sync.py` | PPO 训练：设置 critic 的 `max_token_len_per_gpu` |

---

## 10. 整体数据流

```
用户配置
  ├─ actor_rollout_ref.actor.use_dynamic_bsz = True
  ├─ actor_rollout_ref.actor.max_token_len_per_gpu = 16384
  └─ actor_rollout_ref.ref.log_prob_use_dynamic_bsz = True

数据流 (Training Loop):

1. engine_workers.py: train_batch()
   → 将 use_dynamic_bsz, max_token_len_per_gpu 注入 TensorDict

2. engine/utils.py: prepare_micro_batches()
   → 如果是 use_dynamic_bsz:
     a. 从 batch 中提取 attention_mask / input_ids
     b. 计算每个样本的 seq_len_effective
     c. 调用 calculate_workload() 计算每个样本的负载
     d. 用 get_seqlen_balanced_partitions() (Karmarkar-Karp) 进行分区
     e. 按负载排序并交错排列（大-小交错）
     f. 返回 micro_batches + batch_idx_list
   → 否则: 按固定 micro_batch_size_per_gpu 切分

3. 后端（FSDP/Megatron/...）: forward + backward 每个 micro-batch

4. engine/utils.py: postprocess_batch_func()
   → 如果是 use_dynamic_bsz:
     a. 收集各 micro-batch 的输出
     b. 调用 restore_dynamic_batch() 恢复原始顺序
   → 聚合 loss 和 metrics
```

---

## 11. 配置示例

```yaml
# 启用 Dynamic Batch Size（默认）
actor_rollout_ref:
  actor:
    use_dynamic_bsz: True
    max_token_len_per_gpu: 16384  # 每个 GPU per micro-batch 最多 16K tokens
  ref:
    log_prob_use_dynamic_bsz: True

# 禁用 Dynamic Batch Size（传统固定样本数）
actor_rollout_ref:
  actor:
    use_dynamic_bsz: False
    ppo_micro_batch_size_per_gpu: 8  # 每个 GPU 固定 8 条序列 per micro-batch
  ref:
    log_prob_use_dynamic_bsz: False  # ref 也关闭
```

---

## 12. 总结

| 维度 | 说明 |
|---|---|
| **技术核心** | 按 Token 总数而非样本数切分 micro-batch，结合 Karmarkar-Karp 分区算法实现计算负载均衡 |
| **性能收益** | 消除变长序列导致的 GPU 利用率波动，减少流水线气泡 |
| **后端兼容** | 统一通过 `prepare_micro_batches()` / `postprocess_batch_func()` 集成，FSDP/Megatron/AutoModel/Torchlitan/VeOmni 均支持 |
| **PP 优化** | 重-轻交错排列减少 warm-up/cool-down 气泡；micro-batch 数对齐 VPP 大小 |
| **DP 同步** | 通过 all_reduce(MAX) 确保所有 DP rank micro-batch 数一致 |
| **特殊场景** | `force_group_size` 支持 RM pair 等分组约束 |
