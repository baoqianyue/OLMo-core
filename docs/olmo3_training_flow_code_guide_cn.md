# OLMo3 训练全流程与 OLMo-core 核心代码导读

本文面向“彻底掌握 OLMo3 模型训练全流程，尤其理解 OLMo-core 核心代码”。内容依据本地技术报告 `OLMo3.pdf` 和当前仓库代码整理，重点覆盖 OLMo3 Base 的预训练、midtraining、长上下文扩展，以及 OLMo-core 已支持的 SFT 训练入口。DPO、RL、RL-Zero 在技术报告中属于 post-training 主线，但报告也明确指出预训练代码是 OLMo-core、post-training 主要在 Open Instruct；因此本文会把边界说清楚，不把 Open Instruct 的逻辑强行归到 OLMo-core。

## 1. 技术报告中的 OLMo3 model flow

OLMo3 的核心思想不是只发布最终权重，而是发布完整的 model flow：训练数据、训练代码、中间 checkpoint、评测和依赖。报告把训练分成两大部分：

```text
OLMo3 model flow
  ├─ Base model training
  │   ├─ Stage 1: Pretraining
  │   ├─ Stage 2: Midtraining
  │   └─ Stage 3: Long-context extension
  └─ Post-training
      ├─ Think / Instruct SFT
      ├─ DPO / preference tuning
      ├─ RL with OlmoRL
      └─ RL-Zero from base
```

对 OLMo-core 来说，最关键的是左半边：

- **Pretraining**：在 Dolma 3 Mix 上训练 Base 模型，主上下文长度 8192。
- **Midtraining**：继续在 Dolma 3 Dolmino Mix 上训练 100B tokens，数据更偏向数学、代码、QA、instruction、thinking traces 等能力增强。
- **Long-context extension**：继续在 Dolma 3 Longmino Mix 上训练，把上下文扩到 65536，并使用 YaRN RoPE scaling、context parallelism 和文档边界 masking。
- **SFT**：报告提到 OLMo-core 支持 SFT，并且 Think SFT 阶段从 Open Instruct 切到 OLMo-core 获得训练速度提升；但当前本文主要聚焦 Base 模型三阶段。

报告中的几个关键数值可以直接在脚本里看到：

| 阶段 | 7B | 32B | OLMo-core 对应入口 |
|---|---:|---:|---|
| Pretraining | 8K seq, 4M tokens/batch, peak LR 3e-4, 约 5.93T tokens | 8K seq, 8M tokens/batch, peak LR 6e-4, 约 5.5T tokens | `src/scripts/train/OLMo3/OLMo3-7B.py`, `OLMo3-32B.py` |
| Midtraining | 8K seq, 2M tokens/batch, 100B tokens, LR 2.071e-4 线性衰减到 0 | 8K seq, 4M tokens/batch, 两次 100B runs 后 model soup | `OLMo3-7B-midtraining.py`, `OLMo3-32B-midtraining.py` |
| Long-context | 65K seq, 4M tokens/batch, 50B tokens, CP=8 | 65K seq, 8M tokens/batch, 100B tokens, CP=8 | `OLMo3-7B-long-context.py`, `OLMo3-32B-long-context.py` |

## 2. OLMo-core 的整体框架

从代码组织看，OLMo-core 是一个“配置驱动的训练框架”。训练脚本本身通常不写训练循环，而是构造一个 `ExperimentConfig`，交给通用入口启动。

```text
src/scripts/train/OLMo3/*.py
  -> build_experiment_config() 或 build_config(...)
  -> ExperimentConfig
      ├─ model: TransformerConfig
      ├─ dataset: NumpyDatasetConfig
      ├─ data_loader: NumpyDataLoaderConfig
      ├─ train_module: TransformerTrainModuleConfig
      └─ trainer: TrainerConfig
  -> internal.experiment.main()
  -> train()
      ├─ model = config.model.build(init_device="meta")
      ├─ train_module = config.train_module.build(model)
      ├─ data_loader = _build_data_loader(config, train_module.dp_process_group)
      ├─ trainer = config.trainer.build(train_module, data_loader)
      └─ trainer.fit()
```

核心代码路径：

- `src/olmo_core/internal/experiment.py`：Ai2 内部训练脚本的 CLI、Beaker launch、配置组装和训练入口。
- `src/olmo_core/internal/cookbook.py`：常用训练配置 helper，midtraining 和 long-context 脚本大量使用。
- `src/olmo_core/train/trainer.py`：外层训练循环、checkpoint 恢复、callback 生命周期、metrics、epoch/step 调度。
- `src/olmo_core/train/train_module/transformer/train_module.py`：Transformer 的 forward/backward、micro-batch、loss、optimizer step。
- `src/olmo_core/nn/transformer/*.py`：模型结构、block、初始化、FSDP/TP/CP/AC/compile 改造。
- `src/olmo_core/data/*.py`：numpy token 数据集、source mixture、DataLoader、collator。
- `src/olmo_core/optim/*.py`：AdamW、SkipStepAdamW、scheduler。
- `src/olmo_core/train/callbacks/*.py`：checkpoint、W&B、eval、memory/speed monitor 等。

## 3. OLMo3 官方脚本怎么表达三个阶段

### 3.1 Pretraining

7B 入口：`src/scripts/train/OLMo3/OLMo3-7B.py`

关键配置：

```python
SEQUENCE_LENGTH = 8 * 1024
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024

TransformerConfig.olmo3_7B(...)
SkipStepAdamWConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
CosWithWarmup(warmup_steps=2000)
DataMix.OLMo_mix_0625
max_duration=Duration.tokens(int(5e12))
hard_stop=Duration.tokens(int(4e12))
```

32B 入口：`src/scripts/train/OLMo3/OLMo3-32B.py`

关键差异：

```python
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024
TransformerConfig.olmo3_32B(...)
SkipStepAdamWConfig(lr=6e-4, ...)
DataMix.OLMo_mix_0925
dp_config=HSDP(shard_degree=64, wrapping_strategy=full)
activation_memory_budget=0.5
max_duration=Duration.epochs(1)
```

理解重点：

- 预训练阶段数据来自官方 data mix 文件，如 `src/olmo_core/data/mixes/OLMo-mix-0925.txt`。
- `global_batch_size` 的单位是 token，不是样本数。
- `rank_microbatch_size` 的单位也是 token，用来控制单卡一次 forward/backward 的本地 token 数。
- 7B 和 32B 都用 `SkipStepAdamWConfig`，当 loss 或 grad norm 异常偏离滚动统计时可以跳过不稳定 step。

#### 3.1.1 `OLMo3-7B.py` 中 `build_train_module_config()` 逐参数解释

这一段代码是 7B 预训练脚本里最核心的“训练 step 配方”。它不定义模型结构，也不定义数据来源；它定义的是：模型拿到一个 batch 后，如何切 micro-batch、如何前向/反向、如何并行包裹、如何做优化器更新、如何稳定 loss。

```python
rank_microbatch_size = common.max_sequence_length
```

**`rank_microbatch_size` 的单位是 token。`common.max_sequence_length` 来自脚本底部传给 `build_config()` 的 `SEQUENCE_LENGTH = 8 * 1024`，所以默认值是 8192 tokens。因为每条训练样本长度也是 8192，这等价于“每个 rank 每个 micro-batch 处理 1 条序列”**。

举例：假设 `global_batch_size = 4 * 1024 * 1024`，即约 419 万 tokens，而序列长度是 8192，那么一个全局 batch 大约包含 `4194304 / 8192 = 512` 条 8K 序列。这 512 条序列会先按 data parallel rank 分给不同 GPU；如果某个 rank 分到多条序列，再由 `rank_microbatch_size` 决定一次 forward/backward 放几条进去。micro-batch 只影响显存和梯度累积，不改变数学意义上的全局 batch size。

```python
if common.launch is not None:
    gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
    if all("B200" in g for g in gpus):
        rank_microbatch_size *= 2
```

这段是硬件条件分支：如果通过 Beaker launch 启动，并且所有 cluster 都映射到 B200 GPU，就把 rank micro-batch 从 8192 tokens 提到 16384 tokens。含义是每个 rank 每次处理 2 条 8K 序列。B200 显存/吞吐更强，能承受更大的本地 micro-batch；好处是一次全局 batch 需要的梯度累积轮数更少，通信和调度开销也更低。

```python
TransformerTrainModuleConfig(
    rank_microbatch_size=rank_microbatch_size,
    max_sequence_length=common.max_sequence_length,
    ...
)
```

`TransformerTrainModuleConfig` 会在 `ExperimentConfig` 进入训练后 build 成 `TransformerTrainModule`。它对应 `src/olmo_core/train/train_module/transformer/train_module.py`，负责真正的一次训练 step：切 micro-batch、调用模型 forward、backward、梯度同步、梯度裁剪、scheduler 写入 LR、optimizer step。外层 `Trainer` 则负责循环、checkpoint、callback 和评测调度。

```python
optim=SkipStepAdamWConfig(
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    group_overrides=[
        OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
    ],
)
```

这里使用的是 OLMo3 预训练的 AdamW 变体：

- `lr=3e-4`：7B 预训练 peak learning rate。它和前面的表格对应，32B 脚本里会用更高的 `6e-4`。
- `weight_decay=0.1`：对大多数权重施加 AdamW decoupled weight decay，抑制权重无约束变大。
- `betas=(0.9, 0.95)`：Adam 动量参数。`0.9` 是一阶动量平滑，`0.95` 是二阶矩平滑；LLM 预训练常用比 Adam 默认 `0.999` 更低的 beta2，让二阶统计更快跟上训练早期和数据分布变化。
- `group_overrides`：把 `embeddings.weight` 单独拉到一个 param group，并设置 `weight_decay=0.0`。这对应报告里的“embedding 不做 weight decay”。直觉上，embedding 矩阵直接承载 token 表示，过强的 decay 可能不必要地压缩词向量尺度。

`SkipStepAdamWConfig` 额外维护 loss/grad norm 的滚动统计。如果某一步异常偏离统计范围，可以把更新因子变成 0 来跳过这次不稳定 optimizer step。脚本底部也写了 `include_instance_filter=False  # We use SkipStepOptimizer for this problem.`，意思是这里更依赖 optimizer 层面的异常 step 保护，而不是在数据入口先过滤重复/异常 instance。

```python
compile_model=True
```

启用 `torch.compile`。模型会在并行改造之后被编译，以减少 Python 调度开销并让 PyTorch 编译器做 kernel/图层面的优化。对 7B/32B 这类长时间训练，编译的启动成本可以被后续大量 step 摊薄。

```python
dp_config=TransformerDataParallelConfig(
    name=DataParallelType.hsdp,
    param_dtype=DType.bfloat16,
    reduce_dtype=DType.float32,
    wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
)
```

这里使用 HSDP，也就是 hybrid sharded data parallel。可以把它理解成 FSDP 和数据并行复制的组合：参数、梯度、optimizer state 在 shard 组内切分，同时在 replica 维度上复制模型副本。`param_dtype=bfloat16` 表示模型参数通信/存储主要用 bf16，省显存；`reduce_dtype=float32` 表示梯度归约用 fp32，提高数值稳定性；`wrapping_strategy=blocks` 表示主要按 Transformer blocks 粒度做 FSDP/HSDP 包裹，而不是把 block 内部 attention/MLP 再切得更细。

`bf16 -> fp32` 这个转换本身不会新增精度损失，因为 fp32 可以精确表示所有 bf16 数值；但它也不能恢复 bf16 阶段已经丢掉的小数位。`reduce_dtype=float32` 的价值在于跨 rank 梯度累加时用 fp32 做加和，减少归约过程中继续使用 bf16 累加带来的舍入误差。可以理解为：参数常驻/通信用 bf16 省显存和带宽，梯度归约时提升到 fp32 是为了让“多卡求和”这一步更稳。

这里几个词容易混在一起，可以拆开看：

- **shard / 分片**：把同一份模型参数拆开存在多张 GPU 上。比如一个 block 的参数有 8 份，`shard_degree=8` 时，每张 GPU 常驻其中一份；真正计算这个 block 前，FSDP 会临时 all-gather 拼出当前计算需要的完整参数，因此显存会短暂升高，计算完再 reshard 释放。FSDP 节省的是“常驻显存”，不是完全消除计算瞬间的完整参数峰值。
- **replica / 副本**：一整组 shard 合起来构成一份完整模型。HSDP 里可以有多组这样的 shard 组，每组都能独立处理不同数据 batch。比如 16 张 GPU、2 个节点、每节点 8 卡时，默认可能是 `num_replicas=2, shard_degree=8`：每个节点内 8 卡组成一份完整模型分片组，两个节点之间是两份 replica。它们处理不同数据，反向后再同步梯度，让两份 replica 保持一致。
- **包裹 / wrap**：这里不是抽象说法，而是代码里真的调用 `fully_shard(module, ...)` 给某个 PyTorch `nn.Module` 加 FSDP 管理层。被包裹后的 module 会拥有 FSDP 的 hook 和状态：forward 前自动 all-gather 参数，forward 后按配置 reshard，backward 时做梯度 reduce-scatter，并让 optimizer state 按 shard 管理。

这里的 **hook** 可以理解成“框架挂在 module 生命周期上的自动回调”。你写训练代码时只是调用：

```python
output = block(x)
loss.backward()
```

但 FSDP 包裹后的 `block` 会在这些调用前后自动插入额外动作：

```text
forward pre-hook:
  进入 block.forward() 之前，all-gather 当前 block 的完整参数

forward post-hook:
  block.forward() 结束后，按配置 reshard/释放完整参数

backward hook:
  反向传播产生梯度后，把梯度 reduce-scatter 回各自 shard
```

所以 hook 不是模型结构里的新层，也不是你手动调用的函数；它更像 PyTorch/FSDP 在 module 上注册的一组“事件监听器”，在 forward/backward 的关键时间点自动执行通信和状态切换。

在一次 forward 里，同一个 FSDP shard group 内的每个 rank 通常都会 all-gather 当前 block 的完整参数。原因是这些 rank 虽然处理的是不同数据切片，但使用的是同一份模型权重。可以这样理解：

```text
常驻状态：
  rank0: block 参数 shard A + 自己的数据切片
  rank1: block 参数 shard B + 自己的数据切片
  ...
  rank7: block 参数 shard H + 自己的数据切片

计算当前 block 前：
  rank0..rank7 互相 all-gather
  每个 rank 临时拿到 A+B+...+H，也就是当前 block 的完整参数

计算当前 block 时：
  rank0 用完整 block 参数处理 rank0 的数据切片
  rank1 用完整 block 参数处理 rank1 的数据切片
  ...
  rank7 用完整 block 参数处理 rank7 的数据切片

计算后：
  完整参数被 reshard/释放
  每个 rank 回到只常驻自己那份 shard 的状态
```

所以你说的“每个 rank 都 gather 当前 block 的参数，只是每个 rank 处理各自分到的数据”是对的。需要补充的是：这个 all-gather 发生在同一个 shard group 内；HSDP 如果有多个 replica，每个 replica 内部各自做 shard group 的 all-gather，不同 replica 处理不同数据，反向后再在 replica 维度同步梯度。

为什么只在每个 replica 内部 all-gather，而不是所有 rank 一起 all-gather？因为一个 replica 的 shard group 已经能拼出一份完整模型参数了。跨 replica 再 all-gather 会把其它副本的同一份逻辑参数也拿过来，信息是重复的，通信和显存都会浪费。

这不会导致模型参数不一致，原因是 HSDP 仍然是数据并行语义：

```text
初始时：
  replica0 和 replica1 持有同一份模型参数的不同分片副本，逻辑权重一致

forward/backward 时：
  replica0 处理数据 batch A，得到梯度 shard
  replica1 处理数据 batch B，得到梯度 shard

梯度同步时：
  replica 维度做梯度同步/平均
  shard 维度做 reduce-scatter，把对应梯度留在对应 shard 上

optimizer step 后：
  每个 replica 对同一份逻辑参数应用相同的平均梯度更新
  因此逻辑模型继续保持一致
```

换句话说，**all-gather 解决的是“一个 replica 内如何临时拼出完整参数来计算”**；**replica 之间的一致性靠反向后的梯度同步和相同的 optimizer 更新来保证**。不同 replica 在 forward 时看到的是同一份逻辑权重，只是处理的数据不同。

`wrapping_strategy=blocks` 的具体含义来自 `Transformer.apply_fsdp()` 和 `TransformerBlock.apply_fsdp()`：

```text
Transformer
  embeddings         -> 单独 fully_shard
  block 0            -> fully_shard 整个 block
  block 1            -> fully_shard 整个 block
  ...
  block N            -> fully_shard 整个 block
  lm_head            -> blocks 策略下不单独 fully_shard
  whole Transformer  -> 最外层再 fully_shard 一次
```

也就是说，一个 Transformer block 作为一个相对完整的计算单元被 FSDP 管理。这个 block 内部仍然包含 attention、feed-forward/MLP、norm 等子模块，但它们不会分别再被 `fully_shard()` 包一层。

如果选择更细的 `fine_grained`，block 内部会更像这样：

```text
TransformerBlock
  attention     -> fully_shard
  feed_forward  -> fully_shard
  block root    -> fully_shard
```

更细粒度包裹通常能降低峰值显存，因为 attention 算完后可以更早释放它的完整参数，MLP 再单独 all-gather；代价是 FSDP wrapper 更多、all-gather/reduce-scatter 更碎，通信调度开销也更高。`blocks` 是一个更粗、更简单的折中：显存不一定最低，但通信粒度更大，训练系统更容易跑稳。

```python
float8_config=Float8Config(enabled=False)
```

显式关闭 FP8。框架支持把部分 Linear 转成 torchao Float8/MX 格式，但这个 7B 预训练脚本保持 bf16 主线，避免引入额外量化动态 scale、硬件兼容性和数值验证成本。长上下文脚本里会看到更积极的 FP8 使用。

```python
z_loss_multiplier=1e-5
```

z-loss 是作用在 logits log-sum-exp 上的稳定化正则，防止 logits 尺度不断膨胀。训练时 `LMHead` 会返回 CE loss 和 z-loss，`TransformerTrainModule` 用这个 multiplier 把 z-loss 加进总 loss。报告 Table 33 中的 z-loss weight 也对应 `1e-5`。

更具体地说，语言模型最后会输出一组未归一化分数 `logits`，形状大致是：

```text
logits[token_position] = [vocab_0_score, vocab_1_score, ..., vocab_N_score]
```

softmax 概率是：

```text
softmax(logits_i) = exp(logits_i) / sum_j exp(logits_j)
```

其中分母的 log 形式就是：

```text
z = logsumexp(logits) = log(sum_j exp(logits_j))
```

z-loss 惩罚的是这个 `z` 的平方：

```text
z_loss = z_loss_multiplier * (logsumexp(logits) ** 2)
```

在代码里对应 `src/olmo_core/nn/functional/cross_entropy_loss.py`：

```python
z_squared = logits.logsumexp(-1).pow(2)
mask = labels != ignore_index
z_loss = z_loss_multiplier * z_squared
```

为什么需要它？CE loss 关心的是正确 token 相对其它 token 的概率，而 softmax 对“整体平移”不敏感：如果所有 logits 都加上同一个常数，softmax 概率不变，CE 也几乎不变。但 logits 的绝对尺度如果长期漂大，会让 `exp/logsumexp`、低精度训练、分布式归约和 fused loss kernel 更容易遇到数值压力。z-loss 就像给 logits 的 softmax normalizer 加一个很小的刹车，鼓励模型不要把所有分数整体推得过大。

举个简化例子：

```text
logits A = [2, 1, 0]
logits B = [102, 101, 100]
```

这两组 logits 的 softmax 概率几乎一样，因为 B 只是 A 整体加了 100；CE 主要看相对差距，所以二者 CE 接近。但 B 的 `logsumexp(logits)` 大很多，z-loss 会显著惩罚 B。这正是它要限制的方向：不改变模型学习“哪个 token 更可能”的主目标，只约束 logits 的绝对尺度。

在当前训练链路里，操作顺序是：

```text
LMHead.forward()
  -> 计算 logits
  -> cross_entropy_loss(..., compute_z_loss=True, z_loss_multiplier=1e-5)
  -> ce_loss
  -> z_loss
  -> loss = ce_loss + z_loss

TransformerTrainModule.train_batch()
  -> 对 loss backward
  -> 同时把 ce_loss 和 z_loss 分开记录到 metrics
```

注意这里的 `1e-5` 很小，说明 z-loss 不是主训练目标；主目标仍然是 next-token CE。它只是一个稳定化正则项，作用类似“轻轻压住 logits 尺度”，避免训练后期或异常 batch 把分数推到不必要的大。

```python
max_grad_norm=1.0
```

梯度裁剪阈值。`optim_step()` 前会计算梯度范数，如果超过 1.0 就按比例缩放。它和 `SkipStepAdamWConfig` 是两层保护：梯度裁剪处理“偏大但仍可更新”的梯度，SkipStep 处理“明显异常、不想更新”的 step。

这里的“梯度范数”默认是所有可训练参数梯度拼在一起后的 L2 norm，可以粗略理解为“这次更新向量的整体长度”：

```text
total_grad_norm = sqrt(sum_i grad_i^2)
```

如果 `max_grad_norm=1.0`，裁剪规则大致是：

```text
如果 total_grad_norm <= 1.0:
  梯度不变

如果 total_grad_norm > 1.0:
  scale = 1.0 / total_grad_norm
  所有梯度都乘以同一个 scale
```

例如某一步算出来的总梯度范数是 5.0，那么所有参数梯度都会乘以 `1.0 / 5.0 = 0.2`。这样做不会改变梯度方向，只是把这次更新的长度压回阈值以内。直觉上，它像是给 optimizer update 前加了一个速度上限：方向仍然由反向传播决定，但单步不能冲得太猛。

在当前代码里，执行顺序在 `src/olmo_core/train/train_module/transformer/train_module.py::optim_step()`：

```text
1. 如果 max_grad_norm 不为空：
     grad_norm = _clip_grad_norm(max_grad_norm)
     记录 optim/total grad norm
     如果 optimizer 是 SkipStepOptimizer，把 grad_norm 写入 optim.latest_grad_norm

2. scheduler 根据当前 step/tokens 写入学习率

3. optimizer.step()
```

注意 `_clip_grad_norm()` 返回的是裁剪前的 `total_norm`。所以日志里的 `optim/total grad norm` 可以用来观察“原始梯度有多大”；而实际传给 optimizer 的梯度已经可能被缩放过。

和 `SkipStepAdamW` 的关系也很重要：

- **梯度裁剪**：当前 step 仍然会更新，只是把梯度整体缩小。适合处理“梯度偏大但还可信”的情况。
- **SkipStep**：当前 step 可能完全不更新，`step_factor=0`。适合处理 loss 或 grad norm 明显异常、可能污染动量和参数的情况。

所以两者不是重复机制，而是从轻到重的两道保护。先裁剪可以限制普通尖峰的影响；随后 SkipStep 还能根据 loss/grad norm 的滚动统计决定这一步是否已经异常到应该跳过。

```python
scheduler=CosWithWarmup(warmup_steps=2000)
```

预训练使用 warmup + cosine decay。前 2000 个 optimizer step 从较低 LR 线性升到 `lr=3e-4`，之后按 cosine 曲线下降。`CosWithWarmup` 的默认 `alpha_f=0.1`，所以如果没有额外覆盖，末端 LR 会降到初始 LR 的 10%。注意这里的 step 是 optimizer step，不是 micro-batch；如果全局 batch 被切成多个 micro-batch，只有累积完成并执行一次 optimizer step 后，scheduler 才前进一次。

### 3.2 Midtraining

7B 入口：`src/scripts/train/OLMo3/OLMo3-7B-midtraining.py`

```python
SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**21
MAX_TOKENS = 100_000_000_000
LR = 0.00020712352850360292

train_module_config = cookbook.configure_train_module(
    max_sequence_length=SEQ_LENGTH,
    rank_microbatch_size=SEQ_LENGTH * 2,
    learning_rate=LR,
    scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
    activation_memory_budget=0.5,
)
```

32B 入口：`src/scripts/train/OLMo3/OLMo3-32B-midtraining.py`

```python
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024
MAX_TOKENS = 100_000_000_000
dp_shard_degree=64
load_path="gs://ai2-llm/checkpoints/stego32-highlr-filter3/step656000"
```

midtraining 最重要的数据入口是：

```python
source_list = SourceMixtureList.from_yaml(
    "src/olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
)

NumpyFSLDatasetConfig.from_src_mix(
    src_mix=SourceMixtureDatasetConfig(
        source_list=source_list,
        requested_tokens=MAX_TOKENS,
        global_batch_size=GLOBAL_BATCH_SIZE,
        processes=16,
        seed=SEED,
    ),
    ...
)
```

这对应报告里的 Dolma 3 Dolmino Mix：通过 YAML 给每个 source 设置 `target_ratio`，再由 `SourceMixtureDatasetConfig.build()` 根据 requested tokens、global batch size 和 sequence length 计算每个源应抽取多少 token。

32B 报告中说 midtraining 会跑两次、不同数据顺序 seed 后做 model soup。OLMo-core 里相关工具包括：

- `src/scripts/merge_core_checkpoints.py`
- `src/scripts/merge_hf_checkpoints.py`
- `src/scripts/reshard_core_checkpoint.py`

这里的 **model soup** 指的是把多个训练得到的 checkpoint 权重做平均/合并，得到一个新的单模型 checkpoint。最简单的形式是：

```text
W_soup = (W_1 + W_2 + ... + W_n) / n
```

它不是 ensemble。ensemble 是推理时同时跑多个模型再合并输出；model soup 是在训练后离线合并权重，最终推理时仍然只加载和运行一个模型。

在 32B midtraining 的语境里，可以理解为：用相同架构和相近训练配方跑多个 100B-token midtraining run，它们可能只在数据顺序、seed 或 ingredient 上略有不同；最后把这些 checkpoint 的参数平均，试图抵消单个 run 的随机噪声，让最终模型更稳。这个方法通常要求参与 soup 的模型处在相近的参数区域：同架构、同 tokenizer、同参数命名、训练阶段接近，否则直接平均权重可能没有意义甚至损坏模型。

### 3.3 Long-context extension

7B 入口：`src/scripts/train/OLMo3/OLMo3-7B-long-context.py`

```python
SEQ_LENGTH = 65536
GLOBAL_BATCH_SIZE = 2**22
MAX_TOKENS = 50_000_000_000

TransformerConfig.olmo3_7B(...).with_rope_scaling(
    YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
)

cookbook.configure_train_module(
    max_sequence_length=SEQ_LENGTH,
    rank_microbatch_size=SEQ_LENGTH,
    float8_enabled=True,
    activation_memory_budget=0.7,
    cp_degree=8,
    dp_shard_degree=1,
)

NumpyPackedFSLDatasetConfig.glob(..., generate_doc_lengths=True)
```

32B 入口：`src/scripts/train/OLMo3/OLMo3-32B-long-context.py`

```python
SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024
MAX_TOKENS = 100_000_000_000

cp_config=TransformerContextParallelConfig.llama3(
    degree=8,
    head_stride=4,
)

dp_config=HSDP(shard_degree=8)
activation_memory_budget=0.3
```

长上下文阶段的三件事要绑定理解：

1. `YaRNRoPEScalingConfig(factor=8, old_context_len=8192)`：把 RoPE 从 8K 扩到 64K。
2. `cp_degree=8`：把单条 64K 序列切到 8 个 CP rank，每个 rank 约 8K tokens。
3. `generate_doc_lengths=True`：保留 packed sample 内部文档边界，模型 forward 时生成 `cu_doc_lens`，让 attention/CP 能避免不合理跨文档依赖。

#### 3.3.1 RoPE / YaRN 如何实现上下文扩充

RoPE，也就是 rotary positional embedding，不是把一个“位置向量”加到 token embedding 上，而是在 attention 里对 Q/K 做旋转。每个位置 `pos` 会对应一组 sin/cos 角度，模型用这些角度旋转 query/key。旋转角度由位置和频率共同决定：

```text
angle(pos, dim) = pos * inv_freq(dim)
```

这样做的效果是：attention score 里会自然带上 token 之间的相对位置信息。预训练时模型只见过 8K 长度，所以原始 RoPE 频率主要在 0..8191 这个位置范围内被优化过。直接把位置推到 64K，角度会进入模型没怎么见过的区域，可能导致长距离 attention 表现不稳。

长上下文扩展不是简单把 `max_sequence_length` 改成 65536；还要告诉 RoPE 如何在更长位置上产生“模型能接受”的角度。这里用的是 YaRN：

```python
YaRNRoPEScalingConfig(
    factor=8,
    beta_fast=32,
    beta_slow=1,
    old_context_len=8192,
)
```

`factor=8` 的直觉是把位置尺度压缩 8 倍：64K 新上下文大约映射回 8K 旧上下文的频率范围。最朴素的插值可以理解成：

```text
inv_freq_scaled = inv_freq_original / 8
angle_new(pos) = pos * inv_freq_scaled
```

这样位置 65536 对应的旋转角度，大致像原来位置 8192 附近的角度，模型更容易迁移。但纯粹把所有频率都除以 8 也有问题：高频维度负责短距离、细粒度位置信息，全部压缩可能伤害局部建模。

YaRN 的做法是混合两套频率：

```text
extrapolation frequency = 原始 RoPE 频率
interpolation frequency = 原始 RoPE 频率 / factor
```

然后用 `beta_fast` / `beta_slow` 控制一个线性 ramp：一部分高频维度更接近原始频率，保留短距离能力；一部分低频维度更接近缩放后的频率，负责支持更长上下文。代码里对应 `src/olmo_core/nn/rope.py::YaRNRoPEScalingConfig.compute_scaled_inv_freq()`：

```text
inv_freq = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1 - ramp)
```

此外 YaRN 还会计算 `attention_rescale_factor = 0.1 * log(factor) + 1.0`，并把 sin/cos buffer 乘上这个系数，用来补偿上下文变长后 attention logit 尺度的变化。RoPE buffer 真正生成的位置在 `RotaryEmbedding._get_rotary_embedding()`：它根据新的 `seq_len=65536` 生成 sin/cos，再在 CP 开启时沿 sequence 维切给不同 CP rank。

在 OLMo3 代码里，`.with_rope_scaling(...)` 默认只把 scaling 加到 full attention 层，而跳过 sliding-window attention 层。原因是 sliding-window 层本来只看局部窗口，例如 4096；真正需要跨越 64K 全局距离的是 full attention 层。

#### 3.3.2 `doc_lens` / `cu_doc_lens` 是什么

长上下文训练通常会把多个文档 pack 到一条固定长度样本里。比如一条 64K sequence 可能不是一个完整文档，而是多个文档拼起来：

```text
sample tokens:
  [docA 12000 tokens][docB 8000 tokens][docC 45536 tokens]

doc_lens:
  [12000, 8000, 45536]

cu_doc_lens:
  [0, 12000, 20000, 65536]
```

`doc_lens` 就是 packed sample 内每个原始文档的长度。它来自数据集侧的 `generate_doc_lengths=True`，代码会根据 EOS/BOS 这类文档边界 token 调用 `get_document_lengths()` 计算。collator 会把 batch 内不同数量的文档长度 pad 到同一形状，并产生：

```text
batch["doc_lens"]
batch["max_doc_lens"]
```

进入 `Transformer.forward()` 后，模型会调用 `get_cumulative_document_lengths(doc_lens)`，把每个文档长度转成累计边界 `cu_doc_lens`。`cu` 是 cumulative 的意思，即“累计长度”。attention 后端拿到这些边界后，就知道哪些 token 属于同一个文档，哪些 token 虽然在同一条 packed sequence 里但其实来自不同文档。

为什么这很重要？如果不保留文档边界，模型可能会让 docB 的开头 token attend 到 docA 的结尾 token，好像它们是同一个连续文本。这对 packed training 来说是不合理的跨文档依赖。`cu_doc_lens` 让 attention/CP 能按文档边界做 intra-document masking：同一文档内正常 causal attention，不同文档之间避免错误连接。

在 CP 场景下它还更关键。64K sequence 被切到 8 个 CP rank 后，一个文档可能跨 rank，文档边界也可能落在某个 rank 中间。CP load balancer 需要 `cu_doc_lens` 来正确切分输入和 RoPE buffer，并生成每个 rank 本地的文档边界信息，否则跨 rank attention 很难知道哪些位置属于同一文档。

## 4. 模型架构：报告参数如何落到代码

报告 Table 33 的架构信息：

- decoder-only Transformer。
- 7B：32 layers, hidden size 4096, 32 Q heads, 32 KV heads。
- 32B：64 layers, hidden size 5120, 40 Q heads, 8 KV heads，即 GQA。
- SwiGLU。
- QK-Norm。
- RMSNorm。
- RoPE theta = 500000。
- 3/4 层使用 4096 sliding window attention，最后一层强制 full attention。
- LM head 前有输出 norm。
- embedding 不做 weight decay。
- z-loss weight 为 `1e-5`。

这些主要在 `src/olmo_core/nn/transformer/config.py` 中落地：

```python
TransformerConfig.olmo3_7B(...)
  -> TransformerConfig.olmo2_7B(...)
  -> llama2_7B(...)
  -> llama_like(...)

TransformerConfig.olmo3_32B(...)
  -> TransformerConfig.olmo2_32B(...)
  -> llama_like(...)
```

OLMo3 相比 OLMo2 的核心代码差异是 OLMo3 wrapper 传入了 sliding window attention：

```python
SlidingWindowAttentionConfig(
    force_full_attention_on_first_layer=False,
    force_full_attention_on_last_layer=True,
    pattern=[4096, 4096, 4096, -1],
)
```

`pattern=[4096, 4096, 4096, -1]` 的含义是每 4 层中前 3 层用 4096 窗口，后一层 full attention；同时最后一层总是 full attention。对应报告说的 “SWA at three out of every four layers, last layer full attention”。

`llama_like()` 负责拼出 block：

```python
TransformerBlockConfig(
    sequence_mixer=AttentionConfig(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope=RoPEConfig(theta=rope_theta, scaling=rope_scaling),
        qk_norm=layer_norm if qk_norm else None,
        sliding_window=sliding_window,
    ),
    feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False),
    layer_norm=RMSNorm(...),
)
```

### 4.1 Transformer.forward()

路径：`src/olmo_core/nn/transformer/model.py`

主拓扑：

```text
input_ids
  -> _prepare_inputs()
  -> embeddings
  -> optional embedding_norm
  -> blocks[0..n_layers-1]
  -> lm_head
  -> logits or LMOutputWithLoss
```

`_prepare_inputs()` 做的事情比名字重要：

- 把 `input_ids/labels` 移到训练设备。
- 处理 `loss_div_factor`，传给 block 和 lm_head。
- 如果 batch 有 `doc_lens/max_doc_lens`，转成 `cu_doc_lens`，用于文档边界。
- 如果启用 CP，沿 sequence 维切分 `input_ids`、`labels` 和 RoPE buffers。
- 为每层准备 `per_block_kwargs`，尤其是 CP 分片后的 RoPE buffer。

`forward()` 本体保持简单：

```python
h = self.embeddings(input_ids)
for block_key, block in self.blocks.items():
    h = block(h, **all_block_kwargs, **block_kwargs)
return self.lm_head(h, labels=labels, ...)
```

### 4.2 TransformerBlock.forward()

路径：`src/olmo_core/nn/transformer/block.py`

普通 block 是标准 pre-norm：

```text
x
  -> attention_norm
  -> sequence_mixer / attention
  -> residual add
  -> feed_forward_norm
  -> feed_forward
  -> residual add
```

代码中字段名仍叫 `self.attention`，但注释说明这是为了旧 checkpoint 兼容；实际类型是 `SequenceMixer`，可以是 attention、recurrent、convolution 或其他序列混合模块。这给二次研发留了扩展点。

## 5. 训练启动：从脚本到 Trainer.fit()

OLMo3 脚本有两种风格：

1. 直接写 `build_experiment_config(cli_context)`，如 `OLMo3-7B-midtraining.py`。
2. 用 `functools.partial(build_config, ...)` 组装，如 `OLMo3-7B.py`、`OLMo3-32B-long-context.py`。

两者最终都进入 `src/olmo_core/internal/experiment.py`：

```python
main(config_builder=...)
  -> parse SubCmd: launch / train / train_single / prep / dry_run
  -> cmd.prepare_environment(config)
  -> cmd.run(config)
```

真正训练在 `train(config)`：

```python
seed_all(config.init_seed)
model = config.model.build(init_device="meta")
train_module = config.train_module.build(model)
data_loader = _build_data_loader(config, dp_process_group=train_module.dp_process_group)
trainer = config.trainer.build(train_module, data_loader)
trainer.fit()
```

`init_device="meta"` 很关键。它先创建没有真实参数存储的模型壳，随后 `TransformerTrainModule` 根据 DP/TP/CP/FP8/AC/compile 配置改造模型，最后在 `parallelize_model()` 末尾调用 `init_weights()` materialize 参数。这样能降低大模型初始化的显存峰值。

## 6. Trainer：外层训练调度器

路径：`src/olmo_core/train/trainer.py`

`Trainer` 不关心 Transformer 内部结构，它只调度训练：

```text
fit()
  -> maybe_load_checkpoint(save_folder)
  -> maybe_load_checkpoint(load_path)
  -> callbacks.pre_train()
  -> train_module.pre_train()
  -> _dry_run_batch()
  -> while not training_complete:
       _fit_epoch()
  -> callbacks.post_train()
  -> shutdown
```

`_fit_epoch()` 是每个 step 的骨架：

```text
data_loader.reshuffle(epoch)
train_module.zero_grads()

for batch in data_loader:
  global_step += 1
  global_train_tokens_seen += global_num_tokens
  callbacks.pre_step(batch)
  train_module.train_batch(batch)
  callbacks.pre_optim_step()
  train_module.optim_step()
  train_module.zero_grads()
  callbacks.post_train_batch()
  callbacks.post_step()
  maybe log metrics
```

注意这里的 optimizer step 发生在完整 batch 之后，而不是每个 micro-batch 之后。micro-batch 只是显存层面的切分。

## 7. TransformerTrainModule：一次训练 step 的核心

路径：`src/olmo_core/train/train_module/transformer/train_module.py`

这是理解训练细节最重要的类。

### 7.1 初始化阶段

`TransformerTrainModule.__init__()` 做五件事：

1. 校验 `rank_microbatch_size % max_sequence_length == 0`。
2. 如果分布式训练，调用 `build_world_mesh()` 建立 DP/TP/CP/EP 设备网格。
3. 校验 activation checkpointing 的 budget 模式必须配合 `compile_model=True`。
4. 调用 `parallelize_model()` 改造模型并初始化权重。
5. 构建 optimizer。

`rank_microbatch_size` 的单位是 token。比如 `max_sequence_length=8192`，`rank_microbatch_size=16384` 表示每个 rank 每个 micro-batch 放 2 条 8192 长度序列。

### 7.2 train_batch()

`train_batch(batch)` 是核心训练计算：

```text
batch
  -> set model.train()
  -> 如果没有 labels，则由 input_ids 右移生成 labels
  -> 统计完整 batch 的有效 loss token 数
  -> 按 rank_microbatch_size 切 micro-batches
  -> 对每个 micro-batch:
       _prepare_batch()
       model_forward(...)
       loss.backward()
  -> model.post_batch()
  -> 记录 CE / z-loss / auxiliary metrics
```

labels 逻辑来自 `olmo_core.data.utils.get_labels()`。自回归 LM 通常是：

```text
input_ids = [11, 12, 13, 14]
labels    = [12, 13, 14, -100]
```

`-100` 是 `label_ignore_index`，不参与 CE loss。

### 7.3 micro-batch 的 loss 归一化

这是代码里最容易误解但很关键的点。

`train_batch()` 先在完整 batch 上算：

```python
batch_num_tokens_for_loss = (batch["labels"] != -100).sum()
```

然后每个 micro-batch 都用同一个分母：

```python
loss_reduction="sum"
loss_div_factor=batch_num_tokens_for_loss
```

因此第 i 个 micro-batch 的梯度贡献是：

```text
sum(loss_tokens_in_micro_batch_i) / batch_num_tokens_for_loss
```

所有 micro-batch backward 累积后等价于：

```text
完整 batch 的 token loss 总和 / 完整 batch 的有效 token 数
```

这保证 micro-batch 切分不改变数学上的 batch loss。如果每个 micro-batch 各自除以自己的 token 数，padding、mask 或最后一个较小 micro-batch 会改变梯度权重。

### 7.4 分布式下的梯度同步

`_train_microbatch_context()` 控制 micro-batch 间的同步：

- FSDP/HSDP：非最后一个 micro-batch 不做完整梯度同步，最后一个 backward 再同步。
- HSDP：通过 `set_requires_all_reduce(is_last_mb)` 延迟 all-reduce。
- DDP：非最后一个 micro-batch 进入 `no_sync()`。

所以一次完整 batch 如果切成 4 个 micro-batch，行为是：

```text
forward/backward x 4
gradient sync only around final micro-batch
optimizer.step x 1
zero_grad x 1
```

### 7.5 optim_step()

`optim_step()` 做：

1. `max_grad_norm` 梯度裁剪。
2. scheduler 根据 `trainer.global_step` 或 `global_train_tokens_seen` 写入 LR。
3. `optim.step()`。
4. 如果是 `SkipStepOptimizer`，记录 step 是否跳过。
5. `model.post_optim_step()`，例如 FP8/FSDP 动态 scale 预计算。

## 8. parallelize_model()：并行与显存控制的总开关

路径：`src/olmo_core/train/train_module/transformer/common.py`

顺序非常重要：

```text
parallelize_model()
  -> apply pipeline parallelism if enabled
  -> apply FP8
  -> apply context parallelism
  -> apply tensor parallelism
  -> apply expert parallelism
  -> apply activation checkpointing
  -> apply torch.compile
  -> apply DDP/FSDP/HSDP
  -> init_weights()
```

不能随意调整顺序。CP/TP/EP 会改变模块布局或 DTensor layout；AC 和 compile 要在 FSDP/DDP wrapper 之前；FSDP/DDP 最后包模型并接管参数同步。

### 8.1 Data Parallel: DDP / FSDP / HSDP

配置入口：

```python
TransformerDataParallelConfig(
    name=DataParallelType.hsdp,
    param_dtype=DType.bfloat16,
    reduce_dtype=DType.float32,
    wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
    shard_degree=64,
)
```

区别：

- DDP：每个 rank 保存完整模型，反向后 all-reduce 梯度，简单但显存高。
- FSDP：参数、梯度、optimizer state 按 DP rank 分片，计算时临时 all-gather。
- HSDP：hybrid sharded data parallel，通常一维 shard、一维 replicate；报告 Table 34 里的 DP-shard / DP-rep 就对应这个思路。

`Transformer.apply_fsdp()` 会包：

- 每个 block。
- embeddings。
- optional embedding norm。
- lm_head。
- root module。

`wrapping_strategy` 控制粒度：

- `full`：block、lm_head 等都包。
- `blocks`：主要包 blocks，lm_head 不单独包。
- `fine_grained`：block 内部 attention/MLP 也单独包，更省峰值显存但通信更碎。

### 8.2 Tensor Parallel

TP 把单层内部的大矩阵计算切到多个 rank。入口是 `Transformer.apply_tp()`：

- embeddings 用 `RowwiseParallel`，输出变成 sequence-sharded。
- norm 和 dropout 用 `SequenceParallel`。
- 每个 block 调 `block.apply_tp()`。
- lm_head 调 `lm_head.apply_tp()`。

OLMo3 Base 官方脚本主要依赖 HSDP 和 CP，TP 不是这几个脚本的主配置，但框架支持。

### 8.3 Context Parallel

CP 是长上下文阶段的关键。入口：

```python
TransformerContextParallelConfig.llama3(degree=8, head_stride=4)
```

作用：

- 把 sequence/context 维度切到多个 rank。
- 每个 CP rank 只处理一段局部上下文。
- attention 通过 ring/ulysses 风格通信获得跨分片的 K/V 或 attention 信息。
- lm_head 知道 CP 存在，loss 分母会按 CP degree 调整。

在 `Transformer._prepare_inputs()` 中，CP 会切：

- `input_ids`
- `labels`
- RoPE buffers
- 可选的文档边界信息

如果同时启用 CP 和 intra-document masking，代码要求 rank micro-batch 只有一条 instance，因为要按文档边界准确 shard。

### 8.4 Activation Checkpointing

入口：

```python
TransformerActivationCheckpointingConfig(
    mode=TransformerActivationCheckpointingMode.budget,
    activation_memory_budget=0.5,
)
```

模式：

- `full`：每个 block 都 checkpoint。
- `selected_blocks`：按 block interval 包一部分层。
- `selected_modules`：按模块名 glob 包。
- `budget`：交给 `torch.compile` 的 activation memory budget 机制。

OLMo3 32B 预训练和 midtraining 使用 budget 模式；长上下文阶段也使用更激进的 memory budget。

## 9. 数据路径：从 Dolma mix 到 batch

### 9.1 预训练 data mix

入口：

```python
NumpyFSLDatasetConfig.from_data_mix(
    DataMix.OLMo_mix_0925,
    tokenizer=common.tokenizer,
    mix_base_dir=common.root_dir,
    sequence_length=8192,
)
```

`DataMix` 对应 `src/olmo_core/data/mixes/*.txt`，每行大致是：

```text
source_label,preprocessed/.../{TOKENIZER}/.../*.npy
```

`NumpyDatasetConfig._resolve_paths_metadata()` 会：

1. 根据 tokenizer identifier 替换 `{TOKENIZER}`。
2. 拼出真实路径。
3. 生成 metadata label。
4. 可选按 `source_permutation_seed` 打乱 source 文件顺序。

### 9.2 midtraining source mixture

入口：

```python
SourceMixtureList.from_yaml(...)
SourceMixtureDatasetConfig(...).build(...)
NumpyFSLDatasetConfig.from_src_mix(...)
```

`SourceMixtureList.validate()` 要求所有 `target_ratio` 加起来约等于 1。

`SourceMixtureDatasetConfig.build()` 会：

1. 展开每个 source 的 glob。
2. 按 numpy dtype 和文件大小估算 token 数。
3. 根据 `requested_tokens * target_ratio` 计算每个 source 需要多少 token。
4. 检查 `max_repetition_ratio` 和 `max_source_fraction` 是否足够。
5. 根据 `global_batch_size / sequence_length` 把 token 数 round 到完整 instance。
6. 返回可被 `NumpyFSLDatasetMixture` 使用的路径和 offset index。

这就是报告里“100B-token midtraining mix”的代码表达。

### 9.3 long-context packed dataset

入口：

```python
NumpyPackedFSLDatasetConfig.glob(
    ".../*.npy",
    sequence_length=65536,
    generate_doc_lengths=True,
    source_group_size=8,
)
```

`NumpyPackedFSLDataset` 会把文档 pack 成固定长度 instance，并返回：

- `input_ids`
- `label_mask`
- `doc_lens`，如果 `generate_doc_lengths=True`
- `instance_mask`，如果开启 instance filter

`doc_lens` 经过 collator 后进入 `Transformer._prepare_inputs()`，变成 `cu_doc_lens`，服务长上下文文档边界处理。

### 9.4 DataLoader 和 Collator

`NumpyDataLoaderConfig.build()` 最终会包装成 `NumpyFSLDataLoader` 或 `NumpyVSLDataLoader`。

对 OLMo3 主线，固定长度 FSL 最关键：

```text
NumpyFSLDataLoader
  -> reshuffle(epoch)
  -> 构造 global_indices
  -> 按 global batch 切 batch
  -> 按 dp_rank 切本 rank 的样本
  -> dataset[idx]
  -> DataCollator(items)
```

`DataCollator` 输出常见字段：

- `input_ids`: `(rank_batch_instances, seq_len)`
- `label_mask`: 可选，控制哪些 token 参与 label。
- `doc_lens`: 可选，packed documents 的文档长度。
- `max_doc_lens`: 可选，每条 instance 的最大文档长度。
- `instance_mask`: 可选，过滤重复/异常样本。
- `metadata/index`: 可选，用于追踪数据来源。

`TextDataLoaderBase.__iter__()` 会校验本 rank batch 的 token 数必须等于：

```text
rank_batch_size = global_batch_size // dp_world_size
```

这解释了为什么 OLMo-core 中 batch size 几乎都用 token 表达。

## 10. LM head 与 loss

路径：`src/olmo_core/nn/lm_head.py`

`LMHead.forward()` 在传入 labels 时直接返回 `LMOutputWithLoss`：

```python
LMOutputWithLoss(
    logits,
    loss,
    ce_loss,
    z_loss,
)
```

loss 有两种实现：

- `default`：先算 logits，再用 cross entropy。
- `fused_linear`：把 final linear 和 CE 融合，节省显存。

OLMo3 训练脚本里 `TransformerTrainModule.train_batch()` 调用：

```python
return_logits=False
loss_reduction="sum"
loss_div_factor=batch_num_tokens_for_loss
z_loss_multiplier=1e-5
```

因此训练时不会保留完整 logits，显存更低。`z_loss` 是稳定训练的正则项，报告中也列出 z-loss weight 为 `1e-5`。

如果启用 TP/CP，`LMHead._finalize_loss()` 会调整 `loss_div_factor`，避免同一个 token 分片后重复或漏计。

## 11. Optimizer 和 scheduler

OLMo3 脚本主要用：

```python
SkipStepAdamWConfig(
    lr=...,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    group_overrides=[
        OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
    ],
)
```

报告 Table 33 中 “Weight decay on embeddings: No” 对应这个 override。

`SkipStepOptimizer` 的核心思想：

- 维护最近一段 loss 和 grad norm。
- 当前 step 如果超过均值若干 sigma，则 `step_factor=0`。
- AdamW update 中所有更新乘以 `step_factor`。
- 这样跳过异常 step 时尽量避免 host-device sync。

### 11.1 `SkipStepAdamW` 的原理和实现

`SkipStepAdamW` 分成两层理解：

第一层是父类 `SkipStepOptimizer`，路径是 `src/olmo_core/optim/skip_step_optimizer.py`。它不关心 AdamW 的公式，只负责判断“这一步要不要跳”。训练循环在 `TransformerTrainModule.optim_step()` 里会先把当前 batch 的 loss 和 grad norm 写到 optimizer：

```python
optim.latest_loss = ...
optim.latest_grad_norm = ...
optim.step()
```

父类内部维护两个滚动窗口：

```python
self._losses
self._grad_norms
```

窗口长度由 `rolling_interval_length` 控制，默认是 128。`get_step_factor()` 的逻辑是：

```text
如果历史步数还不够：
  step_factor = 1.0
否则：
  loss_mean/loss_std = 过去窗口 loss 的均值和标准差
  grad_mean/grad_std = 过去窗口 grad norm 的均值和标准差

  如果当前 loss 没有超过 loss_mean + sigma_factor * loss_std
  且当前 grad norm 没有超过 grad_mean + sigma_factor * grad_std:
      step_factor = 1.0
  否则:
      step_factor = 0.0
```

默认 `sigma_factor=6`，意思是只跳过非常极端的尖峰。`rolling_interval_length=128` 时，代码会等历史至少达到 `max(2, 128 // 2) = 64` 条以后才开始判断；前 64 步直接更新，避免统计窗口太小导致误判。

举个简化例子：假设过去一段 loss 均值是 2.0，标准差是 0.1，`sigma_factor=6`，阈值就是 `2.0 + 6 * 0.1 = 2.6`。如果当前 loss 是 2.3，正常更新；如果当前 loss 突然变成 20.0，就会得到 `step_factor=0`，这一步不会更新参数。

第二层是 `SkipStepAdamW` 本身，路径是 `src/olmo_core/optim/adamw.py`。它把 `step_factor` 乘进 AdamW 的每个关键位置：

```python
p.mul_(1 - step_factor * (lr * weight_decay))
exp_avg.lerp_(grad, step_factor * (1 - beta1))
exp_avg_sq.mul_(1 - step_factor * (1 - beta2))
exp_avg_sq.add_(step_factor * grad * grad, alpha=1 - beta2)
update.mul_(step_factor)
step.add_(step_factor)
```

当 `step_factor=1` 时，这就是标准 AdamW：做 weight decay，更新一阶/二阶动量，计算 bias correction，更新参数，并把 Adam step 加 1。

当 `step_factor=0` 时，所有关键动作都变成 no-op：

- weight decay 不发生。
- `exp_avg` 和 `exp_avg_sq` 不被异常梯度污染。
- 参数更新量是 0。
- Adam 的 `step` 不增加，bias correction 不会被一个“没有真实更新”的 step 推进。

这和“在 Python 里写 `if should_skip: return`”很像，但实现上更适合大规模 GPU 训练。`step_factor` 是一个在设备上的 tensor，直接参与 CUDA kernel 计算，不需要把 GPU 上的判断结果同步回 CPU。注释里提到的 host-device sync 就是这个意思：如果每一步都把布尔值从 GPU 拿回 Python 决策，会引入同步等待，影响吞吐。

`foreach=True` 时走 `_step_foreach()`，它把许多参数 tensor 收集成列表，用 `torch._foreach_*` multi-tensor kernel 批量更新；`foreach=False` 时走 `_step()`，逐参数调用 `adamw_step()`。两条路径的数学语义相同，默认配置里 `foreach=True` 是为了更好的性能。

`step_increment_bugfix=True` 是一个兼容开关。注释说明如果设成 `False`，旧行为不会正确增加 Adam step，等效学习率会偏高且 bias correction 不正确；新训练应保持默认 `True`。

### 11.2 `@OptimConfig.register("skip_step_adamw")` 和 `@dataclass`

这两行是 Python 装饰器。装饰器的形式是 `@something`，会在类定义完成后接收这个类，并返回一个类。这里它们叠在一起：

```python
@OptimConfig.register("skip_step_adamw")
@dataclass
class SkipStepAdamWConfig(OptimConfig[SkipStepAdamW]):
    ...
```

执行顺序是从下往上：

1. `@dataclass` 先处理 `SkipStepAdamWConfig`。
2. `@OptimConfig.register("skip_step_adamw")` 再处理 dataclass 化后的类。

`@dataclass` 来自 Python 标准库 `dataclasses`。它会读取类里的字段声明：

```python
lr: float = 1e-3
betas: Tuple[float, float] = (0.9, 0.999)
weight_decay: float = 1e-2
```

然后自动生成初始化函数和一些样板方法。没有它，你需要手写类似下面的代码：

```python
def __init__(self, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, ...):
    self.lr = lr
    self.betas = betas
    self.weight_decay = weight_decay
```

有了 `@dataclass`，训练脚本就可以自然地写：

```python
SkipStepAdamWConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
```

`@OptimConfig.register("skip_step_adamw")` 是 OLMo-core 的可注册配置机制。`OptimConfig` 继承了 `Registrable`，所以每个具体 optimizer config 都可以用一个字符串名字注册。注册后，配置序列化时能带上：

```yaml
type: skip_step_adamw
lr: 0.0003
weight_decay: 0.1
```

反序列化或从配置文件加载时，框架看到 `type: skip_step_adamw`，就能从注册表找到 `SkipStepAdamWConfig`，再根据其 dataclass 字段构造对象。这个设计让配置文件不必写完整 Python import 路径，也让同一个 `optim` 字段可以承载不同优化器，比如 `adamw`、`skip_step_adamw`、`lion`。

最后，`SkipStepAdamWConfig.optimizer()` 返回真正的优化器类：

```python
@classmethod
def optimizer(cls) -> Type[SkipStepAdamW]:
    return SkipStepAdamW
```

所以完整链路是：

```text
配置文件/脚本里的 SkipStepAdamWConfig
  -> OptimConfig.build(model)
  -> build_groups(model)，处理 group_overrides
  -> self.optimizer() 得到 SkipStepAdamW
  -> SkipStepAdamW(param_groups, lr=..., betas=..., ...)
```

scheduler：

- Pretraining：`CosWithWarmup(warmup_steps=2000)`，对应报告里的 cosine schedule。
- Midtraining：`LinearWithWarmup(warmup=0, alpha_f=0.0)`，从当前 LR 线性衰减到 0。
- Long-context：`LinearWithWarmup(warmup=200, alpha_f=0.0)`。

`Scheduler.set_lr()` 根据 `trainer.global_step` 或 `trainer.global_train_tokens_seen` 动态写 optimizer param group。

## 12. Checkpoint、恢复和 model soup

训练恢复逻辑在 `Trainer.fit()`：

1. 优先从 `save_folder` 恢复，用于断点续训，会恢复 trainer state、optimizer state、data loader state。
2. 如果 `save_folder` 没有 checkpoint，再尝试 `load_path`。
3. 如果 `load_strategy=always` 且找不到 checkpoint，直接报错。

`CheckpointerCallback` 负责：

- `pre_train()` 可保存 step 0 checkpoint。
- `post_train_batch()` 按 `save_interval` 或 `ephemeral_save_interval` 保存。
- 支持 async checkpoint。
- 训练结束时如果当前 step 尚未保存，会再保存一次。

checkpoint 内容包括：

- model state
- optimizer state
- trainer state
- data loader state
- RNG state
- callbacks state

因此继续训练时不仅恢复权重，还能恢复数据顺序和 step/tokens 计数。

报告里的 32B midtraining model soup 可结合脚本工具理解：

- `merge_core_checkpoints.py`：合并 OLMo-core checkpoint。
- `reshard_core_checkpoint.py`：改变 checkpoint shard 布局，适配新并行配置。
- `merge_hf_checkpoints.py`：Hugging Face checkpoint 层面的合并。

## 13. 评测和 callbacks

OLMo3 脚本常用：

```python
TrainerConfig(...).with_recommended_evals(
    tokenizer,
    sequence_length,
    cluster,
    task_set="fast",
    eval_interval=1000,
)
```

`with_recommended_evals()` 添加：

- `DownstreamEvaluatorCallbackConfig`
- `LMEvaluatorCallbackConfig`

`LMEvaluator` 通常用 `DataMix.v3_small_ppl_validation` 作为 PPL 验证集。报告中 OlmoBaseEval 的完整评测体系远超这个 callback，但 OLMo-core 的训练循环通过 callbacks 提供了周期性评测接口。

常见工程 callbacks：

- `ConfigSaverCallback`：保存 config。
- `CheckpointerCallback`：保存/清理 checkpoint。
- `WandBCallback` / `CometCallback`：实验记录。
- `GPUMemoryMonitorCallback`：显存监控。
- `SpeedMonitorCallback`：吞吐和 MFU 统计。
- `GarbageCollectorCallback`：控制 GC。
- `SlackNotifierCallback`：通知。

## 14. 如何按代码精读 OLMo3

建议按下面顺序读，不要一开始陷进所有细枝末节。

### 第一遍：跑通主链路

1. `src/scripts/train/OLMo3/README.md`
2. `src/scripts/train/OLMo3/OLMo3-7B.py`
3. `src/olmo_core/internal/experiment.py`
4. `src/olmo_core/train/config.py`
5. `src/olmo_core/train/trainer.py`

目标：知道一个脚本如何变成 `Trainer.fit()`。

### 第二遍：掌握一次 step

1. `src/olmo_core/train/train_module/transformer/config.py`
2. `src/olmo_core/train/train_module/transformer/train_module.py`
3. `src/olmo_core/nn/lm_head.py`
4. `src/olmo_core/data/utils.py`

目标：彻底理解 labels、micro-batch、loss 分母、backward、optimizer step。

### 第三遍：掌握模型结构

1. `src/olmo_core/nn/transformer/config.py`
2. `src/olmo_core/nn/transformer/model.py`
3. `src/olmo_core/nn/transformer/block.py`
4. `src/olmo_core/nn/attention/__init__.py`
5. `src/olmo_core/nn/feed_forward.py`
6. `src/olmo_core/nn/rope.py`

目标：把报告 Table 33 的结构逐项映射到代码。

### 第四遍：掌握并行和长上下文

1. `src/olmo_core/train/train_module/transformer/common.py`
2. `src/olmo_core/distributed/parallel/__init__.py`
3. `src/olmo_core/distributed/parallel/data_parallel.py`
4. `src/olmo_core/distributed/parallel/context_parallel.py`
5. `src/olmo_core/nn/attention/ring.py`
6. `src/scripts/train/OLMo3/OLMo3-32B-long-context.py`

目标：理解 HSDP、CP、YaRN、doc_lens 如何共同支撑 64K 训练。

### 第五遍：掌握数据

1. `src/olmo_core/data/mixes/*.txt`
2. `src/olmo_core/data/source_mixture.py`
3. `src/olmo_core/data/numpy_dataset.py`
4. `src/olmo_core/data/data_loader.py`
5. `src/olmo_core/data/collator.py`

目标：知道 Dolma 3 Mix / Dolmino Mix / Longmino Mix 如何从文件变成 batch。

## 15. 一张总览图

```text
OLMo3 report
  ├─ architecture choices
  │   └─ nn/transformer/config.py
  │       └─ olmo3_7B / olmo3_32B / llama_like
  │
  ├─ pretraining recipe
  │   └─ scripts/train/OLMo3/OLMo3-{7B,32B}.py
  │       ├─ DataMix.OLMo_mix_0625/0925
  │       ├─ CosWithWarmup
  │       └─ HSDP + compile + z-loss
  │
  ├─ midtraining recipe
  │   └─ scripts/train/OLMo3/OLMo3-{7B,32B}-midtraining.py
  │       ├─ SourceMixtureList YAML
  │       ├─ requested_tokens=100B
  │       ├─ LinearWithWarmup(alpha_f=0)
  │       └─ load pretrained checkpoint
  │
  ├─ long-context recipe
  │   └─ scripts/train/OLMo3/OLMo3-{7B,32B}-long-context.py
  │       ├─ seq_len=65536
  │       ├─ YaRNRoPEScalingConfig(factor=8)
  │       ├─ ContextParallelConfig.llama3(degree=8)
  │       └─ NumpyPackedFSLDatasetConfig(generate_doc_lengths=True)
  │
  └─ runtime execution
      ├─ internal/experiment.py
      ├─ train/trainer.py
      ├─ train/train_module/transformer/train_module.py
      ├─ train/train_module/transformer/common.py
      ├─ nn/transformer/model.py
      ├─ nn/transformer/block.py
      ├─ nn/lm_head.py
      └─ data/{numpy_dataset,data_loader,collator,source_mixture}.py
```

## 16. 需要特别留意的边界

- OLMo-core 当前主要覆盖 pretraining、midtraining、long-context extension 和 SFT；DPO/RL/RL-Zero 的完整训练框架不在这条代码主线里。
- `global_batch_size` 和 `rank_microbatch_size` 都是 token 数，不是样本数。
- CP 不是数据并行。CP rank 处理同一条长序列的不同片段；参数仍需同步，所以 `get_dp_model_mesh()` 会把 CP 维度纳入模型 DP 同步维度。
- `DataLoader` 的 `dp_process_group` 必须和 `TrainModule.dp_process_group` 对齐，否则会出现数据重复或错分。
- `save_folder` 用于 checkpoint，`work_dir` 用于本地数据缓存；远程 `save_folder` 不能直接当 `work_dir`。
- midtraining 的 YAML mix 是“目标比例”，最终 token 数会按 sequence length 和 global batch size round 到完整 instance。
- 长上下文的 `generate_doc_lengths=True` 不是普通 metadata，它会影响模型 forward 的文档边界处理和 CP 分片。

