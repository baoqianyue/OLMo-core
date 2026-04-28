# OLMo-core 主链路精读笔记

这份笔记面向二次研发和个人工作站迁移，重点解释 OLMo-core 从官方训练脚本到一次训练 step 的执行路径。

## 1. 总入口

官方脚本通常位于 `src/scripts/official/`，例如 `src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py`。

每个脚本的核心职责是实现 `build_config(opts, overrides)`，返回 `ExperimentConfig`。真正的通用启动逻辑在 `src/olmo_core/script_utils.py`：

```text
official script
  -> build_config()
  -> ExperimentConfig
  -> script_utils.main()
```

`ExperimentConfig` 包含五个主组件：

- `model`: `TransformerConfig`
- `dataset`: `NumpyDatasetConfig`
- `data_loader`: `NumpyDataLoaderConfig`
- `train_module`: `TransformerTrainModuleConfig`
- `trainer`: `TrainerConfig`

`script_utils.main()` 会依次执行：

```text
model = config.model.build(init_device="meta")
train_module = config.train_module.build(model)
dataset = config.dataset.build()
data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
trainer = config.trainer.build(train_module, data_loader)
trainer.fit()
```

`init_device="meta"` 是大模型训练常见技巧：先创建没有真实参数存储的 module，再由并行策略决定如何 materialize 到 GPU，减少初始化阶段显存峰值。

## 2. Trainer 的职责

`src/olmo_core/train/config.py` 中的 `TrainerConfig.build()` 负责把 TrainModule、DataLoader、Checkpointer 和 callbacks 组装成 `Trainer`。

`src/olmo_core/train/trainer.py` 中的 `Trainer.fit()` 是外层训练循环。它不直接知道 Transformer 的内部结构，主要负责：

- 从 `save_folder` 或 `load_path` 恢复 checkpoint。
- 调用 callbacks 的 `pre_train/post_train/on_error/close` 生命周期。
- 做 dry-run，提前触发 compile/FSDP 初始化和显存分配。
- 循环 epoch 和 batch。
- 调用 `train_module.train_batch()` 完成 forward/backward。
- 调用 optimizer step、zero grad、metrics 聚合和 checkpoint 保存。

可以把 `Trainer` 理解为训练调度器，把 `TrainModule` 理解为“怎么训练这个模型”的适配层。

## 3. TransformerTrainModule

Transformer 专用训练逻辑在 `src/olmo_core/train/train_module/transformer/train_module.py`。

`TransformerTrainModuleConfig` 是迁移到个人工作站时最重要的配置层：

- `rank_microbatch_size`: 每个 rank 单次 micro-batch 的 token 数，直接影响显存。
- `max_sequence_length`: 训练允许的最大序列长度。
- `dp_config`: DDP/FSDP/HSDP 等数据并行配置。
- `tp_config`: tensor parallel。
- `cp_config`: context parallel，主要服务长上下文。
- `ac_config`: activation checkpointing，个人工作站训练大模型时很关键。
- `compile_model`: 是否使用 `torch.compile()`。
- `float8_config`: FP8 线性层配置。
- `z_loss_multiplier`: LM head 的 z-loss 稳定项。

`TransformerTrainModule.train_batch()` 的关键步骤：

```text
batch
  -> 生成 labels
  -> 统计有效 loss token 数
  -> 按 rank_microbatch_size 切 micro-batch
  -> 每个 micro-batch 调 model_forward()
  -> lm_head 返回 loss / ce_loss / z_loss
  -> loss.backward()
  -> 完整 batch 后记录指标
```

这里的 loss 归一化使用整个 batch 的有效 token 数，而不是单个 micro-batch 的 token 数。这样在梯度累积时，不同 micro-batch 大小不会改变 loss 权重。

更具体地说，代码会先在完整 batch 上计算：

```python
batch_num_tokens_for_loss = (batch["labels"] != label_ignore_index).sum()
```

这里的“有效 token”指 label 不是 `-100` 的位置。自回归 LM 通常会把每条序列最后一个 label 设成 `-100`，padding、被 mask 掉的位置也可能是 `-100`。这些位置不会参与 cross entropy。

假设单个 rank 当前 batch 是 8 条序列，每条 1024 token：

```text
batch["labels"].shape = (8, 1024)
```

如果每条序列最后 1 个位置是 `-100`，那么总 token 数是：

```text
8 * 1024 = 8192
```

但真正参与 loss 的 token 数是：

```text
8 * 1023 = 8184
```

随后这个 batch 可能因为显存限制被切成多个 micro-batch。比如：

```text
rank_microbatch_size = 2048 tokens
seq_len = 1024
每个 micro-batch 样本数 = 2048 // 1024 = 2
完整 batch 被切成 4 个 micro-batch
```

训练时每个 micro-batch 都会 forward/backward 一次，但 optimizer 只在完整 batch 结束后 step 一次。关键点是，每个 micro-batch 的 loss 都不是除以“自己这个 micro-batch 的有效 token 数”，而是都传入同一个完整 batch 分母：

```python
loss_reduction="sum"
loss_div_factor=batch_num_tokens_for_loss
```

所以第 `i` 个 micro-batch 实际贡献的是：

```text
sum(loss_tokens_in_micro_batch_i) / batch_num_tokens_for_loss
```

4 个 micro-batch 的梯度累积起来后，等价于：

```text
(
  sum(loss_tokens_in_micro_batch_1)
  + sum(loss_tokens_in_micro_batch_2)
  + sum(loss_tokens_in_micro_batch_3)
  + sum(loss_tokens_in_micro_batch_4)
) / batch_num_tokens_for_loss
```

也就是：

```text
完整 batch 的 token loss 总和 / 完整 batch 的有效 token 数
```

这和“不切 micro-batch、一次性跑完整 batch”得到的归一化尺度一致。

如果反过来，让每个 micro-batch 都除以自己的有效 token 数，就会变成“先求每个 micro-batch 的平均 loss，再把这些平均值相加”。这样每个 micro-batch 的权重会接近相同，而不是按有效 token 数加权。当前几个 micro-batch 的有效 token 数不同，比如有些样本 padding 更多、label mask 更多、最后一个 micro-batch 更小的时候，这会改变整体梯度的权重。

一个极简数字例子：

```text
micro-batch 1: 有效 token = 1000, token loss 总和 = 2000, 平均 loss = 2.0
micro-batch 2: 有效 token = 100,  token loss 总和 = 300,  平均 loss = 3.0
```

正确的完整 batch token 平均是：

```text
(2000 + 300) / (1000 + 100) = 2.09
```

如果每个 micro-batch 各自平均再相加，两个 micro-batch 会被赋予近似相同权重：

```text
2.0 + 3.0
```

即使再手动除以 micro-batch 数，也会得到：

```text
(2.0 + 3.0) / 2 = 2.5
```

这已经不是按 token 加权的完整 batch 平均了。OLMo-core 使用完整 batch 的 `batch_num_tokens_for_loss` 做统一分母，就是为了让 micro-batch 只是显存层面的切分，不改变数学上的 batch loss。

## 4. Transformer 模型主干

模型定义主要在：

- `src/olmo_core/nn/transformer/config.py`
- `src/olmo_core/nn/transformer/model.py`
- `src/olmo_core/nn/transformer/block.py`

`Transformer.forward()` 的拓扑很直接：

```text
input_ids
  -> embeddings
  -> optional embedding_norm
  -> blocks[0..n_layers-1]
  -> lm_head
  -> logits 或 LMOutputWithLoss
```

如果传入 `labels`，`lm_head` 会直接计算 loss，并返回 `LMOutputWithLoss(logits, loss, ce_loss, z_loss)`。训练时通常 `return_logits=False`，避免保存完整 logits，降低显存占用。

`TransformerBlock.forward()` 是标准 pre-norm block：

```text
x
  -> attention_norm
  -> sequence_mixer
  -> residual add
  -> feed_forward_norm
  -> feed_forward
  -> residual add
```

代码里字段名仍叫 `attention`，但实际类型是 `SequenceMixer`。这保留了旧 checkpoint 兼容性，也给二次研发留下了接口：可以替换成 attention、recurrent、convolution、linear attention 等不同序列混合模块。

## 5. 并行和显存控制

个人工作站迁移时优先关注这些入口：

- `TransformerTrainModule.__init__()`: 构建 world mesh，并调用 `parallelize_model()`。
- `src/olmo_core/train/train_module/transformer/common.py`: 统一应用 DP/TP/CP/EP/AC/compile/FP8。
- `Transformer.apply_fsdp()`: FSDP2 包裹策略。
- `Transformer.apply_activation_checkpointing()`: activation checkpointing 策略。
- `Transformer.apply_tp()` 和 `Transformer.apply_cp()`: tensor/context parallel。

这部分可以分成两层理解：

1. **world mesh 层**：把所有 GPU/rank 组织成命名维度，比如 `dp=2, cp=2, tp=2`。
2. **模型改造层**：根据这些维度把模型包成 FSDP/DDP、切 TP、切 CP、加 activation checkpointing、compile 或 FP8。

### 5.1 world mesh 是什么

`build_world_mesh()` 的输入是 DP/TP/CP/PP/EP 配置，输出是一个 PyTorch `DeviceMesh`。后续代码不会手工用 global rank 拼 process group，而是从这个 mesh 里拿对应子网格：

```text
get_dp_model_mesh(world_mesh) -> 给 FSDP/DDP 用
get_dp_mesh(world_mesh)       -> 给 data loader 分数据用
get_tp_mesh(world_mesh)       -> 给 tensor parallel 用
get_cp_mesh(world_mesh)       -> 给 context parallel 用
get_ep_mesh(world_mesh)       -> 给 expert parallel 用
```

可以先用一句话区分 DP、TP、CP：

```text
DP: 多份完整训练副本，各自处理不同样本。
TP: 一个模型层太宽，把层内矩阵/heads/hidden 维切到多卡上。
CP: 一个序列太长，把 sequence/context 维切到多卡上。
```

更直观地说：

```text
DP 解决“数据多”：同一个模型，喂不同 batch。
TP 解决“层太大”：同一个 token 的计算，分摊到多个 rank。
CP 解决“上下文太长”：同一条长序列，分段放到多个 rank。
```

下面用 8 张 GPU，也就是 8 个 rank，说明几种不同 mesh。

#### 例子 A：纯 DP，`world_mesh = (dp=8)`

如果只开数据并行：

```text
world_size = 8
world_mesh = (dp=8)
```

可以理解为：

```text
rank0 -> 模型副本 0，处理 batch shard A
rank1 -> 模型副本 1，处理 batch shard B
rank2 -> 模型副本 2，处理 batch shard C
...
rank7 -> 模型副本 7，处理 batch shard H
```

每个 rank 都处理不同样本。区别在于：

- DDP 下，每个 rank 都保存完整模型参数，backward 后同步梯度。
- FSDP 下，逻辑上仍是 8 个数据并行 rank，但参数/梯度/optimizer state 被 8 卡分片保存。

纯 DP 的关键点：**每张卡看到不同样本，但每张卡都在跑完整序列、完整层计算。**

#### 例子 B：DP + TP，`world_mesh = (dp=4, tp=2)`

如果配置：

```text
tp.degree = 2
dp.name = fsdp
world_size = 8
```

那么：

```text
dp = 8 / tp.degree = 4
world_mesh = (dp=4, tp=2)
```

可以把 8 张卡看成 4 组，每组 2 张卡共同跑一个模型副本的“层内计算”：

```text
DP group 0: rank0, rank1 -> 共同处理 batch shard A，其中 rank0/1 做 2-way TP
DP group 1: rank2, rank3 -> 共同处理 batch shard B，其中 rank2/3 做 2-way TP
DP group 2: rank4, rank5 -> 共同处理 batch shard C，其中 rank4/5 做 2-way TP
DP group 3: rank6, rank7 -> 共同处理 batch shard D，其中 rank6/7 做 2-way TP
```

如果模型有 32 个 attention heads，`tp.degree=2` 时可以直观理解为每个 TP rank 处理约 16 个 heads。真实实现还会涉及 embedding、MLP、LM head、sequence parallel 和 DTensor layout，但直觉上就是：**同一个样本、同一层的计算，被 TP 组内两张卡共同完成。**

DP + TP 的关键点：

```text
DP 维度之间：处理不同数据。
TP 维度之间：处理同一批数据的同一层计算的不同切片。
```

#### 例子 C：DP + CP，`world_mesh = (dp=4, cp=2)`

如果配置：

```text
cp.degree = 2
dp.name = fsdp
world_size = 8
```

那么：

```text
dp = 8 / cp.degree = 4
world_mesh = (dp=4, cp=2)
```

可以看成 4 组数据并行副本，每组 2 张卡共同处理同一批样本的长序列：

```text
DP group 0: rank0, rank1 -> 共同处理 batch shard A 的长序列
DP group 1: rank2, rank3 -> 共同处理 batch shard B 的长序列
DP group 2: rank4, rank5 -> 共同处理 batch shard C 的长序列
DP group 3: rank6, rank7 -> 共同处理 batch shard D 的长序列
```

假设 `seq_len=8192`，`cp.degree=2`：

```text
rank0 看到 batch shard A 的 token 0..4095
rank1 看到 batch shard A 的 token 4096..8191
```

这两个 rank 不是在处理不同样本，而是在处理**同一样本的不同上下文片段**。attention 需要跨片段通信，所以 CP 会在 attention 内部做 ring 或 Ulysses 风格的通信。

CP 的关键点：**CP 切的是序列长度，不是参数。**每个 CP rank 仍然有模型参数副本。因此：

- data loader 视角：rank0 和 rank1 应该拿同一批样本，因为它们要共同切同一条序列。
- 模型同步视角：rank0 和 rank1 都有参数副本，所以梯度同步必须把 CP rank 也纳入。

这就是 `get_dp_mesh()` 和 `get_dp_model_mesh()` 会不同的原因。

#### 例子 D：DP + CP + TP，`world_mesh = (dp=2, cp=2, tp=2)`

你原来这段里的配置是：

```text
cp.degree = 2
tp.degree = 2
dp.name = fsdp
```

那么 mesh 维度是：

```text
world_size = 8
dp = 8 / cp.degree / tp.degree = 2
world_mesh = (dp=2, cp=2, tp=2)
```

可以把 8 张卡看成 2 个数据并行副本，每个副本内部有 2-way CP，每个 CP 分片内部再有 2-way TP：

```text
DP group 0，处理 batch shard A:
  CP chunk 0，处理序列前半段:
    rank0, rank1 -> 2-way TP，共同算前半段的层内矩阵/heads
  CP chunk 1，处理序列后半段:
    rank2, rank3 -> 2-way TP，共同算后半段的层内矩阵/heads

DP group 1，处理 batch shard B:
  CP chunk 0，处理序列前半段:
    rank4, rank5 -> 2-way TP
  CP chunk 1，处理序列后半段:
    rank6, rank7 -> 2-way TP
```

如果 `seq_len=8192`、`attention heads=32`，可以粗略理解成：

```text
rank0/rank1: batch A，token 0..4095，各处理一部分 heads/hidden
rank2/rank3: batch A，token 4096..8191，各处理一部分 heads/hidden
rank4/rank5: batch B，token 0..4095，各处理一部分 heads/hidden
rank6/rank7: batch B，token 4096..8191，各处理一部分 heads/hidden
```

这时三个维度各自的含义是：

- `dp=2`: 有 2 份数据并行副本，处理不同数据样本。
- `cp=2`: 每份数据并行副本内部，把同一条长序列切成 2 段。
- `tp=2`: 每个 CP 分片内部，再把层内矩阵、attention heads 或 hidden 维切成 2 份。

#### 为什么 `get_dp_mesh()` 和 `get_dp_model_mesh()` 不一样

还是用 `world_mesh = (dp=2, cp=2, tp=2)`：

```text
rank0 rank1 rank2 rank3 -> 共同服务 DP 副本 0 的 batch shard A
rank4 rank5 rank6 rank7 -> 共同服务 DP 副本 1 的 batch shard B
```

data loader 只关心“哪些 rank 应该拿不同数据”。因为 CP rank 处理的是同一条序列的不同片段，TP rank 处理的是同一层计算的不同切片，所以它们不应该拿不同样本。真正应该拿不同样本的只有 DP 维度：

```text
get_dp_mesh(world_mesh) -> data loader 用

DP 数据副本 0: rank0/1/2/3 拿 batch shard A
DP 数据副本 1: rank4/5/6/7 拿 batch shard B
```

但 FSDP/DDP 关心的是“哪些 rank 上有需要同步的参数副本”。TP 会切参数或张量布局，通常不作为普通 DP 梯度副本来 flatten；CP 不切参数，每个 CP rank 都有参数副本，所以 CP 必须并入模型同步维度。

```text
get_dp_model_mesh(world_mesh) -> FSDP/DDP 用

把 dp 和 cp flatten:
  dp_cp = dp * cp = 2 * 2 = 4

再保留 tp=2:
  model mesh 近似理解为 (dp_cp=4, tp=2)
```

这就是 `distributed/parallel/__init__.py` 里会把 CP 维度 flatten 到 DP 模型 mesh 的原因。简化记忆：

```text
data loader:
  CP/TP rank 是“协作处理同一批数据”，不要分配不同样本。

模型同步:
  CP rank 有参数副本，要参与 DP/FSDP 梯度同步。
  TP rank 是层内切分，走 TP 自己的通信逻辑。
```

### 5.2 DP、DDP、FSDP、HSDP

DP 这里主要指“数据并行维度”，配置类是 `TransformerDataParallelConfig`，底层继承 `DataParallelConfig`。关键字段：

- `name`: `ddp`、`fsdp` 或 `hsdp`。
- `param_dtype`: 参数 materialize 的 dtype，例如 bf16。
- `reduce_dtype`: 梯度 reduce 的 dtype，默认 float32。
- `wrapping_strategy`: FSDP 包裹粒度。
- `prefetch_factor`: FSDP 预取后续模块参数的激进程度。

三种模式的差异：

```text
DDP:
  每张卡都有完整模型参数。
  backward 后 all-reduce 梯度。
  简单稳定，但大模型显存压力最大。

FSDP:
  参数、梯度、optimizer state 在 DP rank 间分片。
  计算某个模块前 all-gather 参数，计算后 reshard。
  更省显存，是单机多卡迁移大模型时优先考虑的方案。

HSDP:
  Hybrid Sharded Data Parallel。
  常见形态是节点内 FSDP shard，节点间 replica 同步。
  更适合多节点训练；个人单机工作站一般不先考虑。
```

`wrapping_strategy` 影响 FSDP 包裹粒度：

```text
full:
  block 和 lm_head 等模块都会单独包裹。

blocks:
  主要包 transformer blocks，lm_head 不单独包。

fine_grained:
  block 内部的 attention、feed_forward 等也更细粒度包裹。
  可能更省峰值显存，但通信和 wrapper 开销更高。
```

个人工作站常见起点：

```text
单卡:
  不配 dp_config，先靠小 batch、小 sequence length、activation checkpointing 跑通。

单机 2-8 卡:
  优先 fsdp + blocks/full wrapping。
  如果仍 OOM，再调小 rank_microbatch_size 或加 activation checkpointing。
```

### 5.3 rank_microbatch_size 和显存

`rank_microbatch_size` 是每个 rank 单次 micro-batch 的 token 数，不是全局 batch，也不是样本数。它直接决定单次 forward/backward 的激活规模。

```text
seq_len = 2048
rank_microbatch_size = 8192
每个 micro-batch 的样本数 = 8192 // 2048 = 4
```

如果 `global_batch_size` 很大，OLMo-core 会把一个 rank batch 切成多个 micro-batch，逐个 forward/backward，最后只做一次 optimizer step。这样可以保持较大的有效 batch，同时控制单次显存峰值。

调显存时通常先改：

```text
sequence_length / max_sequence_length
rank_microbatch_size
activation checkpointing
FSDP wrapping_strategy
```

### 5.4 TP: Tensor Parallel

TP 把单层内部的大矩阵计算切到多个 rank。入口是 `Transformer.apply_tp()`，配置是 `TransformerTensorParallelConfig`：

- `degree`: TP 分片数。
- `enable_async`: 是否启用实验性 async TP。

直观例子：

```text
tp.degree = 2
attention heads = 32
每个 TP rank 约处理 16 个 heads
```

TP 能降低单卡参数和层内激活压力，但会增加层内通信。它通常用于单张 GPU 放不下单层矩阵的模型。个人工作站迁移时，除非 FSDP + AC 仍然不够，否则不建议一开始就上 TP，因为 TP 对模型结构、shape、通信性能更敏感。

### 5.5 CP: Context Parallel

CP 把 sequence/context 维度切到多个 rank，主要用于长上下文训练。入口是 `Transformer.apply_cp()`，配置是 `TransformerContextParallelConfig`。

OLMo-core 提供几个构造方式：

```python
TransformerContextParallelConfig.zig_zag(degree=2)
TransformerContextParallelConfig.llama3(degree=2)
TransformerContextParallelConfig.ulysses(degree=2)
```

直观例子：

```text
seq_len = 8192
cp.degree = 4
每个 CP rank 先持有约 2048 token 的局部序列片段
```

CP 的重点不是切参数，而是切长序列。每个 CP rank 仍持有模型参数副本，所以它对参数同步的关系更像“额外的数据并行副本”。这也是为什么代码里模型同步 mesh 会把 CP 维度并入 DP 维度。

CP 适合长上下文场景；如果只是普通 2K/4K 上下文，小工作站通常先不打开 CP。

### 5.6 EP: Expert Parallel

EP 只对 MoE 模型有效。它把不同 experts 分到不同 rank 上，入口在 `parallelize_model()` 里的 `ep_config` 分支。

注意当前实现里：

- dense Transformer 不能用 EP。
- TP + EP 当前不支持同时启用。
- EP 当前更偏大规模 MoE 训练，不是个人工作站迁移的第一优先级。

### 5.7 AC: Activation Checkpointing

Activation checkpointing 用计算换显存：forward 时不保存某些中间激活，backward 时重新计算这些 forward。

配置类是 `TransformerActivationCheckpointingConfig`，关键字段：

- `mode`: checkpoint 策略。
- `block_interval`: `selected_blocks` 模式下每隔几层包一次。
- `modules`: `selected_modules` 模式下按模块名/glob 选择。
- `activation_memory_budget`: `budget` 模式下的预算，需要 `compile_model=True`。

常见模式：

```text
full:
  每个 transformer block 都 checkpoint。
  显存最省，速度最慢。

selected_blocks:
  例如 block_interval=2，只 checkpoint 第 0、2、4... 层。
  显存和速度折中。

selected_modules:
  例如 modules=["blocks.*.feed_forward"]。
  只包匹配的子模块，适合知道显存热点时精细控制。

selected_ops:
  使用 selective checkpointing，只针对特定 op。

budget:
  使用 torch.compile 相关的 activation memory budget。
  activation_memory_budget 越接近 0，重算越激进；越接近 1，越接近不 checkpoint。
```

个人工作站上最实用的是：

```text
先试 full。
如果能跑但太慢，再改 selected_blocks。
```

### 5.8 compile 和 FP8

`compile_model=True` 会调用 `Transformer.apply_compile()`，通常在 AC 之后、FSDP/DDP 之前执行。它可能提升性能，但首次编译慢，也更容易暴露动态 shape 或分布式 wrapper 的兼容问题。

迁移调试建议：

```text
先 compile_model=False 跑通。
确认 batch、数据、checkpoint、loss 都正常后，再打开 compile_model。
```

FP8 由 `float8_config` 控制，入口是 `Transformer.apply_fp8()`。它会把大部分 linear 层换成 FP8 训练实现，降低显存和带宽压力，但依赖硬件、torchao/TransformerEngine 等栈，调试成本更高。

个人工作站建议：

```text
先 bf16/fp32 跑通。
最后再考虑 FP8。
```

推荐调参顺序：

1. 先关闭 `compile_model`，用单卡或小 FSDP 配置跑通。
2. 降低 `sequence_length`。
3. 降低 `rank_microbatch_size`。
4. 开启 `activation_checkpointing=full`。
5. 使用 FSDP blocks wrapping。
6. 再考虑 `torch.compile()`、TP、CP 或 FP8。

## 6. 数据路径

数据抽象主要在：

- `src/olmo_core/data/numpy_dataset.py`
- `src/olmo_core/data/data_loader.py`
- `src/olmo_core/data/collator.py`

`DataLoaderBase` 是分布式、确定性、可恢复的数据加载器。它的 `global_batch_size` 是所有 data-parallel rank 的总 batch，单个 rank 得到：

```text
rank_batch_size = global_batch_size // dp_world_size
```

`DataCollator` 负责把样本 pad 到同一长度，并输出 batch dict。常见字段包括：

- `input_ids`: token id，形状为 `(batch, seq_len)`。
- `attention_mask`: padding mask。
- `attention_bias`: 更细粒度的 attention mask/bias。
- `label_mask`: 控制哪些 token 参与 loss。
- `doc_lens`: packed sequence 内部的文档长度，用于文档边界 mask 或 context parallel。

对于本地复现实验，最容易出错的是数据格式。OLMo-core 的 Numpy FSL 数据通常期望 flat raw binary token id 文件；文件名可以是 `.npy`，但内容不一定是带 NumPy header 的标准 `.npy`。

