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

