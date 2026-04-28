import logging
from typing import List, Optional, TypeVar, cast

import torch
from torch.distributed import DeviceMesh

from olmo_core.distributed.parallel import (
    DataParallelType,
    get_cp_mesh,
    get_device_mesh_info,
    get_dp_model_mesh,
    get_ep_mesh,
    get_pp_mesh,
    get_tp_mesh,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import MoETransformer, Transformer

from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)


M = TypeVar("M", Transformer, List[Transformer])


def parallelize_model(
    model: M,
    *,
    world_mesh: Optional[DeviceMesh],
    device: torch.device,
    max_sequence_length: Optional[int] = None,
    rank_microbatch_size: Optional[int] = None,
    compile_model: bool = False,
    float8_config: Optional[Float8Config] = None,
    dp_config: Optional[TransformerDataParallelConfig] = None,
    tp_config: Optional[TransformerTensorParallelConfig] = None,
    cp_config: Optional[TransformerContextParallelConfig] = None,
    ep_config: Optional[TransformerExpertParallelConfig] = None,
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
    pp_enabled: bool = False,
) -> M:
    model_parts: List[Transformer] = [model] if isinstance(model, Transformer) else model

    # 中文导读：parallelize_model() 是训练启动阶段的“模型改造流水线”。
    # 它不是训练循环的一部分，而是在 optimizer 构建之前执行一次。
    # 顺序很重要：
    #   - 先做 CP/TP/EP 这类改变张量布局或通信组的改造；
    #   - 再做 activation checkpointing 和 torch.compile；
    #   - 最后做 DDP/FSDP，因为它们会把模块包起来并接管参数同步/切分。
    # 如果顺序反过来，后续很难再安全地找到原始子模块或改变 DTensor layout。
    pp_mesh: Optional[DeviceMesh] = None
    if pp_enabled:
        assert world_mesh is not None
        pp_mesh = get_pp_mesh(world_mesh)
        for m in model_parts:
            m.apply_pp(pp_mesh)

    # Maybe apply FP8 training.
    if float8_config is not None and float8_config.enabled:
        for m in model_parts:
            m.apply_fp8(float8_config)
            log.info("Swapped linear layers to Float8 linear layers\n%s", m)

    # Maybe apply context parallelism.
    # 中文导读：CP(Context Parallel) 主要服务长上下文。它把序列长度维度 T
    # 切到多个 rank 上，例如 T=8192、cp.degree=4 时，每个 CP rank 只持有
    # 约 2048 个 token 的局部上下文。注意 CP rank 仍然持有一份模型参数副本，
    # 所以做参数同步时 CP 维度会被看作额外的数据并行副本。
    if cp_config is not None:
        assert world_mesh is not None
        cp_mesh = get_cp_mesh(world_mesh)
        for m in model_parts:
            m.apply_cp(cp_mesh, ring=cp_config.ring, uly=cp_config.uly)
        log.info(f"Applied context parallelism to the model with {get_device_mesh_info(cp_mesh)}")

    # Maybe apply tensor.
    # 中文导读：TP(Tensor Parallel) 把单层内部的大矩阵/heads 切到多个 rank 上。
    # 例如 tp.degree=2 时，attention heads、MLP hidden 维度、LM head vocab/hidden
    # 相关矩阵会按实现策略拆分，单卡参数和中间激活下降，但层内通信增加。
    if tp_config is not None:
        if ep_config is not None:
            raise NotImplementedError("TP + EP is not implemented yet")
        assert world_mesh is not None
        tp_mesh = get_tp_mesh(world_mesh)
        for m in model_parts:
            m.apply_tp(tp_mesh)
        tp_config.maybe_enable_async_tp(tp_mesh)
        log.info(f"Applied tensor parallelism to the model with {get_device_mesh_info(tp_mesh)}")

    # Maybe apply expert parallelism.
    # 中文导读：EP(Expert Parallel) 只对 MoE 模型有效，用来把不同 experts
    # 分散到不同 rank。普通 dense Transformer 没有 expert，因此这里会直接报错。
    # 当前实现还限制 TP 和 EP 不能同时用。
    if ep_config is not None:
        assert world_mesh is not None
        ep_mesh = get_ep_mesh(world_mesh)
        for m in model_parts:
            if not m.is_moe:
                raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
            cast(MoETransformer, m).apply_ep(ep_mesh)
        log.info(f"Applied expert parallelism to the model with {get_device_mesh_info(ep_mesh)}")

    # Maybe apply activation checkpointing.
    # 中文导读：AC(Activation Checkpointing) 用“反向传播时重算 forward”
    # 换取更低的激活显存。full 模式最省显存但最慢；selected_blocks/modules
    # 可以只包一部分层；budget 模式让 PyTorch compile 根据显存预算自动取舍。
    if ac_config is not None:
        for m in model_parts:
            m.apply_activation_checkpointing(
                ac_config.mode,
                block_interval=ac_config.block_interval,
                modules=ac_config.modules,
                activation_memory_budget=ac_config.activation_memory_budget,
            )
        log.info(f"Applied '{ac_config.mode}' activation checkpointing to the model")

    # Maybe compile.
    # 中文导读：torch.compile() 通常能减少 Python overhead 并融合部分算子，
    # 但首次编译慢，且对动态 shape、分布式 wrapper 顺序更敏感。因此这里在
    # AC 之后、FSDP/DDP 之前调用。
    if compile_model:
        if torch.cuda.is_available():
            for m in model_parts:
                m.apply_compile()
            log.info("Applied torch.compile() to the model")
        else:
            log.warning("Skipping model compilation since CUDA is not available")

    # Maybe shard/replicate according to data parallel config.
    # 中文导读：DP(Data Parallel) 决定不同数据副本之间如何保存参数并同步梯度：
    #   - DDP：每个 DP rank 保存完整模型，反向后 all-reduce 梯度；
    #   - FSDP：参数/梯度/优化器状态按 DP rank 分片，用 all-gather 临时还原计算；
    #   - HSDP：混合方式，通常节点内 shard、节点间 replicate。
    # 对个人工作站，单机多卡优先考虑 FSDP；单卡则通常不配置 dp_config。
    if dp_config is not None:
        assert world_mesh is not None
        # 中文导读：这里特意取 get_dp_model_mesh()，不是 get_dp_mesh()。
        # get_dp_mesh() 给 data loader 用，只决定哪些 rank 拿不同样本；
        # get_dp_model_mesh() 给 FSDP/DDP 用，会把 CP 维度并入 DP 同步维度，
        # 因为 CP rank 持有参数副本，梯度也需要一起同步。
        dp_mesh = get_dp_model_mesh(world_mesh)
        param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
        if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
            # 中文导读：FSDP 和 HSDP 都走 apply_fsdp()。
            # 区别主要体现在 dp_mesh 的形状：
            #   fsdp:  通常是 1D dp mesh，例如 (dp=8)
            #   hsdp:  通常是 2D mesh，例如 (dp_replicate=2, dp_shard=8)
            # apply_fsdp() 内部的 fully_shard 会根据 mesh 决定参数如何分片/复制。
            for m in model_parts:
                if m.is_moe:
                    # 中文导读：MoE 模型在 FSDP 前要先整理 expert 参数布局，
                    # 确保 expert parallel / sharded state dict / FSDP 包裹能兼容。
                    cast(MoETransformer, m).prepare_experts_for_fsdp(
                        world_mesh,
                        param_dtype=param_dtype,
                        reduce_dtype=dp_config.reduce_dtype.as_pt(),
                        pp_enabled=pp_enabled,
                    )
                m.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                    wrapping_strategy=dp_config.wrapping_strategy,
                    pp_enabled=pp_enabled,
                    prefetch_factor=dp_config.prefetch_factor,
                )
            log.info(f"Applied FSDP to the model with {get_device_mesh_info(dp_mesh)}")
        elif dp_config.name == DataParallelType.ddp:
            # 中文导读：DDP 分支不切参数，每个 DP rank 保存完整模型副本。
            # backward 后 DDP reducer 在 dp_mesh 上 all-reduce 梯度。
            # 它更简单，但显存占用通常高于 FSDP。
            for m in model_parts:
                if m.is_moe:
                    cast(MoETransformer, m).prepare_experts_for_ddp(world_mesh)
                m.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")
        else:
            raise NotImplementedError(dp_config.name)

    # Materialize and init parameters.
    # 中文导读：这一步才真正初始化权重。前面如果模型是在 init_device="meta"
    # 上创建的，参数此前没有真实存储；这里会结合 max_sequence_length、
    # rank_microbatch_size 和 world_mesh 初始化 buffer/cache/参数。
    log.info("Initializing model weights...")
    for m in model_parts:
        m.init_weights(
            max_seq_len=max_sequence_length,
            max_local_microbatch_size=rank_microbatch_size,
            device=device,
            world_mesh=world_mesh,
        )

    return model
