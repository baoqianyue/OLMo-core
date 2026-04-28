import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Shard, distribute_module
from torch.distributed.tensor.parallel import SequenceParallel as _SequenceParallel

from olmo_core.config import Config

log = logging.getLogger(__name__)


@dataclass
class TensorParallelConfig(Config):
    """
    Configuration class for tensor parallelism (TP).
    """

    # 中文导读：TP 的 degree 表示“同一个模型层的计算切成几份”。
    # 例如 degree=2 时，一个 DP/CP 分片内部会有 2 个 rank 协作计算
    # attention heads、MLP hidden、LM head 等层内大张量。
    # build_world_mesh() 会把这个 degree 变成名为 "tp" 的 mesh 维度；
    # common.py::parallelize_model() 会调用 get_tp_mesh() 再传给 Transformer.apply_tp()。
    degree: int
    """
    The TP degree.
    """

    # 中文导读：实验性的 async TP。开启后会启用 torch inductor 的
    # micro-pipeline TP 相关配置，并为 TP process group 打开 symmetric memory。
    # 它是性能优化，不改变“tp.degree 如何分组”的基本语义。
    enable_async: bool = False
    """
    Enable experimental async tensor parallelism.
    """

    def maybe_enable_async_tp(self, tp_mesh: DeviceMesh):
        if self.enable_async:
            log.info("Enabling async tensor parallel")

            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            torch._inductor.config._micro_pipeline_tp = True  # type: ignore
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)


class SequenceParallel(_SequenceParallel):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
        output_layouts: Optional[Placement] = None,
    ):
        super().__init__(sequence_dim=sequence_dim, use_local_output=use_local_output)
        # 中文导读：SequenceParallel 是 TP 里的常见配套布局。
        # 它把张量沿 sequence 维 shard，例如 hidden [B, S, D] 在 degree=2 时
        # 变成每个 rank 本地大致 [B, S/2, D]。这样 norm/dropout 等按 token
        # 独立的操作可以在本地 sequence shard 上完成。
        self.output_layouts = (output_layouts or Shard(sequence_dim),)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        del mod, device_mesh
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.sequence_sharding),  # type: ignore
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),  # type: ignore
        )
