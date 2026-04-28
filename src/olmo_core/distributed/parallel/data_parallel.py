import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import get_num_nodes
from olmo_core.exceptions import OLMoConfigurationError

log = logging.getLogger(__name__)


class DPMeshDimName(StrEnum):
    """
    ``DeviceMesh`` dimension names for data parallelism.
    """

    replicate = "dp_replicate"
    """
    The device mesh dimension over which the model is replicated.
    """
    shard = "dp_shard"
    """
    The device mesh dimension over which the model is sharded.
    """


class DataParallelType(StrEnum):
    # 中文导读：这里枚举的是“数据并行模型包装方式”。
    #   ddp:  每个 DP rank 保留完整参数，反向后 all-reduce 梯度。
    #   fsdp: 参数/梯度/optimizer state 在 DP rank 间分片，计算时按模块 all-gather。
    #   hsdp: Hybrid FSDP，通常把 DP 维度拆成 replicate 和 shard 两维。
    # 具体分支在 train/train_module/transformer/common.py::parallelize_model()。
    fsdp = "fsdp"
    hsdp = "hsdp"
    ddp = "ddp"


@dataclass
class DataParallelConfig(Config):
    # 中文导读：这是所有 DP 类配置的基类。Transformer 专用配置
    # TransformerDataParallelConfig 会继承它并额外增加 wrapping_strategy。
    #
    # name 决定走 DDP、FSDP 还是 HSDP；param_dtype/reduce_dtype 分别控制
    # 参数 materialize dtype 和梯度通信 dtype；num_replicas/shard_degree
    # 只在 HSDP 拆分 replicate/shard 维度时使用。
    name: DataParallelType
    param_dtype: Optional[DType] = None
    reduce_dtype: DType = DType.float32
    num_replicas: Optional[int] = None
    shard_degree: Optional[int] = None

    def get_replicate_and_shard_degree(self, dp_world_size: int) -> Tuple[int, int]:
        """
        Defaults to one replica per node, with the shard degree set to the number of gpus per node.

        :param dp_world_size: The data parallel world size.
        :return: A tuple of (num_replicas, shard_degree)
        """
        # 中文导读：HSDP 会把数据并行维度拆成：
        #   num_replicas: 有几份 FSDP 分片组副本，通常跨节点；
        #   shard_degree: 每个副本内部用几张卡做 FSDP 参数分片，通常节点内。
        #
        # 例子：dp_world_size=16，2 个节点，每节点 8 卡，默认 get_num_nodes()=2：
        #   num_replicas = 2
        #   shard_degree = 16 // 2 = 8
        # 含义是节点内 8 卡做 shard，两个节点之间做副本级同步。
        if self.num_replicas is None and self.shard_degree is None:
            return get_num_nodes(), dp_world_size // get_num_nodes()
        elif self.num_replicas is not None and self.shard_degree is not None:
            return _check_num_replicas(self.num_replicas, dp_world_size), _check_shard_degree(
                self.shard_degree, dp_world_size
            )
        elif self.num_replicas is not None:
            return (
                _check_num_replicas(self.num_replicas, dp_world_size),
                dp_world_size // self.num_replicas,
            )
        else:
            assert self.shard_degree is not None
            return dp_world_size // self.shard_degree, _check_shard_degree(
                self.shard_degree, dp_world_size
            )


def _check_num_replicas(num_replicas: int, dp_world_size: int) -> int:
    if dp_world_size % num_replicas != 0:
        raise OLMoConfigurationError(
            f"data parallel world size ({dp_world_size}) must be "
            f"divisible by 'num_replicas' ({num_replicas})"
        )
    return num_replicas


def _check_shard_degree(shard_degree: int, dp_world_size: int) -> int:
    if dp_world_size % shard_degree != 0:
        raise OLMoConfigurationError(
            f"data parallel world size ({dp_world_size}) must be "
            f"divisible by 'shard_degree' ({shard_degree})"
        )
    return shard_degree
