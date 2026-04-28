from dataclasses import dataclass

from olmo_core.config import Config


@dataclass
class ExpertParallelConfig(Config):
    """
    Configuration class for expert parallelism (EP).
    """

    # 中文导读：EP 的 degree 表示 MoE experts 分到几个 rank 上。
    # 例如 64 个 experts、degree=8 时，可以直观理解为每个 EP rank 管一部分 experts。
    # build_world_mesh() 会把它变成 expert parallel mesh；common.py::parallelize_model()
    # 会调用 MoETransformer.apply_ep()。dense Transformer 没有 experts，不能使用 EP。
    #
    # 当前 OLMo-core 还限制 TP + EP 不能同时启用，且 EP 主要面向 MoE 大规模训练。
    degree: int
    """
    The EP degree.
    """
