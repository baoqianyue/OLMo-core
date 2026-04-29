from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import torch

from ..config import DType
from ..distributed.utils import get_local_tensor
from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer


def adamw_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    step_factor: torch.Tensor,
    step_increment_bugfix: bool = True,
):
    beta1, beta2 = betas

    # Perform step weight decay.
    # 中文导读：普通 AdamW 是 p *= (1 - lr * weight_decay)。
    # 这里额外乘 step_factor：如果 step_factor=0，本轮连 weight decay 也不做。
    p.mul_(1 - step_factor * (lr * weight_decay))

    # Decay the first and second moment running average coefficient.
    # step_factor=1 时就是标准 AdamW 动量更新；step_factor=0 时
    # exp_avg / exp_avg_sq 保持不变，异常梯度不会污染动量统计。
    exp_avg.lerp_(grad.type_as(exp_avg), (step_factor * (1 - beta1)).type_as(exp_avg))
    exp_avg_sq.mul_(1 - step_factor * (1 - beta2))
    exp_avg_sq.add_(step_factor * grad * grad, alpha=1 - beta2)

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)

    step_size = lr / bias_correction1

    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

    update = -step_size * torch.div(exp_avg, denom)
    # 最终参数更新也乘 step_factor。为 0 时 p.add_(0)，参数完全不变。
    update.mul_(step_factor)
    p.add_(update)
    if step_increment_bugfix:
        # 只在真正更新时增加 Adam step。跳步时 step 不变，bias correction
        # 也不会被一个“没有发生的更新”推进。
        step.add_(step_factor)


def foreach_adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    steps: list[torch.Tensor],
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    step_factor: torch.Tensor,
    step_increment_bugfix: bool = True,
):
    """Perform a single AdamW update with multi-tensor (*foreach*) kernels."""
    if not params:
        return  # nothing to do

    beta1, beta2 = betas

    # Perform step weight decay.
    # foreach 路径和 adamw_step() 语义相同，只是一次批量处理多个参数 tensor。
    torch._foreach_mul_(params, 1 - step_factor * (lr * weight_decay))

    grads = [g.type_as(ea) for g, ea in zip(grads, exp_avgs)]

    # Decay the first and second moment running average coefficient.
    # foreach_lerp_ has issues when DTensor is enabled (see https://github.com/pytorch/pytorch/issues/132017).
    # Implement the lerp(a, b, w) = a + w * (b - a) with basic _foreach_mul_/add_ ops instead:
    w1 = step_factor * (1 - beta1)
    torch._foreach_mul_(exp_avgs, 1.0 - w1)
    torch._foreach_add_(exp_avgs, torch._foreach_mul(grads, w1))

    grad_squares = torch._foreach_mul(grads, grads)

    w2 = step_factor * (1 - beta2)
    torch._foreach_mul_(exp_avg_sqs, 1.0 - w2)
    torch._foreach_add_(exp_avg_sqs, torch._foreach_mul(grad_squares, w2))

    steps_t = torch.stack(steps)
    bias_corrections1 = 1 - torch.pow(beta1, steps_t + 1)
    bias_corrections2 = 1 - torch.pow(beta2, steps_t + 1)

    step_sizes = lr / bias_corrections1

    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denoms, bias_corrections2.sqrt().unbind())
    torch._foreach_add_(denoms, eps)

    updates = torch._foreach_div(exp_avgs, denoms)
    torch._foreach_mul_(updates, (-step_factor * step_sizes).unbind())
    torch._foreach_add_(params, updates)
    if step_increment_bugfix:
        torch._foreach_add_(steps, [step_factor] * len(steps))


class SkipStepAdamW(SkipStepOptimizer):
    """
    A "skip step" version of :class:`AdamW`.

    中文导读：
    SkipStepAdamW = AdamW 更新公式 + “这一轮是否应该更新”的 step_factor。
    step_factor 由父类 SkipStepOptimizer.get_step_factor() 根据最近一段 loss
    和 grad norm 的滚动统计给出：
      - step_factor = 1.0：当前 step 看起来正常，执行标准 AdamW 更新。
      - step_factor = 0.0：当前 step 是异常尖峰，weight decay、动量更新、
        参数更新和 step 计数都乘以 0，相当于跳过这次 optimizer step。

    这样写的好处是 step_factor 仍然是设备上的 tensor，可以直接参与 CUDA
    计算，避免把“是否跳过”的布尔值搬回 CPU 做 if 判断造成 host-device sync。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        foreach: bool = False,
        step_increment_bugfix: bool = True,
    ) -> None:
        assert lr >= 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(
            params,
            defaults,
            rolling_interval_length=rolling_interval_length,
            sigma_factor=sigma_factor,
        )
        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        self.foreach = foreach
        self.stepfix = step_increment_bugfix
        self._step_skipped: Optional[torch.Tensor] = None

    @property
    def step_skipped(self) -> torch.Tensor:
        # 训练循环和 metrics 会读取这个值。1 表示刚才那步被跳过，0 表示正常更新。
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        # foreach=True 走 PyTorch multi-tensor kernel，一次处理一组 tensor，
        # 通常更快；foreach=False 逐参数更新，逻辑更直观。
        if self.foreach:
            self._step_foreach(closure)
        else:
            self._step(closure)

    def _step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        # 父类根据最新 loss / grad norm 和历史窗口计算是否跳步。
        # 注意 get_step_factor() 返回 tensor，而不是 Python bool。
        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    # AdamW 为每个参数维护 step、一阶动量 exp_avg、二阶矩 exp_avg_sq。
                    # dtype 可用于把动量状态存成 bf16 等格式以节省 optimizer state 显存。
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                adamw_step(
                    get_local_tensor(p),
                    get_local_tensor(p.grad),
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    exp_avg=get_local_tensor(state["exp_avg"]),
                    exp_avg_sq=get_local_tensor(state["exp_avg_sq"]),
                    step=state["step"],
                    step_factor=step_factor,
                    step_increment_bugfix=self.stepfix,
                )

    def _step_foreach(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        # foreach 路径和逐参数路径使用完全相同的 step_factor，只是把同一组公式
        # 换成 torch._foreach_* multi-tensor kernel。
        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            steps_list = []  # create list outside loops

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    # foreach 版本同样延迟初始化 optimizer state，只有参数第一次有梯度时才建状态。
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                params_with_grad.append(get_local_tensor(p))
                grads.append(get_local_tensor(p.grad))
                exp_avgs.append(get_local_tensor(state["exp_avg"]))
                exp_avg_sqs.append(get_local_tensor(state["exp_avg_sq"]))
                steps_list.append(state["step"])

            if not params_with_grad:
                continue  # nothing to update in this group

            foreach_adamw_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                steps_list,
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                step_factor=step_factor,
                step_increment_bugfix=self.stepfix,
            )


@OptimConfig.register("adamw")
@dataclass
class AdamWConfig(OptimConfig[torch.optim.AdamW]):
    """
    Configuration class for building an :class:`torch.optim.AdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    foreach: Optional[bool] = None
    fused: Optional[bool] = None

    @classmethod
    def optimizer(cls) -> Type[torch.optim.AdamW]:
        return torch.optim.AdamW


@OptimConfig.register("skip_step_adamw")
@dataclass
class SkipStepAdamWConfig(OptimConfig[SkipStepAdamW]):
    """
    Configuration class for building a :class:`SkipStepAdamW` optimizer.

    中文导读：
    上面两行是 Python 装饰器，不是普通注释。

    @dataclass 来自标准库 dataclasses。它会根据下面声明的字段自动生成
    __init__、__repr__、相等性比较等样板代码，所以可以直接写
    SkipStepAdamWConfig(lr=3e-4, weight_decay=0.1)。

    @OptimConfig.register("skip_step_adamw") 来自 OLMo-core 的 Registrable
    机制。它把这个 config 类登记到 OptimConfig 的注册表里，名字叫
    "skip_step_adamw"。这样配置文件/CLI/序列化字典里只要出现
    type: skip_step_adamw，框架就能反查到 SkipStepAdamWConfig 类。

    两者配合后，这个类既是“可序列化/可合并的配置对象”，也是“可通过
    字符串 type 找到的 optimizer 配置类型”。
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    dtype: Optional[DType] = None
    foreach: bool = True
    """
    Whether to use multi-tensor (*foreach*) kernels for the AdamW update.
    Faster than the non-foreach version.
    """

    step_increment_bugfix: bool = True
    """
    Whether or not to fix the step-incrementing bug discovered in SkipStepAdamW.

    If this flag is set to False, the step will not be incremented, which
    gives the optimizer an effective lr that is 2.2x higher than the specified lr,
    and no bias correction is applied.
    """

    rolling_interval_length: int = 128
    """
    The length of the rolling interval to use for computing the mean and standard deviation of the loss.
    """

    sigma_factor: int = 6
    """
    The number of standard deviations above the mean loss to skip a step.
    """

    # 中文导读：@classmethod 表示这是“类方法”，调用时不需要先创建实例。
    # Python 会把当前类本身作为第一个参数传进来，按惯例命名为 cls。
    # 例如 SkipStepAdamWConfig.optimizer() 调用时，cls 就是 SkipStepAdamWConfig。
    #
    # 这里不用 self，是因为这个方法只回答一个固定问题：
    # “这个 config 对应要构建哪个 optimizer 类？”答案就是 SkipStepAdamW。
    # OptimConfig.build(model) 会调用 self.optimizer() 拿到这个类，
    # 然后再实例化成真正的优化器对象。
    @classmethod
    def optimizer(cls) -> Type[SkipStepAdamW]:
        return SkipStepAdamW
