from __future__ import annotations

"""
SFT（监督微调）相关的 microbatch 训练步。

单元测试的设计是：传入已经算好的 policy_log_probs（batch, seq_len），
以及 response_mask（batch, seq_len），函数内部负责：
- 构造 loss
- 做归约（只在 response token 上）
- 做 gradient_accumulation_steps 的缩放
- 调用 backward()
"""

import torch
from torch import Tensor

from .masking import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    计算 SFT microbatch loss 并反传。

    最大似然目标（只在 response tokens 上）：
      maximize sum_t logπ(y_t | x, y_<t)
      loss = - sum_t logπ(...)

    这里采用 masked_normalize：
    - 先对 response token 的 per-token loss 求和
    - 再除以一个外部给定 normalize_constant（可能是 1.0，也可能是固定长度）
    - 最后除以 gradient_accumulation_steps，确保梯度累积尺度正确
    """
    token_loss = -policy_log_probs

    denom = 1.0 if normalize_constant is None else float(normalize_constant)
    loss = masked_normalize(token_loss, response_mask, dim=None, normalize_constant=denom)
    loss = loss / float(gradient_accumulation_steps)

    loss.backward()

    # 额外返回一些日志指标（不影响单元测试输出结构也没问题）
    token_loss_sum = masked_normalize(
        token_loss, response_mask, dim=None, normalize_constant=1.0
    )
    return loss, {"token_loss_sum": token_loss_sum}

