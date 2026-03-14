from __future__ import annotations

"""
信息熵相关工具。

在本作业中，常用的是“对 logits 的最后一维（vocab 维）计算熵”。
熵越大代表分布越不确定，熵越小代表模型越自信。
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_entropy(logits: Tensor) -> Tensor:
    """
    计算 logits 的熵（沿最后一维）。

    Args:
        logits: (..., vocab_size) 未归一化的分数。

    Returns:
        (...,) 每个位置的熵值。
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)

