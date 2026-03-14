from __future__ import annotations

"""
mask 相关的张量归约函数。

在语言模型训练里，我们经常只在“有效 token”上计算 loss：
- prompt token 不算（只监督 response）
- padding token 不算（避免把 padding 当成真实数据）

因此需要 masked_mean / masked_normalize 这类工具。
"""

import torch
from torch import Tensor


def masked_mean(tensor: Tensor, mask: Tensor, dim: int | None = None) -> Tensor:
    """
    在 mask==True 的位置上取平均。

    Args:
        tensor: 需要做 mean 的张量
        mask: 与 tensor broadcast 兼容的布尔 mask（True 表示参与平均）
        dim:
            - None：对所有元素做全局 masked mean
            - int：沿 dim 做 masked mean

    Returns:
        与 torch.mean 类似的形状，但忽略 mask==False 的元素。

    注意：当某个 slice 的有效元素数为 0 时，返回 0（避免 NaN）。
    """
    mask_f = mask.to(dtype=tensor.dtype)
    masked = tensor * mask_f

    if dim is None:
        denom = mask_f.sum()
        if denom.item() == 0:
            return torch.zeros((), dtype=tensor.dtype, device=tensor.device)
        return masked.sum() / denom

    denom = mask_f.sum(dim=dim)
    numer = masked.sum(dim=dim)
    return torch.where(denom > 0, numer / denom, torch.zeros_like(numer))


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """
    对 mask==True 的元素求和，然后除以一个“外部给定”的常数 normalize_constant。

    与 masked_mean 的区别：
    - masked_mean 的分母是 mask 的计数（每个样本可能不同）
    - masked_normalize 的分母是固定常数（例如固定的 max response length）

    这在一些实现（例如 Dr. GRPO 的做法）里很常见：用固定常数控制损失尺度。
    """
    mask_f = mask.to(dtype=tensor.dtype)
    masked = tensor * mask_f
    summed = masked.sum() if dim is None else masked.sum(dim=dim)
    return summed / float(normalize_constant)

