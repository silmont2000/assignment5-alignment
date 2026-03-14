from __future__ import annotations

"""
策略梯度相关 loss（按 token 维度返回，不在这里做 mask）。

本作业中单元测试关心的是：
- naive policy gradient（REINFORCE，无 baseline）
- GRPO-Clip（PPO-style clipping 版本）
- 一个 wrapper：根据 loss_type 分发到对应实现
"""

import torch
from torch import Tensor


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """
    最朴素的 policy gradient（REINFORCE）per-token loss：

      L_t = - R * log π(a_t | s_t)

    这里的 R 可以是：
    - raw reward（无 baseline）
    - advantage（例如做了组内去均值/除 std 的归一化）

    Args:
        raw_rewards_or_advantages: (batch, 1)
        policy_log_probs: (batch, seq_len)

    Returns:
        (batch, seq_len) 的 per-token loss
    """
    if raw_rewards_or_advantages.ndim != 2 or raw_rewards_or_advantages.shape[1] != 1:
        raise ValueError(
            "raw_rewards_or_advantages must have shape (batch_size, 1), got "
            f"{tuple(raw_rewards_or_advantages.shape)}"
        )
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    GRPO-Clip：与 PPO-Clip 非常相似，只是 advantage 的来源不同（GRPO 用组内归一化）。

    ratio_t = exp(logπ_t - logπ_old_t)

    PPO-style clipping 的关键点：
    - advantage >= 0：取 min(ratio*A, clip(ratio)*A)
    - advantage <  0：取 max(ratio*A, clip(ratio)*A)

    这个“按 advantage 正负选 min/max”是为了保证 clipping 的方向正确。

    Returns:
        loss: (batch, seq_len) per-token loss
        metadata: 用于统计 clip fraction 等
    """
    if advantages.ndim != 2 or advantages.shape[1] != 1:
        raise ValueError(
            "advantages must have shape (batch_size, 1), got "
            f"{tuple(advantages.shape)}"
        )
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError(
            "policy_log_probs and old_log_probs must have the same shape, got "
            f"{tuple(policy_log_probs.shape)} and {tuple(old_log_probs.shape)}"
        )

    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    adv = advantages
    surrogate = ratio * adv
    surrogate_clipped = clipped_ratio * adv
    obj = torch.where(
        adv >= 0,
        torch.minimum(surrogate, surrogate_clipped),
        torch.maximum(surrogate, surrogate_clipped),
    )
    loss = -obj

    metadata = {
        "ratio": ratio.detach(),
        "clipped_ratio": clipped_ratio.detach(),
        "clip_mask": ((ratio - clipped_ratio).abs() > 0).detach(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: str,
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    根据 loss_type 分发到不同的 policy gradient loss。

    loss_type 取值约定：
    - "no_baseline": 使用 raw reward
    - "reinforce_with_baseline": 使用 advantages（相当于 reward - baseline）
    - "grpo_clip": 使用 GRPO-Clip（需要 old_log_probs 与 cliprange）
    """
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}

    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    raise ValueError(
        f"Unknown loss_type={loss_type!r}. Expected one of "
        f"'no_baseline', 'reinforce_with_baseline', 'grpo_clip'."
    )
