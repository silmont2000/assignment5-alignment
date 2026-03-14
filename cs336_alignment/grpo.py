from __future__ import annotations

"""
GRPO（Group Relative Policy Optimization）相关的 microbatch 训练步。

该函数的输入是“已经算好的”每 token log-prob（policy_log_probs），以及 response_mask。
我们只需要负责：
- 根据 loss_type 选择合适的 per-token loss
- 只在 response token 上做 masked mean
- 按 gradient_accumulation_steps 做缩放，并 backward()
"""

from typing import Literal

import torch
from torch import Tensor

from .masking import masked_mean
from .policy_losses import compute_policy_gradient_loss


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    计算 GRPO microbatch 的 policy gradient loss 并反传。

    Args:
        loss_type:
            - "no_baseline": 使用 raw_rewards（shape: batch x 1）
            - "reinforce_with_baseline": 使用 advantages（shape: batch x 1）
            - "grpo_clip": PPO-style clipping，需要 advantages/old_log_probs/cliprange

    Returns:
        loss: 标量张量
        metadata: 一些额外统计信息（例如 clip_fraction）
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards must be provided for loss_type='no_baseline'")
        _raw_rewards = raw_rewards
        _advantages = raw_rewards  # 占位（不会被使用）
        _old_log_probs = policy_log_probs  # 占位（不会被使用）
        _cliprange = 0.0
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                "advantages must be provided for loss_type='reinforce_with_baseline'"
            )
        _raw_rewards = advantages  # 占位（不会被使用）
        _advantages = advantages
        _old_log_probs = policy_log_probs  # 占位（不会被使用）
        _cliprange = 0.0
    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError(
                "advantages, old_log_probs, and cliprange must be provided for loss_type='grpo_clip'"
            )
        _raw_rewards = advantages  # 占位（不会被使用）
        _advantages = advantages
        _old_log_probs = old_log_probs
        _cliprange = float(cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=_raw_rewards,
        advantages=_advantages,
        old_log_probs=_old_log_probs,
        cliprange=_cliprange,
    )

    loss = masked_mean(per_token_loss, response_mask, dim=None)
    loss = loss / float(gradient_accumulation_steps)
    loss.backward()

    # clip_fraction：有多少 response token 触发了 clipping（仅对 grpo_clip 有意义）
    if loss_type == "grpo_clip" and "clip_mask" in metadata:
        clip_fraction = masked_mean(
            metadata["clip_mask"].to(loss.dtype), response_mask, dim=None
        )
        metadata = dict(metadata)
        metadata["clip_fraction"] = clip_fraction

    return loss, metadata

