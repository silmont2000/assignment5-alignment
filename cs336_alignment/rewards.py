from __future__ import annotations

"""
奖励计算与组内归一化（GRPO 相关）。

GRPO 的核心思想之一：同一个 prompt 采样出多条 response（一个 group），
在组内做归一化，把“相对更好”的 response 赋予正 advantage，把“相对更差”的
response 赋予负 advantage，从而不需要额外训练一个 value baseline。
"""

from typing import Callable

import torch
from torch import Tensor


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """
    计算 raw rewards，并在每个 group 内做归一化。

    Args:
        reward_fn:
            输入 (response, ground_truth) -> dict，至少包含 key "reward"。
            可选包含 "format_reward" / "answer_reward"（用于日志统计）。
        rollout_responses:
            policy 采样得到的 response 列表，长度为 rollout_batch_size。
        repeated_ground_truths:
            对应的 ground truth，长度同 rollout_batch_size。
            注意：同一个 prompt 的 ground truth 会重复 group_size 次。
        group_size:
            每个 prompt 采样多少条 response。
        advantage_eps:
            防止除以 0 的 epsilon。
        normalize_by_std:
            True 则除以组内 std；False 则只做减均值。

    Returns:
        normalized_rewards:
            shape (rollout_batch_size,) 的组内归一化结果（即 advantages）。
        raw_rewards:
            shape (rollout_batch_size,) 的原始 reward。
        metadata:
            一些统计信息（可用于 wandb/logging）。
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length, "
            f"got {len(rollout_responses)} and {len(repeated_ground_truths)}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if len(rollout_responses) % group_size != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by group_size, got "
            f"{len(rollout_responses)} % {group_size} != 0"
        )

    rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, gt)
        r = float(scores["reward"])
        rewards.append(r)
        format_rewards.append(float(scores.get("format_reward", r)))
        answer_rewards.append(float(scores.get("answer_reward", r)))

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)

    grouped = raw_rewards.view(-1, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - group_mean

    if normalize_by_std:
        # 用 sample std（unbiased=True），与 torch.std 默认行为一致。
        group_std = grouped.std(dim=1, unbiased=True, keepdim=True)
        advantages = advantages / (group_std + advantage_eps)

    normalized_rewards = advantages.view(-1)

    metadata = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=False).item()),
        "format_reward_mean": float(torch.tensor(format_rewards).mean().item()),
        "answer_reward_mean": float(torch.tensor(answer_rewards).mean().item()),
    }

    return normalized_rewards, raw_rewards, metadata
