from __future__ import annotations

"""
CS336 Assignment 5（Alignment）核心实现的导出入口。

你要求把实现“按很细粒度拆分成多个文件”，所以具体逻辑被拆到了多个模块：
- tokenization.py：分词与 response_mask
- entropy.py：熵
- logprobs.py：log-prob
- rewards.py：GRPO 组内归一化奖励
- policy_losses.py：policy gradient / GRPO-Clip loss
- masking.py：masked mean/normalize
- sft.py：SFT microbatch train step
- grpo.py：GRPO microbatch train step
- packed_sft_dataset.py：可选数据集 packing + iterate_batches
- parsing.py：MMLU/GSM8K 输出解析
- dpo.py：DPO 单样本 loss

单元测试通过 tests/adapters.py 从本包导入这些符号，因此这里负责“集中 re-export”。
"""

from .dpo import compute_per_instance_dpo_loss
from .entropy import compute_entropy
from .grpo import grpo_microbatch_train_step
from .logprobs import get_response_log_probs
from .masking import masked_mean, masked_normalize
from .packed_sft_dataset import PackedSFTDataset, get_packed_sft_dataset, iterate_batches
from .parsing import parse_gsm8k_response, parse_mmlu_response
from .policy_losses import (
    compute_grpo_clip_loss,
    compute_naive_policy_gradient_loss,
    compute_policy_gradient_loss,
)
from .rewards import compute_group_normalized_rewards
from .sft import sft_microbatch_train_step
from .tokenization import tokenize_prompt_and_output

__all__ = [
    "PackedSFTDataset",
    "compute_entropy",
    "compute_group_normalized_rewards",
    "compute_grpo_clip_loss",
    "compute_naive_policy_gradient_loss",
    "compute_per_instance_dpo_loss",
    "compute_policy_gradient_loss",
    "get_packed_sft_dataset",
    "get_response_log_probs",
    "grpo_microbatch_train_step",
    "iterate_batches",
    "masked_mean",
    "masked_normalize",
    "parse_gsm8k_response",
    "parse_mmlu_response",
    "sft_microbatch_train_step",
    "tokenize_prompt_and_output",
]

