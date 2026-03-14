from __future__ import annotations

"""
计算条件 log-prob（以及可选的 token 熵）。

给定：
  input_ids: 模型输入序列（通常是 prompt+response 去掉最后一个 token）
  labels:    next-token 目标（通常是同一序列去掉第一个 token）

我们计算每个位置 label token 的 log-prob：
  log p(labels_t | input_ids_<=t)
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .entropy import compute_entropy
from .model_io import model_forward_logits


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    """
    返回每个位置的 label token log-prob。

    这里不对 prompt/padding 做 mask；mask 一般在训练循环中用 response_mask 处理。
    """
    logits = model_forward_logits(model, input_ids)
    log_probs_vocab = F.log_softmax(logits, dim=-1)

    # 用 labels 在 vocab 维上 gather 出“真实 token 的 log-prob”
    token_log_probs = log_probs_vocab.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    out: dict[str, Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out

