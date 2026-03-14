from __future__ import annotations

"""
DPO（Direct Preference Optimization）单样本 loss。

输入：
- prompt
- response_chosen（人类偏好更好的回复）
- response_rejected（人类偏好更差的回复）
- lm（要训练的模型）
- lm_ref（reference 模型，固定不训练）

目标：
  让 lm 相比 lm_ref 更偏向 chosen 而不是 rejected。
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .model_io import model_forward_logits
from .tokenization import tokenize_prompt_and_output


def _sequence_log_prob_sum(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> Tensor:
    """
    计算 logπ(response | prompt) 的“按 token 求和”的标量值。

    关键点：
    - 只在 response tokens 上求和（prompt tokens 不算）
    - 这样才能把不同 prompt 的长度影响去掉，使偏好学习更稳定
    """
    batch = tokenize_prompt_and_output([prompt], [response], tokenizer=tokenizer)

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    logits = model_forward_logits(model, input_ids)
    log_probs_vocab = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_vocab.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    token_log_probs = token_log_probs * response_mask.to(token_log_probs.dtype)
    return token_log_probs.sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    """
    计算单个偏好对的 DPO loss：

      L = -log σ( β * [ (logπ(chosen)-logπ(rejected)) - (logπ_ref(chosen)-logπ_ref(rejected)) ] )

    实现细节：
    - lm 的 logprob 需要梯度（用于训练）
    - lm_ref 的 logprob 不需要梯度（with torch.no_grad）
    """
    logp_pi_chosen = _sequence_log_prob_sum(lm, tokenizer, prompt, response_chosen)
    logp_pi_rejected = _sequence_log_prob_sum(lm, tokenizer, prompt, response_rejected)

    with torch.no_grad():
        logp_ref_chosen = _sequence_log_prob_sum(
            lm_ref, tokenizer, prompt, response_chosen
        )
        logp_ref_rejected = _sequence_log_prob_sum(
            lm_ref, tokenizer, prompt, response_rejected
        )

    preference_logit = float(beta) * (
        (logp_pi_chosen - logp_pi_rejected) - (logp_ref_chosen - logp_ref_rejected)
    )

    return -F.logsigmoid(preference_logit)

