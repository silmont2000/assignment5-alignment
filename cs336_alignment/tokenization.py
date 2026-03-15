from __future__ import annotations

"""
分词与 response mask 构造。

本作业里很多函数都需要：
- input_ids：作为模型输入（通常是完整序列去掉最后一个 token）
- labels：作为 next-token 监督信号（通常是完整序列去掉第一个 token）
- response_mask：标记哪些 label 位置属于“回复部分”（用于只在回复 token 上计算 loss）

注意：response_mask 的对齐最容易写错，因为 labels 相对于 input_ids 是左移一位的。
"""

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    将 prompt 与 output 分别分词，并构造用于训练的 input_ids/labels/response_mask。

    关键对齐关系（非常重要）：
    - full_input_ids：对 prompt+output 的分词结果（含 padding）
    - input_ids = full_input_ids[:, :-1]
    - labels    = full_input_ids[:, 1:]
    - 若 prompt 在 full_input_ids 中占用 P 个 token，那么：
        - output 的第一个 token 在 full_input_ids 的索引为 P
        - 在 labels 中对应的索引为 P-1（因为 labels 左移一位）

    返回张量的形状（与单元测试一致）：
    - input_ids / labels / response_mask: (batch_size, max_seq_len - 1)
    """
    if len(prompt_strs) != len(output_strs): # 这里，实际上看的是batch size
        raise ValueError(
            "prompt_strs and output_strs must have the same length, "
            f"got {len(prompt_strs)} and {len(output_strs)}"
        )

    prompt_enc = tokenizer(
        prompt_strs,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    output_enc = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    prompt_ids_list: list[list[int]] = prompt_enc["input_ids"]
    output_ids_list: list[list[int]] = output_enc["input_ids"]

    full_ids_list: list[list[int]] = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list)
    ]

    max_len = max((len(ids) for ids in full_ids_list), default=0)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id for padding")

    full_input_ids = torch.full(
        (len(full_ids_list), max_len),
        fill_value=int(pad_id),
        dtype=torch.long,
    )
    full_attention_mask = torch.zeros(
        (len(full_ids_list), max_len),
        dtype=torch.long,
    )

    for i, ids in enumerate(full_ids_list):
        if not ids:
            continue
        full_input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        full_attention_mask[i, : len(ids)] = 1

    input_ids = full_input_ids[:, :-1].contiguous()
    labels = full_input_ids[:, 1:].contiguous()

    # response_mask 与 labels 对齐：True 表示该 label 位置属于 output 部分。
    response_mask = torch.zeros_like(labels, dtype=torch.bool)

    for i, prompt_ids in enumerate(prompt_ids_list):
        prompt_len = len(prompt_ids)
        full_len = int(full_attention_mask[i].sum().item())

        # labels 的有效长度是 full_len - 1（去掉了第一个 token）
        # output 的起始索引在 labels-space 中是 prompt_len - 1
        start = max(prompt_len - 1, 0)
        end = max(full_len - 1, 0)
        if end > start:
            response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
