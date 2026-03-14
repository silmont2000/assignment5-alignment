from __future__ import annotations

"""
可选部分：SFT 数据集的“packing”实现，以及简单的 batch 迭代器。

为什么要 packing？
- 语言模型训练通常希望输入是固定长度 seq_length 的块
- 把多个短文档拼接成连续 token stream，可以减少 padding 浪费

本实现刻意与测试夹具的构造方式对齐（非常关键）：
1) 每条样本是 {"prompt": ..., "response": ...}
2) 先拼成 document: prompt + "\\n\\n" + response
3) 将所有 document 再用 "\\n\\n" 连接成 full_text
4) tokenizer.encode(full_text, add_special_tokens=True) 只会在最开头加一次 BOS
5) 用 (seq_length + 1) 的窗口分块，生成 input_ids/labels
"""

import json
import os
import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class PackedExample:
    input_ids: Tensor
    labels: Tensor


class PackedSFTDataset(Dataset):
    """
    一个最小的 packed 语言模型数据集。

    每个样本返回：
      - input_ids: (seq_length,)
      - labels:    (seq_length,)
    """

    def __init__(self, examples: list[PackedExample]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ex = self._examples[idx]
        return {"input_ids": ex.input_ids, "labels": ex.labels}


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    从 JSONL 加载 SFT 数据并做 packing。

    注意：
    - shuffle=True 时，我们使用固定随机种子，保证在单元测试/可复现环境中稳定。
    """
    dataset_path = os.fspath(dataset_path)
    documents: list[str] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = str(record["prompt"])
            response = str(record["response"])
            documents.append(f"{prompt}\n\n{response}")

    if shuffle:
        rng = random.Random(0)
        rng.shuffle(documents)

    full_text = "\n\n".join(documents)

    # 关键点：只 tokenize 一次，且 add_special_tokens=True
    token_ids = tokenizer.encode(full_text, add_special_tokens=True)

    examples: list[PackedExample] = []
    # 每个训练样本需要 seq_length+1 个 token 才能构造 input_ids 和 labels
    for start in range(0, len(token_ids) - seq_length, seq_length):
        chunk = token_ids[start : start + seq_length + 1]
        if len(chunk) < seq_length + 1:
            break
        input_ids = torch.tensor(chunk[:seq_length], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        examples.append(PackedExample(input_ids=input_ids, labels=labels))

    return PackedSFTDataset(examples)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    构造一个简单的 DataLoader，用于一个 epoch 的 batch 迭代。
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

