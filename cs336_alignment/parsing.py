from __future__ import annotations

"""
解析模型输出的小工具（可选部分）。

这里的目标不是“完全准确理解文本”，而是实现一个足够鲁棒的解析器，
把模型输出映射成：
- MMLU：选项字母 A/B/C/D
- GSM8K：最后出现的整数（字符串形式）
"""

import re
from typing import Any


_MMLU_OPTION_RE = re.compile(
    r"""
    (?:
        \b(?:answer|correct\s+answer)\b     # 提示词：answer / correct answer
        [^A-Da-d]{0,20}                    # 允许少量噪声字符
        ([A-Da-d])                         # 选项字母
    )
    |
    (?:
        \(([A-Da-d])\)                     # 形如 (B)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    """
    从模型输出中解析出 MMLU 的选项字母。

    单元测试希望：
      "The correct answer is B."  -> "B"

    若解析失败，返回 None。
    """
    del mmlu_example  # 解析用不到，但保留该参数以对齐作业 API

    matches = _MMLU_OPTION_RE.findall(model_output)
    for m1, m2 in matches:
        letter = (m1 or m2).upper()
        if letter in {"A", "B", "C", "D"}:
            return letter
    return None


_GSM8K_NUMBER_RE = re.compile(r"(?<!\w)(-?\d+)(?!\w)")


def parse_gsm8k_response(model_output: str) -> str | None:
    """
    GSM8K：取模型输出里“最后出现的整数”作为预测答案。

    若输出中完全没有数字，返回 None。
    """
    numbers = _GSM8K_NUMBER_RE.findall(model_output)
    if not numbers:
        return None
    return numbers[-1]

