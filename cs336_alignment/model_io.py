from __future__ import annotations

"""
与 HuggingFace/Transformers 模型输出结构相关的兼容层。

不同模型的 forward 输出可能是：
- 一个带 .logits 属性的 ModelOutput
- 或者一个 tuple，其中第 0 个元素是 logits

为了让其它函数不被这些细节干扰，我们统一在这里“抽出 logits”。
"""

from torch import Tensor


def model_forward_logits(model, input_ids: Tensor) -> Tensor:
    """
    对模型做一次 forward，并返回 logits（不做 no_grad）。

    注意：这里刻意不加 no_grad，因为：
    - SFT/GRPO 的训练 loss 需要对 policy_log_probs 反传
    - DPO 里对训练模型 lm 的 logprob 也需要梯度
    """
    out = model(input_ids=input_ids)
    if hasattr(out, "logits"):
        return out.logits
    return out[0]

