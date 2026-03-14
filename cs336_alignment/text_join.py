from __future__ import annotations

"""
文本拼接相关的小工具。

在对话式/指令式数据集中，prompt 和 response 往往是分开存储的：
- 有的 prompt 末尾已经包含空格或换行
- 有的 response 开头也可能自带空白
- 也有不少数据两边都不带任何分隔符

为了让后续的分词结果更稳定，我们在“确实缺少分隔符”的情况下，
自动插入一个 ASCII 空格作为连接符。
"""


def needs_space_between(prompt: str, output: str) -> bool:
    """
    判断是否需要在 prompt 与 output 之间插入一个空格。

    插入空格的条件（尽量保守）：
    - prompt 与 output 都非空
    - prompt 末尾不是空白字符
    - output 开头不是空白字符
    """
    if not prompt or not output:
        return False
    if prompt[-1].isspace():
        return False
    if output[0].isspace():
        return False
    return True


def join_prompt_and_output(prompt: str, output: str) -> str:
    """
    以“尽量不改变原始文本语义”的方式拼接 prompt 与 output。

    该函数只在必要时插入一个空格，避免出现：
    "Hello" + "world"  -> "Helloworld"  (不希望)
    """
    if needs_space_between(prompt, output):
        return f"{prompt} {output}"
    return f"{prompt}{output}"

