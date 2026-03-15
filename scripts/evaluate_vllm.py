from __future__ import annotations

"""
作业 Question (math_baseline) 对应的评测脚本骨架。

目标（对应你截图中的 (a) 要求）：
1) 从 MATH 验证集文件加载样本（默认路径按作业文档：/data/a5-alignment/MATH/validation.json）
2) 使用 r1_zero 提示词把题目格式化为语言模型输入 prompt
3) 用 vLLM 为每个 prompt 生成模型输出
4) 用 reward_fn 计算评估指标（reward / format_reward / answer_reward）
5) 把样本、prompt、模型输出、以及评估分数序列化到磁盘，便于后续分析

说明：
- 作业文档里展示的 evaluate_vllm 签名没有包含 ground_truths，但 reward_fn 需要
  (response, ground_truth) 两个参数。这里我们在函数里增加 ground_truths 这个参数，
  以便把评估逻辑写完整且可复用。
- 你后续可以复用 evaluate_vllm 做 GSM8K/MMLU/AlpacaEval 等任务的评测。
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable

from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


@dataclass(frozen=True)
class MathExample:
    """
    MATH 样本的最小抽象表示。

    我们只关心两件事：
    - question：题目文本，用于构造 prompt
    - ground_truth：标准答案（通常来自 solution 字段，包含 \\boxed{...}）
    """

    question: str
    ground_truth: str
    raw: dict[str, Any]


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    """
    读取 JSON（list）或 JSONL（每行一个 JSON object）。

    作业文档给的文件名是 validation.json，但不同分发版本可能用 JSONL。
    为了鲁棒性，这里两种格式都支持。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    # 如果是 JSONL：第一行就是一个 dict，而不是以 '[' 开头的列表
    if not text.lstrip().startswith("["):
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got: {type(data)}")
    return data


def load_math_validation(path: str) -> list[MathExample]:
    """
    加载 MATH 验证集，并把字段规范化成 (question, ground_truth)。

    常见字段名：
    - question/problem：题目文本
    - solution/answer：标准答案或包含推导过程的 solution（通常含 \\boxed{}）
    """
    raw_records = _read_json_or_jsonl(path)
    examples: list[MathExample] = []
    for rec in raw_records:
        if not isinstance(rec, dict):
            continue

        question = rec.get("problem") or rec.get("question") or rec.get("prompt")
        ground_truth = rec.get("solution") or rec.get("answer") or rec.get("ground_truth")

        if question is None or ground_truth is None:
            continue

        examples.append(
            MathExample(
                question=str(question),
                ground_truth=str(ground_truth),
                raw=rec,
            )
        )
    return examples


def _extract_gsm8k_final_answer(answer_text: str) -> str | None:
    """
    GSM8K 的标准答案通常在最后一行形如：'#### 18'
    这里提取最后的数字/字符串答案，作为 reward_fn 的 ground_truth。
    """
    if "####" not in answer_text:
        return None
    final = answer_text.split("####")[-1].strip()
    if not final:
        return None
    final = final.replace(",", "")
    final = final.replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", final)
    if m is None:
        return None
    return m.group(0)


def load_gsm8k(path: str) -> list[MathExample]:
    """
    从本仓库自带的 GSM8K JSONL（data/gsm8k/*.jsonl）加载样本。

    注意：
    - GSM8K 的 answer 字段通常包含推导过程，最后用 '#### <final>' 给出最终答案
    - 我们把 ground_truth 规范化成最终答案（字符串），用于 r1_zero_reward_fn 的对比
    """
    raw_records = _read_json_or_jsonl(path)
    examples: list[MathExample] = []
    for rec in raw_records:
        if not isinstance(rec, dict):
            continue
        question = rec.get("question")
        answer = rec.get("answer")
        if question is None or answer is None:
            continue
        final = _extract_gsm8k_final_answer(str(answer))
        if final is None:
            continue
        examples.append(
            MathExample(
                question=str(question),
                ground_truth=final,
                raw=rec,
            )
        )
    return examples


def load_prompt_template(path: str) -> str:
    """
    读取 prompt 模板文件（例如 cs336_alignment/prompts/r1_zero.prompt）。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_r1_zero_prompts(examples: list[MathExample], template: str) -> list[str]:
    """
    将 MATH question 注入到 r1_zero 模板里，生成最终 prompt 字符串。
    """
    prompts: list[str] = []
    for ex in examples:
        prompts.append(template.format(question=ex.question))
    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
) -> None:
    """
    在一组 prompts 上评测 vLLM 模型，计算指标并把结果写到磁盘。

    写出的 JSONL 里每行包含：
    - idx: 样本序号
    - prompt: 最终输入给模型的 prompt（已经按 r1_zero 格式化）
    - ground_truth: 标准答案
    - output: 模型生成的文本
    - metrics: reward_fn 输出的字典（含 reward/format_reward/answer_reward）

    同时生成一个 summary JSON，方便你在作业 (b)(c) 里做汇总分析。
    """
    if len(prompts) != len(ground_truths):
        raise ValueError(
            "prompts and ground_truths must have the same length, "
            f"got {len(prompts)} and {len(ground_truths)}"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    raw_outputs = vllm_model.generate(prompts, eval_sampling_params)

    # vLLM 的返回结构：每个 RequestOutput 对应一个 prompt，
    # output.outputs 是一个列表（对应 n 个采样），通常我们取第 0 个。
    outputs_text: list[str] = []
    for out in raw_outputs:
        if not out.outputs:
            outputs_text.append("")
            continue
        outputs_text.append(out.outputs[0].text)

    all_metrics: list[dict[str, float]] = []
    counts = {
        "format1_answer1": 0,
        "format1_answer0": 0,
        "format0_answer0": 0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (prompt, gt, output) in enumerate(
            tqdm(zip(prompts, ground_truths, outputs_text), total=len(prompts))
        ):
            metrics = reward_fn(output, gt)
            all_metrics.append(metrics)

            fmt = float(metrics.get("format_reward", 0.0))
            ans = float(metrics.get("answer_reward", 0.0))
            if fmt == 1.0 and ans == 1.0:
                counts["format1_answer1"] += 1
            elif fmt == 1.0 and ans == 0.0:
                counts["format1_answer0"] += 1
            else:
                counts["format0_answer0"] += 1

            record = {
                "idx": i,
                "prompt": prompt,
                "ground_truth": gt,
                "output": output,
                "metrics": metrics,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_examples": len(prompts),
        "mean_reward": mean([m.get("reward", 0.0) for m in all_metrics]) if all_metrics else 0.0,
        "mean_format_reward": mean([m.get("format_reward", 0.0) for m in all_metrics]) if all_metrics else 0.0,
        "mean_answer_reward": mean([m.get("answer_reward", 0.0) for m in all_metrics]) if all_metrics else 0.0,
        "counts": counts,
    }

    summary_path = output_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Evaluation complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote per-example results to: {output_path}")
    print(f"Wrote summary to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_gsm8k_path = os.path.join(repo_root, "data", "gsm8k", "test.jsonl")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["math", "gsm8k"],
        default="math",
        help="Which dataset format to evaluate. Use 'gsm8k' if you don't have the MATH dataset locally.",
    )
    parser.add_argument(
        "--math-validation-path",
        type=str,
        default="/data/a5-alignment/MATH/validation.json",
        help="MATH validation file path (JSON or JSONL).",
    )
    parser.add_argument(
        "--gsm8k-path",
        type=str,
        default=default_gsm8k_path,
        help="GSM8K JSONL path (defaults to the copy included under ./data/gsm8k).",
    )
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "cs336_alignment",
            "prompts",
            "r1_zero.prompt",
        ),
        help="Path to r1_zero prompt template.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        help="Local path (recommended) or HF repo id for the model.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "outputs", "math_baseline_r1_zero.jsonl"),
        help="Where to write the per-example JSONL results.",
    )

    # vLLM / 采样参数
    parser.add_argument("--num-gpus", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)

    args = parser.parse_args()

    if args.dataset == "math":
        if not os.path.exists(args.math_validation_path):
            raise FileNotFoundError(
                "MATH 验证集文件不存在："
                f"{args.math_validation_path}\n"
                "这通常是因为 MATH 数据集有版权限制，作业仓库不会自带。\n"
                "你有两种选择：\n"
                "1) 把 MATH 数据集下载/放到该路径，或用 --math-validation-path 指向你本机真实路径；\n"
                "2) 直接用仓库自带的 GSM8K 先跑通流程：\n"
                f"   uv run scripts/evaluate_vllm.py --dataset gsm8k --gsm8k-path {default_gsm8k_path}\n"
            )
        examples = load_math_validation(args.math_validation_path)
    else:
        if not os.path.exists(args.gsm8k_path):
            raise FileNotFoundError(
                f"GSM8K 文件不存在：{args.gsm8k_path}\n"
                "请确认你是在仓库根目录运行，或用 --gsm8k-path 指向 ./data/gsm8k/test.jsonl。"
            )
        examples = load_gsm8k(args.gsm8k_path)
    if not examples:
        raise ValueError(
            "未加载到任何样本。请检查你选择的数据集与路径：\n"
            "- 若 --dataset math：检查 --math-validation-path\n"
            "- 若 --dataset gsm8k：检查 --gsm8k-path\n"
        )

    template = load_prompt_template(os.path.abspath(args.prompt_template_path))
    prompts = build_r1_zero_prompts(examples, template)
    ground_truths = [ex.ground_truth for ex in examples]

    vllm_model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=os.path.abspath(args.output_path),
    )


if __name__ == "__main__":
    main()
