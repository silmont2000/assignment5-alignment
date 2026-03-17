from __future__ import annotations

"""
SFT 实验脚本（对应作业 handout 的 sft_experiment）。

你可以把它当作“把前面写好的基础积木拼成一个可跑的训练 + 验证流程”的示例。

整体结构（非常重要）：
1) 训练/评估数据：使用本仓库的 MATH（JSONL）：
   - ./MATH/sft.jsonl（训练）
   - ./MATH/validation.jsonl（评估）
2) 训练（policy 端）：transformers 的 AutoModelForCausalLM + AdamW
   - 每个 microbatch：tokenize -> forward 得到 log_probs -> sft_microbatch_train_step backward
   - 每累积 gradient_accumulation_steps 次 microbatch：clip grad -> optimizer.step() -> zero_grad()
3) 评估（eval 端）：与训练共用同一张 GPU（更省钱）
   - 用 transformers 的 generate 生成 answer
   - 用 r1_zero_reward_fn 打分
4) 输出：把训练与评估指标按行写入 outputs/sft_experiment.jsonl，便于你后续画曲线。

注意：
 - 这个脚本的目标是“把作业要求跑通 + 结构清晰”，而不是追求极致训练效率。
 - 真正达到较高准确率，需要你在有 GPU 的环境里调参并跑足训练步数。
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any
import wandb

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass(frozen=True)
class SFTExample:
    """
    SFT 训练样本的最小表示形式。

    - prompt：提示词（包含题目/指令/上下文）
    - response：目标输出（包含推理轨迹 + 最终答案）
    - ground_truth：可选字段，用于“筛选只保留正确样本”的实验
    """
    prompt: str
    response: str
    ground_truth: str | None
    raw: dict[str, Any]


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """
    读取 JSONL：每行一个 JSON object。
    """
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_prompt_template(path: str) -> str:
    """
    读取 prompt 模板（例如 cs336_alignment/prompts/r1_zero.prompt）。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_math_as_sft(
    path: str,
    prompt_template: str,
    require_ground_truth: bool,
) -> list[SFTExample]:
    """
    加载 MATH 数据为本脚本训练所需的 SFTExample（prompt/response）。

    支持两种格式：
    1) 标准格式：{problem, solution, answer}
    2) SFT 格式（MATH/sft.jsonl）：{input, output}
    """
    raw = _read_jsonl(path)
    out: list[SFTExample] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        if "input" in rec and "output" in rec:
            prompt = str(rec.get("input") or "")
            response = str(rec.get("output") or "")
            if not prompt or not response:
                continue

            response = response.replace("</think><answer>", "</think> <answer>")
            if response and not response[0].isspace():
                response = f"\n{response}"

            out.append(SFTExample(prompt=prompt, response=response, ground_truth=response, raw=rec))
            continue

        problem = rec.get("problem") or rec.get("question") or rec.get("prompt")
        solution = rec.get("solution")
        answer = rec.get("answer")
        if problem is None or solution is None:
            continue
        if require_ground_truth and (answer is None or str(answer).strip() == ""):
            continue

        prompt = prompt_template.format(question=str(problem))
        response = f"\n{solution}\n</think> <answer>{str(answer or '').strip()}</answer>"
        out.append(SFTExample(prompt=prompt, response=response, ground_truth=str(solution), raw=rec))
    return out


def _set_seed(seed: int) -> None:
    """
    设置 torch 随机种子，保证抽样/打乱等行为可复现。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shuffle_in_place(items: list[Any], seed: int) -> None:
    """
    用固定 seed 的方式原地打乱 list，保证“每次跑出来的子集选择一致”。
    """
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(items), generator=g).tolist()
    items[:] = [items[i] for i in perm]


def _select_subset(examples: list[SFTExample], n: int | None, seed: int) -> list[SFTExample]:
    """
    选择训练子集：
    - n=None 表示用全量
    - 否则取打乱后的前 n 条

    对应 handout 的要求：比较不同数据规模（128/256/512/1024/full）下的效果。
    """
    copied = list(examples)
    _shuffle_in_place(copied, seed)
    if n is None:
        return copied
    return copied[:n]


def _iter_minibatches(examples: list[SFTExample], batch_size: int):
    """
    一个极简的 batch 切分器：把 list 按 batch_size 切片。

    这里不使用 DataLoader，是因为训练数据已经是“prompt/response 的 Python list”，
    对这个作业任务来说，直接切片更直观，也更方便阅读学习。
    """
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


@torch.no_grad()
def evaluate_policy_with_transformers(
    tokenizer: AutoTokenizer,
    policy: torch.nn.Module,
    prompts: list[str],
    ground_truths: list[str],
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, float]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    was_training = bool(policy.training)
    policy.eval()

    device = next(policy.parameters()).device
    rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []

    eval_batch_size = 12
    
    for start in range(0, len(prompts), eval_batch_size):
        cur_prompts = prompts[start : start + eval_batch_size]
        cur_gts = ground_truths[start : start + eval_batch_size]

        batch = tokenizer(cur_prompts, return_tensors="pt", padding=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": bool(do_sample),
            "max_new_tokens": int(max_tokens),
            "pad_token_id": int(tokenizer.pad_token_id),
        }
        if bool(do_sample):
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = float(top_p)
            if int(top_k) > 0:
                generate_kwargs["top_k"] = int(top_k)

        gen = policy.generate(
            **generate_kwargs,
        )

        input_lens = attention_mask.sum(dim=1).tolist()
        for i, gt in enumerate(cur_gts):
            out_ids = gen[i, int(input_lens[i]) :]
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            scores = r1_zero_reward_fn(text, gt, fast=True)
            rewards.append(float(scores.get("reward", 0.0)))
            format_rewards.append(float(scores.get("format_reward", 0.0)))
            answer_rewards.append(float(scores.get("answer_reward", 0.0)))

    if was_training:
        policy.train()

    return {
        "eval_reward_mean": mean(rewards) if rewards else 0.0,
        "eval_format_reward_mean": mean(format_rewards) if format_rewards else 0.0,
        "eval_answer_reward_mean": mean(answer_rewards) if answer_rewards else 0.0,
    }


@torch.no_grad()
def evaluate_checkpoint_and_write_reward1_examples(
    checkpoint_path: str,
    math_eval_path: str,
    prompt_template: str,
    device: str,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    output_path: str,
    eval_num_examples: int | None,
) -> dict[str, Any]:
    eval_examples = load_math_as_sft(
        path=math_eval_path,
        prompt_template=prompt_template,
        require_ground_truth=True,
    )
    if not eval_examples:
        raise ValueError("未加载到任何 MATH 评估样本。")

    if eval_num_examples is not None and int(eval_num_examples) > 0:
        eval_examples = eval_examples[: int(eval_num_examples)]

    prompts = [ex.prompt for ex in eval_examples]
    ground_truths = [str(ex.ground_truth) for ex in eval_examples]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    kept = 0

    eval_batch_size = 12
    with open(os.path.abspath(output_path), "w", encoding="utf-8") as f:
        for start in range(0, len(prompts), eval_batch_size):
            cur_prompts = prompts[start : start + eval_batch_size]
            cur_gts = ground_truths[start : start + eval_batch_size]
            cur_examples = eval_examples[start : start + eval_batch_size]

            batch = tokenizer(cur_prompts, return_tensors="pt", padding=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generate_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "do_sample": bool(do_sample),
                "max_new_tokens": int(max_tokens),
                "pad_token_id": int(tokenizer.pad_token_id),
            }
            if bool(do_sample):
                generate_kwargs["temperature"] = float(temperature)
                generate_kwargs["top_p"] = float(top_p)
                if int(top_k) > 0:
                    generate_kwargs["top_k"] = int(top_k)

            gen = model.generate(**generate_kwargs)

            input_lens = attention_mask.sum(dim=1).tolist()
            for i, (ex, gt) in enumerate(zip(cur_examples, cur_gts)):
                out_ids = gen[i, int(input_lens[i]) :]
                text = tokenizer.decode(out_ids, skip_special_tokens=True)
                scores = r1_zero_reward_fn(text, gt, fast=True)
                r = float(scores.get("reward", 0.0))
                rewards.append(r)
                format_rewards.append(float(scores.get("format_reward", 0.0)))
                answer_rewards.append(float(scores.get("answer_reward", 0.0)))
                if r >= 1.0:
                    kept += 1
                    f.write(
                        json.dumps(
                            {
                                "prompt": ex.prompt,
                                "ground_truth": gt,
                                "output": text,
                                "scores": scores,
                                "raw": ex.raw,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "event": "collect_reward1_summary",
        "checkpoint": checkpoint_path,
        "eval_num_examples": len(eval_examples),
        "reward1_count": kept,
        "reward1_rate": float(kept / max(1, len(eval_examples))),
        "eval_reward_mean": mean(rewards) if rewards else 0.0,
        "eval_format_reward_mean": mean(format_rewards) if format_rewards else 0.0,
        "eval_answer_reward_mean": mean(answer_rewards) if answer_rewards else 0.0,
        "output_path": os.path.abspath(output_path),
    }


# def _parse_subset_sizes(raw: str) -> list[int | None]:
#     raw_sizes: list[str] = [s.strip() for s in str(raw).split(",") if s.strip()]
#     subset_sizes: list[int | None] = []
#     for s in raw_sizes:
#         if s.lower() in {"full", "all"}:
#             subset_sizes.append(None)
#         else:
#             subset_sizes.append(int(s))
#     return subset_sizes


# def _parse_int_list(raw: str | None) -> list[int] | None:
#     if raw in (None, "", "none", "null"):
#         return None
#     parts = [p.strip() for p in str(raw).split(",") if p.strip()]
#     return [int(p) for p in parts]



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name-or-path", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join("outputs", "checkpoints", "sft_math"))

    parser.add_argument("--collect-reward1-checkpoint", type=str, default=None)
    parser.add_argument(
        "--collect-reward1-output-path",
        type=str,
        default=os.path.join("outputs", f"math_reward1_examples_{int(time.time())}.jsonl"),
    )
    parser.add_argument("--eval-num-examples", type=int, default=None)

    parser.add_argument("--wandb-project", type=str, default="cs336-a5-sft-math-Math")

    # 单卡：训练与评估都在同一张 GPU 上跑
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    # 训练超参（你后续做实验时主要会调这些）
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    # 评估相关参数
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--eval-subset-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--eval-do-sample", action="store_true")
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-top-k", type=int, default=0)
    
    # Filtering argument for Part 2
    parser.add_argument("--filter-data", action="store_true", help="Filter training data to keep only correct examples.")

    args = parser.parse_args()

    _set_seed(int(args.seed))

    math_train_path = "/root/autodl-tmp/assignment5-alignment/MATH/sft.jsonl"
    math_eval_path = "/root/autodl-tmp/assignment5-alignment/MATH/validation.jsonl"
    prompt_template_path = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    prompt_template = load_prompt_template(os.path.abspath(prompt_template_path))


    if args.collect_reward1_checkpoint not in (None, "", "none", "null"):
        summary = evaluate_checkpoint_and_write_reward1_examples(
            checkpoint_path=str(args.collect_reward1_checkpoint),
            math_eval_path=math_eval_path,
            prompt_template=prompt_template,
            device=str(args.policy_device),
            max_tokens=int(args.max_tokens),
            do_sample=bool(args.eval_do_sample),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            top_k=int(args.eval_top_k),
            output_path=str(args.collect_reward1_output_path),
            eval_num_examples=int(args.eval_num_examples) if args.eval_num_examples is not None else None,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return

    examples = load_math_as_sft(
        path=math_train_path,
        prompt_template=prompt_template,
        require_ground_truth=False,
    )
    math_eval = load_math_as_sft(
        path=math_eval_path,
        prompt_template=prompt_template,
        require_ground_truth=True,
    )

    # subset_size_grid = [128, 256, 512, 1024, None]
    subset_size_grid = [128, 256, 512, 1024, None]

    eval_subset_size = args.eval_subset_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # =========================
    # 核心：subset size 扫描（每个 subset size 单独一个 W&B run）
    # =========================
    
    # Pre-load ground truth for filtering if needed
    prompt_to_gt = {}
    if args.filter_data:
        print("Loading train.jsonl for filtering ground truth...")
        train_examples_gt = load_math_as_sft(
            path="/root/autodl-tmp/assignment5-alignment/MATH/train.jsonl",
            prompt_template=prompt_template,
            require_ground_truth=True,
        )
        prompt_to_gt = {ex.prompt: ex.ground_truth for ex in train_examples_gt}
        print(f"Loaded {len(prompt_to_gt)} ground truth examples.")

    for subset_n in subset_size_grid:
        # Select subset FIRST, then filter? Or Filter then select subset?
        # Assignment: "Filter... Run SFT on the (full) filtered dataset."
        # And also "Run SFT on ... range {128, ...} ... and full dataset."
        # The filtering part seems to be a separate experiment ("2. Filter...").
        # If args.filter_data is set, we assume we are doing the filtering experiment.
        # In that case, we should filter the FULL dataset first, then maybe subsample?
        # The assignment says: "Run SFT on the (full) filtered dataset, and report the size... and accuracy."
        # It does NOT say to run subset sizes on the filtered dataset.
        # So if filter is on, we just run on the FULL filtered dataset.
        # But to support general usage, I will apply filtering to the source examples, 
        # and then subset logic applies to the filtered list.
        
        current_examples = list(examples)
        if args.filter_data:
            print(f"Filtering {len(current_examples)} examples...")
            filtered = []
            for ex in current_examples:
                gt = prompt_to_gt.get(ex.prompt)
                if gt:
                    # check correctness
                    score = r1_zero_reward_fn(ex.response, gt)
                    if score.get("reward", 0.0) == 1.0:
                        filtered.append(ex)
            print(f"Filtered down to {len(filtered)} correct examples.")
            current_examples = filtered
            
            # For filtering experiment, we usually just want the full filtered dataset.
            # But the loop iterates subset_size_grid.
            # If we want just the full filtered run, we should probably ignore the other sizes or let the user specify.
            # The user can control subset_size_grid via code or arguments, but here it's hardcoded.
            # If filtering is ON, and subset_n is NOT None, we effectively subsample the filtered set.
            # This is fine.

        subset = _select_subset(current_examples, subset_n, seed=int(args.seed))
        run_tag = "full" if subset_n is None else str(subset_n)
        if args.filter_data:
            run_tag += "_filtered"

        # Fixed epochs for consistent training
        epochs_eff = args.num_epochs

        wandb_run = None
        run_name = f"subset{run_tag}"
        if args.filter_data:
            run_name += "_filtered"

        wandb_run = wandb.init(
            project=str(args.wandb_project),
            name=run_name,
            config={
                "seed": int(args.seed),
                "learning_rate": float(args.learning_rate),
                "epochs": float(epochs_eff),
                "micro_batch_size": int(args.micro_batch_size),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "max_tokens": int(args.max_tokens),
                "eval_do_sample": bool(args.eval_do_sample),
                "eval_temperature": float(args.eval_temperature),
                "eval_top_p": float(args.eval_top_p),
                "eval_top_k": int(args.eval_top_k),
                "filter_data": args.filter_data,
            },)

        policy = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        policy.to(args.policy_device)
        policy.train()

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=float(args.learning_rate),
            weight_decay=0.0,
        )

        optimizer.zero_grad(set_to_none=True)
        opt_step = 0
        accum_in_group = 0

        microbatch_size = int(args.micro_batch_size)
        gradient_accumulation_steps = int(args.gradient_accumulation_steps)
        micro_total_step = int(math.ceil(len(subset) / (microbatch_size * gradient_accumulation_steps)))
        total_optimizer_steps = int(max(1, math.ceil(float(epochs_eff) * micro_total_step))) # 要跑完指定的轮次需要多少个batch


        pbar = tqdm(
            total=int(total_optimizer_steps),
            desc=f"sft(subset={run_tag})",
            dynamic_ncols=True,
        )
        epoch_pass = 0
        while opt_step < int(total_optimizer_steps):
            _shuffle_in_place(subset, seed=int(args.seed) + int(epoch_pass))
            epoch_pass += 1
            for batch in _iter_minibatches(subset, batch_size=int(microbatch_size)):
                # 现在一个batch就是一次小循环里的所有样本。

                batch_prompts = [ex.prompt for ex in batch]
                batch_responses = [ex.response for ex in batch]
                tok = tokenize_prompt_and_output(
                    prompt_strs=batch_prompts, output_strs=batch_responses, tokenizer=tokenizer
                )
                input_ids = tok["input_ids"].to(args.policy_device)
                labels = tok["labels"].to(args.policy_device)
                response_mask = tok["response_mask"].to(args.policy_device)
                logprob_out = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                policy_log_probs = logprob_out["log_probs"]
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=int(gradient_accumulation_steps),
                    normalize_constant=1.0,
                )

                accum_in_group += 1
                if accum_in_group >= int(gradient_accumulation_steps):
                    accum_in_group = 0
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                    pbar.update(1)

                    record: dict[str, Any] = {
                        "train_step": opt_step,
                        "train_loss": float(loss.detach().cpu().item()),
                    }
                    if int(args.eval_every_steps) > 0 and opt_step % int(args.eval_every_steps) == 0:
                        cur_eval = _select_subset(
                            math_eval,
                            int(eval_subset_size),
                            seed=int(args.seed) + int(opt_step),
                        )
                        eval_prompts = [ex.prompt for ex in cur_eval]
                        eval_gts = [str(ex.ground_truth) for ex in cur_eval]
                        metrics = evaluate_policy_with_transformers(
                            tokenizer=tokenizer,
                            policy=policy,
                            prompts=eval_prompts,
                            ground_truths=eval_gts,
                            max_tokens=int(args.max_tokens),
                            do_sample=bool(args.eval_do_sample),
                            temperature=float(args.eval_temperature),
                            top_p=float(args.eval_top_p),
                            top_k=int(args.eval_top_k),
                        )
                        record.update(metrics)
                    wandb.log(dict(record), step=int(opt_step))
                    if opt_step >= int(total_optimizer_steps):
                        break
            if accum_in_group > 0 and opt_step < int(total_optimizer_steps):
                accum_in_group = 0
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                pbar.update(1)
            if opt_step >= int(total_optimizer_steps):
                break
        pbar.close()

        policy.eval()
        cur_eval = _select_subset(
            math_eval,
            int(eval_subset_size),
            seed=int(args.seed) + int(opt_step) + 10_000,
        )
        eval_prompts = [ex.prompt for ex in cur_eval]
        eval_gts = [str(ex.ground_truth) for ex in cur_eval]
        final_metrics = evaluate_policy_with_transformers(
            tokenizer=tokenizer,
            policy=policy,
            prompts=eval_prompts,
            ground_truths=eval_gts,
            max_tokens=int(args.max_tokens),
            do_sample=bool(args.eval_do_sample),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            top_k=int(args.eval_top_k),
        )
        policy.train()
        final_record: dict[str, Any] = {
            "train_step": opt_step,
            "train_loss": float(loss.detach().cpu().item()),
            **final_metrics,
            "eval_accuracy": float(final_metrics.get("eval_reward_mean", 0.0)),
        }
        wandb.log(dict(final_record), step=int(opt_step))
        print(json.dumps(final_record, ensure_ascii=False))

        ckpt_dir = os.path.join(
            os.path.abspath(str(args.checkpoint_dir)),
            f"subset_{run_tag}_bs{int(microbatch_size * gradient_accumulation_steps)}_epochs{str(epochs_eff).replace('.', 'p')}_{int(time.time())}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        policy.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        del policy
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wandb_run.finish()

if __name__ == "__main__":
    main()
