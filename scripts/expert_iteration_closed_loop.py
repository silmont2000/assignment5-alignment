from __future__ import annotations

"""
闭环 Expert Iteration（EI）脚本：模型自探索 -> 奖励筛选 -> 用“新产生的专家数据”做 SFT -> 再自探索。

本脚本实现作业图里 (Expert Iteration / self-improvement) 的最小可运行闭环，并且数据集使用
本仓库内的 MATH（./MATH/train.jsonl / ./MATH/validation.jsonl）。

核心流程（每个 EI step）：
1) Sampling Engine：
   - 对训练集每个问题，用当前模型 do_sample=True 采样 G 条候选 response（best-of-G）
   - 通过 temperature (0.6~1.0) 提升多样性
   - 生成后在“第二个 </answer> 标签”处截断，避免复读/废话（即便模型继续生成也会被裁掉）
2) Expert Filter：
   - 用 r1_zero_reward_fn(response, ground_truth) 检查正确性（reward==1.0）
   - 策略 A：保留所有正确样本
   - 策略 B：每题只保留 log_prob( response | prompt ) 最高的那条正确样本
3) Iterative Trainer：
   - 对筛选出的专家数据做 1~2 个 epoch 的 SFT
   - 总共循环 n_ei_steps=5 次
   - 在不同 EI step 使用不同的有效 batch size Db（Db = microbatch_size * grad_accum_steps）
4) Monitoring & Logging：
   - 训练时同时记录 “Response Entropy”（只在 response tokens 上做 masked mean）
   - 关键指标同步到 W&B（尽量少传无关配置/指标）
"""

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import (
    get_response_log_probs,
    masked_mean,
    masked_normalize,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

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


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass(frozen=True)
class Problem:
    prompt: str
    ground_truth: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ExpertSample:
    prompt: str
    response: str
    ground_truth: str


def load_math_problems(path: str, prompt_template: str) -> list[Problem]:
    """
    只加载 “(prompt, ground_truth)” 这两件事：
    - prompt：r1_zero 模板 + problem（题目文本）
    - ground_truth：优先用 solution（通常包含 \\boxed{}），否则退化到 answer/ground_truth

    注意：这里不构造标准 response，因为 EI 需要模型自己生成 response。
    """
    raw = _read_json_or_jsonl(path)
    out: list[Problem] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        problem = rec.get("problem") or rec.get("question") or rec.get("prompt")
        ground_truth = rec.get("solution") or rec.get("answer") or rec.get("ground_truth")
        if problem is None or ground_truth is None:
            continue

        prompt = prompt_template.format(question=str(problem))
        gt = str(ground_truth)
        out.append(Problem(prompt=prompt, ground_truth=gt, raw=rec))
    return out


def _iter_minibatches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _truncate_at_second_answer_tag(text: str) -> str:
    """
    作业要求：确保推理在第二个 </answer> 标签处终止（防止模型复读/废话）。

    这里用一个非常朴素但鲁棒的文本裁剪：
    - 若能找到第 2 个 </answer>，就截断到它（包含该标签）
    - 否则保持原样（模型可能只输出了 1 个标签，甚至没输出标签）
    """
    tag = "</answer>"
    first = text.find(tag)
    if first < 0:
        return text
    second = text.find(tag, first + len(tag))
    if second < 0:
        return text
    return text[: second + len(tag)]


@torch.no_grad()
def sample_best_of_g(
    policy: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    group_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    prompt_batch_size: int,
) -> list[list[str]]:
    """
    Sampling Engine：对每个 prompt 采样 G 条候选 response。

    返回：
      responses_per_prompt: List[ List[str] ]，外层长度=len(prompts)，内层长度=G

    重要点：
    - do_sample=True + temperature/top_p 让同一题产生多样化轨迹
    - 生成后做 </answer> 第二次出现处的截断（满足作业 EOS 控制要求）
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = next(policy.parameters()).device
    was_training = bool(policy.training)
    policy.eval()

    all_out: list[list[str]] = []
    total_batches = int(math.ceil(len(prompts) / max(1, int(prompt_batch_size))))
    for batch_prompts in tqdm(
        _iter_minibatches(prompts, int(prompt_batch_size)),
        total=total_batches,
        desc=f"sampling(best-of-{group_size})",
        dynamic_ncols=True,
    ):
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        gen = policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(group_size),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=int(tokenizer.pad_token_id),
        )

        input_lens = attention_mask.sum(dim=1).tolist()
        input_lens_rep = [int(l) for l in input_lens for _ in range(int(group_size))]

        decoded: list[str] = []
        for i in range(gen.shape[0]):
            out_ids = gen[i, int(input_lens_rep[i]) :]
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            decoded.append(_truncate_at_second_answer_tag(text))

        for i in range(len(batch_prompts)):
            start = i * int(group_size)
            all_out.append(decoded[start : start + int(group_size)])

    if was_training:
        policy.train()
    return all_out


@torch.no_grad()
def evaluate_policy_with_transformers(
    tokenizer: AutoTokenizer,
    policy: torch.nn.Module,
    prompts: list[str],
    ground_truths: list[str],
    max_tokens: int,
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

        gen = policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=int(max_tokens),
            pad_token_id=int(tokenizer.pad_token_id),
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
def _compute_response_logprob_sums(
    policy: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    responses: list[str],
    microbatch_size: int,
) -> list[float]:
    """
    用当前 policy 计算 log p(response | prompt) 的总和（只在 response tokens 上求和）。

    为什么需要它？
    - Expert Filter 的策略 B：同一题多个正确答案时，只保留 log_prob 最高的那一个。
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")

    device = next(policy.parameters()).device
    was_training = bool(policy.training)
    policy.eval()

    out: list[float] = []
    for batch_prompts, batch_responses in zip(
        _iter_minibatches(prompts, int(microbatch_size)),
        _iter_minibatches(responses, int(microbatch_size)),
    ):
        tok = tokenize_prompt_and_output(
            prompt_strs=list(batch_prompts),
            output_strs=list(batch_responses),
            tokenizer=tokenizer,
        )
        input_ids = tok["input_ids"].to(device)
        labels = tok["labels"].to(device)
        response_mask = tok["response_mask"].to(device)

        logprob_out = get_response_log_probs(
            model=policy,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
        )
        token_log_probs = logprob_out["log_probs"]
        sums = masked_normalize(token_log_probs, response_mask, dim=1, normalize_constant=1.0)
        out.extend([float(x) for x in sums.detach().cpu().tolist()])

    if was_training:
        policy.train()
    return out


@torch.no_grad()
def filter_experts(
    policy: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problems: list[Problem],
    responses_per_prompt: list[list[str]],
    expert_strategy: str,
    logprob_microbatch_size: int,
) -> tuple[list[ExpertSample], dict[str, float]]:
    """
    Expert Filter：
    - 先用 r1_zero_reward_fn 判断正确性（reward==1.0）
    - 再按策略 A/B 产出最终专家数据集
    """
    if len(problems) != len(responses_per_prompt):
        raise ValueError("problems and responses_per_prompt must have the same length")

    correct_prompt_idx: list[int] = []
    correct_prompts: list[str] = []
    correct_responses: list[str] = []
    correct_gts: list[str] = []

    total = 0
    correct = 0
    for i, (prob, candidates) in enumerate(zip(problems, responses_per_prompt)):
        for resp in candidates:
            total += 1
            scores = r1_zero_reward_fn(resp, prob.ground_truth, fast=True)
            r = float(scores.get("reward", 0.0))
            if r >= 1.0:
                correct += 1
                correct_prompt_idx.append(i)
                correct_prompts.append(prob.prompt)
                correct_responses.append(resp)
                correct_gts.append(prob.ground_truth)

    if expert_strategy.lower() in {"all", "a"}:
        experts = [
            ExpertSample(prompt=p, response=r, ground_truth=gt)
            for p, r, gt in zip(correct_prompts, correct_responses, correct_gts)
        ]
        meta = {
            "num_candidates": float(total),
            "num_correct_candidates": float(correct),
            "expert_dataset_size": float(len(experts)),
        }
        return experts, meta

    if expert_strategy.lower() not in {"best", "b"}:
        raise ValueError("expert_strategy must be one of: all/best")

    if not correct_prompts:
        return [], {
            "num_candidates": float(total),
            "num_correct_candidates": float(correct),
            "expert_dataset_size": 0.0,
        }

    logprob_sums = _compute_response_logprob_sums(
        policy=policy,
        tokenizer=tokenizer,
        prompts=correct_prompts,
        responses=correct_responses,
        microbatch_size=int(logprob_microbatch_size),
    )

    best_by_problem: dict[int, tuple[float, ExpertSample]] = {}
    for idx, p_idx in enumerate(correct_prompt_idx):
        sample = ExpertSample(
            prompt=correct_prompts[idx],
            response=correct_responses[idx],
            ground_truth=correct_gts[idx],
        )
        lp = float(logprob_sums[idx])
        cur = best_by_problem.get(p_idx)
        if cur is None or lp > cur[0]:
            best_by_problem[p_idx] = (lp, sample)

    experts = [v[1] for v in best_by_problem.values()]
    meta = {
        "num_candidates": float(total),
        "num_correct_candidates": float(correct),
        "expert_dataset_size": float(len(experts)),
    }
    return experts, meta


def _parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("expected a non-empty comma-separated int list")
    return out


def _compute_grad_accum_steps(target_effective_batch_size: int, microbatch_size: int) -> int:
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be > 0")
    if target_effective_batch_size <= 0:
        raise ValueError("target_effective_batch_size must be > 0")
    return max(1, int(target_effective_batch_size) // int(microbatch_size))


def _is_cuda_oom_error(err: BaseException) -> bool:
    if isinstance(err, torch.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "cuda" in msg and "out of memory" in msg


def train_sft_on_experts(
    policy: torch.nn.Module,
    tokenizer: AutoTokenizer,
    experts: list[ExpertSample],
    learning_rate: float,
    microbatch_size: int,
    target_effective_batch_size: int,
    max_grad_norm: float,
    sft_epochs: int,
    log_every_opt_steps: int,
    log_response_entropy: bool,
    ei_step: int,
    global_opt_step: int,
) -> int:
    """
    Iterative Trainer（SFT 部分）。

    这里的实现刻意“像 sft_experiment 一样直白”，方便你对照阅读：
    - 每个 microbatch：tokenize -> forward 得到 log_probs(+entropy) -> sft_microbatch_train_step backward
    - 每累积 gradient_accumulation_steps 次：clip -> step -> zero_grad

    Response Entropy 的计算：
    - get_response_log_probs(return_token_entropy=True) 会返回 token_entropy（每个位置的熵）
    - 只在 response_mask==True 的位置做 masked mean，得到 response_entropy
    """
    if not experts:
        return global_opt_step

    device = next(policy.parameters()).device
    policy.train()

    grad_accum_steps = _compute_grad_accum_steps(
        target_effective_batch_size=int(target_effective_batch_size),
        microbatch_size=int(microbatch_size),
    )
    effective_batch_size = int(microbatch_size) * int(grad_accum_steps)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(learning_rate), weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)

    opt_steps_this_train = 0
    accum = 0

    for epoch in range(int(sft_epochs)):
        pbar = tqdm(
            list(_iter_minibatches(experts, int(microbatch_size))),
            desc=f"sft(ei_step={ei_step},epoch={epoch},Db={effective_batch_size})",
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch_prompts = [ex.prompt for ex in batch]
            batch_responses = [ex.response for ex in batch]
            tok = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
            input_ids = tok["input_ids"].to(device)
            labels = tok["labels"].to(device)
            response_mask = tok["response_mask"].to(device)

            try:
                logprob_out = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=bool(log_response_entropy),
                )
            except Exception as e:
                if not _is_cuda_oom_error(e):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logprob_out = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
            policy_log_probs = logprob_out["log_probs"]
            response_entropy_value: float | None = None
            token_entropy = logprob_out.get("token_entropy")
            if token_entropy is not None:
                response_entropy = masked_mean(token_entropy, response_mask, dim=None)
                response_entropy_value = float(response_entropy.detach().cpu().item())

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=int(grad_accum_steps),
                normalize_constant=1.0,
            )

            accum += 1
            if accum >= int(grad_accum_steps):
                accum = 0
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_opt_step += 1
                opt_steps_this_train += 1

                if int(log_every_opt_steps) > 0 and global_opt_step % int(log_every_opt_steps) == 0:
                    payload: dict[str, float | int] = {
                        "ei_step": int(ei_step),
                        "train_step": int(global_opt_step),
                        "train_loss": float(loss.detach().cpu().item()),
                        "effective_batch_size": int(effective_batch_size),
                    }
                    if response_entropy_value is not None:
                        payload["train_response_entropy"] = float(response_entropy_value)
                    wandb.log(payload, step=int(global_opt_step))

        pbar.close()

    return global_opt_step


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name-or-path", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
    )

    parser.add_argument("--wandb-project", type=str, default="cs336-a5-ei-math")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--prompt-batch-size", type=int, default=8)

    parser.add_argument("--expert-strategy", type=str, default="best", choices=["all", "best"])

    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--effective-batch-sizes", type=str, default="512,1024,2048")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every-opt-steps", type=int, default=5)
    parser.add_argument("--disable-train-response-entropy", action="store_true")

    parser.add_argument("--train-num-problems", type=int, default=1024)
    parser.add_argument("--eval-num-problems", type=int, default=128)

    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join("outputs", "checkpoints", "ei_math"))

    args = parser.parse_args()

    _set_seed(int(args.seed))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = _repo_root()
    math_train_path = os.path.join(repo_root, "MATH", "train.jsonl")
    math_eval_path = os.path.join(repo_root, "MATH", "validation.jsonl")
    prompt_template_path = os.path.join(repo_root, "cs336_alignment", "prompts", "r1_zero.prompt")

    prompt_template = load_prompt_template(os.path.abspath(prompt_template_path))
    train_problems = load_math_problems(math_train_path, prompt_template)
    eval_problems = load_math_problems(math_eval_path, prompt_template)

    if int(args.train_num_problems) > 0:
        train_problems = train_problems[: int(args.train_num_problems)]
    if int(args.eval_num_problems) > 0:
        eval_problems = eval_problems[: int(args.eval_num_problems)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    attn_implementation = str(args.attn_implementation)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    use_flash = attn_implementation.lower() == "flash_attention_2"
    on_cuda = str(args.policy_device).lower().startswith("cuda") and torch.cuda.is_available()
    if use_flash and on_cuda:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model_kwargs["device_map"] = "auto"
    elif use_flash and not on_cuda:
        warnings.warn(
            "FlashAttention requires GPU. Falling back to sdpa because policy-device is not CUDA."
        )
        model_kwargs["attn_implementation"] = "sdpa"
    elif attn_implementation.lower() != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        policy = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    except Exception as e:
        if "attn_implementation" not in model_kwargs:
            raise
        model_kwargs.pop("attn_implementation", None)
        warnings.warn(
            f"Failed to load model with attn_implementation={attn_implementation!r} "
            f"({type(e).__name__}: {e}). Falling back to transformers default."
        )
        policy = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    policy.to(str(args.policy_device))

    effective_batch_sizes = _parse_int_list(str(args.effective_batch_sizes))

    run_name = (
        str(args.wandb_run_name)
        if args.wandb_run_name not in (None, "", "none", "null")
        else f"ei_math_g{int(args.group_size)}_{int(time.time())}"
    )
    wandb.init(
        project=str(args.wandb_project),
        name=run_name,
        config={
            "seed": int(args.seed),
            "model_name_or_path": str(args.model_name_or_path),
            "attn_implementation": str(args.attn_implementation),
            "n_ei_steps": int(args.n_ei_steps),
            "group_size": int(args.group_size),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
            "expert_strategy": str(args.expert_strategy),
            "sft_epochs": int(args.sft_epochs),
            "learning_rate": float(args.learning_rate),
            "microbatch_size": int(args.microbatch_size),
            "effective_batch_sizes": effective_batch_sizes,
            "disable_train_response_entropy": bool(args.disable_train_response_entropy),
            "train_num_problems": int(len(train_problems)),
            "eval_num_problems": int(len(eval_problems)),
        },
    )

    eval_prompts = [p.prompt for p in eval_problems]
    eval_gts = [p.ground_truth for p in eval_problems]

    global_opt_step = 0
    base_eval = evaluate_policy_with_transformers(
        tokenizer=tokenizer,
        policy=policy,
        prompts=eval_prompts,
        ground_truths=eval_gts,
        max_tokens=int(args.max_new_tokens),
    )
    wandb.log({"ei_step": -1, **base_eval}, step=0)

    for ei_step in range(int(args.n_ei_steps)):
        prompts = [p.prompt for p in train_problems]

        responses_per_prompt = sample_best_of_g(
            policy=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            group_size=int(args.group_size),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            prompt_batch_size=int(args.prompt_batch_size),
        )

        experts, filter_meta = filter_experts(
            policy=policy,
            tokenizer=tokenizer,
            problems=train_problems,
            responses_per_prompt=responses_per_prompt,
            expert_strategy=str(args.expert_strategy),
            logprob_microbatch_size=int(args.microbatch_size),
        )

        wandb.log(
            {
                "ei_step": int(ei_step),
                "num_candidates": float(filter_meta["num_candidates"]),
                "num_correct_candidates": float(filter_meta["num_correct_candidates"]),
                "expert_dataset_size": float(filter_meta["expert_dataset_size"]),
            },
            step=int(global_opt_step),
        )

        target_db = effective_batch_sizes[ei_step % len(effective_batch_sizes)]
        global_opt_step = train_sft_on_experts(
            policy=policy,
            tokenizer=tokenizer,
            experts=experts,
            learning_rate=float(args.learning_rate),
            microbatch_size=int(args.microbatch_size),
            target_effective_batch_size=int(target_db),
            max_grad_norm=float(args.max_grad_norm),
            sft_epochs=int(args.sft_epochs),
            log_every_opt_steps=int(args.log_every_opt_steps),
            log_response_entropy=not bool(args.disable_train_response_entropy),
            ei_step=int(ei_step),
            global_opt_step=int(global_opt_step),
        )

        eval_metrics = evaluate_policy_with_transformers(
            tokenizer=tokenizer,
            policy=policy,
            prompts=eval_prompts,
            ground_truths=eval_gts,
            max_tokens=int(args.max_new_tokens),
        )
        wandb.log({"ei_step": int(ei_step), **eval_metrics}, step=int(global_opt_step))

        if bool(args.save_checkpoints):
            out_dir = os.path.join(
                os.path.abspath(str(args.checkpoint_dir)),
                run_name,
                f"ei_step{ei_step}",
            )
            os.makedirs(out_dir, exist_ok=True)
            policy.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
