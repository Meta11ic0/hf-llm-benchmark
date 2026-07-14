#!/usr/bin/env python3
"""Benchmark Qwen3-0.6B: 3 prompt × 3 次，输出 avg/P50/P99/tok/s。"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 常量 ──────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-0.6B"
REPEAT = 3  # 每条 prompt 重复次数

PROMPTS = [
    {"content": "用一句话解释什么是 API"},
    {"content": "请翻译成英文：今天天气真好"},
    {"content": "用中文解释人工智能的三个应用"},
]


# ── 加载模型 ──────────────────────────────────────
def load_model(max_new_tokens = 4096):
    """加载 tokenizer 和 model，返回 (tokenizer, model)。"""
    print("加载 tokenizer 和 model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    m.eval()
    m.generation_config.max_new_tokens = max_new_tokens  # 让 EOS 自然终止
    return tok, m


# ── 单次推理 ──────────────────────────────────────
def run_one(tok, m, content):
    """执行一次推理，返回 (耗时秒, 生成token数)。"""
    messages = [{"role": "user", "content": content}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )

    t0 = time.time()
    with torch.no_grad():
        output_ids = m.generate(**inputs)
    elapsed = time.time() - t0

    n_tokens = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
    return elapsed, n_tokens


# ── 统计计算 ──────────────────────────────────────
def stats(times):
    """给定耗时列表，返回 (avg, p50, p99)。"""
    s = sorted(times)
    avg = sum(s) / len(s)
    p50 = s[len(s) // 2]
    p99 = s[min(int(len(s) * 0.99), len(s) - 1)]
    return avg, p50, p99


# ── 主流程 ────────────────────────────────────────
def main():
    tokenizer, model = load_model()

    # 预热
    print("\n预热中...")
    run_one(tokenizer, model, "预热")
    print("预热完成\n")

    # 正式 benchmark
    print(f"{'='*60}")
    print(f"  benchmark: {len(PROMPTS)} prompt × {REPEAT} 次")
    print(f"{'='*60}\n")

    for i, cfg in enumerate(PROMPTS):
        times = []
        total_tok = 0
        for run in range(REPEAT):
            elapsed, n = run_one(tokenizer, model, cfg["content"])
            times.append(elapsed)
            total_tok += n
            print(f"  [{i+1}/{len(PROMPTS)}] {cfg['content'][:25]}... "
                  f"run {run+1}: {elapsed:.2f}s, {n} tok")

        avg, p50, p99 = stats(times)
        tok_s = total_tok / sum(times) if sum(times) > 0 else 0
        print(f"  → avg={avg:.2f}s  P50={p50:.2f}s  P99={p99:.2f}s  "
              f"tok/s={tok_s:.1f}\n")


if __name__ == "__main__":
    main()
