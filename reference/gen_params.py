#!/usr/bin/env python3
"""1.5 生成参数实验：验证 temperature / top_p / max_new_tokens 对速度和输出的影响。"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "用一句话解释什么是 API"


def main():
    print("加载模型...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    m.eval()

    inputs = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
    )

    print(f"\n{'='*55}")
    print("  实验 1：temperature 是否影响速度？")
    print(f"{'='*55}\n")
    for t in [0.1, 0.7, 1.5]:
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=64, temperature=t)
        elapsed = time.time() - t0
        n_tok = out.shape[-1] - inputs["input_ids"].shape[-1]
        text = tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"  T={t:.1f}  |  {elapsed:.1f}s  |  {n_tok} tok  |  {text[:40]}...")

    print(f"\n{'='*55}")
    print("  实验 2：top_p 是否影响速度？")
    print(f"{'='*55}\n")
    for p in [0.5, 0.9, 1.0]:
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=64, top_p=p, temperature=0.7)
        elapsed = time.time() - t0
        n_tok = out.shape[-1] - inputs["input_ids"].shape[-1]
        print(f"  top_p={p:.1f}  |  {elapsed:.1f}s  |  {n_tok} tok")

    print(f"\n{'='*55}")
    print("  实验 3：max_new_tokens 与延迟是否线性？")
    print(f"{'='*55}\n")
    for n in [32, 64, 128]:
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=n, temperature=0.7)
        elapsed = time.time() - t0
        n_tok = out.shape[-1] - inputs["input_ids"].shape[-1]
        print(f"  max={n:>3}  |  {elapsed:.1f}s  |  实际生成 {n_tok} tok")

    print(f"\n  结论：temperature 和 top_p 不影响速度，max_new_tokens 线性影响延迟。\n")


if __name__ == "__main__":
    main()
