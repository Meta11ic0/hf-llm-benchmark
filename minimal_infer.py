#!/usr/bin/env python3
"""最小推理：验证 prompt → generate → decode 全链路。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

print("加载 tokenizer 和 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # CPU 用 FP32 最稳
)
model.eval()

messages = [{"role": "user", "content": "用一句话解释什么是 API"}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("\n=== chat_template 结果 ===")
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")
print("\n=== input_ids ===")
print(inputs["input_ids"])

print("\n生成中...")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
text = tokenizer.decode(new_ids, skip_special_tokens=True)
print("\n=== 模型回复 ===")
print(text)