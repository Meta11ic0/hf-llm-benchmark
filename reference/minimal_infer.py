#!/usr/bin/env python3
# Shebang：声明用 Python 3 解释器；`python minimal_infer.py` 时系统自动选用
"""最小推理：验证 prompt → generate → decode 全链路。"""  # 模块 docstring，说明脚本目的

import torch  # PyTorch 是 transformers 的底层计算引擎；tensor 操作和 no_grad 都依赖它
from transformers import AutoModelForCausalLM, AutoTokenizer
# AutoModelForCausalLM — 因果语言模型自动加载器（Causal = 自回归，只看当前 token 之前的上下文）
# AutoTokenizer — 根据 MODEL_ID 自动找到对应词表和分词规则，无需手动指定 tokenizer 类

MODEL_ID = "Qwen/Qwen3-0.6B"
# HuggingFace Hub 仓库名；from_pretrained 先查 ~/.cache/huggingface/，有则加载，无则下载
# Qwen3 系列不区分 base/Instruct 仓库名；0.6B 已内置 chat_template

print("加载 tokenizer 和 model...")  # UX 提示：模型加载约 10–30 秒，避免用户以为卡死

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# 只下载/读取 tokenizer 文件（~11MB）；此时不加载 1.4GB 模型权重
# 提供 encode / decode / apply_chat_template 方法

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,   # CPU 推理用 FP32 最稳；float16 在部分 CPU 上支持不完整
)
# 下载/读取 model.safetensors（~1.4GB），构建 PyTorch 计算图；FP32 内存约 2–3GB

model.eval()  # 切换推理模式：关闭 Dropout 等训练层；同样输入行为确定（类比 release 构建）
print("generation_config:", model.generation_config)  # 全部生成默认参数

messages = [{"role": "user", "content": "用一句话解释什么是 API"}]
# OpenAI 风格对话格式；模型训练时看到的是 chat_template 转换后的原始字符串

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,              # 只返回字符串，不转 token id（便于下一步单独 tokenize 并打印）
    add_generation_prompt=True,  # 关键：末尾追加 <|im_start|>assistant\n，告诉模型轮到你生成了
)
# 漏掉 add_generation_prompt 会重复 prompt 或输出为空
# Qwen3 输出示例：
#   <|im_start|>user\n用一句话解释什么是 API\n<|im_start|>assistant\n

print("\n=== chat_template 结果 ===")
print(prompt)  # 调试：确认 messages → 带特殊标记的字符串 转换正确

inputs = tokenizer(prompt, return_tensors="pt")
# 编码 prompt 为 token id；return_tensors="pt" 返回 PyTorch tensor 供 generate 使用
# 返回值含 input_ids [1, seq_len] 和 attention_mask（无 padding 时全 1）

print("\n=== input_ids ===")
print(inputs["input_ids"])  # 典型约 14 个 token，展示「人类语言 → 数字序列」
print("输入 token 数:", inputs["input_ids"].shape[-1])  # seq_len 维度，直接用已有 tensor

print("\n生成中...")  # generate 在 CPU 上约 10 tok/s，64 token 需 ~6 秒，属正常

with torch.no_grad():  # 推理时不建计算图、不算梯度，省内存（只读调用）
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True)
    # **inputs 解包 input_ids 和 attention_mask
    # max_new_tokens=64 — 最多新生成 64 个 token（不含输入长度）
    # do_sample=False — greedy decoding，每步选最高概率 token，输出确定可复现
    # output_ids shape [1, input_len + new_len]，包含输入 + 生成的全部 token

new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
# output_ids[0] 取 batch 第一条；[input_len:] 切片只保留新生成的 token，去掉 prompt 部分

text = tokenizer.decode(new_ids, skip_special_tokens=True)
# token id → 人类可读文本；skip_special_tokens=True 去掉 <|im_start|> 等控制符

print("\n=== 模型回复 ===")
print(text)  # 验收：应看到关于 API 的中文解释，而非重复 prompt 或乱码