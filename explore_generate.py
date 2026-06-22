"""
任务 1.4 + 1.5: 单次推理跑通 + 生成参数实验
===========================================
1.4: 加载 Qwen3-0.6B → chat_template → model.generate() → decode
1.5: 系统对比 temperature / top_p / max_new_tokens / do_sample

用法:
    python explore_generate.py                    # 完整实验（含参数扫描）
    python explore_generate.py --quick            # 快速冒烟测试（只跑 1 次推理）
    python explore_generate.py --param-sweep-only # 只跑参数扫描

预计耗时 (CPU):
    --quick:          ~30-60 秒
    完整参数扫描:     ~5-10 分钟（取决于 CPU 性能）

模型加载: FP32，约 1.4GB 内存
推理速度: ~2-5 tokens/s（取决于 CPU）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
if not HF_ENDPOINT:
    print("⚠️  HF_ENDPOINT 未设置！执行: export HF_ENDPOINT=https://hf-mirror.com")
    print()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"


# ============================================================
# 测试 Prompt
# ============================================================
# 长 prompt 用来测试 max_new_tokens 的影响更明显
SHORT_PROMPT_MESSAGES = [
    {"role": "user", "content": "用一句话解释什么是大语言模型。"},
]

LONG_PROMPT_MESSAGES = [
    {"role": "user", "content": "请用中文详细解释：Transformer 模型的注意力机制是如何工作的？请从 Query、Key、Value 三个矩阵开始讲起。"},
]


def load_model():
    """
    加载 Qwen3-0.6B 模型和 tokenizer。

    AutoModelForCausalLM.from_pretrained() 的工作流程:
      1. 读取 config.json → 构建模型架构（1024 hidden, 28 layers, GQA...）
      2. 读取 model.safetensors → 加载全部权重到内存
      3. torch_dtype=torch.float32 → 用 FP32 精度（每个参数 4 字节）
      4. device_map="cpu" → 强制放在 CPU 内存
      5. 返回一个可调用 generate() 的模型对象

    内存占用: ~0.6B × 4 字节 ≈ 2.4GB（含优化器状态预留）
    实际观察: ~1.4GB（safetensors 文件大小）到 ~2.4GB（展开后）
    """
    print(f"正在加载模型: {QWEN_MODEL_ID}")
    print(f"  torch dtype: float32 (FP32)")
    print(f"  device: CPU")
    print(f"  预计内存占用: ~1.4-2.4 GB")

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    print(f"  ✓ tokenizer 加载完成 ({time.time() - t0:.1f}s)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    # model.eval() 关闭 dropout / batch norm 训练行为，切换到推理模式
    # 不设这个可能输出不稳定（虽然 dropout 概率很小的扰动在微调中才明显）
    model.eval()
    print(f"  ✓ 模型加载完成 ({time.time() - t0:.1f}s)")

    return tokenizer, model


def generate_and_measure(tokenizer, model, messages, **gen_kwargs):
    """
    执行一次完整的推理并测量时间。

    参数:
        tokenizer, model: 已加载的 tokenizer 和模型
        messages: 符合 OpenAI 格式的对话列表
        **gen_kwargs: 生成参数 (temperature, top_p, max_new_tokens...)

    返回:
        dict: 包含 input/output/timing 的完整记录
    """
    # 第一步: messages → prompt 字符串（通过 chat_template）
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 第二步: prompt → token IDs
    #   return_tensors="pt": 返回 PyTorch tensor [1, seq_len]
    inputs = tokenizer(prompt, return_tensors="pt")

    input_token_count = inputs["input_ids"].shape[1]

    # 第三步: model.generate() —— 核心推理步骤
    #   这是整个流程中最耗时的部分（自回归逐 token 生成）
    #   输入 shape: [1, input_len] 的 token ids
    #   输出 shape: [1, input_len + generated_len] 的 token ids
    t_start = time.time()

    with torch.no_grad():  # 关闭梯度计算——推理不需要梯度，能省内存
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )

    t_end = time.time()

    # 第四步: output token IDs → 文本
    #   outputs[0] = [input_len + generated_len] 的完整序列
    #   只取生成的 token（跳过输入部分）
    generated_ids = outputs[0][input_token_count:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 计算指标
    generated_tokens = len(generated_ids)
    total_time = t_end - t_start
    tokens_per_sec = generated_tokens / total_time if total_time > 0 else 0

    result = {
        "prompt_chars": len(prompt),
        "input_tokens": input_token_count,
        "generated_tokens": generated_tokens,
        "total_time_s": round(total_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "generated_text": generated_text,
        "gen_kwargs": gen_kwargs.copy(),
    }

    return result


# ============================================================
# 任务 1.4: 单次推理跑通
# ============================================================
def task14_single_inference(tokenizer, model):
    """
    跑通完整推理流程并展示每一步的中间产物。

    整个流程:
    用户输入 (messages)
        │
        ▼
    tokenizer.apply_chat_template()  →  prompt 字符串
        │
        ▼
    tokenizer.__call__()  →  input_ids (token id 序列)
        │
        ▼
    model.generate()  →  output_ids (input + 生成的 token)
        │
        ▼
    tokenizer.decode()  →  生成的文本
    """
    print(f"\n{'='*70}")
    print(f"  任务 1.4: 单次推理跑通")
    print(f"{'='*70}")

    messages = [
        {"role": "user", "content": "用一句话解释什么是API。"},
    ]

    # Step 1: chat_template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n  📝 Step 1 - apply_chat_template → prompt 字符串:")
    print(f"  {'─'*50}")
    for line in prompt.split("\n"):
        print(f"  │ {line}")
    print(f"  {'─'*50}")

    # Step 2: tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"\n  🔢 Step 2 - tokenize → input_ids:")
    print(f"    shape: {inputs['input_ids'].shape}  ← [batch=1, sequence_length]")
    print(f"    前 5 个 token IDs: {inputs['input_ids'][0][:5].tolist()}")
    print(f"    后 5 个 token IDs: {inputs['input_ids'][0][-5:].tolist()}")

    # Step 3: generate
    print(f"\n  🤖 Step 3 - model.generate() —— 正在推理...")
    result = generate_and_measure(
        tokenizer, model, messages,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
    )

    # Step 4: decode
    print(f"\n  📄 Step 4 - decode → 生成文本:")
    print(f"  {'─'*50}")
    print(f"  {result['generated_text'][:300]}")
    print(f"  {'─'*50}")

    print(f"\n  ⏱️  性能指标:")
    print(f"    输入 token 数:   {result['input_tokens']}")
    print(f"    生成 token 数:   {result['generated_tokens']}")
    print(f"    总耗时:          {result['total_time_s']:.1f}s")
    print(f"    生成速度:        {result['tokens_per_sec']:.1f} tokens/s")
    print(f"    (如果生成 512 token 的完整回复，预计需要 "
          f"{512 / max(result['tokens_per_sec'], 0.1):.0f}s)")

    return result


# ============================================================
# 任务 1.5: 生成参数实验
# ============================================================
def task15_param_sweep(tokenizer, model):
    """
    系统对比 4 个生成参数对输出质量和速度的影响。

    ┌────────────────┬─────────────────────────────────────────────┐
    │ 参数           │ 作用                                         │
    ├────────────────┼─────────────────────────────────────────────┤
    │ temperature    │ 控制「创造力」——低=确定性输出, 高=更多样性     │
    │ top_p          │ 核采样阈值——只从累积概率 ≤ top_p 的候选选      │
    │ max_new_tokens │ 最多生成多少个 token（硬上限，控制延迟）       │
    │ do_sample      │ True=从概率分布采样, False=每步选最可能token  │
    └────────────────┴─────────────────────────────────────────────┘

    这些参数在 API 层面也都会暴露给用户（OpenAI API 的 temperature/top_p/max_tokens）
    """
    print(f"\n{'='*70}")
    print(f"  任务 1.5: 生成参数系统实验")
    print(f"{'='*70}")

    # 使用固定 prompt 保证对比的公平性
    messages = [
        {"role": "user", "content": "请用中文简要介绍人工智能的三个主要应用领域。"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n  📝 统一 Prompt ( {len(prompt)} 字符 ):")
    for line in prompt.split("\n"):
        if line.strip():
            print(f"     {line}")

    results = []
    configs = []

    # ---- 实验组 1: Temperature 对比 ----
    # temperature → 1.0 时所有候选 token 的概率按原始分布
    # temperature → 0.1 时高概率 token 被极大加强（接近 greedy）
    # temperature → 1.5 时概率分布被「压平」——低概率 token 被推高
    print(f"\n  {'─'*60}")
    print(f"  实验组 1: Temperature (top_p=0.9, max_new_tokens=128)")
    print(f"  {'─'*60}")
    for temp in [0.1, 0.7, 1.5]:
        configs.append({
            "group": "Temperature",
            "label": f"T={temp}",
            "params": {"temperature": temp, "top_p": 0.9, "max_new_tokens": 128, "do_sample": True},
        })

    # ---- 实验组 2: top_p 对比 ----
    # top_p=nucleus sampling —— 按概率降序排列候选 token，累加到概率和 ≥ p 为止
    # p=1.0 ≈ 不过滤（全部候选都参与采样）
    # p=0.5 ≈ 只保留"最可能的一半"
    print(f"\n  {'─'*60}")
    print(f"  实验组 2: top_p (temperature=0.7, max_new_tokens=128)")
    print(f"  {'─'*60}")
    for p in [0.5, 0.9, 1.0]:
        configs.append({
            "group": "top_p",
            "label": f"p={p}",
            "params": {"temperature": 0.7, "top_p": p, "max_new_tokens": 128, "do_sample": True},
        })

    # ---- 实验组 3: max_new_tokens 对比 ----
    # 控制最大生成长度——相当于告诉模型"回答不要太长"
    # 所有主流 LLM API 都有这个限制
    print(f"\n  {'─'*60}")
    print(f"  实验组 3: max_new_tokens (temperature=0.7, top_p=0.9)")
    print(f"  {'─'*60}")
    for mnt in [32, 128, 256]:
        configs.append({
            "group": "max_new_tokens",
            "label": f"max={mnt}",
            "params": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": mnt, "do_sample": True},
        })

    # ---- 实验组 4: do_sample (sampling vs greedy) ----
    # do_sample=False → greedy decoding：每步选概率最高的 token
    #   优点：确定性、最快
    #   缺点：可能陷入重复循环，输出单一
    # do_sample=True → 从概率分布中随机采样
    #   优点：输出有变化、更像人类
    #   缺点：每次结果不同、可能产生不连贯内容
    print(f"\n  {'─'*60}")
    print(f"  实验组 4: do_sample (temperature=0.7, top_p=0.9, max_new_tokens=128)")
    print(f"  {'─'*60}")
    for sample in [False, True]:
        label = "sampling" if sample else "greedy"
        configs.append({
            "group": "do_sample",
            "label": label,
            "params": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 128, "do_sample": sample},
        })

    # ---- 实验组 5: 极端确定性 vs 极端随机性 ----
    print(f"\n  {'─'*60}")
    print(f"  实验组 5: 极端对比")
    print(f"  {'─'*60}")
    # Greedy: 完全确定性
    configs.append({
        "group": "Extreme",
        "label": "greedy_deterministic",
        "params": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 128, "do_sample": False},
    })
    # High randomness: 高温度 + 低 top_p
    configs.append({
        "group": "Extreme",
        "label": "high_randomness",
        "params": {"temperature": 1.5, "top_p": 0.5, "max_new_tokens": 128, "do_sample": True},
    })

    # ---- 运行所有实验 ----
    print(f"\n  🏃 开始参数扫描（共 {len(configs)} 组）...\n")

    for i, cfg in enumerate(configs):
        group = cfg["group"]
        label = cfg["label"]
        params = cfg["params"]

        print(f"  [{i+1:2d}/{len(configs)}] {group:15s} | {label:25s} ...", end=" ", flush=True)

        try:
            result = generate_and_measure(tokenizer, model, messages, **params)
            result["group"] = group
            result["label"] = label
            results.append(result)

            # 展示关键指标 + 生成文本的前 80 字符
            preview = result["generated_text"].replace("\n", " ")[:80]
            print(f"⏱ {result['total_time_s']:5.1f}s | "
                  f"{result['generated_tokens']:3d} tok | "
                  f"{result['tokens_per_sec']:4.1f} tok/s | "
                  f"「{preview}...」")
        except Exception as e:
            print(f"❌ 失败: {e}")

    # ---- 汇总表 ----
    print(f"\n{'='*70}")
    print(f"  参数实验汇总表")
    print(f"{'='*70}")
    print(f"\n{'组别':<15s} {'参数':<25s} {'耗时(s)':>8s} {'生成Tok':>7s} {'速度(tok/s)':>10s} {'前60字'}")
    print(f"{'─'*15} {'─'*25} {'─'*8} {'─'*7} {'─'*10} {'─'*60}")
    for r in results:
        preview = r["generated_text"].replace("\n", " ")[:60]
        print(f"{r['group']:<15s} {r['label']:<25s} "
              f"{r['total_time_s']:>7.1f}s {r['generated_tokens']:>6d} "
              f"{r['tokens_per_sec']:>9.1f} {preview}")

    # ---- 关键结论 ----
    print(f"\n{'='*70}")
    print(f"  💡 关键发现")
    print(f"{'='*70}")

    # 提取每组数据做分析
    temp_results = [r for r in results if r["group"] == "Temperature"]
    top_p_results = [r for r in results if r["group"] == "top_p"]
    maxlen_results = [r for r in results if r["group"] == "max_new_tokens"]
    sample_results = [r for r in results if r["group"] == "do_sample"]
    extreme_results = [r for r in results if r["group"] == "Extreme"]

    if temp_results:
        best = max(temp_results, key=lambda r: r["tokens_per_sec"])
        print(f"\n  Temperature 影响:")
        print(f"    T 对生成速度基本无影响（所有 T 值 ≈ {temp_results[0]['tokens_per_sec']:.1f} tok/s）")
        print(f"    T 影响的是输出的「多样性」，不是「速度」")
        print(f"    低 T (0.1): 输出更确定，接近 greedy")
        print(f"    高 T (1.5): 输出更多样，但也可能更随机")

    if sample_results:
        greedy = [r for r in sample_results if r["label"] == "greedy"][0]
        sampling = [r for r in sample_results if r["label"] == "sampling"][0]
        print(f"\n  Sampling vs Greedy:")
        print(f"    Greedy:   {greedy['total_time_s']:.1f}s, {greedy['tokens_per_sec']:.1f} tok/s")
        print(f"    Sampling: {sampling['total_time_s']:.1f}s, {sampling['tokens_per_sec']:.1f} tok/s")
        print(f"    速度基本一致（每 token 的计算量相同，只是选择方式不同）")

    if extreme_results:
        greedy_det = [r for r in extreme_results if "greedy" in r["label"]][0]
        high_rand = [r for r in extreme_results if "randomness" in r["label"]][0]
        print(f"\n  极端对比 (确定性 vs 高随机性):")
        print(f"    Greedy 确定:        {greedy_det['generated_text'][:80]}...")
        print(f"    高随机性(T=1.5):    {high_rand['generated_text'][:80]}...")
        print(f"    高随机性下可能出现重复/不连贯——这是 temperature 过高的典型症状")

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM 推理实验")
    parser.add_argument("--quick", action="store_true", help="只跑 1 次推理（冒烟测试）")
    parser.add_argument("--param-sweep-only", action="store_true", help="只跑参数扫描")
    args = parser.parse_args()

    print("=" * 70)
    print("  任务 1.4 + 1.5: LLM 推理跑通 + 生成参数实验")
    print(f"  模型: {QWEN_MODEL_ID}")
    print(f"  设备: CPU")
    print("=" * 70)

    # ---- 加载模型（最耗时的一步，约 10-30 秒） ----
    print("\n⚙️  加载模型...")
    t_load = time.time()
    tokenizer, model = load_model()
    print(f"  模型加载总耗时: {time.time() - t_load:.0f}s\n")

    # ---- 任务 1.4: 单次推理 ----
    if not args.param_sweep_only:
        task14_single_inference(tokenizer, model)

    if args.quick:
        print(f"\n✅ --quick 模式完成！运行 python explore_generate.py 做完整参数扫描。")
        return

    # ---- 任务 1.5: 参数扫描 ----
    results = task15_param_sweep(tokenizer, model)

    # ---- 保存结果 ----
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": QWEN_MODEL_ID,
        "device": "CPU",
        "results": [{
            "group": r["group"],
            "label": r["label"],
            "gen_kwargs": r["gen_kwargs"],
            "input_tokens": r["input_tokens"],
            "generated_tokens": r["generated_tokens"],
            "total_time_s": r["total_time_s"],
            "tokens_per_sec": r["tokens_per_sec"],
            "text_preview": r["generated_text"][:200],
        } for r in results],
    }
    os.makedirs("results", exist_ok=True)
    out_path = "results/param_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已保存到 {out_path}")

    print(f"\n{'='*70}")
    print(f"  ✅ 任务 1.4 + 1.5 完成！")
    print(f"  下一步 → 任务 1.6: python explore_quantization.py (8-bit 量化)")
    print(f"  下一步 → 任务 1.7: python benchmark.py (批量压测)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
