"""
任务 1.3: 分词器入门
====================
深入理解 Qwen3 tokenizer 的工作机制:
  - tokenize 中英文 → 观察 token 数量差异（为什么中文 token 数多？）
  - 理解 input_ids / attention_mask / 特殊 token
  - chat_template: messages → prompt 字符串的转换过程
  - encode → decode 往返实验（信息是否无损？）
  - Llama tokenizer 对比（如果已下载）

用法:
    python explore_tokenizer.py
"""

import os
import sys

# 检查镜像
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
if not HF_ENDPOINT:
    print("⚠️  HF_ENDPOINT 未设置！执行: export HF_ENDPOINT=https://hf-mirror.com")
    print()

from transformers import AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# ============================================================
# 测试语料——精心挑选的句子，用来展示 tokenizer 的核心行为
# ============================================================
TEST_TEXTS = {
    # 中文：短句、长句、文言文风格、混合数字符号
    "中文短句": "你好，今天天气怎么样？",
    "中文长句": "人工智能正在深刻地改变着各行各业的运作方式，从医疗诊断到自动驾驶，从金融风控到教育辅导，大语言模型的出现进一步加速了这一进程。",
    "中英混合": "今天我要写一个 Python 脚本，用 HuggingFace Transformers 加载 LLM 模型，然后做 benchmark 测试。",
    # 英文：短句、长句
    "英文短句": "Hello, how are you today?",
    "英文长句": "Artificial intelligence is profoundly transforming the way industries operate. From medical diagnosis to autonomous driving, from financial risk control to educational tutoring, the emergence of large language models has further accelerated this process.",
    # 特殊场景
    "代码片段": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "纯数字": "2026年6月22日，价格为￥39.90元，占总量的5.2%。",
}


def print_section(title: str):
    """打印分节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_kv(key: str, value, indent: int = 2):
    """打印键值对"""
    prefix = " " * indent
    val_str = str(value)
    # 截断过长的输出
    if len(val_str) > 120:
        val_str = val_str[:120] + "..."
    print(f"{prefix}{key}: {val_str}")


# ============================================================
# 第 1 部分: 加载 tokenizer + 基础概念
# ============================================================
def part1_load_tokenizer():
    """
    加载 Qwen3 tokenizer。

    AutoTokenizer.from_pretrained() 的工作流程:
      1. 从 HF Hub（走镜像）下载 tokenizer 文件到本地缓存
      2. 根据 tokenizer_config.json 自动选择正确的 tokenizer 类
         (Qwen3 → Qwen2Tokenizer)
      3. 加载词表（tokenizer.json / vocab.json + merges.txt）
      4. 设置 chat_template（对话格式模板）
      5. 返回一个可用的 tokenizer 对象
    """
    print_section("第 1 部分: 加载 Tokenizer")

    print("\n正在加载 Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)

    print_kv("Tokenizer 类", type(tokenizer).__name__)
    print_kv("词表大小", tokenizer.vocab_size)
    print_kv("特殊 token 数量", len(tokenizer.all_special_tokens))

    # 展示所有特殊 token
    print("\n  特殊 token 列表:")
    for tok in tokenizer.all_special_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        print(f"    {repr(tok):30s} → id={tok_id}")

    # BOS/EOS/PAD 是什么
    print(f"\n  💡 关键概念:")
    print(f"    BOS (Beginning of Sequence):   序列开始标记 — ID {tokenizer.bos_token_id} {repr(tokenizer.bos_token)}")
    print(f"    EOS (End of Sequence):         序列结束标记 — ID {tokenizer.eos_token_id} {repr(tokenizer.eos_token)}")
    print(f"    PAD (Padding):                 填充标记(补齐不等长序列) — ID {tokenizer.pad_token_id} {repr(tokenizer.pad_token)}")
    print(f"    UNK (Unknown):                 未知标记(词表中不存在的字符) — {repr(tokenizer.unk_token)}")
    print(f"")
    print(f"    类比 C++ 字符串: BOS/EOS ≈ 字符串的起止哨兵字符('\\0')")
    print(f"                    PAD ≈ 结构体对齐填充字节(padding bytes)")

    return tokenizer


# ============================================================
# 第 2 部分: tokenize 中英文——观察 token 数差异
# ============================================================
def part2_tokenize_texts(tokenizer):
    """
    tokenize 的本质: 把人类可读的文本 → 模型可计算的 token id 序列。

    为什么中文的 token 数通常比英文多？
      - 英文有天然的空格分词边界，BPE 可以从空格切分开始
      - 中文没有空格，每个汉字本身就是一个"字符"，BPE 需要额外处理
      - 更根本的原因: 词表的覆盖度。
        Qwen3 的词表(151936)为中文做了优化，所以中文 token 效率比
        Llama(128256，英文为主)好很多。
    """
    print_section("第 2 部分: Tokenize 中英文对比")

    results = []

    for label, text in TEST_TEXTS.items():
        # encode(): 文本 → token IDs 列表
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # convert_ids_to_tokens(): token IDs → 原始 token 字符串
        #   Qwen3 使用 byte-level BPE, token 是字节级别的，中文会显示为乱码
        #   这是正常行为！不影响 encode/decode 的正确性
        raw_tokens = tokenizer.convert_ids_to_tokens(tokens)
        # 用 decode 单独解码每个 token 以获得人类可读的版本
        token_strs = [tokenizer.decode([tid], skip_special_tokens=False) for tid in tokens]

        # 计算字符数 vs token 数的比率
        char_count = len(text)
        token_count = len(tokens)
        ratio = char_count / token_count  # 每个 token 平均覆盖几个字符

        results.append((label, text, char_count, token_count, ratio, tokens, token_strs))

    # ---- 打印结果 ----
    print(f"\n{'语料':<12s} {'文本(截断)':<35s} {'字符数':>6s} {'Token数':>6s} {'字符/Token':>10s}")
    print("-" * 75)
    for label, text, chars, toks, ratio, _, _ in results:
        text_short = text[:35] + "..." if len(text) > 35 else text
        print(f"{label:<12s} {text_short:<35s} {chars:>6d} {toks:>6d} {ratio:>9.1f}")

    print(f"\n  💡 观察:")
    print(f"    中文文本的「字符/Token」比通常 ≈ 0.5-1.5 (一个汉字 ≈ 1 个 token)")
    print(f"    英文文本的「字符/Token」比通常 ≈ 3.0-4.5 (多个字母才组成 1 个 token)")
    print(f"    代码/数字: 比英文更高效，因为常见关键字和模式被 BPE 完整收录")
    print(f"")
    print(f"    这意味着：同样的语义内容，中文需要更多 token")
    print(f"    → 推理时 latency 更高 (每个 token 都要过一次 Transformer)")
    print(f"    → API 计费更贵 (按 token 数收费)")

    # ---- 重点展示一个中文句子 vs 一个英文句子的 token 切分 ----
    print(f"\n{'─'*70}")
    print(f"  🔍 显微镜: 中英混合句子的逐 token 展示")
    print(f"{'─'*70}")
    for label, text, chars, toks, ratio, _, token_strs in results:
        if label == "中英混合":
            print(f"\n  原始文本: {text}")
            print(f"  Token 数: {toks}, 字符/Token: {ratio:.1f}")
            print(f"\n  逐 token 分解 (前 30 个):")
            for i, t in enumerate(token_strs[:30]):
                # Ġ 是 BPE 的前缀，表示"这个 token 前面有空格"
                clean = t.replace("Ġ", "␣")  # 用可见符号替换空格标记
                print(f"    [{i:3d}] {clean}", end="")
                if (i + 1) % 5 == 0:
                    print()
            if len(token_strs) > 30:
                print(f"    ... 还有 {len(token_strs) - 30} 个 token")
            print()

    return results


# ============================================================
# 第 3 部分: input_ids / attention_mask / 返回值的完整含义
# ============================================================
def part3_encoding_details(tokenizer):
    """
    tokenizer.__call__() (即 tokenizer(text, ...)) 返回一个 BatchEncoding 对象。
    它包含三个核心字段:

    ┌──────────────┬──────────────────────────────────────────────────┐
    │ 字段         │ 含义                                              │
    ├──────────────┼──────────────────────────────────────────────────┤
    │ input_ids    │ token id 序列，形状 [batch_size, seq_len]          │
    │ attention_mask│ 注意力掩码，1=真 token, 0=padding(需忽略)         │
    │ token_type_ids│ (仅 BERT 系) 句子 A vs 句子 B 的区分              │
    └──────────────┴──────────────────────────────────────────────────┘

    attention_mask 为什么重要:
      - 不同句子 token 数不同，batch 时需要 pad 到等长
      - 模型必须在注意力计算时屏蔽掉 padding token（否则噪声进入计算）
      - mask=0 的位置，注意力权重被设为 -inf，softmax 后变成 0
    """
    print_section("第 3 部分: input_ids / attention_mask 完整含义")

    # 三条长度不同的文本一起编码——展示 padding 的效果
    texts = [
        "你好",                                           # 很短
        "今天我想讨论一下关于人工智能的未来发展趋势。",    # 中等
        "从前有座山山里有座庙庙里有个老和尚",              # 中等
    ]

    # padding=True: 把短序列用 pad_token 补齐到最长序列的长度
    # truncation=True: 超过 max_length 的序列截断
    # return_tensors="pt": 返回 PyTorch tensor (默认是 Python list)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    print(f"\n  输入文本数: {len(texts)}")
    print(f"  Padding 后形状: {encoded['input_ids'].shape}  ← [batch_size=3, max_seq_len]")

    print(f"\n  input_ids (token id 序列):")
    print(f"  {encoded['input_ids']}")
    print(f"\n  attention_mask (1=有效token, 0=padding需忽略):")
    print(f"  {encoded['attention_mask']}")

    # 可视化 padding 的作用
    print(f"\n  💡 attention_mask 的工作方式:")
    print(f"    文本1「你好」只有 2 个 token，但被 pad 到了和文本2/3 一样长")
    print(f"    attention_mask[0] = [1,1,0,0,0,...] ← 模型只会关注前 2 个位置")
    print(f"    类比 C++: 相当于给数组标记了「有效长度」——超出部分不参与计算")

    # 解码回来验证无损
    print(f"\n  decode 验证 (encode → decode 往返):")
    for i, text in enumerate(texts):
        # skip_special_tokens=True: 去掉 BOS/EOS/PAD
        decoded = tokenizer.decode(encoded["input_ids"][i], skip_special_tokens=True)
        match = "✅" if decoded.strip() == text.strip() else "⚠️"
        print(f"    [{i}] {match} 原始: {repr(text)}")
        print(f"        解码: {repr(decoded)}")


# ============================================================
# 第 4 部分: chat_template——对话模板的魔法
# ============================================================
def part4_chat_template(tokenizer):
    """
    chat_template 是 LLM 推理中最重要的抽象层之一。

    它的作用: 把结构化的对话消息列表 → 模型能理解的纯文本 prompt 字符串。

    为什么需要 chat_template？
      - 每个模型的对话格式不同（Qwen 用 <|im_start|>...<|im_end|>，
        Llama 用 <|begin_of_text|>...<|eot_id|>）
      - 开发者不需要记住每种格式——用 messages 数组 + apply_chat_template()
        自动生成正确的 prompt

    Qwen3 的聊天格式:
      <|im_start|>system
      You are a helpful assistant.<|im_end|>
      <|im_start|>user
      你好<|im_end|>
      <|im_start|>assistant
      你好！有什么我可以帮你的吗？<|im_end|>
    """
    print_section("第 4 部分: chat_template——对话模板")

    # 构造一个标准的对话消息列表
    # 这就是 OpenAI API /v1/chat/completions 中 messages 字段的格式
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答。"},
        {"role": "user", "content": "解释一下什么是 Transformer 模型？"},
    ]

    # apply_chat_template(): messages → 模型训练/推理时看到的实际 prompt 字符串
    # tokenize=True: 同时 tokenize 成 input_ids
    # add_generation_prompt=True: 在末尾追加 assistant 的开始标记
    #   这样 model.generate() 就会从 assistant 的位置开始生成
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # 查看 raw prompt 字符串
    raw_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"\n  📝 原始 messages:")
    for m in messages:
        print(f"    [{m['role']}] {m['content']}")

    print(f"\n  🔄 经过 chat_template 处理后 → 模型看到的实际 prompt:")
    print(f"  {'─'*60}")
    # 用缩进展示特殊 token
    fromatted = raw_prompt.replace("<|im_start|>", "\n    <|im_start|>")
    fromatted = fromatted.replace("<|im_end|>", "<|im_end|>\n    ")
    print(f"    {fromatted}")
    print(f"  {'─'*60}")

    # apply_chat_template(tokenize=True) 返回的是 list of list 或 dict
    # 当 return_tensors="pt" 时返回 BatchEncoding (dict-like)
    if hasattr(result, 'shape'):
        input_ids = result
    elif isinstance(result, dict):
        input_ids = result['input_ids']
    else:
        input_ids = result

    user_tokens = len(tokenizer.encode(messages[1]['content'], add_special_tokens=False))
    sys_tokens = len(tokenizer.encode(messages[0]['content'], add_special_tokens=False))

    # 获取形状
    if hasattr(input_ids, 'shape'):
        shape = input_ids.shape
        total = input_ids.shape[1]
    else:
        shape = (1, len(input_ids[0]))
        total = len(input_ids[0])

    print(f"\n  📊 Token 化后 stats:")
    print(f"    input_ids 形状: {shape}")
    print(f"    总 token 数: {total}")
    print(f"    其中用户输入部分: {user_tokens} tokens")
    print(f"    系统提示部分: {sys_tokens} tokens")
    print(f"    特殊 token 开销: {total - user_tokens - sys_tokens} tokens")

    print(f"\n  💡 chat_template 是理解 LLM API 的关键:")
    print(f"    你发给 API 的是 messages 数组 → chat_template 把它变成 prompt 字符串")
    print(f"    → tokenizer 把 prompt 变成 input_ids → model.generate() 生成 output_ids")
    print(f"    → tokenizer.decode() 把 output_ids 变成文本 → 返回给用户")


# ============================================================
# 第 5 部分: Llama tokenizer 对比（如果已下载）
# ============================================================
def part5_compare_llama(tokenizer_qwen, comparison_data):
    """
    对比 Qwen3 和 Llama 的 tokenizer 差异。

    核心差异:
    ┌───────────────────┬───────────────────┬───────────────────┐
    │                   │ Qwen3             │ Llama 3.2         │
    ├───────────────────┼───────────────────┼───────────────────┤
    │ 词表大小          │ 151,936           │ 128,256           │
    │ 分词算法          │ BPE (字节级)      │ BPE (字节级)      │
    │ 中文优化          │ ✅ 大量中文词条   │ ⚠️ 以英文为主      │
    │ 特殊 token 风格   │ <|im_start|> 系   │ <|begin_of_text|> 系│
    │ 中文 token 数     │ 较少 (中文友好)   │ 较多 (中文弱势)   │
    └───────────────────┴───────────────────┴───────────────────┘
    """
    print_section("第 5 部分: Llama Tokenizer 对比")

    # 尝试加载 Llama tokenizer
    llama_available = False
    try:
        tokenizer_llama = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
        llama_available = True
        print(f"\n  ✅ Llama tokenizer 加载成功")
    except Exception as e:
        print(f"\n  ⚠️  Llama tokenizer 加载失败: {type(e).__name__}")
        print(f"  原因: Llama 是门控模型，需要先在 huggingface.co 申请访问权限")
        print(f"  步骤: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print(f"        → 填写申请表单 → 生成 Access Token → huggingface-cli login")
        print(f"\n  下面用 Qwen3 自身做一次「词表密度」的演示分析\n")

    # ---- Qwen3 自身的中文 vs 英文 token 效率 ----
    print(f"\n  📊 Qwen3 tokenizer 中文 vs 英文 token 效率:")
    print(f"{'─'*65}")
    print(f"  {'文本':<30s} {'字符数':>6s} {'Token数':>6s} {'字符/Token':>10s}")
    print(f"  {'─'*65}")

    cn_texts = [t for t in TEST_TEXTS.items() if "中文" in t[0] or "混合" in t[0] or "数字" in t[0]]
    en_texts = [t for t in TEST_TEXTS.items() if "英文" in t[0] or "代码" in t[0]]

    cn_ratios = []
    en_ratios = []

    for label, text in cn_texts:
        toks = len(tokenizer_qwen.encode(text, add_special_tokens=False))
        ratio = len(text) / toks
        cn_ratios.append(ratio)
        print(f"  [{label}] {text[:25]:<25s} {len(text):>6d} {toks:>6d} {ratio:>9.1f}")

    for label, text in en_texts:
        toks = len(tokenizer_qwen.encode(text, add_special_tokens=False))
        ratio = len(text) / toks
        en_ratios.append(ratio)
        print(f"  [{label}] {text[:25]:<25s} {len(text):>6d} {toks:>6d} {ratio:>9.1f}")

    if cn_ratios and en_ratios:
        avg_cn = sum(cn_ratios) / len(cn_ratios)
        avg_en = sum(en_ratios) / len(en_ratios)
        print(f"  {'─'*65}")
        print(f"  中文平均字符/Token: {avg_cn:.1f}")
        print(f"  英文平均字符/Token: {avg_en:.1f}")
        print(f"  中文「token 效率」是英文的 {avg_cn/avg_en*100:.0f}%")
        print(f"  → 中文每个 token 覆盖的字符数少 ≈{(avg_en/avg_cn):.1f}倍")
        print(f"  → 表示相同信息量，中文需要 ≈{(avg_en/avg_cn):.1f}倍 的 token 数")
        print(f"  → 推理时间相应更长，API 成本更高")

    if llama_available:
        # Llama 的中文效率对比
        print(f"\n  📊 同一句中文: Qwen3 vs Llama 的 token 数对比")
        cn_long = "人工智能正在深刻地改变着各行各业的运作方式"
        qwen_toks = len(tokenizer_qwen.encode(cn_long, add_special_tokens=False))
        llama_toks = len(tokenizer_llama.encode(cn_long, add_special_tokens=False))
        print(f"    文本: {cn_long}")
        print(f"    Qwen3 token 数: {qwen_toks}")
        print(f"    Llama  token 数: {llama_toks}")
        print(f"    Llama 多用了 {(llama_toks/qwen_toks - 1)*100:.0f}% 的 token")
        print(f"    → 词表大且覆盖中文的模型，token 效率明显更高")
        print(f"    → 选型时 tokenizer 的中文能力直接影响推理成本和延迟")


# ============================================================
# 第 6 部分: encode → decode 往返实验
# ============================================================
def part6_roundtrip(tokenizer):
    """
    验证 tokenize 的信息完整性: 任意文本 → encode → decode → 是否完全一致？

    可能出现的偏差:
      1. 末尾空格丢失 (部分 tokenizer 的 normalize 行为)
      2. 特殊字符被标准化 (全角→半角)
      3. 超出词表的字符被替换为 UNK
    """
    print_section("第 6 部分: Encode → Decode 往返实验")

    test_cases = [
        "Hello, world!",
        "你好，世界！",
        "  ",
        "\n\n",
        "Hello 你好 123 !@#$%^&*()",
        "   leading and trailing spaces   ",
        "emoji: 😀 🎉 🔥",               # emoji 的 tokenize 行为
        "数学符号: ∑ ∫ √ ∞ ≈ ≠ ≤ ≥",     # 数学符号
        "<|im_start|>system",            # 如果用户输入中包含特殊 token 怎么办？
    ]

    print()
    for text in test_cases:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        match = "✅" if text == decoded else "⚠️ 不一致"
        print(f"  {match} 原始: {repr(text):40s} → tokens={len(ids):3d} → decode: {repr(decoded)}")

    print(f"\n  💡 注意事项:")
    print(f"    如果用户输入中包含特殊 token 字符串（如 '<|im_start|>'），")
    print(f"    tokenizer 可能把它当作真正的特殊 token 处理。")
    print(f"    生产环境中需要做输入净化（sanitization），防止 prompt injection。")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  任务 1.3: 分词器入门 — 深入理解 LLM 的「文字→数字」转换")
    print("=" * 70)

    # Part 1: 加载
    tokenizer = part1_load_tokenizer()

    # Part 2: Tokenize 中英文
    comparison_data = part2_tokenize_texts(tokenizer)

    # Part 3: input_ids / attention_mask
    part3_encoding_details(tokenizer)

    # Part 4: chat_template
    part4_chat_template(tokenizer)

    # Part 5: Llama 对比（或 Qwen3 自身的效率分析）
    part5_compare_llama(tokenizer, comparison_data)

    # Part 6: 往返实验
    part6_roundtrip(tokenizer)

    print(f"\n{'='*70}")
    print(f"  ✅ 任务 1.3 完成！")
    print(f"  下一步 → 任务 1.4: python explore_generate.py (模型推理跑通)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
