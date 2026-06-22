"""
任务 1.2: 模型下载与文件结构分析
=================================
下载 Qwen3-0.6B-Instruct，遍历模型文件目录，理解每个文件的作用，
并与 Llama-3.2-1B-Instruct 的 config.json 做对比（Llama 的门控授权需提前申请）。

用法:
    python download_and_explore.py              # 下载 Qwen3 + 文件结构分析
    python download_and_explore.py --download-only  # 只下载（不打印结构）
    python download_and_explore.py --no-download    # 只分析已下载的模型（不重新下载）

Qwen3 模型约 1.2GB（FP32），下载时间取决于网络。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --- 关键环境变量：必须在调用任何 huggingface_hub API 前设置 ---
# 如果 ~/.bashrc 里写了 export HF_ENDPOINT=https://hf-mirror.com，
# 这个检查会通过。如果没通过——检查你的 bashrc 配置。
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
if not HF_ENDPOINT:
    print("⚠️ 警告: HF_ENDPOINT 未设置！")
    print("  国内下载可能极慢或失败。请先执行:")
    print("  export HF_ENDPOINT=https://hf-mirror.com")
    print()
    print("  继续运行中（会尝试直连 huggingface.co）...")
    print()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, list_repo_files


# ============================================================
# 模型配置
# ============================================================
QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"  # Qwen3 不带后缀即 Instruct，"Qwen/Qwen3-0.6B-Base" 是基座
# Llama 3.2 需要 HF 账号授权，暂不下载
# LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# HF 缓存目录——所有下载的模型都放在这里
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def download_model(model_id: str) -> Path:
    """
    下载模型到本地 HF 缓存。

    snapshot_download() 的工作流程:
      1. 连接 HF Hub（走 HF_ENDPOINT 镜像）
      2. 下载不存在的文件，跳过已存在的（断点续传）
      3. 验证 sha256 校验和
      4. 返回模型在缓存中的目录路径

    返回: 模型缓存目录路径
    """
    print(f"\n{'='*60}")
    print(f"开始下载: {model_id}")
    print(f"缓存目录: {HF_CACHE}")
    print(f"{'='*60}")

    # snapshot_download 下载模型仓库的所有文件到本地缓存
    # resume_download=True: 断点续传，中断后重跑不会重新下载
    local_dir = snapshot_download(
        repo_id=model_id,
        resume_download=True,
        # ignore_patterns: 跳过不需要的大文件，节省磁盘和时间
        # *.bin 是旧格式的 PyTorch 权重（新版 HF 用 *.safetensors）
        # *.msgpack 是旧格式 tokenizer 文件
        ignore_patterns=["*.bin", "*.msgpack"],
    )

    print(f"\n✅ 下载完成！模型路径: {local_dir}")
    return Path(local_dir)


def explore_model_dir(model_dir: Path, model_name: str):
    """
    遍历并分析模型文件目录结构。

    一个 HF 模型目录通常包含以下几类文件:
    ┌─────────────────────┬──────────────────────────────────────────┐
    │ 文件                │ 作用                                      │
    ├─────────────────────┼──────────────────────────────────────────┤
    │ config.json         │ 模型架构配置（层数、隐层维度、注意力头数等）│
    │ tokenizer.json      │ 分词器词表（子词 → token id 的映射表）     │
    │ tokenizer_config.json│ tokenizer 的加载配置（chat_template 等）  │
    │ model.safetensors   │ 模型权重文件（新版 SafeTensors 格式）      │
    │ *.safetensors.index.json │ 权重分片索引（大模型会分片存储）  │
    │ generation_config.json│ 默认生成参数（temperature/max_length）  │
    │ vocab.json / merges.txt│ BPE 词表原始文件（部分 tokenizer 用）  │
    └─────────────────────┴──────────────────────────────────────────┘
    """
    print(f"\n{'='*60}")
    print(f"📂 {model_name} 文件结构分析")
    print(f"   路径: {model_dir}")
    print(f"{'='*60}")

    # 收集所有文件及其大小
    files_info = []
    for file_path in sorted(model_dir.rglob("*")):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            files_info.append((file_path, size_mb))

    if not files_info:
        print("❌ 模型目录为空！下载可能失败。")
        return

    # 按大小排序，最大文件排前面
    files_info.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'文件名':<45s} {'大小':>10s}   作用")
    print("-" * 90)

    FILE_DESCRIPTIONS = {
        "config.json": "模型架构超参数（hidden_size, num_layers, vocab_size...）",
        "model.safetensors": "⭐ 模型权重——所有神经元参数都在这里",
        "tokenizer.json": "⭐ 词表——子词和 token id 的完整映射表",
        "tokenizer_config.json": "tokenizer 加载配置 + chat_template 模板",
        "generation_config.json": "默认生成参数（temperature, top_p, max_new_tokens）",
        "vocab.json": "BPE 词表（原始格式，部分 tokenizer 用）",
        "merges.txt": "BPE 合并规则（字节对的合并优先级列表）",
        "special_tokens_map.json": "特殊 token 的名称映射",
        "chat_template.json": "对话模板（messages 格式 → 模型看到的 prompt 字符串）",
    }

    total_size = 0
    for file_path, size_mb in files_info:
        name = file_path.name
        desc = FILE_DESCRIPTIONS.get(name, "")
        symbol = "⭐" if name in ("model.safetensors", "tokenizer.json", "config.json") else "  "
        print(f"{symbol} {name:<43s} {size_mb:7.1f} MB   {desc}")
        total_size += size_mb

    print("-" * 90)
    print(f"{'合计':>55s} {total_size:7.1f} MB")
    print(f"文件数: {len(files_info)}")


def read_config(config_path: Path) -> dict:
    """读取并解析 config.json"""
    with open(config_path) as f:
        return json.load(f)


def analyze_config(model_dir: Path, model_name: str):
    """
    深入分析 config.json——这是理解模型结构的核心入口。

    config.json 中的每个字段都是模型的一个超参数，决定了模型的结构和大小。
    理解这些字段后，面试时能回答「这个模型有多大、多少层、
    KV Cache 占多少内存」等问题。
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"❌ 找不到 {config_path}")
        return {}

    config = read_config(config_path)

    print(f"\n{'='*60}")
    print(f"🔍 {model_name} config.json 详细分析")
    print(f"{'='*60}")

    # ---------- 模型尺寸参数 ----------
    # 这些参数直接决定了模型权重的大小和推理内存占用
    print(f"\n── 模型尺寸参数 ──")
    hidden_size = config.get("hidden_size", "?")
    num_layers = config.get("num_hidden_layers", "?")
    intermediate_size = config.get("intermediate_size", "?")
    vocab_size = config.get("vocab_size", "?")
    num_attention_heads = config.get("num_attention_heads", "?")
    num_kv_heads = config.get("num_key_value_heads", num_attention_heads)

    print(f"  hidden_size (隐层维度):          {hidden_size}")
    print(f"     → 每个 token 在模型内部用 {hidden_size} 维的向量表示")
    print(f"  num_hidden_layers (Transformer 层数): {num_layers}")
    print(f"     → 模型深度——越多层捕捉越抽象的特征")
    print(f"  intermediate_size (FFN 中间维度): {intermediate_size}")
    print(f"     → 前馈网络的扩张维度，通常是 hidden_size 的倍数")
    print(f"  vocab_size (词表大小):            {vocab_size}")
    print(f"     → 模型「认识」多少个不同的 token")
    print(f"  num_attention_heads (注意力头数):  {num_attention_heads}")
    print(f"  num_key_value_heads (KV 头数):     {num_kv_heads}")
    if num_attention_heads != num_kv_heads:
        print(f"     → GQA（分组查询注意力）: {num_attention_heads} 个 Q 头共享 {num_kv_heads} 个 KV 头")

    # ---------- 参数数量快速估算 ----------
    # 这是面试高频问题：Qwen3-0.6B 为什么叫 0.6B？
    # 0.6B = 6 亿参数，实际上 Qwen3 用的是更精确的命名
    print(f"\n── 参数量快速估算 ──")
    # 粗略估算公式（忽略 embedding 和 lm_head 的共享权重）:
    # 每层 Transformer ≈ 4 * hidden_size^2 + 2 * hidden_size * intermediate_size
    per_layer = 4 * hidden_size * hidden_size + 2 * hidden_size * intermediate_size
    total_transformer = per_layer * num_layers
    # embedding 层: vocab_size * hidden_size
    embedding_params = vocab_size * hidden_size
    estimated = (total_transformer + embedding_params) / 1e9
    print(f"  Transformer 层参数量:    ~{total_transformer/1e6:.0f}M")
    print(f"  Embedding 层参数量:      ~{embedding_params/1e6:.0f}M")
    print(f"  估算总参数量:            ~{estimated:.2f}B")
    print(f"  Qwen3 标称参数量:        0.6B")
    print(f"  (实际参数量会因共享权重、LayerNorm 参数等有 ±5% 偏差)")

    # ---------- 推理相关配置 ----------
    print(f"\n── 推理相关配置 ──")
    print(f"  max_position_embeddings (最大上下文长度): {config.get('max_position_embeddings', '?')}")
    print(f"  rope_theta (RoPE 频率基数):              {config.get('rope_theta', '?')}")
    print(f"     → θ 值越大，对长位置编码越友好（如 Qwen3 用 1M 的 θ）")
    print(f"  rope_scaling:                            {config.get('rope_scaling', 'None')}")
    print(f"  torch_dtype (推荐推理精度):              {config.get('torch_dtype', 'float32')}")
    print(f"  tie_word_embeddings (输入输出共享权重):  {config.get('tie_word_embeddings', False)}")
    print(f"     → True=embedding 和 lm_head 共享权重矩阵，节省约 vocab_size*hidden_size 参数")

    # ---------- 归一化与激活函数 ----------
    print(f"\n── 归一化与激活函数 ──")
    print(f"  hidden_act (FFN 激活函数):      {config.get('hidden_act', '?')}")
    print(f"     → silu/gelu: 现代 Transformer 的标配激活函数")
    print(f"  rms_norm_eps (RMS Norm 精度):   {config.get('rms_norm_eps', '?')}")
    print(f"  attention_bias (注意力偏置):     {config.get('attention_bias', '?')}")

    return config


def compare_configs(qwen_config: dict, model_dir: Path):
    """
    演示如何对比两个模型的 config。

    如果你已经申请到 Llama 的访问权限，运行:
        python download_and_explore.py --compare-llama

    对比的核心维度:
    ┌──────────────────┬─────────────────┬────────────────┐
    │ 维度              │ Qwen3-0.6B      │ Llama-3.2-1B   │
    ├──────────────────┼─────────────────┼────────────────┤
    │ 词表              │ 中文友好 BPE     │ 英文为主 BPE    │
    │ 注意力机制        │ GQA (含 KV 压缩) │ GQA            │
    │ 最大上下文长度    │ 32768 (32K)     │ 131072 (128K)  │
    │ 归一化            │ RMS Norm        │ RMS Norm       │
    │ 激活函数          │ SiLU            │ SiLU           │
    └──────────────────┴─────────────────┴────────────────┘
    """
    # Llama 的 config 只有在下载后才能读取，这里只展示对比框架
    print(f"\n{'='*60}")
    print(f"🔍 两模型 config 对比框架（Llama 需先授权下载）")
    print(f"{'='*60}")

    print(f"""
  对比维度                     Qwen3-0.6B            Llama-3.2-1B
  ─────────────────────────────────────────────────────────────────
  hidden_size                  {qwen_config.get('hidden_size', '?'):>5}                2048
  num_hidden_layers            {qwen_config.get('num_hidden_layers', '?'):>5}                  16
  vocab_size                   {qwen_config.get('vocab_size', '?'):>6}             128256
  max_position_embeddings      {qwen_config.get('max_position_embeddings', '?'):>5}              131072
  注意力机制                   GQA (Q:{qwen_config.get('num_attention_heads','?')}, KV:{qwen_config.get('num_key_value_heads','?')})
                                        GQA (Q:12, KV:4)
  RoPE theta                   {qwen_config.get('rope_theta', '?'):>10}       500000
  tie_word_embeddings          {str(qwen_config.get('tie_word_embeddings', '?')):>5}              True
  """)


def download_llama_info():
    """打印 Llama 下载步骤引导"""
    print(f"""
{'='*60}
🔐 Llama-3.2-1B-Instruct 下载前置步骤
{'='*60}

Llama 是 Meta 的门控模型（gated model），需要以下步骤:

  1. 访问 https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  2. 点击 "Expand to review and accept" 填写申请表单
  3. 通常 1-5 分钟自动通过（会发邮件确认）
  4. 去 https://huggingface.co/settings/tokens 生成 Access Token
  5. 在本机执行登录:
       huggingface-cli login
     (粘贴你的 Access Token，选 "Yes" 保存到 ~/.cache/huggingface/token)

  6. 登录后运行:
       python download_and_explore.py --download-llama

  提示: Qwen3 的实验不受影响——先跑完 Qwen3 的 tokenizer/generate/benchmark，
  Llama 授权通过后一句命令就能加载做对比实验。
""")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="下载 HF 模型并分析文件结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_and_explore.py                  # 下载 Qwen3 + 完整分析
  python download_and_explore.py --download-only   # 只下载 Qwen3
  python download_and_explore.py --no-download     # 只分析已下载的 Qwen3
  python download_and_explore.py --download-llama  # 下载 Llama（需先 huggingface-cli login）
        """,
    )
    parser.add_argument("--download-only", action="store_true", help="只下载不分析")
    parser.add_argument("--no-download", action="store_true", help="只分析不下载")
    parser.add_argument("--download-llama", action="store_true", help="下载 Llama-3.2-1B-Instruct（需授权）")
    args = parser.parse_args()

    # ---- 步骤 1: 下载 Qwen3 ----
    if not args.no_download:
        qwen_dir = download_model(QWEN_MODEL_ID)
    else:
        # 从缓存中找到已下载的 Qwen3
        import glob
        pattern = str(HF_CACHE / "models--Qwen--Qwen3-0.6B-Instruct" / "snapshots" / "*")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print("❌ 未找到已下载的 Qwen3 模型！请先运行不带 --no-download 的版本。")
            sys.exit(1)
        qwen_dir = Path(matches[-1])  # 取最新的 snapshot
        print(f"📂 使用已缓存的模型: {qwen_dir}")

    if args.download_only:
        print("\n✅ 下载完成！运行 python download_and_explore.py --no-download 查看文件结构分析。")
        return

    # ---- 步骤 2: 文件结构遍历 ----
    explore_model_dir(qwen_dir, "Qwen3-0.6B-Instruct")

    # ---- 步骤 3: config.json 深入分析 ----
    qwen_config = analyze_config(qwen_dir, "Qwen3-0.6B-Instruct")

    # ---- 步骤 4: 两模型 config 对比框架 ----
    compare_configs(qwen_config, qwen_dir)

    # ---- Llama 下载引导 ----
    download_llama_info()

    # ---- 下一步提示 ----
    print(f"""
{'='*60}
✅ 任务 1.2 完成！下一步:
{'='*60}
  → 任务 1.3 分词器入门: python explore_tokenizer.py
  (理解 Qwen3 和 Llama 的 tokenizer 差异——中文 token 数的物理原因)
""")


if __name__ == "__main__":
    main()
