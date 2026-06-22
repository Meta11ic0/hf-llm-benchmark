# HF LLM Benchmark — Step 1：HuggingFace Transformers LLM 推理基础

> 来源：[AI Infra 项目实战计划](https://github.com/Meta11ic0/hf-llm-benchmark) Step 1
>
> 定位：用最小成本获得 LLM 工程的底层手感——这是后续 FastAPI 服务化、RAG、vLLM 的共同地基。

---

## 新机器环境准备（3 步跑起来）

```bash
# 1. 克隆 + 虚拟环境
git clone https://github.com/Meta11ic0/hf-llm-benchmark.git
cd hf-llm-benchmark
python3 -m venv venv
source venv/bin/activate

# 2. 配置国内镜像（国内网络必需！）
export HF_ENDPOINT=https://hf-mirror.com
# 建议把这一行加到 ~/.bashrc 或 ~/.zshrc

# 3. 安装依赖（先装 torch CPU 版，再装其他）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. 验证
python3 -c "import torch; print('CPU only:', not torch.cuda.is_available())"
```

> **注意**：模型文件（~1.5GB）首次运行时会自动下载到 `~/.cache/huggingface/`。
> 如果已经有模型缓存，可以直接拷贝到新机器的相同路径。

---

## 目标

用 HuggingFace Transformers 加载 Qwen3-0.6B 和 Llama-3.2-1B 两个大语言模型，深入理解分词器、生成流程、参数影响、量化效果，编写 benchmark 脚本测量推理性能。

---

## 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| 主模型 | Qwen3-0.6B-Instruct（~1.2GB） | 2026 主流开源，Apache 2.0 协议，CPU 友好 |
| 对比模型 | Llama-3.2-1B-Instruct（~2.5GB） | Meta 开源，不同 tokenizer 体系（BPE vs 中文优化），展示多样性 |
| 框架 | HuggingFace Transformers | 行业标准 |
| 精度 | FP32 + 8-bit（BitsAndBytesConfig） | CPU 推理默认 + 量化入门 |
| 模型下载 | HF Mirror (`hf-mirror.com`) | 国内网络必需 |
| 语言 | Python 3.10+ | 岗位要求 |

---

## 详细任务清单

| # | 任务 | 内容 | 预计 |
|---|------|------|------|
| 1.1 | 环境准备 | 创建项目，git init，虚拟环境，配置 HF Mirror，安装依赖。**注意用 CPU 版 PyTorch** | 0.5h |
| 1.2 | 模型下载与文件结构 | 下载 Qwen3-0.6B-Instruct → 理解 `config.json` / `tokenizer.json` / `model.safetensors` / `*.safetensors.index.json`。对比两模型 config 差异 | 1.5h |
| 1.3 | 分词器入门 | Qwen3 tokenizer → tokenize 中英文示例 → 理解 input_ids / attention_mask / 特殊 token → **对比 Llama tokenizer**（BPE 体系差异、同一句中文的 token 数差异） | 2h |
| 1.4 | 单次推理跑通 | 加载 Qwen3 → `tokenizer.apply_chat_template()` → `model.generate()` → decode，打通完整推理流程。再切换到 Llama 跑同一条 prompt | 1h |
| 1.5 | 生成参数实验 | Qwen3 上系统对比：① temperature(0.1/0.7/1.5) ② top_p(0.5/0.9/1.0) ③ max_new_tokens(32/128/512) ④ do_sample(True/False)。记录每组输出质量和 tokens/s | 3h |
| 1.6 | 8-bit 量化实验 | 用 `BitsAndBytesConfig(load_in_8bit=True)` 加载 Qwen3 → 对比 FP32 vs INT8：模型加载内存、推理速度、输出质量 → 理解用精度换显存/内存 | 1.5h |
| 1.7 | Benchmark 脚本 | 准备 10-15 条不同长度 prompt（短问答/中篇总结/长文分析）→ 预热 10 次 → 每条 10 次取统计 → 两模型分别跑 → 输出 avg/P50/P99 延迟 + tokens/s + 生成 token 数 | 3h 编码 + ~5h wall clock |
| 1.8 | 结果整理 + 提交 | README（项目目的+环境+镜像配置+核心结果）+ 两模型对比表 + 量化对比表 | 1h |

---

## 产出物

```
hf-llm-benchmark/
├── README.md                      # 项目说明 + benchmark 结果
├── requirements.txt
├── benchmark.py                   # 核心 benchmark（支持 --model 参数）
├── explore_tokenizer.py           # 分词器实验（两模型对比）
├── explore_params.py              # 生成参数实验
├── explore_quantization.py        # 8-bit 量化对比
└── results/
    └── report.md
```

---

## 验收标准

- [ ] `python benchmark.py --model qwen` 一键运行
- [ ] 理解 tokenizer 的 chat_template 和特殊 token
- [ ] 能解释 temperature / top_p 的影响
- [ ] 能说出 Qwen3 和 Llama tokenizer 的一个具体差异（如中文切分方式）
- [ ] 量化实验：INT8 内存降幅有具体数字
- [ ] 报告包含两个模型的 avg/P50/P99 延迟和 tokens/s

---

## 执行提示

**CPU benchmark 太慢怎么办。** Qwen3-0.6B 在普通笔记本 CPU 上约 2-5 tokens/s，生成 512 token 需要 1-3 分钟。完整 benchmark 约 300 次推理 ≈ 5 小时 wall clock。建议首次降采样：10 prompt × 5 次 = 50 次，先出数据，完整版挂一晚上跑。

**tokenizer 对比抓住一个点。** Qwen3 和 Llama tokenizer 对中文的处理差异最容易观察——同一句中文，Llama 的 token 数通常是 Qwen3 的 1.5-2 倍（因为 Llama 的 BPE 词表以英文为主）。把这个对比数字写进报告，面试时一句话就能展示"我理解 tokenizer 对推理成本和速度的影响"。

**量化实验注意。** `BitsAndBytesConfig(load_in_8bit=True)` 在纯 CPU 环境下可能需要额外配置（默认依赖 CUDA）。如果遇到报错，备选方案：① 改 `torch_dtype=torch.float16` 加载 ② 不改代码，用模型权重文件大小推算理论内存差异（FP32 ≈ 参数数 × 4 字节，INT8 ≈ 参数数 × 1 字节）。

