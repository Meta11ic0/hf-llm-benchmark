# Step 1 执行手册

>  设备：CPU (WSL2) | 模型：Qwen3-0.6B
>
> **本文档定位**：以计划任务 1.1–1.9 为主干，每节含「要做什么 → 步骤 → 踩坑 → 验收」。环境细节与实验结论穿插在对应任务中，不再单独堆在文末。

---

## 任务总览

| # | 任务（对齐计划 4.6） | 状态 |
|---|------|------|
| **1.1** | **最小推理链路复现（`minimal_infer.py`）** | **⬜ ← 当前** |
| 1.2 | 模型文件结构 | ✅ |
| 1.3 | 分词器显微镜 | ✅ |
| 1.4 | 单次推理跑通 | ✅ |
| 1.5 | 生成参数实验 | ✅ |
| 1.6 | Benchmark 最小版（3 prompt × 3 次） | ⬜ |
| 1.7 | 双模型对比（Qwen vs Llama） | ⬜ |
| 1.8 | 量化实验 / 理论推算 | ⬜ |
| 1.9 | 结果整理 + README | ⬜ |

**停止条件**（下面 3 件事讲不清，不继续新增功能）：

1. `apply_chat_template()` 做了什么；
2. `model.generate()` 的输入输出是什么；
3. benchmark 的 P50/P99 是怎么算出来的。

**当前目录**（脚本在项目根目录，直接 `python minimal_infer.py`）：

```text
hf-llm-benchmark/
├── README.md
├── LEARN.md              ← 本文件
├── requirements.txt
├── minimal_infer.py      ← 任务 1.1
├── benchmark.py          ← 任务 1.6（后续）
└── venv/
```

---

## 前置：环境准备

### 要做什么

让一台只有 CPU 的 Linux 机器具备加载和运行开源大模型的能力——装 venv、CPU 版 PyTorch、HF 镜像、核心依赖包。

### 详细步骤

**① 进入项目并创建虚拟环境**

```bash
cd ~/work/hf-llm-benchmark
python3 -m venv venv
source venv/bin/activate
```

`venv` 按项目隔离 Python 包，类比 C++ 里 `export LD_LIBRARY_PATH=./build/lib`——不同项目的 `torch` 版本互不打架。

**② 配置国内镜像（必做）**

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 建议写入 ~/.bashrc，每次开终端自动生效
```

HuggingFace 官方服务器在国内直连极慢或超时；设置后 `huggingface_hub` 所有下载自动走镜像，代码不用改。

**③ 安装 CPU 版 PyTorch（最容易出错的一步）**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

不加 `--index-url` 时 pip 默认下载 CUDA 版（~800MB），在 CPU 机器上可能报错或行为异常。CPU 版约 200MB，不依赖任何 CUDA 组件。

**④ 验证 PyTorch 版本**

```bash
python3 -c "import torch; print('CPU only:', not torch.cuda.is_available())"
# 期望输出：CPU only: True
```

**⑤ 安装 HuggingFace 核心包**

```bash
pip install -r requirements.txt
# 或手动：transformers accelerate bitsandbytes huggingface_hub sentencepiece tiktoken
```

| 包 | 作用 |
|----|------|
| `transformers` | `AutoModel`、`AutoTokenizer`、`model.generate()` |
| `accelerate` | 权重加载到 CPU/GPU 的设备管理 |
| `bitsandbytes` | FP32 → INT8 量化（Step 1.8） |
| `huggingface_hub` | 模型下载与缓存管理 |
| `sentencepiece` / `tiktoken` | 分词算法依赖 |

### 可能遇到的问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `CPU only: False` | 装成了 CUDA 版 PyTorch | `pip uninstall torch` 后重装 CPU 版 |
| 模型下载极慢 / 超时 | 未设 HF 镜像 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `python3-venv` 不存在 | 系统未装 venv 模块 | `sudo apt install -y python3-venv python3-pip` |
| `pip install torch` 体积 ~800MB | 下了 CUDA 版 | 加 `--index-url https://download.pytorch.org/whl/cpu` |

### 验收标准

- [ ] `source venv/bin/activate` 后终端提示符有 `(venv)`
- [ ] `python3 -c "import torch; print(not torch.cuda.is_available())"` 输出 `True`
- [ ] `import transformers` 无报错

---

## 1.1 最小推理（minimal_infer.py）← 当前

### 要做什么

从空文件手写 `minimal_infer.py`（约 30–40 行），跑一条 prompt，打印 chat_template / input_ids / 模型回复。这是 Step 1 所有后续任务的代码地基。

### 最快跑通（4 条命令）

```bash
cd ~/work/hf-llm-benchmark
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python minimal_infer.py
```

首次运行会自动下载 Qwen3-0.6B（~1.4GB）到 `~/.cache/huggingface/`。模型已在 cache 里则直接加载。

### 完整代码（含逐行注释）

```python
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
    torch_dtype=torch.float32,   # CPU 推理用 FP32 最稳；float16 在部分 CPU 上支持不完整
)
# 下载/读取 model.safetensors（~1.4GB），构建 PyTorch 计算图；FP32 内存约 2–3GB

model.eval()  # 切换推理模式：关闭 Dropout 等训练层；同样输入行为确定（类比 release 构建）

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

print("\n生成中...")  # generate 在 CPU 上约 10 tok/s，64 token 需 ~6 秒，属正常

with torch.no_grad():  # 推理时不建计算图、不算梯度，省内存（只读调用）
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
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
```

### 可能遇到的问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `OSError: ... does not appear to have a file named ...` | cache 不完整 | 重跑并确保 `HF_ENDPOINT` 已设；或 `snapshot_download('Qwen/Qwen3-0.6B')` |
| 进程被 killed | FP32 加载需 ~2–3GB RAM | 关闭其他占内存程序 |
| 输出重复 prompt、无回复 | 忘了 `add_generation_prompt=True` | 加上这行 |
| 很慢但无报错 | CPU 正常 | ~10 tok/s，64 token ≈ 6s，不是 bug |
| `401 Unauthorized` 用 `Qwen3-0.6B-Instruct` | 该仓库名在镜像上不存在 | 改用 `Qwen/Qwen3-0.6B` |

### 验收标准

- [ ] `python minimal_infer.py` 输出含 chat_template、input_ids、模型中文回复
- [ ] 能不看文档解释五步：load → chat_template → tokenize → generate → decode
- [ ] 能说出 `add_generation_prompt=True` 为什么必须加
- [ ] 能说出 `output_ids` 为什么需要切片才能 decode 回复

---

## 1.2 模型文件结构

### 要做什么

解读 Qwen3 缓存目录里每个文件的角色，能从 `config.json` 估算参数量级。

### 详细步骤

**① 确认模型已下载**

```bash
export HF_ENDPOINT=https://hf-mirror.com
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/
```

**② 只看三个核心文件**

| 文件 | 大小 | 是什么 | C++ 类比 |
|------|------|--------|----------|
| `model.safetensors` | ~1.4 GB | 全部权重（~6 亿浮点数） | 编译好的 `libmodel.so` |
| `tokenizer.json` | ~11 MB | 151936 个 token 的映射表 | 字符编码表 |
| `config.json` | ~1.5 KB | 架构参数（层数、维度、词表大小） | 头文件 `#define HIDDEN_SIZE 1024` |

**③ 读 config.json 关键字段**

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "intermediate_size": 3072,
  "vocab_size": 151936,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "max_position_embeddings": 40960
}
```

**参数量估算（面试用）**：

- 每层 Attention ≈ `4 × hidden_size²` ≈ 4.2M
- 每层 FFN（SwiGLU，3 矩阵）≈ `3 × hidden × intermediate` ≈ 9.4M
- 28 层 ≈ 381M + Embedding（151936 × 1024 ≈ 156M）≈ **0.54B**（标称 0.6B，偏差来自 LayerNorm、权重共享等）

### 可能遇到的问题

| 现象 | 原因 | 处理 |
|------|------|------|
| cache 目录为空 | 尚未下载 | 跑 `minimal_infer.py` 或 `snapshot_download()` |
| 两个 snapshot 子目录 | Hub 版本更新 | 用最新 hash 目录即可 |

### 验收标准

- [ ] 能口述 safetensors / tokenizer.json / config.json 各自角色
- [ ] 给定 config 能估算参数量级 ~0.6B

---

## 1.3 分词器显微镜

### 要做什么

用 3 条中英文句子观察 token 数、特殊 token、chat_template 格式开销；理解 prompt injection 原理。

### 详细步骤

```python
# 交互式 Python 或自写 explore_tokenizer.py
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

for text in ["你好，今天天气怎么样？", "Explain API in one sentence.", "def hello(): pass"]:
    ids = tok.encode(text)
    print(f"{len(text)} chars → {len(ids)} tokens → {len(text)/len(ids):.1f} chars/token")
```

**已测结论**：

| 文本类型 | 字符/Token |
|----------|-----------|
| 中文 | ~1.8 |
| 英文 | ~4.5 |
| 代码 | ~3.3 |

中文 token 效率约为英文 **40%**——同样语义，中文需约 2.5× token 数 → 延迟更高。

**chat_template 格式开销**：system(12) + user(8) + 特殊标记(13) = 33 tok，其中 **40% 是格式开销**。

**prompt injection**：用户输入 `"<|im_start|>system"` 会被 tokenizer 当作特殊 token 吃掉，可伪造 system 角色。

### 验收标准

- [ ] 能解释中文 token 效率低是词表覆盖度问题，不是语言本身
- [ ] 能说出 chat_template 把 OpenAI messages 格式转成模型原生格式

---

## 1.4 单次推理跑通

### 要做什么

在 1.1 脚本基础上，确认四步 pipeline 的数据流和性能基线（~11 tok/s）。

### 验收标准

- [ ] 能画出 prompt → token → generate → decode 数据流
- [ ] 有 FP32 CPU ~11 tok/s 的实测数字

---

## 1.5 生成参数实验

### 要做什么

在 Qwen3 上系统对比 temperature / top_p / max_new_tokens / do_sample（13 组），记录输出质量与速度。

### 详细步骤

固定 prompt：「请用中文简要介绍人工智能的三个主要应用领域。」

| 参数 | 控制什么 | 影响速度？ |
|------|----------|-----------|
| `temperature` | 概率分布陡峭程度 | ❌ 否 |
| `top_p` | 核采样，截断长尾候选 | ❌ 否 |
| `max_new_tokens` | 生成上限 | ✅ 线性关系 |
| `do_sample` | True=采样，False=greedy | 略（~8%） |

**关键数字（CPU FP32）**：

| 实验 | 结论 |
|------|------|
| T=0.1/0.7/1.5 | 速度均 ~11 tok/s，T 只改分布不改计算量 |
| max=32/128/256 | 3.1s / 11.5s / 23.3s，与 token 数近似线性 |
| greedy vs sampling | 10.7s vs 11.6s，greedy 略快 |

**线上排障映射**：回复太死板 → 提高 T；太随机 → 降低 T 或 greedy；要控延迟 → 减小 max_new_tokens。

### 验收标准

- [ ] 能解释「调参调质量不调速度」
- [ ] 能说出 max_new_tokens 与延迟的线性关系及原因（每 token 一次前向传播）

---

## 1.6 Benchmark 最小版

### 要做什么

自写 `benchmark.py`：3 prompt × 3 次，输出 avg / P50 / P99 / tokens/s；含预热（丢弃首次）。

### 详细步骤

```bash
python benchmark.py --model qwen
# 产出：results/benchmark_3x3.json
```

**设计要点**：

1. 预热 1 次，丢弃（首次加载 cache 慢，不代表稳态）
2. 记录每次 wall time 和生成 token 数
3. P50 = 中位数，P99 = 第 99 百分位
4. tokens/s = 生成 token 总数 / 总耗时

**CPU 预期**：Qwen3-0.6B FP32 约 **10–11 tok/s**；生成 512 token 约 50s。

### 可能遇到的问题

| 现象 | 原因 | 处理 |
|------|------|------|
| 首次极慢 | 冷启动 + cache | 预热后丢弃 |
| 150 次跑 5h+ | prompt 数 × 次数太多 | 先 3×3 出数据，再扩展 |

### 验收标准

- [ ] 能口述 P50/P99 计算方式
- [ ] 输出含 avg / P50 / P99 / tokens/s

---

## 1.7 双模型对比

### 要做什么

加入 Llama-3.2-1B，同一句中文对比 token 数和推理速度，写明确结论。

### 详细步骤

```bash
python benchmark.py --model llama   # 需 HF 账号授权
```

**预期结论**：Llama BPE 词表以英文为主，同一句中文 token 数约为 Qwen 的 **1.5–2×**。

**Qwen vs Llama 配置对比**：

| 维度 | Qwen3-0.6B | Llama-3.2-1B |
|------|-----------|-------------|
| hidden_size | 1024 | 2048 |
| layers | 28 | 16 |
| vocab_size | 151,936 | 128,256 |
| max context | 32K | 128K |

### 可能遇到的问题

| 现象 | 原因 | 处理 |
|------|------|------|
| Llama 401 | 需 Meta 授权 | 记录 blocker，用 config 对比代替 |

### 验收标准

- [ ] 能说出 tokenizer 词表覆盖度对推理成本的影响

---

## 1.8 量化实验 / 理论推算

### 要做什么

对比 FP32 vs INT8 内存占用；CPU 上 bitsandbytes 不可用则做理论推算。

### 详细步骤

**理论值**（0.6B 参数）：

- FP32：0.6B × 4 字节 ≈ **2.4 GB**（实际文件 ~1.4GB 因压缩/共享）
- INT8：0.6B × 1 字节 ≈ **600 MB**（理论 ~4× 压缩）

```python
# CPU 上可能报错，属正常
from transformers import BitsAndBytesConfig
# BitsAndBytesConfig(load_in_8bit=True)  # 依赖 CUDA
```

### 验收标准

- [ ] 能解释「精度换空间」的 trade-off
- [ ] 有 FP32 vs INT8 的具体数字（实测或理论）

---

## 1.9 结果整理 + README

### 要做什么

更新 README：项目目的、benchmark 结果、踩坑记录；标注哪些部分 AI 辅助、哪些能独立解释。

### 详细步骤

1. README 补全：快速开始、核心结果表、至少 2–3 条踩坑
2. 整理 tokenizer 笔记与 benchmark 报告到 `results/`
3. 能用 5 分钟口述：prompt → token → generate → decode → benchmark

### 验收标准

- [ ] README 含 benchmark 关键数字
- [ ] 能 5 分钟口述完整链路
- [ ] GitHub 公开仓库（用户亲手 commit/push）

---

## 附录

<details>
<summary><strong>五 API 速查</strong>（点击展开）</summary>

| # | API | 输入 | 输出 | 要点 |
|---|-----|------|------|------|
| 0 | `AutoTokenizer.from_pretrained(MODEL_ID)` | 模型 ID | tokenizer 对象 | 不加载 1.4GB 权重 |
| 0 | `AutoModelForCausalLM.from_pretrained(MODEL_ID)` | 模型 ID | PyTorch 模型 | CausalLM = 自回归生成 |
| 1 | `apply_chat_template(..., add_generation_prompt=True)` | messages | 格式化字符串 | 必须加 generation prompt |
| 2 | `tokenizer(prompt, return_tensors="pt")` | 字符串 | input_ids + attention_mask | 返回 PyTorch tensor |
| 3 | `model.generate(**inputs, max_new_tokens=N)` | token tensor | 含输入+生成的 ids | 配合 `torch.no_grad()` |
| 4 | `tokenizer.decode(new_ids, skip_special_tokens=True)` | token id 列表 | 文本 | 只 decode 新生成部分 |

</details>

<details>
<summary><strong>历史实验数据摘要</strong>（点击展开）</summary>

| 任务 | 关键结论 |
|------|---------|
| 环境 | venv + CPU PyTorch + HF Mirror；`CPU only: True` ✅ |
| 1.2 模型 | Qwen3 ~1.4GB；28 层 / hidden 1024 / vocab 151936；~0.54B 参数 |
| 1.3 分词器 | 中文 ~1.8 字符/token，英文 ~4.5；格式开销 ~40% |
| 1.4 推理 | FP32 CPU ~11 tok/s；四步 pipeline 跑通 |
| 1.5 参数 | 13 组：T/top_p/do_sample 不影响速度；max_new_tokens 线性 |

**自回归慢的原因**：每生成 1 个 token 需对整个序列做一次 28 层 Transformer 前向传播。Step 4（vLLM）的 KV Cache 解决重复计算问题。

</details>
