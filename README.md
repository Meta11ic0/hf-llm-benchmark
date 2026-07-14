# HF LLM Benchmark — LLM 推理入门教学项目

> 以教学为目的编排，9 个任务逐步深入，每个任务聚焦一个核心概念。纯 CPU、最小成本获得 LLM 推理底层手感。
>
> **怎么学**：跟着 [LEARN.md](LEARN.md) 逐步操作即可。代码可以手写，也可以让 AI 生成——脚本都不超过 30 行。但核心是**每行代码你都能解释它在做什么**，而不是"AI 帮我写的，能跑就行"。
>
> **CLAUDE.md** 是本项目配套的 AI 助教设定，融入了作者的学习偏好。Clone 后可根据自己的习惯微调（比如去掉追问环节、调整解释深度等）。

---

## 快速开始

```bash
git clone https://github.com/Meta11ic0/hf-llm-benchmark.git && cd hf-llm-benchmark
python3 -m venv venv && source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# 然后打开 LEARN.md，从任务 1.1 开始逐步操作。每个任务需要写脚本时，文档会指引你。
# 写完对照 reference/ 里的参考实现——确保你理解每行在做什么，而不是直接复制粘贴。
```

---

## 预期结果（Qwen3-0.6B，CPU FP32）

| 指标 | 预期范围 |
|------|---------|
| 推理速度 | 10-12 tok/s（稳态，预热后） |
| 预热开销 | 首次比稳态慢 20-30%，正常丢弃 |
| P99 vs avg | P99 > avg > P50，正常长尾分布 |

你跑出来的具体数字因硬件而异，只要在预期范围附近即可。

### 量化

| 精度 | 内存 | 说明 |
|------|------|------|
| FP32 | ~2.4 GB | 用 `dtype=float32` 加载后的内存占用 |
| BF16 | ~1.2 GB（理论） | 模型文件的官方发布格式，CPU 加载时自动转 FP32 计算 |
| INT8 | ~0.6 GB（理论） | 量化后的理论值，CPU 上无法直接用 `from_pretrained` 加载 |

磁盘文件（safetensors）1.5 GB——部分参数以 fp32 存储，并非纯 bf16。详见 [LEARN.md §1.8](LEARN.md#18-量化实验--理论推算)。

---

## 项目结构

```
hf-llm-benchmark/
├── README.md              ← 本文件（项目简介）
├── LEARN.md               ← 完整教程，9 个任务逐步深入
├── CLAUDE.md              ← AI 助教设定（可选，可定制）
├── requirements.txt
├── reference/             ← 参考实现，每个任务做完再看
│   ├── minimal_infer.py   ← 任务 1.1
│   ├── gen_params.py      ← 任务 1.5
│   ├── benchmark.py       ← 任务 1.6
│   └── quantization.py    ← 任务 1.8
└── venv/
```

---

## 踩坑记录

| 坑 | 现象 | 解决 |
|----|------|------|
| CUDA 版 PyTorch 在 CPU 上报错 | `pip install torch` 默认装 CUDA 版（~800MB） | 加 `--index-url https://download.pytorch.org/whl/cpu` |
| 模型下载极慢 | HF 官方服务器在国内直连慢 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 首次推理比稳态慢 20%+ | 首次加载需填充物理内存页 | 预热一次，丢弃首次结果 |
| `do_sample=True` 时 tok 数波动 | 随机采样天然不稳定 | tok/s 用 total_tok / total_time 算 |
| Llama 模型可能无法下载 | Meta 管制部分地区访问 | 用 config + tokenizer 公开信息做理论对比 |
| CPU 上无法加载 int8 | bitsandbytes 依赖 CUDA | 理论推算代替实测 |

---

## 学到什么

- `apply_chat_template()` → tokenize → `model.generate()` → `decode` 四步数据流
- tokenizer 词表覆盖度决定推理成本（Qwen 中文 vs Llama 英文 token 数差 2-3×）
- P50/P99 统计 + 预热 + 多次重复的 benchmark 方法论
- 推理瓶颈在访存带宽（memory-bound），不在计算——量化加速主要来自省带宽
- 模型磁盘存储（bf16）≠ 内存加载精度（用户 `dtype` 决定）≠ 量化后精度（int8）
