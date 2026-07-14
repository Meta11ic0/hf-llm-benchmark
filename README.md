# HF LLM Benchmark — LLM 推理前置学习项目

> 纯 CPU（WSL2）、最小成本获得 LLM 推理底层手感。Step 1 完整记录见 [LEARN.md](LEARN.md)。

---

## 快速开始

```bash
git clone https://github.com/Meta11ic0/hf-llm-benchmark.git && cd hf-llm-benchmark
python3 -m venv venv && source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python minimal_infer.py    # 单次推理
python benchmark.py        # 性能测试
python quantization.py     # 量化推算
```

---

## 核心结果（Qwen3-0.6B，CPU FP32，WSL2）

### Benchmark（3 prompt × 3 次）

| prompt | avg(s) | P50(s) | P99(s) | tok/s |
|--------|--------|--------|--------|-------|
| 短问答 | 19.29 | 18.26 | 21.75 | 12.1 |
| 中翻译 | 14.10 | 13.86 | 15.08 | 12.3 |
| 长总结 | 36.28 | 36.82 | 41.14 | 11.6 |

**稳态 ~12 tok/s**，含预热（丢弃首次）和 P50/P99 统计。

### 量化

| 精度 | 内存 | 备注 |
|------|------|------|
| FP32 | 2.38 GB | 当前加载方式 |
| BF16 | 1.19 GB（理论） | 官方发布格式，CPU 无硬件加速 |
| INT8 | 0.60 GB（理论） | CPU 不可用 bitsandbytes，需 llama.cpp |

磁盘文件（safetensors）1.5 GB——部分参数以 fp32 存储，并非纯 bf16。详见 [LEARN.md §1.8](LEARN.md#18-量化实验--理论推算)。

---

## 项目结构

```
hf-llm-benchmark/
├── README.md              ← 本文件
├── LEARN.md               ← Step 1 完整执行手册
├── CLAUDE.md              ← AI 导师角色定义
├── requirements.txt
├── minimal_infer.py       ← 任务 1.1：最小推理链路
├── benchmark.py           ← 任务 1.6：性能 benchmark
├── quantization.py        ← 任务 1.8：量化内存推算
└── venv/
```

---

## 踩坑记录

| 坑 | 现象 | 解决 |
|----|------|------|
| CUDA 版 PyTorch 在 CPU 上报错 | `pip install torch` 默认下 CUDA 版（~800MB） | 加 `--index-url https://download.pytorch.org/whl/cpu` |
| 模型下载极慢 / 超时 | HF 官方服务器在国内直连慢 | `export HF_ENDPOINT=https://hf-mirror.com` |
| Llama-3.2-1B 申请两次被拒 | Meta 遵守美国出口管制，中国 IP 自动拒 | 通过 config + tokenizer 公开信息做理论对比 |
| `do_sample=True` 导致 tok 数不稳定 | 采样天然随机 | tok/s 用 total_tok / total_time 算 |
| 首次推理比稳态慢 20%+ | 内存页填充、Python JIT 冷路径 | 预热一次，丢弃 |
| CPU 上 bitsandbytes 不可用 | bitsandbytes 依赖 CUDA | 理论推算代替实测 |

---

## Step 1 学到什么

- `apply_chat_template()` → tokenize → `model.generate()` → `decode` 四步数据流
- tokenizer 词表覆盖度决定推理成本（Qwen 中文 vs Llama 英文 token 数差 2-3×）
- P50/P99 统计 + 预热 + 多次重复的 benchmark 方法论
- 推理瓶颈在访存带宽（memory-bound），不在计算——量化加速主要来自省带宽
- 模型磁盘存储（bf16）≠ 内存加载精度（用户 `dtype` 决定）≠ 量化后精度（int8）
