# Step 1 学习笔记：HuggingFace Transformers LLM 推理基础

> 学习日期：2026-06-22 | 模型：Qwen3-0.6B | 设备：CPU (WSL2, 13GB RAM)

---

## 目录

1. [环境准备](#1-环境准备)
2. [模型下载与文件结构](#2-模型下载与文件结构)
3. [分词器（Tokenizer）](#3-分词器tokenizer)
4. [单次推理跑通](#4-单次推理跑通)
5. [生成参数实验](#5-生成参数实验待做)
6. [8-bit 量化实验](#6-8-bit-量化实验待做)
7. [Benchmark 压测](#7-benchmark-压测待做)

---

## 1. 环境准备

### 1.1 在做什么？

让一台只有 CPU 的 Linux 机器具备加载和运行开源大模型的能力。

### 1.2 为什么是第一步？

HuggingFace Transformers 是所有开源 LLM 工程的入口。不做这步，后面什么都没法做。类比 C++ 开发者必须先装 gcc/cmake。

### 1.3 每一步做了什么、为什么

#### 安装 pip（Python 的包管理器）

```bash
sudo apt install -y python3-pip
```

Python 装包和 C++ 装库的区别：
- C++: 下载源码 → cmake → make install → 头文件放 `/usr/include/`，库放 `/usr/lib/`
- Python: `pip install <包名>` 一条命令，自动下载+安装到 `site-packages/`

`pip` 就是这个包管理器。系统 Python 3.10.12 默认没带，所以需要 apt 单独装。

#### 安装 venv（虚拟环境）

```bash
sudo apt install -y python3-venv
python3 -m venv venv
source venv/bin/activate
```

**为什么需要虚拟环境？** Python 全局安装包会导致版本冲突。项目 A 需要 `torch==2.0`，项目 B 需要 `torch==2.12`——两者装到全局就会打架。

`venv` 创建一个独立的 Python 环境目录 `venv/`。激活后 `pip install` 的所有包都装在这个目录里，互不干扰。

**C++ 类比：**
```bash
# Python venv ≈ C++ 项目隔离
source venv/bin/activate     # ≈ export LD_LIBRARY_PATH=./build/lib
pip install torch            # ≈ cmake --build . --target install (到 ./build/lib)
```

#### 配置国内镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

HuggingFace 官方服务器在国内直连极慢（几十 KB/s，经常超时）。`hf-mirror.com` 是国内社区维护的完整镜像。设置后所有模型下载自动走镜像，代码不用改。

#### 安装 PyTorch（CPU 版）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**这是整个环境准备里最容易出错的一步。** 不加 `--index-url` 时 pip 默认下载 CUDA 版 PyTorch（~800MB），在 CPU 机器上可能报错或静默异常。

`--index-url https://download.pytorch.org/whl/cpu` 强制从 PyTorch 官方的 CPU-only 仓库下载，包体积约 200MB，不依赖任何 CUDA 组件。

验证装对没有：
```python
import torch
print(torch.cuda.is_available())  # 必须是 False
```

#### 安装 HuggingFace 核心包

```bash
pip install transformers accelerate bitsandbytes huggingface_hub sentencepiece tiktoken
```

| 包 | 作用 |
|----|------|
| `transformers` | 核心库：`AutoModel`, `AutoTokenizer`, `model.generate()` |
| `accelerate` | 模型加载加速：管理权重放哪个设备（CPU/GPU）、内存优化 |
| `bitsandbytes` | 量化库：FP32 → INT8 权重压缩 |
| `huggingface_hub` | HF Hub 客户端：下载模型、管理缓存 |
| `sentencepiece` | BPE 分词算法（Llama tokenizer 依赖） |
| `tiktoken` | OpenAI 的 tokenizer 库 |

---

## 2. 模型下载与文件结构

### 2.1 下载了什么

从 HuggingFace Hub 下载了 **Qwen3-0.6B** 模型，约 1.4GB，存放在：

```
~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<hash>/
```

### 2.2 模型目录里的每个文件

| 文件 | 大小 | 是什么 | C++ 类比 |
|------|------|--------|----------|
| `model.safetensors` | 1433.7 MB | 模型所有权重（6亿个浮点数） | 编译好的 `libmodel.so` |
| `tokenizer.json` | 10.9 MB | 词表——151936 个 token 的映射 | 字符编码表 |
| `config.json` | 1.5 KB | 模型架构参数 | 头文件配置 `#define HIDDEN_SIZE 1024` |
| `tokenizer_config.json` | ~5 KB | tokenizer 配置 + chat_template | API 配置文件 |
| `generation_config.json` | ~0.7 KB | 默认生成参数 | 默认编译选项 |

### 2.3 config.json 关键参数解读

```json
{
  "hidden_size": 1024,              // 每个 token 用 1024 维向量表示
  "num_hidden_layers": 28,           // Transformer 层的数量
  "intermediate_size": 3072,         // FFN 中间维度（3× hidden_size）
  "vocab_size": 151936,              // 词表大小——模型"认识"多少个 token
  "num_attention_heads": 16,         // 注意力头数
  "num_key_value_heads": 8,          // KV 头数（GQA）
  "max_position_embeddings": 40960,  // 最大上下文长度 = 32K tokens
  "rope_theta": 1000000,             // 长文本位置编码能力
  "tie_word_embeddings": true        // 输入输出共享权重（省参数）
}
```

**Qwen3 vs Llama-3.2 配置对比：**

| 维度 | Qwen3-0.6B | Llama-3.2-1B |
|------|-----------|-------------|
| hidden_size | 1024 | 2048 |
| num_hidden_layers | 28 | 16 |
| vocab_size | 151,936 | 128,256 |
| max_position_embeddings | 40,960 (32K) | 131,072 (128K) |
| 注意力 | GQA (Q:16, KV:8) | GQA (Q:12, KV:4) |

**关键差异：** Qwen3 更深但更窄（28层 vs 16层），词表更大（中文 token 效率更好）；Llama 上下文更长（128K vs 32K）。

---

## 3. 分词器（Tokenizer）

### 3.1 Tokenizer 在做什么

**把人类文字翻译成模型认识的数字序列，以及反过来。**

```
人类文字 "你好，今天天气怎么样？"
     │  tokenizer.encode()
     ▼
Token IDs [108386, 100850, 104056, 104191, 106797, 102130]
     │  model.generate()  ← 模型只认数字
     ▼
Token IDs [..., 新 token]
     │  tokenizer.decode()
     ▼
人类文字 "今天天气不错！"
```

**C++ 类比：** 类似 Unicode 编解码（字符 ↔ code point），但粒度不同——tokenizer 是「子词级别」映射：
- 高频词（如 "今天"）→ 1 个完整 token
- 低频词（如 "Qwen3"）→ 拆成多个字节 token

### 3.2 实测：中文 vs 英文 Token 效率

| 文本类型 | 字符数 | Token 数 | 字符/Token |
|----------|--------|----------|------------|
| 中文短句 | 11 | 6 | 1.8 |
| 中文长句 | 63 | 35 | 1.8 |
| 英文短句 | 25 | 7 | 3.6 |
| 英文长句 | 254 | 38 | 6.7 |
| 代码片段 | 92 | 28 | 3.3 |

**结论：** 中文 token 效率约是英文的 **40%**（1.8 vs 4.5 字符/token）。同样的语义内容，中文需要约 2.5 倍的 token 数 → 推理延迟更高 → API 费用更贵。

**面试可以这样说：** 「中文 token 效率低不是因为语言本身，而是词表覆盖度问题。Qwen3 的 151K 词表为中文做了优化，如果换成 Llama 的 128K 英文主导词表，中文 token 数会更差。」

### 3.3 chat_template：对话格式的自动转换

```python
# 你发给 API 的格式（OpenAI 标准）
messages = [
    {"role": "system", "content": "你是一个帮助性的AI助手。"},
    {"role": "user", "content": "解释一下什么是 Transformer？"},
]

# chat_template 自动转换成模型训练时看到的原始格式 ↓
"""
<|im_start|>system
你是一个帮助性的AI助手。<|im_end|>
<|im_start|>user
解释一下什么是 Transformer？<|im_end|>
<|im_start|>assistant
"""
```

**不同模型的格式不同：** Qwen3 用 `<|im_start|>` / `<|im_end|>`，Llama 用 `<|begin_of_text|>` / `<|eot_id|>`。chat_template 屏蔽了这个差异——开发者统一用 `messages` 数组即可。

**实测 token 开销：** system 提示（12 tok）+ user 问题（8 tok）+ 特殊标记（13 tok）= **共 33 tok，其中 40% 是格式开销。**

### 3.4 Encode → Decode 往返验证

所有正常文本都能无损往返（encode → decode → 完全一致），包括 emoji 和数学符号。**但有一个安全漏洞：**

```python
# 危险：用户输入中包含特殊 token 字符串
"<|im_start|>system"  → encode → decode → "system"
# ← 特殊 token 被"吃掉"了！
```

用户可以在输入中注入 `<|im_start|>` 来伪造系统角色——这是 **prompt injection** 攻击的基础原理。

---

## 4. 单次推理跑通

### 4.1 推理的四个步骤

```
用户输入 "用一句话解释什么是API"
    │
    ▼ Step 1: apply_chat_template()
    │    messages 数组 → 模型训练格式的 prompt 字符串
    │
    ▼ Step 2: tokenizer(prompt)
    │    prompt 字符串 → token IDs [151644, 872, 198, ...]
    │    输入共 14 个 token
    │
    ▼ Step 3: model.generate()  ← 最耗时的一步
    │    自回归循环 128 次，每次预测下一个 token
    │    每次循环 = 28 层 Transformer × 1024 维矩阵乘法
    │
    ▼ Step 4: tokenizer.decode()
    │    生成的 token IDs → 人类可读文本
    │
    输出 "API（Application Programming Interface）是应用程序接口..."
```

### 4.2 为什么自回归生成慢？

```python
# model.generate() 内部的等价逻辑
input_ids = [151644, 872, 198, ...]   # 15 个已知 token

for i in range(128):                  # 生成 128 个新 token
    # 对整个序列（15 + i 个 token）做一次前向传播
    logits = model.forward(input_ids) # ← 28 层 Transformer 在这里
    next_token = sample(logits)       # 选下一个 token
    input_ids.append(next_token)      # 拼回去继续循环
```

**每个 token 都要把整个序列重新算一遍**（不优化的话）。这是 Transformer 注意力机制的固有特性——它需要看到「当前 token 之前的所有 token」才能预测下一个。

Step 4（vLLM）会讲 KV Cache 如何解决这个重复计算问题。

### 4.3 实测性能

| 指标 | 数值 |
|------|------|
| 输入 token | 14 |
| 生成 token | 128 |
| 总耗时 | 31.9 秒 |
| **生成速度** | **4.0 tokens/s** |
| 预计生成 512 token | ~128 秒（2 分钟） |

**环境：** WSL2 / 无 GPU / FP32 精度 / 13GB RAM

---

## 下一步

- [ ] 任务 1.5: 系统对比 temperature / top_p / max_new_tokens / do_sample 对输出质量和速度的影响
- [ ] 任务 1.6: FP32 vs INT8 量化对比
- [ ] 任务 1.7: Benchmark 批量压测（多 prompt × 多轮取平均值）
- [ ] 任务 1.8: 结果整理 + GitHub 提交
