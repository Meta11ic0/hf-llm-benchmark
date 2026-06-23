# Step 1 学习笔记：HuggingFace Transformers LLM 推理基础

> 学习日期：2026-06-22 | 模型：Qwen3-0.6B | 设备：CPU (WSL2, 13GB RAM)

---

## 目录

1. [环境准备](#1-环境准备)
2. [模型下载与文件结构](#2-模型下载与文件结构)
3. [分词器（Tokenizer）](#3-分词器tokenizer)
4. [单次推理跑通](#4-单次推理跑通)
5. [生成参数实验](#5-生成参数实验)
6. [学习进度](#6-学习进度)
7. [后续学习路线](#7-后续学习路线)
8. [附注：关于 Llama 和 Instruct](#附注关于-llama-和-instruct)
9. [附录：项目中的三个预生成脚本](#附录项目中的三个预生成脚本)

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

装完后在终端验证（一行命令，不需要写文件）：
```bash
python3 -c "import torch; print('CPU only:', not torch.cuda.is_available())"
# 输出 CPU only: True 表示装的是 CPU 版 PyTorch，正确 ✅
# 输出 CPU only: False 表示装成了 CUDA 版，需要卸载重装 CPU 版
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

### 2.0 下载命令与两种方式对比

**两种下载方式及其区别：**

| | 方式一：`snapshot_download()` | 方式二：`AutoModel.from_pretrained()` |
|---|---|---|
| 触发方式 | 主动调用 huggingface_hub API | transformers 加载模型时自动触发 |
| 下载了什么 | 只下载文件，不加载到内存 | 下载 + 解析 config + 创建模型对象 |
| 适合场景 | 先下载后研究，分步理解 | 一步到位跑推理 |
| 类比 C++ | `wget` 下载源码包 | `cmake --build . && make install` 一条龙 |

**推荐学习顺序：先用方式一下载，确认文件结构后再用方式二加载推理。** 这在面试时也更好讲——你能说清楚"下载"和"加载"是两步。

> **关于模型命名：** `Qwen/Qwen3-0.6B` 本身就是 Instruct 模型——Qwen3 系列在 HuggingFace 上不区分 base 和 Instruct 仓库，chat_template 已内置在 `tokenizer_config.json` 中。最初计划使用的 `Qwen/Qwen3-0.6B-Instruct` 在 hf-mirror 镜像上返回 401（该仓库名不存在），改为 `Qwen/Qwen3-0.6B` 即可，不影响所有实验。

```bash
# 确保镜像已设置
export HF_ENDPOINT=https://hf-mirror.com

# 方式一：只下载，不加载（推荐先用这个）
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen3-0.6B')
print(f'下载完成，路径: {path}')
"

# 方式二：下载+加载一条龙（后续推理时实际用的）
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
```

下载后检查缓存目录：
```bash
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/
```

> **CPU 指令集问题：** PyTorch CPU 版不要求特殊指令集（AVX/AVX-512 等），任何 x86-64 CPU 都能跑。PyTorch 启动时自动检测并启用可用指令集，无需配置。唯一注意：`bitsandbytes` 8-bit 量化在纯 CPU 环境可能受限（Step 1.6 详述）。

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

### 2.4 参数量估算：为什么叫 0.6B？

从 config.json 的几个数字可以手动推算模型总参数量。

每层 Transformer 包含两个模块：

**Attention 模块（QKV + 输出投影）：**
```
W_Q: 1024 × 1024 = 1,048,576
W_K: 1024 × 1024 = 1,048,576
W_V: 1024 × 1024 = 1,048,576
W_O: 1024 × 1024 = 1,048,576
────────────────────────────
合计: 4 × 1024² = 4,194,304 ≈ 4.2M
```

**FFN 模块（扩张 → 收缩，SwiGLU 变体多一个 gate 矩阵）：**
```
W_up:   1024 × 3072 = 3,145,728   ← 扩张到 3× hidden_size
W_gate: 1024 × 3072 = 3,145,728   ← SwiGLU 门控
W_down: 3072 × 1024 = 3,145,728   ← 缩回 hidden_size
────────────────────────────
合计: 3 × 1024 × 3072 = 9,437,184 ≈ 9.4M
```

**每层总计 ≈ 4.2M + 9.4M ≈ 13.6M**

> 注：旧公式用 `2 × hidden × intermediate`（~6.3M），那是标准 FFN（无 gate）。Qwen3 用 SwiGLU，多了 gate 矩阵，所以是 3 个矩阵而非 2 个。

**28 层 ≈ 13.6M × 28 ≈ 381M**

**Embedding 层：** `vocab_size × hidden_size = 151936 × 1024 ≈ 156M`

**总计 ≈ 381M + 156M ≈ 537M ≈ 0.54B**

Qwen3 标称 0.6B（6 亿）。偏差来自：LayerNorm 参数、`tie_word_embeddings=true` 下 lm_head 和 embedding 共享权重（省了 ~156M），以及偏差项的累积。

**面试可以这样说**：「给定 config.json，我能估算模型参数量级。核心公式是 Attention 块 4×hidden_size² + FFN 块 3×hidden_size×intermediate_size（SwiGLU 多一个 gate），乘以层数再加 embedding。算出来约 0.54B，与标称 0.6B 在合理范围内。」

**C++ 类比**：相当于看 `struct` 定义推算 `sizeof()`——知道每个字段类型就能心算出总字节数，不需要真的跑编译器。

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

| 指标 | 第一次 | 第二次（参数实验同批） |
|------|--------|----------------------|
| 输入 token | 14 | 14 |
| 生成 token | 128 | 128 |
| 总耗时 | 12.4s | 11.5s |
| **生成速度** | **10.3 tok/s** | **11.1 tok/s** |
| 预计生成 512 token | ~50 秒 | ~46 秒 |

两次运行速度都在 ~11 tok/s 左右，差异来自系统负载波动。纯 CPU 推理下这是正常水平。

**环境：** WSL2 / 无 GPU / FP32 精度 / 13GB RAM

---

## 5. 生成参数实验

> 实验日期：2026-06-23 | 模型：Qwen3-0.6B | Prompt：统一使用"请用中文简要介绍人工智能的三个主要应用领域。"

### 5.1 四个参数分别控制什么

| 参数 | 作用 | 系统类比 |
|------|------|----------|
| `temperature` | 控制概率分布的"陡峭程度"——低=高概率 token 更突出，高=概率被压平 | 类似 `softmax(x/T)`，T 越小分布越集中 |
| `top_p` | 核采样——只从累积概率 ≤ p 的候选 token 中选，截断长尾 | 类似剪枝：忽略概率极低的候选 |
| `max_new_tokens` | 硬上限——模型最多生成多少个 token，到了就停 | 类似 `for` 循环的最大迭代次数 |
| `do_sample` | True=按概率采样，False=每步直接选概率最高的 token (greedy) | `argmax` vs `random_choice(weights)` |

### 5.2 实测结果（13 组配置）

#### Temperature 组（固定 top_p=0.9, max=128）

| 配置 | 耗时 | 速度 | 输出特点 |
|------|------|------|---------|
| T=0.1 | 11.6s | 11.0 tok/s | 输出确定，接近 greedy |
| T=0.7 | 11.2s | 11.4 tok/s | 有一定多样性 |
| T=1.5 | 11.0s | 11.7 tok/s | 更多样，但本测试中未出现明显不连贯 |

**结论：Temperature 不影响推理速度。** 无论 T 是多少，28 层 Transformer 的矩阵乘法计算量完全相同。T 只在最后 `softmax` 之后改变采样分布，这个操作耗时可忽略。

#### top_p 组（固定 T=0.7, max=128）

| 配置 | 耗时 | 速度 |
|------|------|------|
| p=0.5 | 10.9s | 11.7 tok/s |
| p=0.9 | 10.8s | 11.8 tok/s |
| p=1.0 | 10.9s | 11.7 tok/s |

**结论：top_p 也不影响速度。** 它只过滤候选 token 集合，不改变计算量。

#### max_new_tokens 组（固定 T=0.7, p=0.9）

| 配置 | 耗时 | 生成 token | 速度 |
|------|------|-----------|------|
| max=32 | 3.1s | 32 | 10.2 tok/s |
| max=128 | 11.5s | 128 | 11.1 tok/s |
| max=256 | 23.3s | 255 | 11.0 tok/s |

**结论：耗时与 max_new_tokens 几乎线性关系。** 每多生成 1 个 token 约多花 0.09s。这就是自回归生成的特征——每个 token 都需要一次完整的前向传播。

注意 max=256 实际只生成了 255 个——模型在到达上限前触发了 EOS（结束符）。

#### do_sample 组（固定 T=0.7, p=0.9, max=128）

| 配置 | 耗时 | 速度 |
|------|------|------|
| greedy | 10.7s | 12.0 tok/s |
| sampling | 11.6s | 11.1 tok/s |

**结论：Greedy 略快（~8%），因为只做 argmax 不需要随机采样。** 但差距很小——大头仍然是 Transformer 前向传播。

#### 极端对比

| 配置 | 耗时 | 速度 |
|------|------|------|
| greedy_deterministic (T=1.0, p=1.0, sample=False) | 9.8s | 13.0 tok/s |
| high_randomness (T=1.5, p=0.5, sample=True) | 11.2s | 11.5 tok/s |

Greedy deterministic 是 13 组中最快的——不做任何采样，每步直接选最高概率 token。

### 5.3 核心结论

1. **所有生成参数（temperature/top_p/do_sample）对推理速度的影响都可以忽略。** 延迟由 `max_new_tokens` 和模型大小决定。
2. **调参数调的是"输出质量/多样性"，不是"速度"。**
3. **线上服务质量问题对应关系：**
   - 用户抱怨"回复太死板" → 提高 temperature（如 0.7→1.0）或开启 sampling
   - 用户抱怨"回复太随机/胡说八道" → 降低 temperature（如 0.7→0.3）或用 greedy
   - 需要控制延迟预算 → 减小 max_new_tokens

### 5.4 面试角度

> **面试官**：「你做过生成参数调优吗？temperature 和 top_p 的区别是什么？」
>
> **你可以答**：「temperature 改变的是概率分布的 shape——低 T 让高概率 token 更突出，接近 greedy；高 T 压平分布，让低概率 token 有机会被选中。top_p 则是直接截断——只保留累积概率不超过 p 的那些候选。两者可以组合使用：先用 top_p 过滤长尾，再用 temperature 调节分布陡峭程度。而且我从实测数据验证了这些参数不影响推理延迟——延迟由 max_new_tokens 和模型大小决定。」

---

## 6. 学习进度

| # | 任务 | 状态 | 关键产出 |
|---|------|------|---------|
| 1.1 | 环境准备 | ✅ | venv + CPU PyTorch + HF Mirror |
| 1.2 | 模型文件结构 | ✅ | config.json 字段解读、Qwen vs Llama 对比表 |
| 1.3 | 分词器 | ✅ | 中英 token 效率 2.5x、chat_template、prompt injection |
| 1.4 | 单次推理 | ✅ | 4 步 pipeline（template→tokenize→generate→decode） |
| 1.5 | 生成参数实验 | ✅ | 13 组对比，temperature/top_p 不影响速度 |
| 1.6 | 量化实验 | ⬜ | 目标：FP32 vs INT8 内存/速度/质量对比 |
| 1.7 | Benchmark 压测 | ⬜ | 目标：多 prompt × 多轮，P50/P99 延迟 |
| 1.8 | 整理提交 | ⬜ | 对比表 + README 更新 + push |

---

## 7. 后续学习路线

### 下一站：任务 1.6 量化实验

**不需要预生成脚本。** 你要回答的核心问题：

> FP32 权重（每个参数 4 字节）→ INT8（1 字节），理论上内存能省多少？实际会牺牲什么？

用你已有的数据估算：Qwen3-0.6B 约 6 亿参数，`model.safetensors` 文件 1.4GB。INT8 后应该是多大？加载到内存后实际占用多少？

**学习目标**（面试级）：
- 能解释量化的基本思想：把权重从 FP32 压缩到 INT8，用精度换空间
- 能说出 trade-off：内存 ↓75%，但可能有精度损失，且 CPU 上 INT8 推理不一定比 FP32 快
- 知道 `bitsandbytes` 库的作用

**预计时间**：1-1.5h（包括读 `bitsandbytes` 文档 + 实验 + 记录）

### 再下一站：任务 1.7 Benchmark 压测

这是你**第一次自己写代码**的任务。目标：写一个 `benchmark.py`，要求：

- 支持 `--model qwen` 参数
- 对多条 prompt 各跑多轮
- 每轮做预热（丢弃第一次推理结果）
- 输出 avg / P50 / P99 延迟 + tokens/s
- 结果保存到 JSON

这就是面试官会追问的：「你的 benchmark 怎么设计的？怎么保证可复现？」

### 最后一站：任务 1.8 整理提交

- 更新 README.md：项目目的、使用方法、关键发现
- 确保 LEARN.md 完整（你现在就在做这件事）
- `git commit` + push

---

## 附注：关于 Llama 和 Instruct

- **Qwen3-0.6B-Instruct**：hf-mirror 镜像上不存在（返回 401）。Qwen3-0.6B 本身就是 Instruct 模型（Qwen3 系列 base 和 Instruct 在 HuggingFace 上是同一个仓库，`chat_template` 已内置），不影响所有实验。
- **Llama-3.2-1B**：需要 HuggingFace 账号授权，暂不可用。Qwen3 足够覆盖所有学习目标。后续如果申请到授权，可以用相同脚本跑 Llama 做对比。

---

## 附录：项目中的三个预生成脚本

项目最初包含三个 AI 生成的脚本，它们的设计目标和对应的 LEARN.md 章节：

| 脚本 | 设计目标 | 学习内容记录在 |
|------|---------|-------------|
| `download_and_explore.py` | 下载模型 → 遍历文件结构 → 解读 config.json → 参数量估算 → Qwen vs Llama 对比 | 第 2 节（含 2.4 参数量估算） |
| `explore_tokenizer.py` | 中英 tokenize 对比 → chat_template 机制 → encode→decode 往返 → prompt injection 演示 | 第 3 节 |
| `explore_generate.py` | 单次推理 4 步流程 → 13 组生成参数实验 → 速度/质量分析 | 第 4 + 5 节 |

三个脚本的全部学习内容已提取到本文档，脚本可以删除。后续实验（1.6 量化、1.7 Benchmark）由学习者自己编写代码，而非依赖预生成脚本。
