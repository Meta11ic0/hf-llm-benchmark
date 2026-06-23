# HF LLM Benchmark — AI Infra 实战学习项目

> 目标：用最小成本（纯 CPU、单机）获得 LLM 推理工程的底层手感。
> 这不是一个"跑通的 demo"，而是一份能拿去面试讲清楚的实战经历。
>
> 学习笔记见 [LEARN.md](LEARN.md)——所有实验过程、数据、结论都在里面。

---

## 新机器从零跑起来（4 步）

```bash
# 1. 克隆 + 虚拟环境
git clone https://github.com/Meta11ic0/hf-llm-benchmark.git
cd hf-llm-benchmark
python3 -m venv venv
source venv/bin/activate

# 2. 配置国内镜像（国内网络必需！）
export HF_ENDPOINT=https://hf-mirror.com
# 建议加入 ~/.bashrc 持久化

# 3. 安装依赖（先装 torch CPU 版，再装其他）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. 验证环境
python3 -c "import torch; print('CPU only:', not torch.cuda.is_available())"
# 输出 CPU only: True ✅
```

模型文件（~1.4GB）首次使用 `AutoModel.from_pretrained()` 或 `snapshot_download()` 时自动下载到 `~/.cache/huggingface/`。

---

## 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| 模型 | Qwen3-0.6B（~1.4GB） | Apache 2.0 协议，CPU 可跑，中文友好 |
| 框架 | HuggingFace Transformers | 行业标准，`AutoModel` / `AutoTokenizer` 统一接口 |
| 精度 | FP32 + 8-bit（bitsandbytes） | 先跑通再量化，对比学习 |
| 设备 | 纯 CPU（WSL2 / 13GB RAM） | 无需 GPU，降低硬件门槛 |
| 镜像 | HF Mirror (`hf-mirror.com`) | 国内网络必需 |

> **关于对比模型**：原计划包含 Llama-3.2-1B，但该模型需要 HuggingFace 账号授权，暂未获取。Qwen3-0.6B 已能覆盖所有学习目标。后续获取授权后可追加对比实验。

---

## 学习路线（8 步）

| # | 任务 | 学什么 | 状态 |
|---|------|--------|------|
| 1.1 | 环境准备 | venv 隔离、CPU 版 PyTorch、HF Mirror 镜像 | ✅ |
| 1.2 | 模型下载与文件结构 | config.json 解读、参数量估算（0.6B 怎么算出来的）、safetensors 权重格式 | ✅ |
| 1.3 | 分词器入门 | encode/decode 原理、中英 token 效率对比、chat_template、prompt injection | ✅ |
| 1.4 | 单次推理跑通 | 4 步 pipeline（template → tokenize → generate → decode）、自回归循环机制 | ✅ |
| 1.5 | 生成参数实验 | temperature / top_p / max_new_tokens / do_sample 对输出质量和速度的影响 | ✅ |
| 1.6 | 8-bit 量化实验 | FP32 vs INT8 内存/速度/质量 trade-off | ⬜ |
| 1.7 | Benchmark 压测 | 多 prompt × 多轮统计、P50/P99 延迟、结果可复现设计 | ⬜ |
| 1.8 | 结果整理 + 提交 | 完整对比表、LEARN.md 定稿、push | ⬜ |

详细实验过程、数据和面试要点见 [LEARN.md](LEARN.md)。

---

## 产出物

```
hf-llm-benchmark/
├── README.md           # 项目概览 + 快速开始
├── LEARN.md            # ⭐ 学习笔记（主要产出——所有实验+分析+面试准备）
├── requirements.txt    # Python 依赖列表
├── results/            # 实验原始数据（JSON）
│   └── param_sweep.json
└── (后续) benchmark.py # 自写的压测脚本（任务 1.7）
```

---

## 验收标准

- [ ] 能解释 `model.generate()` 内部的自回归循环
- [ ] 能说出 temperature 和 top_p 的区别，以及它们为什么不影响推理速度
- [ ] 能从 config.json 估算模型参数量（面试高频题）
- [ ] 量化实验：FP32 vs INT8 内存降幅有具体数字
- [ ] Benchmark 输出 avg/P50/P99 延迟 + tokens/s
- [ ] LEARN.md 记录了全部实验过程和结论

---

## 运行提示

**CPU 推理速度预期。** Qwen3-0.6B 在普通笔记本 CPU（WSL2）上约 10-13 tokens/s。生成 512 token 约需 40-50 秒。完整 benchmark（10 prompt × 10 轮 = 100 次推理）约需 20-30 分钟。

**量化实验注意。** `bitsandbytes` 的 8-bit 量化在纯 CPU 环境下可能需要额外配置（默认依赖 CUDA）。遇到报错时，可先用模型权重文件大小推算理论差异（FP32 ≈ 参数数 × 4 字节，INT8 ≈ 参数数 × 1 字节），后续在 GPU 环境验证。
