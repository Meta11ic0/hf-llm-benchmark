# HF LLM Benchmark — LLM 服务化前置学习项目

> 用最小成本（纯 CPU、单机）获得 LLM 推理工程的底层手感，为后续 LLM 流式服务化做准备。
> 详细学习计划与进度见 [LEARN.md](LEARN.md)。

---

## 新机器从零跑起来（4 步）

```bash
git clone https://github.com/Meta11ic0/hf-llm-benchmark.git && cd hf-llm-benchmark
python3 -m venv venv && source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像，建议写入 ~/.bashrc
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python3 -c "import torch; print('CPU only:', not torch.cuda.is_available())"  # 应输出 True
```

模型（~1.4GB）首次 `from_pretrained()` 时自动下载到 `~/.cache/huggingface/`。

---

## 技术选型

Qwen3-0.6B + HuggingFace Transformers + FP32/INT8 对比 + 纯 CPU（WSL2）+ HF Mirror 镜像。选型理由见 [LEARN.md §1–2](LEARN.md)。

---

## 产出物

```
hf-llm-benchmark/
├── README.md / LEARN.md / CLAUDE.md / requirements.txt
└── (后续) benchmark.py   # 任务 1.6
```

验收标准、CPU 速度预期与量化注意见 [LEARN.md](LEARN.md)。
