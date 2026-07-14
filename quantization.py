#!/usr/bin/env python3
"""1.8 量化实验：计算 Qwen3-0.6B 在不同精度下的内存占用。"""

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"

def main():
    print("加载模型，计算参数量...")
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

    n_params = sum(p.numel() for p in m.parameters())

    print(f"\n{'='*50}")
    print(f"  参数量:     {n_params / 1e9:.2f}B ({n_params:,} 个)")
    print(f"  FP32 内存:  {n_params * 4 / 1e9:.2f} GB")
    print(f"  INT8 内存:  {n_params * 1 / 1e9:.2f} GB (理论)")
    print(f"  压缩比:     fp32 / int8 = 4×")
    print(f"{'='*50}")

    # 看看磁盘上实际文件大小
    from pathlib import Path
    snapshots_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            latest = snapshots[-1]
            safetensors = sorted(latest.glob("*.safetensors"))
            total_mb = sum(f.stat().st_size for f in safetensors) / 1e6
            print(f"\n  磁盘文件（safetensors）: {total_mb:.0f} MB ({total_mb/1000:.2f} GB)")
            print(f"  磁盘存的是 bf16（2 字节/参数），内存 FP32 是磁盘的 {n_params * 4 / (total_mb * 1e6):.1f}×")
    else:
        print(f"\n  （磁盘缓存目录未找到，跳过文件大小对比）")

    # 为啥加载 FP32 而不是 BF16
    print(f"\n  注：CPU 上 bf16 无硬件加速，PyTorch 内部回退到 fp32 计算")
    print(f"     所以 dtype=float32 是 CPU 推理的正确选择，不是浪费。\n")

if __name__ == "__main__":
    main()
