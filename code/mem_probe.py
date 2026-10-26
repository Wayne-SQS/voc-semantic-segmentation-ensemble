# -*- coding: utf-8 -*-
"""训练显存可行性测试：给指定骨干 + batch + 裁剪尺寸，跑一次前向+反向，报告显存峰值。

为什么需要这个：8 GB 显存是硬约束，R101 的参数和激活都比 R50 大，
直接开训练跑到一半 OOM 会浪费几小时。先用 3 个 iter 探底，几十秒就能给出结论。

用法：
    python mem_probe.py --backbone resnet50 --batch_size 4 --crop_size 513
    python mem_probe.py --backbone resnet101 --batch_size 2 --crop_size 513
"""

import argparse

import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    ap.add_argument("--model", default="lfca", choices=["deeplabv3", "lfca"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--crop_size", type=int, default=513)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("没有可用 GPU")

    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.model == "lfca":
        from model_v2 import build_model_v2
        model = build_model_v2(pretrained=False, aux_loss=True,
                               backbone=args.backbone).to(dev)
    else:
        from model import build_model
        model = build_model(pretrained=False, aux_loss=True,
                            backbone=args.backbone).to(dev)

    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(ignore_index=255)

    x = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size, device=dev)
    y = torch.randint(0, 21, (args.batch_size, args.crop_size, args.crop_size),
                      device=dev)

    model.train()
    for i in range(args.iters):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = crit(out["out"], y)
            if out.get("aux") is not None:
                loss = loss + 0.4 * crit(out["aux"], y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  iter {i + 1}: loss={loss.item():.4f}  peak={peak:.2f} GB", flush=True)

    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"\n{args.model}/{args.backbone}  batch={args.batch_size} crop={args.crop_size}")
    print(f"参数量 {n_param:.1f} M | 峰值 allocated {peak:.2f} GB | "
          f"reserved {reserved:.2f} GB | 显卡总量 {total:.1f} GB")
    if peak > total * 0.85:
        print("判定：显存偏紧，建议降 batch 或 crop（保留 15% 余量给验证和数据加载）")
    else:
        print("判定：显存可行")


if __name__ == "__main__":
    main()
