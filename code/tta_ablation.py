# -*- coding: utf-8 -*-
"""TTA（测试时增强）消融：在验证集子集上对比"单尺度"与"多尺度+翻转"的 mIoU。

目的：验证 TTA 到底能涨多少点，用真实数字决定最终推理是否开 TTA。
只跑验证集的前 N 张（默认 300），控制耗时。

用法：
    python tta_ablation.py --ckpt ..\runs\lfca_res_e30\ckpt\best.pt --num 300
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from metrics import ConfusionMatrix
from model import load_for_infer
from voc_dataset import IGNORE_INDEX, NUM_CLASSES, VOCSegmentation


@torch.no_grad()
def run(model, loader, device, scales, flip, use_amp=True):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
    for images, targets in loader:
        images = images.to(device)
        h, w = targets.shape[-2:]
        probs = torch.zeros((1, NUM_CLASSES, h, w), device=device, dtype=torch.float32)
        n = 0
        for s in scales:
            if s != 1.0:
                hs = max(int(round(images.shape[-2] * s)), 16)
                ws = max(int(round(images.shape[-1] * s)), 16)
                xs = F.interpolate(images, size=(hs, ws), mode="bilinear", align_corners=False)
            else:
                xs = images
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xs)["out"]
            logits = F.interpolate(logits.float(), size=(h, w), mode="bilinear",
                                   align_corners=False)
            probs += logits.softmax(dim=1)
            n += 1
            if flip:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    lg = model(torch.flip(xs, dims=[-1]))["out"]
                lg = F.interpolate(lg.float(), size=(h, w), mode="bilinear", align_corners=False)
                probs += torch.flip(lg, dims=[-1]).softmax(dim=1)
                n += 1
        pred = probs.argmax(dim=1)[0].cpu()
        cm.update(pred, targets)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--num", type=int, default=300, help="验证集前 N 张")
    ap.add_argument("--base_size", type=int, default=513)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_for_infer(args.ckpt, device)

    ds = VOCSegmentation(args.voc_root, "val", train=False, base_size=args.base_size)
    sub = Subset(ds, list(range(min(args.num, len(ds)))))
    loader = DataLoader(sub, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    use_amp = device.type == "cuda"
    print(f"在验证集前 {len(sub)} 张上对比 TTA（权重 {os.path.basename(args.ckpt)}）", flush=True)

    results = []
    for name, scales, flip in [
        ("单尺度 (1.0)", [1.0], False),
        ("单尺度 + 翻转", [1.0], True),
        ("多尺度 (0.75/1.0/1.25)", [0.75, 1.0, 1.25], False),
        ("多尺度 + 翻转", [0.75, 1.0, 1.25], True),
        ("5尺度 (0.5~1.5)", [0.5, 0.75, 1.0, 1.25, 1.5], False),
        ("5尺度 + 翻转", [0.5, 0.75, 1.0, 1.25, 1.5], True),
    ]:
        t0 = time.time()
        cm = run(model, loader, device, scales, flip, use_amp)
        m21, m20 = cm.miou(True), cm.miou(False)
        results.append((name, m21, m20, time.time() - t0))
        print(f"  {name:<24} mIoU21={m21:.4f} mIoU20={m20:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    base = results[0][1]
    print("\n相对单尺度的增益：")
    for name, m21, m20, _ in results:
        print(f"  {name:<24} Δ mIoU21 = {(m21 - base) * 100:+.2f} 点")


if __name__ == "__main__":
    main()
