# -*- coding: utf-8 -*-
"""测试集推理（作业任务点 6）：输出 single channel gray scale mask。

格式要求（PDF P4-6 原文）：
    "Perform inference on the test set, and save the outputs which are
     single channel gray scale masks."
因此：
    - 保存为 PIL mode='L' 的 8-bit 单通道 PNG（**不是调色板图、不是 RGB 彩图**）
    - 像素值 = 类别编号 0..20（0=background）
    - 尺寸还原到**原图尺寸**（先缩放到 base_size 推理，再把 logits 上采样回原尺寸）
    - 文件名与输入图同名（扩展名改 .png）

可选 TTA：多尺度 + 水平翻转，softmax 概率平均后再 argmax（默认关闭，基线跑通后再开）。
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from model import load_for_infer
from voc_dataset import NUM_CLASSES, VOCTestDataset


@torch.no_grad()
def predict_one(model, x, orig_hw, device, scales=(1.0,), flip=True, use_amp=True):
    """x: [1,3,H,W] 已归一化；返回原图尺寸的类别索引图 [H,W] (uint8)。"""
    h, w = orig_hw
    probs = torch.zeros((1, NUM_CLASSES, h, w), device=device, dtype=torch.float32)
    n = 0
    for s in scales:
        if s != 1.0:
            hs, ws = max(int(round(x.shape[-2] * s)), 16), max(int(round(x.shape[-1] * s)), 16)
            xs = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
        else:
            xs = x
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xs)["out"]
        logits = F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
        probs += logits.softmax(dim=1)
        n += 1
        if flip:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_f = model(torch.flip(xs, dims=[-1]))["out"]
            logits_f = F.interpolate(logits_f.float(), size=(h, w), mode="bilinear",
                                     align_corners=False)
            probs += torch.flip(logits_f, dims=[-1]).softmax(dim=1)
            n += 1
    return probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="train.py 保存的 best.pt / last.pt")
    ap.add_argument("--test_dir", default=r"E:\桌面\人工智能技术大作业\data\test_images")
    ap.add_argument("--out_dir", default=r"E:\桌面\人工智能技术大作业\outputs\predict")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0],
                    help="TTA 尺度，如 0.75 1.0 1.25；基线阶段保持 1.0")
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张，用于快速自检")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 按 ckpt 里记录的架构加载，strict=True —— 加载不完整会直接报错，避免静默产出垃圾 mask
    model, ckpt = load_for_infer(args.ckpt, device)
    print("加载权重:", args.ckpt, "| strict 校验通过", flush=True)

    ds = VOCTestDataset(args.test_dir, base_size=args.base_size,
                        limit=args.limit or None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    t0 = time.time()
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i, (x, hw, names) in enumerate(loader):
        x = x.to(device)
        pred = predict_one(model, x, hw, device, scales=tuple(args.scales),
                           flip=not args.no_flip, use_amp=use_amp)
        hist += np.bincount(pred.reshape(-1), minlength=NUM_CLASSES)
        name = names[0]
        stem = os.path.splitext(os.path.basename(name))[0]
        out_path = os.path.join(args.out_dir, stem + ".png")
        # 关键：mode='L' → 真正的单通道 8-bit 灰度图
        Image.fromarray(pred, mode="L").save(out_path)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}  {(time.time() - t0):.0f}s", flush=True)

    print(f"完成：{len(ds)} 张 -> {args.out_dir}（{(time.time() - t0):.0f}s）", flush=True)

    # 兜底自检：如果权重没加载对，模型会输出几乎纯背景，这里把它当场抓出来
    total = hist.sum()
    bg_ratio = hist[0] / max(total, 1)
    n_cls = int((hist > 0).sum())
    print(f"类别分布：background 占 {bg_ratio * 100:.2f}%，共出现 {n_cls} 个类别", flush=True)
    if bg_ratio > 0.995 or n_cls < 3:
        raise RuntimeError("预测结果异常（几乎全背景 / 类别过少），很可能权重未正确加载，已中止")
    print("分布自检通过", flush=True)


if __name__ == "__main__":
    main()
