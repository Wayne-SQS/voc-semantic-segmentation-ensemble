# -*- coding: utf-8 -*-
"""独立评估某个权重在验证集上的表现，输出逐类 IoU 表（报告取数用）。

用法：
    python evaluate.py --ckpt ..\runs\smoke02\ckpt\best.pt
    python evaluate.py --ckpt ..\runs\smoke02\ckpt\best.pt --split trainval
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from metrics import ConfusionMatrix, format_table
from model import load_for_infer
from train import validate
from voc_dataset import IGNORE_INDEX, NUM_CLASSES, VOCSegmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--split", default="val")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_txt", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_for_infer(args.ckpt, device)

    ds = VOCSegmentation(args.voc_root, args.split, train=False, base_size=args.base_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    cm = validate(model, loader, device, num_classes=NUM_CLASSES,
                  use_amp=(device.type == "cuda"))

    print(f"\n=== {args.split} 集评估（{len(ds)} 张，权重 {os.path.basename(args.ckpt)}）===")
    print(f"mIoU (21 类含背景) : {cm.miou(True):.4f}")
    print(f"mIoU (20 类前景)   : {cm.miou(False):.4f}")
    print(f"Pixel Accuracy     : {cm.pixel_accuracy:.4f}")
    print(f"Mean Class Accuracy: {cm.mean_class_accuracy:.4f}")
    table = format_table(cm.per_class_table())
    print("\n逐类 IoU：")
    print(table)

    if args.save_txt:
        with open(args.save_txt, "w", encoding="utf-8") as f:
            f.write(f"split={args.split} ckpt={args.ckpt}\n")
            f.write(f"mIoU21={cm.miou(True):.4f}  mIoU20={cm.miou(False):.4f}  "
                    f"pixAcc={cm.pixel_accuracy:.4f}  mAcc={cm.mean_class_accuracy:.4f}\n\n")
            f.write(table + "\n")
        print(f"\n已保存：{args.save_txt}")


if __name__ == "__main__":
    main()
