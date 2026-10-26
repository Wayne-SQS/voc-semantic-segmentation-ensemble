# -*- coding: utf-8 -*-
"""测量"零样本基线"：torchvision 的 COCO_WITH_VOC_LABELS_V1 预训练权重，
在**不做任何微调**的情况下，直接在 VOC2012 val 上能拿多少 mIoU。

为什么要测这个：
    本项目训练曲线有个反常现象——R101 在第 1 个 epoch 就冲到 0.7601，
    超过了 R50 训练 30 轮才达到的 best（0.7570），之后反而下降。
    如果不测零样本基线，很容易误判成"训练出了问题"或"学习率太大"。
    测出来才知道：这份权重本身就有很高的起点，大学习率是在**破坏**它，
    而不是"从零开始学"。

用法：
    python eval_pretrain.py --backbone resnet50 --num 200
    python eval_pretrain.py --backbone resnet101 --num 200
"""

import argparse

import torch
from torch.utils.data import DataLoader

from metrics import ConfusionMatrix
from train import validate
from voc_dataset import NUM_CLASSES, VOCSegmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=200, help="只测前 N 张，0=全量")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone == "resnet101":
        from model import build_model
    else:
        from model import build_model
    model = build_model(pretrained=True, aux_loss=True, backbone=args.backbone)
    model = model.to(device).eval()
    print(f"已加载原始预训练权重：DeepLabV3-{args.backbone} "
          f"(COCO_WITH_VOC_LABELS_V1，未做任何微调)", flush=True)

    ds = VOCSegmentation(args.voc_root, args.split, train=False, base_size=args.base_size)
    if args.num > 0:
        ds.ids = ds.ids[:args.num]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        pin_memory=True)

    cm = validate(model, loader, device, num_classes=NUM_CLASSES,
                  use_amp=(device.type == "cuda"))
    print(f"\n=== 零样本基线（{args.split} 前 {len(ds)} 张）===")
    print(f"mIoU (21 类)  : {cm.miou(True):.4f}")
    print(f"mIoU (20 类)  : {cm.miou(False):.4f}")
    print(f"Pixel Accuracy: {cm.pixel_accuracy:.4f}")
    print(f"\n结论：这份预训练权重无需任何微调即可达 {cm.miou(True):.4f}，"
          f"这就是训练曲线的真实起点。")


if __name__ == "__main__":
    main()
