# -*- coding: utf-8 -*-
"""把 torchvision 的原始预训练权重导出成项目统一格式的 ckpt，供集成/推理脚本直接加载。

为什么需要这一步：
    实测发现 COCO_WITH_VOC_LABELS_V1 权重**不做任何微调**就有极高的 mIoU
    （R50 0.7768 / R101 0.7951，全量 val），显著高于我们自己微调 30 轮的模型。
    既然它们是最强的单个"模型"，就应该作为集成的一员参与投票。
    但 predict_ensemble.py 需要 ckpt 格式（含 args 字段以决定架构），所以这里做一次格式转换。

注意：导出的 ckpt 里的 args 只含架构信息，没有训练超参——因为它本来就没训练过。

用法：
    python export_pretrain.py --backbone resnet50  --out ..\runs\pretrain_zs_r50\ckpt\zeroshot.pt
    python export_pretrain.py --backbone resnet101 --out ..\runs\pretrain_zs_r101\ckpt\zeroshot.pt
"""

import argparse
import os

import torch

from model import HETERO_MODELS, build_hetero, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    ap.add_argument("--arch", default="", choices=[""] + list(HETERO_MODELS),
                    help="导出异构零样本架构（FCN / MobileNetV3 / LRASPP），留空则为 DeepLabV3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.arch:
        model = build_hetero(args.arch, pretrained=True)
        arch_name, bb_name = args.arch, args.arch
    else:
        model = build_model(pretrained=True, aux_loss=True, backbone=args.backbone)
        arch_name, bb_name = "deeplabv3", args.backbone
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "args": {"model": arch_name, "backbone": bb_name,
                 "note": "torchvision COCO_WITH_VOC_LABELS_V1, zero-shot, never finetuned"},
        "epoch": -1,
        "best_miou": None,
    }
    torch.save(state, args.out)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"已导出 {arch_name} 零样本权重 -> {args.out}（{n:.1f} M 参数）")


if __name__ == "__main__":
    main()
