# -*- coding: utf-8 -*-
"""最终推理管线：多模型概率集成（+多尺度翻转 TTA）+ 可选 YOLOv8 实例融合。

这是提交用的"总装"脚本，把前面每一步的收益串成一条链：

    单模型
      └ + 翻转 TTA                （压边界噪声）
        └ + 多尺度 TTA            （多感受野投票）
          └ + 多模型概率集成       （互抵系统性偏差）
            └ + YOLO 实例融合      （补小目标 / 拆相邻实例，域适配 COCO）
              └ 输出单通道灰度 mask

每一步的单独增益都可以用 --mode val + 开关组合测出来，报告里的消融表就是这么来的。

用法
----
    # 全量 val 消融：只看集成
    python predict_final.py --mode val --ckpts A B C \
        --scales 0.5 0.75 1.0 1.25 1.5 --no_fusion

    # 全量 val：集成 + YOLO 融合
    python predict_final.py --mode val --ckpts A B C \
        --scales 0.5 0.75 1.0 1.25 1.5 --fusion

    # 出提交 mask
    python predict_final.py --mode test --ckpts A B C \
        --scales 0.5 0.75 1.0 1.25 1.5 --fusion \
        --out_dir ..\outputs\predict_final
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from metrics import ConfusionMatrix, format_table
from predict_ensemble import build_models, ensemble_probs
from voc_dataset import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, VOCSegmentation, VOCTestDataset


def denorm_to_uint8(x):
    """[1,3,H,W] 归一化张量 → (H,W,3) uint8 RGB，供 YOLO 使用（必须与语义分支同尺寸）。"""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    rgb = (x * std + mean).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (rgb * 255).astype(np.uint8)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--split", default="val")
    ap.add_argument("--test_dir", default=r"E:\桌面\人工智能技术大作业\data\test_images")
    ap.add_argument("--out_dir", default=r"E:\桌面\人工智能技术大作业\outputs\predict_final")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--fusion", action="store_true", help="启用 YOLOv8 实例融合")
    ap.add_argument("--yolo_weights", default="yolov8n-seg.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.35)
    ap.add_argument("--tau_sem", type=float, default=0.6)
    ap.add_argument("--rule_b", type=float, default=-1, help="<0 表示不启用规则 B")
    ap.add_argument("--num", type=int, default=0, help="val 只跑前 N 张")
    ap.add_argument("--limit", type=int, default=0, help="test 只跑前 N 张")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_txt", default="")
    ap.add_argument("--no_amp", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    weights = args.weights or [1.0] * len(args.ckpts)
    if len(weights) != len(args.ckpts):
        raise SystemExit("--weights 数量必须与 --ckpts 一致")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print(f"构建 {len(args.ckpts)} 个模型：", flush=True)
    models = build_models(args.ckpts, device)

    yolo = None
    if args.fusion:
        from yolo_fusion import YoloSeg, fuse
        yolo = YoloSeg(args.yolo_weights, conf=args.yolo_conf, device=str(device))
        print(f"YOLO 融合已启用：tau_sem={args.tau_sem} rule_b={args.rule_b} "
              f"yolo_conf={args.yolo_conf}", flush=True)
    else:
        fuse = None
        print("模式：纯语义集成（无 YOLO 融合）", flush=True)

    rule_b = None if args.rule_b < 0 else args.rule_b
    tag = "融合" if args.fusion else "纯集成"
    print(f"配置：{tag}，尺度 {args.scales}，翻转 {not args.no_flip}", flush=True)

    if args.mode == "val":
        ds = VOCSegmentation(args.voc_root, args.split, train=False, base_size=args.base_size)
        if args.num > 0:
            ds.ids = ds.ids[:args.num]
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
        cm = ConfusionMatrix(NUM_CLASSES)
        t0 = time.time()
        for i, (x, target) in enumerate(loader):
            x = x.to(device)
            probs = ensemble_probs(models, weights, x, (x.shape[-2], x.shape[-1]),
                                  device, tuple(args.scales), not args.no_flip, use_amp)
            p = probs[0].cpu().numpy()
            if yolo is not None:
                ylab, ycf = yolo(denorm_to_uint8(x), (x.shape[-2], x.shape[-1]))
                pred = fuse(p, ylab, ycf, args.tau_sem, args.yolo_conf, rule_b)
            else:
                pred = p.argmax(axis=0)
            cm.update(pred, target.numpy())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / 60:.1f} min", flush=True)

        table = format_table(cm.per_class_table())
        print(f"\n=== {tag} 评估（{args.split} {len(ds)} 张，{len(models)} 模型）===")
        print(f"mIoU (21 类含背景) : {cm.miou(True):.4f}")
        print(f"mIoU (20 类前景)   : {cm.miou(False):.4f}")
        print(f"Pixel Accuracy     : {cm.pixel_accuracy:.4f}")
        print(f"Mean Class Accuracy: {cm.mean_class_accuracy:.4f}")
        print("\n逐类 IoU：\n" + table)
        print(f"\n耗时 {(time.time() - t0) / 60:.1f} min")

        if args.save_txt:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_txt)), exist_ok=True)
            with open(args.save_txt, "w", encoding="utf-8") as f:
                f.write(f"mode={tag} split={args.split} n={len(ds)}\n")
                f.write(f"ckpts={args.ckpts}\nweights={weights}\n")
                f.write(f"scales={args.scales} flip={not args.no_flip} "
                        f"fusion={args.fusion} tau_sem={args.tau_sem} rule_b={rule_b}\n")
                f.write(f"mIoU21={cm.miou(True):.4f}  mIoU20={cm.miou(False):.4f}  "
                        f"pixAcc={cm.pixel_accuracy:.4f}  "
                        f"mAcc={cm.mean_class_accuracy:.4f}\n\n")
                f.write(table + "\n")
            print(f"已保存：{args.save_txt}")

    else:
        os.makedirs(args.out_dir, exist_ok=True)
        ds = VOCTestDataset(args.test_dir, base_size=args.base_size,
                            limit=args.limit or None)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        t0 = time.time()
        hist = np.zeros(NUM_CLASSES, dtype=np.int64)
        for i, (x, hw, names) in enumerate(loader):
            x = x.to(device)
            oh, ow = int(hw[0]), int(hw[1])
            probs = ensemble_probs(models, weights, x, (oh, ow), device,
                                  tuple(args.scales), not args.no_flip, use_amp)
            p = probs[0].cpu().numpy()
            if yolo is not None:
                # YOLO 在原图分辨率上检测更准；它内部会把 mask 缩放到 hw
                rgb = np.array(Image.open(os.path.join(args.test_dir, names[0]))
                               .convert("RGB"))
                ylab, ycf = yolo(rgb, (oh, ow))
                pred = fuse(p, ylab, ycf, args.tau_sem, args.yolo_conf, rule_b)
            else:
                pred = p.argmax(axis=0)
            pred = pred.astype(np.uint8)
            hist += np.bincount(pred.reshape(-1), minlength=NUM_CLASSES)
            stem = os.path.splitext(os.path.basename(names[0]))[0]
            Image.fromarray(pred, mode="L").save(os.path.join(args.out_dir, stem + ".png"))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / 60:.1f} min", flush=True)

        print(f"\n完成：{len(ds)} 张 -> {args.out_dir}（{(time.time() - t0) / 60:.1f} min）",
              flush=True)
        total = hist.sum()
        bg = hist[0] / max(total, 1)
        n_cls = int((hist > 0).sum())
        print(f"类别分布：background 占 {bg * 100:.2f}%，共出现 {n_cls} 个类别", flush=True)
        if bg > 0.995 or n_cls < 3:
            raise RuntimeError("预测结果异常（几乎全背景 / 类别过少），已中止")
        print("分布自检通过", flush=True)


if __name__ == "__main__":
    main()
