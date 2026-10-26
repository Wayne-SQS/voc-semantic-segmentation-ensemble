# -*- coding: utf-8 -*-
"""多模型概率集成推理（叠加多尺度 + 翻转 TTA）。

思路
----
单模型的误差在两个层面：系统性偏差（某类永远分不清）+ 随机噪声（边界抖动）。
TTA 只能压噪声；换架构 / 换训练数据 / 换初始化的模型，错的地方不完全重合，
概率平均后系统性偏差会被互相抵消一部分——这就是集成通常比单模型稳的原因。

为什么用 **概率平均** 而不是 logits 平均：
    logits 尺度因模型而异（不同 run 的权重范数不同），直接平均会被"话说得大声"的
    模型主导；softmax 之后各模型被归一化到同一概率单纯形上，投票才公平。

为什么这几个模型有互补性：
    base_r50_e30      纯 DeepLabV3 原版解码器，只用 VOC 1464 张
    lfca_res_e30      自研 LFCA 解码器（低层融合 + SE 注意力），只用 VOC 1464 张
    sbd_lfca_e30      LFCA + SBD 增强数据 8284 张
    三者在"解码头结构"和"见过的数据"两个维度上都不同，不是简单的同款复制。

用法
----
验证集评估（全量 1449 张，报告取数用）：
    python predict_ensemble.py --mode val \
        --ckpts ..\runs\base_r50_e30\ckpt\best.pt ^
                ..\runs\lfca_res_e30\ckpt\best.pt ^
                ..\runs\sbd_lfca_e30\ckpt\best.pt \
        --scales 0.5 0.75 1.0 1.25 1.5 --save_txt ..\outputs\ensemble_per_class.txt

生成提交用 mask（400 张）：
    python predict_ensemble.py --mode test --out_dir ..\outputs\predict_ensemble ...
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from metrics import ConfusionMatrix, format_table
from model import load_for_infer
from voc_dataset import NUM_CLASSES, VOCSegmentation, VOCTestDataset


@torch.no_grad()
def multi_scale_flip_probs(model, x, out_hw, device, scales=(1.0,), flip=True, use_amp=True):
    """单个模型在 (多尺度 × 翻转) 上的平均 softmax 概率，输出尺寸固定为 out_hw。

    返回 [1, C, H, W] float32。flip 分支记得把 logits 翻回来再进 softmax，
    否则相当于拿镜像预测去和原图对齐——位置全错。
    """
    h, w = out_hw
    probs = torch.zeros((1, NUM_CLASSES, h, w), device=device, dtype=torch.float32)
    n = 0
    for s in scales:
        if s != 1.0:
            hs = max(int(round(x.shape[-2] * s)), 16)
            ws = max(int(round(x.shape[-1] * s)), 16)
            xs = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
        else:
            xs = x
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xs)["out"]
        probs += F.interpolate(logits.float(), size=(h, w), mode="bilinear",
                               align_corners=False).softmax(dim=1)
        n += 1
        if flip:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_f = model(torch.flip(xs, dims=[-1]))["out"]
            logits_f = F.interpolate(logits_f.float(), size=(h, w), mode="bilinear",
                                     align_corners=False)
            probs += torch.flip(logits_f, dims=[-1]).softmax(dim=1)
            n += 1
    return probs / n


@torch.no_grad()
def ensemble_probs(models, weights, x, out_hw, device, scales, flip, use_amp):
    """多模型加权概率平均。返回 [1, C, H, W]。"""
    total = None
    wsum = 0.0
    for m, w in zip(models, weights):
        p = multi_scale_flip_probs(m, x, out_hw, device, scales, flip, use_amp)
        total = p * w if total is None else total + p * w
        wsum += w
    return total / wsum


def build_models(ckpt_paths, device):
    models = []
    for p in ckpt_paths:
        m, ck = load_for_infer(p, device)
        arch = (ck.get("args", {}) or {}).get("model", "deeplabv3")
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        print(f"  加载 {run:<16s} arch={arch:<12s} {os.path.basename(p)}", flush=True)
        models.append(m)
    return models


def run_val(models, weights, args, device, use_amp):
    ds = VOCSegmentation(args.voc_root, args.split, train=False, base_size=args.base_size)
    if args.num > 0:
        ds.ids = ds.ids[:args.num]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    cm = ConfusionMatrix(NUM_CLASSES)
    t0 = time.time()
    for i, (x, target) in enumerate(loader):
        x = x.to(device)
        out_hw = (x.shape[-2], x.shape[-1])
        probs = ensemble_probs(models, weights, x, out_hw, device,
                              tuple(args.scales), not args.no_flip, use_amp)
        pred = probs.argmax(dim=1)[0].cpu().numpy()
        cm.update(pred, target.numpy() if torch.is_tensor(target) else target)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    table = format_table(cm.per_class_table())
    print(f"\n=== 集成评估（{args.split} 集 {len(ds)} 张，{len(models)} 个模型，"
          f"尺度 {args.scales}，翻转 {not args.no_flip}）===")
    print(f"mIoU (21 类含背景) : {cm.miou(True):.4f}")
    print(f"mIoU (20 类前景)   : {cm.miou(False):.4f}")
    print(f"Pixel Accuracy     : {cm.pixel_accuracy:.4f}")
    print(f"Mean Class Accuracy: {cm.mean_class_accuracy:.4f}")
    print("\n逐类 IoU：")
    print(table)
    print(f"\n耗时 {(time.time() - t0) / 60:.1f} min")

    if args.save_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_txt)), exist_ok=True)
        with open(args.save_txt, "w", encoding="utf-8") as f:
            f.write(f"mode=ensemble split={args.split} n={len(ds)}\n")
            f.write(f"ckpts={args.ckpts}\nweights={weights}\n")
            f.write(f"scales={args.scales} flip={not args.no_flip}\n")
            f.write(f"mIoU21={cm.miou(True):.4f}  mIoU20={cm.miou(False):.4f}  "
                    f"pixAcc={cm.pixel_accuracy:.4f}  mAcc={cm.mean_class_accuracy:.4f}\n\n")
            f.write(table + "\n")
        print(f"已保存：{args.save_txt}")

    return cm


def run_test(models, weights, args, device, use_amp):
    os.makedirs(args.out_dir, exist_ok=True)
    ds = VOCTestDataset(args.test_dir, base_size=args.base_size, limit=args.limit or None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    t0 = time.time()
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i, (x, hw, names) in enumerate(loader):
        x = x.to(device)
        oh, ow = int(hw[0]), int(hw[1])
        probs = ensemble_probs(models, weights, x, (oh, ow), device,
                               tuple(args.scales), not args.no_flip, use_amp)
        pred = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        hist += np.bincount(pred.reshape(-1), minlength=NUM_CLASSES)
        stem = os.path.splitext(os.path.basename(names[0]))[0]
        Image.fromarray(pred, mode="L").save(os.path.join(args.out_dir, stem + ".png"))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    print(f"完成：{len(ds)} 张 -> {args.out_dir}（{(time.time() - t0) / 60:.1f} min）",
          flush=True)
    total = hist.sum()
    bg = hist[0] / max(total, 1)
    n_cls = int((hist > 0).sum())
    print(f"类别分布：background 占 {bg * 100:.2f}%，共出现 {n_cls} 个类别", flush=True)
    if bg > 0.995 or n_cls < 3:
        raise RuntimeError("预测结果异常（几乎全背景 / 类别过少），很可能权重未正确加载，已中止")
    print("分布自检通过", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="与 ckpts 等长的权重，默认全 1")
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--split", default="val")
    ap.add_argument("--test_dir", default=r"E:\桌面\人工智能技术大作业\data\test_images")
    ap.add_argument("--out_dir", default=r"E:\桌面\人工智能技术大作业\outputs\predict_ensemble")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--num", type=int, default=0, help="val 只跑前 N 张，0=全量")
    ap.add_argument("--limit", type=int, default=0, help="test 只跑前 N 张")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_txt", default="")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.ckpts)
    if len(weights) != len(args.ckpts):
        raise SystemExit("--weights 数量必须与 --ckpts 一致")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"构建 {len(args.ckpts)} 个模型：", flush=True)
    models = build_models(args.ckpts, device)

    if args.mode == "val":
        run_val(models, weights, args, device, use_amp)
    else:
        run_test(models, weights, args, device, use_amp)


if __name__ == "__main__":
    main()
