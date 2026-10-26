# -*- coding: utf-8 -*-
"""一次性遍历多个 τ_sem，找出集成语义概率下的融合最优阈值。

为什么必须重调：
    τ_sem=0.6 这个值是在**单模型 TTA** 的概率上网格搜索出来的。
    换成三模型概率平均后，softmax 的峰值被拉平（平均会让分布更接近均匀），
    maxp 整体下移，`maxp < τ_sem` 这个触发条件会变得更容易满足 ——
    同一个 0.6 在集成上等于"更激进地让 YOLO 覆盖"，很可能不是最优。

为什么能一次跑完：
    语义概率与 YOLO 结果与 τ 无关，只有最后一步门限依赖 τ。
    因此每张图算一次 (probs, ylab, ycf)，就能同时累计所有 τ 的混淆矩阵，
    一个 pass 出全部曲线，不必为每个 τ 重跑一遍模型（那是 9 倍的浪费）。

用法：
    python tune_fusion.py --scales 0.5 0.75 1.0 1.25 1.5 \
        --ckpts A B C --taus 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import ConfusionMatrix
from predict_ensemble import build_models, ensemble_probs
from predict_final import denorm_to_uint8
from voc_dataset import NUM_CLASSES, VOCSegmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--split", default="val")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--yolo_weights", default="yolov8n-seg.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.35)
    ap.add_argument("--num", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_txt", default="")
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.ckpts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"构建 {len(args.ckpts)} 个模型：", flush=True)
    models = build_models(args.ckpts, device)

    from yolo_fusion import YoloSeg, fuse
    yolo = YoloSeg(args.yolo_weights, conf=args.yolo_conf, device=str(device))

    ds = VOCSegmentation(args.voc_root, args.split, train=False, base_size=args.base_size)
    if args.num > 0:
        ds.ids = ds.ids[:args.num]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        pin_memory=True)

    cms = {t: ConfusionMatrix(NUM_CLASSES) for t in args.taus}
    cm_no = ConfusionMatrix(NUM_CLASSES)   # 不融合的对照

    t0 = time.time()
    for i, (x, target) in enumerate(loader):
        x = x.to(device)
        hw = (x.shape[-2], x.shape[-1])
        probs = ensemble_probs(models, weights, x, hw, device, tuple(args.scales),
                              not args.no_flip, True)
        p = probs[0].cpu().numpy()
        tgt = target.numpy()
        cm_no.update(p.argmax(axis=0), tgt)

        ylab, ycf = yolo(denorm_to_uint8(x), hw)
        for t in args.taus:
            cms[t].update(fuse(p, ylab, ycf, t, args.yolo_conf, None), tgt)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    print(f"\n=== τ_sem 网格搜索（{args.split} {len(ds)} 张，仅规则 A）===")
    base = cm_no.miou(True)
    print(f"不融合          : {base:.4f}")
    print(f"{'τ_sem':>6s}  {'mIoU21':>8s}  {'Δ':>8s}")
    print("-" * 28)
    rows = []
    for t in args.taus:
        v = cms[t].miou(True)
        rows.append((t, v, v - base))
        print(f"{t:>6.2f}  {v:>8.4f}  {v - base:>+8.4f}")

    best = max(rows, key=lambda r: r[1])
    print(f"\n最优 τ_sem = {best[0]:.2f}，mIoU21 = {best[1]:.4f}（相对不融合 {best[2]:+.4f}）")

    if args.save_txt:
        with open(args.save_txt, "w", encoding="utf-8") as f:
            f.write(f"split={args.split} n={len(ds)} scales={args.scales} "
                    f"flip={not args.no_flip}\n")
            f.write(f"ckpts={args.ckpts}\nno_fusion mIoU21={base:.4f}\n\n")
            f.write("tau_sem,mIoU21,delta\n")
            for t, v, d in rows:
                f.write(f"{t:.2f},{v:.4f},{d:+.4f}\n")
            f.write(f"\nbest tau={best[0]:.2f} mIoU21={best[1]:.4f}\n")
        print(f"已保存：{args.save_txt}")


if __name__ == "__main__":
    main()
