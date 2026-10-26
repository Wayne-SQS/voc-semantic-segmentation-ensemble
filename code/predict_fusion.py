# -*- coding: utf-8 -*-
"""最终推理：语义模型 5 尺度 TTA + YOLOv8-seg 实例融合，输出 400 张单通道 mask。

在 VOC val 子集上实测：TTA-only mIoU≈0.7774 → TTA+fusion≈0.7879（+1.05），故采用。
融合参数取消融最优：tau_sem=0.6, rule_b=None（YOLO 规则 A 补漏，不启用去伪规则 B）。
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from model import load_for_infer
from voc_dataset import NUM_CLASSES, VOCTestDataset
from yolo_fusion import YoloSeg, fuse


@torch.no_grad()
def tta_probs(model, x, orig_hw, device, scales, flip=True, use_amp=True):
    """x:[1,3,H,W] 归一化；返回原图尺寸的平均 softmax 概率 (21,H,W) numpy。"""
    h, w = orig_hw
    probs = torch.zeros((NUM_CLASSES, h, w), device=device, dtype=torch.float32)
    n = 0
    for s in scales:
        xs = x if s == 1.0 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xs)["out"]
        logits = F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)[0]
        probs += logits.softmax(dim=0); n += 1
        if flip:
            with torch.amp.autocast("cuda", enabled=use_amp):
                lf = model(torch.flip(xs, dims=[-1]))["out"]
            lf = F.interpolate(lf.float(), size=(h, w), mode="bilinear", align_corners=False)[0]
            probs += torch.flip(lf, dims=[-1]).softmax(dim=0); n += 1
    return (probs / n).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", default=r"E:\桌面\人工智能技术大作业\data\test_images")
    ap.add_argument("--out_dir", default=r"E:\桌面\人工智能技术大作业\outputs\predict")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--no_fusion", action="store_true", help="消融用：纯 TTA 不融合")
    ap.add_argument("--tau_sem", type=float, default=0.6)
    ap.add_argument("--rule_b", type=float, default=None)
    ap.add_argument("--yolo", default="yolov8n-seg.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    model, _ = load_for_infer(args.ckpt, device)
    print("加载权重:", args.ckpt, "| strict 通过", flush=True)
    yolo = None if args.no_fusion else YoloSeg(args.yolo, conf=args.yolo_conf, device=str(device))
    if yolo is None:
        print("模式：纯 TTA（无融合）", flush=True)
    else:
        print(f"模式：TTA + YOLO 融合  tau_sem={args.tau_sem} rule_b={args.rule_b} yolo_conf={args.yolo_conf}", flush=True)

    ds = VOCTestDataset(args.test_dir, base_size=args.base_size, limit=args.limit or None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    t0 = time.time()
    for i, (x, hw, names) in enumerate(loader):
        x = x.to(device)
        prob = tta_probs(model, x, hw, device, tuple(args.scales),
                         flip=not args.no_flip, use_amp=use_amp)
        if yolo is None:
            pred = prob.argmax(0).astype(np.uint8)
        else:
            rgb = np.array(Image.open(os.path.join(args.test_dir, names[0])).convert("RGB"))
            ylab, ycf = yolo(rgb, hw)
            pred = fuse(prob, ylab, ycf, args.tau_sem,
                        tau_yolo=args.yolo_conf, rule_b_tau=args.rule_b).astype(np.uint8)
        hist += np.bincount(pred.reshape(-1), minlength=NUM_CLASSES)
        stem = os.path.splitext(os.path.basename(names[0]))[0]
        Image.fromarray(pred, mode="L").save(os.path.join(args.out_dir, stem + ".png"))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)}  {time.time()-t0:.0f}s", flush=True)

    print(f"完成：{len(ds)} 张 -> {args.out_dir}（{time.time()-t0:.0f}s）", flush=True)
    total = hist.sum(); bg = hist[0] / max(total, 1); ncls = int((hist > 0).sum())
    print(f"类别分布：background 占 {bg*100:.2f}%，共 {ncls} 个类别", flush=True)
    if bg > 0.995 or ncls < 3:
        raise RuntimeError("预测异常（几乎全背景/类别过少），可能权重没加载对")
    print("分布自检通过", flush=True)


if __name__ == "__main__":
    main()
