# -*- coding: utf-8 -*-
"""向已有的 val 概率缓存里**增量追加**一个成员（不重算其它 5 个）。

用途：P2 轻量微调产出新 ckpt 后，只想评估它作为集成第 6/第 5 成员的边际价值。
cache_val_probs.py 会把 5 个成员全部重算一遍（48 min），这里只算新增的那一个（约 10 min）。

用法：
    python cache_val_add.py --name p2_e0 --ckpt PATH\to\best.pt
之后用 sweep_val_cache.py --members zs_r50 zs_r101 sbd_lfca lfca_res r101_e13 p2_e0 扫描。
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import load_for_infer
from predict_ensemble import multi_scale_flip_probs
from voc_dataset import VOCSegmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--split", default="val")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", default=r"E:\tmp\valcache")
    ap.add_argument("--name", required=True, help="缓存子目录名，如 p2_e0")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    tdir = os.path.join(args.out_dir, "target")
    if not os.path.isdir(tdir) or not os.listdir(tdir):
        raise SystemExit(f"目标掩码缓存不存在：{tdir}，请先跑 cache_val_probs.py")

    ds = VOCSegmentation(args.voc_root, args.split, train=False, base_size=args.base_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        pin_memory=True)

    model, ck = load_for_infer(args.ckpt, device)
    model.eval()
    cfg = ck.get("args", {}) if isinstance(ck, dict) else {}
    print(f"成员 {args.name} <- {args.ckpt}", flush=True)
    print(f"  model={cfg.get('model')} backbone={cfg.get('backbone')} "
          f"split={cfg.get('train_split')} epoch={ck.get('epoch')}", flush=True)

    flip = not args.no_flip
    t0 = time.time()
    with torch.no_grad():
        for i, (x, target) in enumerate(loader):
            x = x.to(device)
            out_hw = (x.shape[-2], x.shape[-1])
            tpath = os.path.join(tdir, f"{i:04d}.npy")
            if not os.path.isfile(tpath):
                np.save(tpath, target.numpy().astype(np.int16))
            probs = multi_scale_flip_probs(model, x, out_hw, device,
                                           tuple(args.scales), flip, True)
            ph, pw = out_hw[0] // 2, out_hw[1] // 2
            probs = F.interpolate(probs, size=(ph, pw), mode="bilinear",
                                  align_corners=False)[0].cpu().numpy()
            np.save(os.path.join(out, f"{i:04d}.npy"), probs.astype(np.float16))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"完成：{len(ds)} 张 -> {out}（{(time.time() - t0) / 60:.1f} min）", flush=True)


if __name__ == "__main__":
    main()
