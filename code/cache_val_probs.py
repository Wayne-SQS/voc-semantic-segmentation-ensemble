# -*- coding: utf-8 -*-
"""缓存 VOC val 全量上各候选成员的 TTA softmax 概率图（半分辨率 float16），
供后续瞬时扫描任意子集组合 / 权重组合，避免反复跑 GPU。

用法：
    python cache_val_probs.py --out_dir E:\tmp\valcache
成员清单与 predict_ensemble.py 的 --ckpts 一致；目标掩码只存一份（按索引对齐，
dataset shuffle=False，顺序确定）。
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
from voc_dataset import NUM_CLASSES, VOCSegmentation

MEMBERS = [
    ("zs_r50",    r"E:\桌面\人工智能技术大作业\runs\pretrain_zs_r50\ckpt\zeroshot.pt"),
    ("zs_r101",   r"E:\桌面\人工智能技术大作业\runs\pretrain_zs_r101\ckpt\zeroshot.pt"),
    ("sbd_lfca",  r"E:\桌面\人工智能技术大作业\runs\sbd_lfca_e30\ckpt\best.pt"),
    ("lfca_res",  r"E:\桌面\人工智能技术大作业\runs\lfca_res_e30\ckpt\best.pt"),
    ("r101_e13",  r"E:\桌面\人工智能技术大作业\runs\sbd_lfca_r101_e45\ckpt\last.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--split", default="val")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", default=r"E:\tmp\valcache")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "target"), exist_ok=True)

    ds = VOCSegmentation(args.voc_root, args.split, train=False, base_size=args.base_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        pin_memory=True)

    # 全部模型同时驻留显存（合计约 1.5GB，8GB 卡无压力）
    models = []
    for name, path in MEMBERS:
        m, _ = load_for_infer(path, device)
        m.eval()
        models.append(m)
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)
        print(f"  加载 {name}", flush=True)

    flip = not args.no_flip
    t0 = time.time()
    with torch.no_grad():
        for i, (x, target) in enumerate(loader):
            x = x.to(device)
            out_hw = (x.shape[-2], x.shape[-1])
            # 目标掩码（全分辨率）只由第一遍保存，后续校验一致后跳过
            tpath = os.path.join(args.out_dir, "target", f"{i:04d}.npy")
            t_np = target.numpy()
            if not os.path.isfile(tpath):
                np.save(tpath, t_np.astype(np.int16))
            for name, m in zip([n for n, _ in MEMBERS], models):
                probs = multi_scale_flip_probs(m, x, out_hw, device,
                                               tuple(args.scales), flip, True)
                # 降半分辨率：ranking 用，最终配置再用全精度实测校验
                ph, pw = out_hw[0] // 2, out_hw[1] // 2
                probs = F.interpolate(probs, size=(ph, pw), mode="bilinear",
                                      align_corners=False)[0].cpu().numpy()
                np.save(os.path.join(args.out_dir, name, f"{i:04d}.npy"),
                        probs.astype(np.float16))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"完成：{len(ds)} 张 × {len(models)} 个成员 -> {args.out_dir}"
          f"（{(time.time() - t0) / 60:.1f} min）", flush=True)


if __name__ == "__main__":
    main()
