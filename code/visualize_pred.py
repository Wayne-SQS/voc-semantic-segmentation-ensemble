# -*- coding: utf-8 -*-
"""把预测 mask 上色并与原图左右拼接，人工肉眼质检（也是报告里的可视化素材）。

用法：
    python visualize_pred.py --mask_dir ..\outputs\predict --num 3
产出：outputs\vis\vis_序号.png（左=原图，中=预测彩色图，右=叠加图）
"""

import argparse
import os

import numpy as np
from PIL import Image

from voc_dataset import PALETTE


def colorize(mask_np):
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(PALETTE):
        out[mask_np == cid] = color
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default=r"E:\桌面\人工智能技术大作业\outputs\predict")
    ap.add_argument("--test_dir", default=r"E:\桌面\人工智能技术大作业\data\test_images")
    ap.add_argument("--out_dir", default=r"E:\桌面\人工智能技术大作业\outputs\vis")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--size", type=int, default=400, help="拼接图里每格的边长")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    masks = sorted([n for n in os.listdir(args.mask_dir) if n.endswith(".png")])[:args.num]
    for n in masks:
        stem = os.path.splitext(n)[0]
        img = Image.open(os.path.join(args.test_dir, stem + ".jpg")).convert("RGB")
        msk = np.array(Image.open(os.path.join(args.mask_dir, n)))
        color = colorize(msk)

        s = args.size
        def fit(im):
            return im.resize((s, s), Image.BILINEAR) if isinstance(im, Image.Image) \
                else Image.fromarray(im).resize((s, s), Image.NEAREST)
        img_s, color_s = fit(img), fit(color)
        overlay = Image.blend(img_s, color_s, alpha=0.55)

        canvas = Image.new("RGB", (s * 3 + 20, s))
        canvas.paste(img_s, (0, 0))
        canvas.paste(color_s, (s + 10, 0))
        canvas.paste(overlay, (2 * s + 20, 0))
        p = os.path.join(args.out_dir, "vis_" + stem + ".png")
        canvas.save(p)
        cls = sorted(np.unique(msk).tolist())
        print(f"{stem}: 出现类别 {cls} -> {p}")


if __name__ == "__main__":
    main()
