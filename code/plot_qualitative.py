# -*- coding: utf-8 -*-
"""从最终 mask 中挑选类别丰富的样本，拼成 2×3 定性对比图（报告用）。
每格 = 原图 | 预测彩色图 | 叠加图；挑选依据：出现类别数最多的 6 张
（域偏移重灾区 person/diningtable 优先展示）。"""
import os

import numpy as np
from PIL import Image

from voc_dataset import PALETTE

MASK_DIR = r"E:\桌面\人工智能技术大作业\outputs\predict_final"
TEST_DIR = r"E:\桌面\人工智能技术大作业\data\test_images"
OUT = r"E:\桌面\人工智能技术大作业\outputs\figures\test_qualitative.png"


def colorize(mask_np):
    out = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cid, color in enumerate(PALETTE):
        out[mask_np == cid] = color
    return out


def main():
    names = sorted(n for n in os.listdir(MASK_DIR) if n.endswith(".png"))
    stats = []
    for n in names:
        m = np.array(Image.open(os.path.join(MASK_DIR, n)))
        ncls = len(np.unique(m))
        npx = m.size
        stats.append((n, ncls, npx))
    # 类别数降序，其次像素量降序（大图细节多）
    stats.sort(key=lambda x: (-x[1], -x[2]))
    picked = [s[0] for s in stats[:6]]
    print("挑选（按类别数）:")
    for n, c, _ in stats[:6]:
        print(f"  {n}: {c} 类")

    S = 320  # 每格边长
    GAP = 8
    W, H = S * 3 + GAP * 2, S
    canvas = Image.new("RGB", (W * 2 + GAP, H * 3 + GAP * 2), "white")
    for k, n in enumerate(picked):
        stem = os.path.splitext(n)[0]
        img = Image.open(os.path.join(TEST_DIR, stem + ".jpg")).convert("RGB")
        msk = np.array(Image.open(os.path.join(MASK_DIR, n)))
        color = colorize(msk)
        img_s = img.resize((S, S), Image.BILINEAR)
        color_s = Image.fromarray(color).resize((S, S), Image.NEAREST)
        overlay = Image.blend(img_s, color_s, alpha=0.55)
        r, c = divmod(k, 2)
        x0, y0 = c * (W + GAP), r * (H + GAP)
        canvas.paste(img_s, (x0, y0))
        canvas.paste(color_s, (x0 + S + GAP, y0))
        canvas.paste(overlay, (x0 + 2 * (S + GAP), y0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
