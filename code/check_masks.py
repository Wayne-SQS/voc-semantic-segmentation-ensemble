# -*- coding: utf-8 -*-
"""提交前自检：确保 400 张预测 mask 完全符合"单通道灰度图"要求。

检查项：
    1. 数量 == 测试集图像数
    2. 文件名与输入图一一对应（扩展名 .png）
    3. PIL mode == 'L'（单通道 8-bit，不是 P 调色板、不是 RGB）
    4. 尺寸 == 对应原图尺寸
    5. 像素值全部落在 0..20（21 类），不出现 255
    6. 输出各类别像素占比，用于 sanity check（有没有全背景的废图）

用法：
    python check_masks.py --mask_dir outputs/predict --test_dir data/test_images
"""

import argparse
import os

import numpy as np
from PIL import Image

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default=r"E:\桌面\人工智能技术大作业\outputs\predict")
    ap.add_argument("--test_dir", default=r"E:\桌面\人工智能技术大作业\data\test_images")
    args = ap.parse_args()

    imgs = sorted([n for n in os.listdir(args.test_dir)
                   if n.lower().endswith((".jpg", ".jpeg", ".png"))])
    masks = sorted([n for n in os.listdir(args.mask_dir) if n.lower().endswith(".png")])
    print(f"图像 {len(imgs)} 张 | mask {len(masks)} 张")

    problems = []
    if len(imgs) != len(masks):
        problems.append(f"数量不一致：{len(imgs)} vs {len(masks)}")

    stem2img = {os.path.splitext(n)[0]: n for n in imgs}
    hist = np.zeros(21, dtype=np.int64)
    for m in masks:
        stem = os.path.splitext(m)[0]
        if stem not in stem2img:
            problems.append(f"无对应原图: {m}")
            continue
        img = Image.open(os.path.join(args.test_dir, stem2img[stem]))
        msk = Image.open(os.path.join(args.mask_dir, m))
        if msk.mode != "L":
            problems.append(f"{m}: mode={msk.mode}（应为 L）")
        if img.size != msk.size:
            problems.append(f"{m}: 尺寸 {msk.size} != 原图 {img.size}")
        arr = np.array(msk)
        vals = np.unique(arr)
        if vals.max() > 20 or vals.min() < 0:
            problems.append(f"{m}: 像素值越界 {vals.tolist()}")
        hist += np.bincount(arr.reshape(-1), minlength=21)

    total = hist.sum()
    print("\n各类别像素占比：")
    for i, name in enumerate(VOC_CLASSES):
        if hist[i] > 0:
            print(f"  {i:>2} {name:<12} {hist[i] / total * 100:6.2f}%")

    if problems:
        print(f"\n❌ 发现 {len(problems)} 个问题：")
        for p in problems[:30]:
            print("   -", p)
    else:
        print("\n✅ 全部检查通过：数量/文件名/单通道/尺寸/值域均合规")


if __name__ == "__main__":
    main()
