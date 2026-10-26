# -*- coding: utf-8 -*-
"""把 SBD (Semantic Boundaries Dataset) 的 .mat 语义标注转为 VOC 格式 mask，
并生成增强训练清单 train_aug.txt。

关键：严格剔除与 VOC val 泄漏的 545 个 ID，避免验证集污染。
SBD GTcls.Segmentation 取值 0..20，与 VOC 0..20 直接对应（已验证 98.4% 一致），
无需偏移；0=背景，无 void(255)。
"""
import os, scipy.io, numpy as np
from PIL import Image
import time

SBD = r"E:\桌面\人工智能技术大作业\data\sbd/benchmark_RELEASE/dataset"
VOC = r"E:\桌面\人工智能技术大作业\data/VOCdevkit/VOC2012"
OUT_MASK = r"E:\桌面\人工智能技术大作业\data\sbd_masks"  # 生成掩码落盘处

def read_ids(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]

sbd_train = read_ids(os.path.join(SBD, "train.txt"))
voc_val = set(read_ids(os.path.join(VOC, "ImageSets/Segmentation/val.txt")))
voc_train = set(read_ids(os.path.join(VOC, "ImageSets/Segmentation/train.txt")))

# 剔除泄漏到 VOC val 的 ID
sbd_clean = [x for x in sbd_train if x not in voc_val]
leaked = len(sbd_train) - len(sbd_clean)
print(f"SBD train={len(sbd_train)}  泄漏到 VOC val 已剔除={leaked}  清洗后={len(sbd_clean)}")

os.makedirs(OUT_MASK, exist_ok=True)

t0 = time.time()
done = 0
for i, imgid in enumerate(sbd_clean):
    m = scipy.io.loadmat(os.path.join(SBD, "cls", imgid + ".mat"))
    seg = np.asarray(m["GTcls"][0, 0]["Segmentation"])
    if seg.dtype == object:
        seg = seg.item()
    seg = seg.astype(np.uint8)  # 0..20
    Image.fromarray(seg, mode="L").save(os.path.join(OUT_MASK, imgid + ".png"))
    done += 1
    if (i + 1) % 1000 == 0:
        print(f"  {done}/{len(sbd_clean)}  ({time.time()-t0:.0f}s)")
print(f"masks 生成完毕: {done} 张 -> {OUT_MASK}  ({time.time()-t0:.0f}s)")

# 生成 train_aug.txt（去重并集）
train_aug = sorted(set(voc_train) | set(sbd_clean))
with open(os.path.join(VOC, "ImageSets/Segmentation/train_aug.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(train_aug) + "\n")
print(f"train_aug.txt 写入: {len(train_aug)} 张 (VOCtrain {len(voc_train)} + SBD新增 {len(set(sbd_clean)-voc_train)})")
