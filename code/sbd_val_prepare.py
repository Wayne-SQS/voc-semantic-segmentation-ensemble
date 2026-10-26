# -*- coding: utf-8 -*-
"""准备「SBD 官方 val split」作为独立标注评估集（泛化诊断用）。

背景：train_aug 只用了 SBD 的 train.txt（8498，剔除 VOC val 泄漏后 7953）；
SBD 官方 val（2857 张）从未参与任何训练与选型，是一份干净的独立标注数据。
用途：若某集成配置只在 VOC val 上涨、在 SBD val 上掉，即为过拟合 VOC val 的信号
（用户明确提示老师会在本数据集之外测试，泛化性优先于刷分）。

处理：
  1) 读 SBD val.txt，剔除与 VOC train / VOC val 重叠的 ID（避免“见过”的数据）
  2) 转换 GTcls.Segmentation -> data/sbd_masks/<id>.png（与训练侧同一套映射）
  3) 写 split 清单 ImageSets/Segmentation/sbdval.txt
     （VOCSegmentation 对 VOC 缺失的 id 自动回退 SBD 图像/掩码目录）
"""
import os

import numpy as np
import scipy.io
from PIL import Image

SBD = r"E:\桌面\人工智能技术大作业\data\sbd\benchmark_RELEASE\dataset"
VOC = r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012"
OUT_MASK = r"E:\桌面\人工智能技术大作业\data\sbd_masks"


def read_ids(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def main():
    sbd_val = read_ids(os.path.join(SBD, "val.txt"))
    voc_val = set(read_ids(os.path.join(VOC, "ImageSets/Segmentation/val.txt")))
    voc_train = set(read_ids(os.path.join(VOC, "ImageSets/Segmentation/train.txt")))
    seen = voc_val | voc_train
    clean = [x for x in sbd_val if x not in seen]
    print(f"SBD val={len(sbd_val)}  与 VOC train/val 重叠剔除={len(sbd_val) - len(clean)}"
          f"  可用={len(clean)}")

    os.makedirs(OUT_MASK, exist_ok=True)
    n_new = 0
    for i, imgid in enumerate(clean):
        dst = os.path.join(OUT_MASK, imgid + ".png")
        if not os.path.isfile(dst):
            m = scipy.io.loadmat(os.path.join(SBD, "cls", imgid + ".mat"))
            seg = np.asarray(m["GTcls"][0, 0]["Segmentation"])
            if seg.dtype == object:
                seg = seg.item()
            Image.fromarray(seg.astype(np.uint8), mode="L").save(dst)
            n_new += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(clean)}", flush=True)
    print(f"新转换 mask: {n_new} 张（已有缓存 {len(clean) - n_new} 张）")

    with open(os.path.join(VOC, "ImageSets/Segmentation/sbdval.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(clean) + "\n")
    print(f"sbdval.txt 写入: {len(clean)} 张")


if __name__ == "__main__":
    main()
