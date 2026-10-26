# -*- coding: utf-8 -*-
"""用引导滤波（guided filter）作为 DenseCRF 的廉价替代，细化集成概率图。

动机（可写进报告）：
    最终误差集中在两类——(1) 孤立误检： Chair/bicycle 上的零散像素被判成前景；
    (2) 边界毛糙。经典解法是 DenseCRF 的全连接条件随机场后处理，但
    `pydensecrf` 在 Python 3.11 / Windows 上没有可用轮子（已实测
    `No matching distribution found`），opencv 也未装 contrib。
    引导滤波（He et al., ECCV 2010）是公认的等价廉价替代：以原图灰度作为
    guidance，在概率图上拟合局部线性模型 q = a·I + b，再由 a、b 的盒滤波
    重构输出。效果与 CRF 的 smoothness 先验一致——在颜色一致的区域内平滑
    （去掉孤立误检），在颜色边界处不平滑（保留物体轮廓）。
    且它是线性算子：gf(c·p, eps) = c·gf(p, eps)，故权重和与归一化平均等价。

q = a·I + b，其中
    a = cov(I, p) / (var(I) + eps)
    b = mean(p) − a·mean(I)

防自欺：奇数号图像选 (r, eps)，偶数号图像验证，选择与评估严格分离。
口径：与 sweep_val_cache.py 一致（半分辨率 argmax + 最近邻上采样），
      仅用于筛选；若有效仍需 predict_ensemble.py 全精度复核。
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics import ConfusionMatrix
from voc_dataset import VOCSegmentation

MEMBERS = ["zs_r50", "zs_r101", "sbd_lfca", "r101_e13"]
WEIGHTS = [1.0, 1.25, 1.0, 1.0]
# (半径, eps)：eps 相对于 guidance 的局部方差（灰度归一化到 0~1 后方差约 0.03~0.08）
GRID = [(2, 1e-3), (4, 1e-3), (4, 1e-2), (8, 1e-2), (8, 1e-1), (16, 1e-1)]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def box(x, r):
    """盒滤波。count_include_pad=True 使边界处行为稳定（等价边缘延拓）。"""
    if r <= 0:
        return x
    return F.avg_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r,
                        count_include_pad=True)


def guided_filter(guide, p, r, eps):
    """guide: [1,h,w]（0~1）；p: [C,h,w]。返回同形状的细化概率图。"""
    mean_I = box(guide, r)
    mean_p = box(p, r)
    var_I = box(guide * guide, r) - mean_I * mean_I
    cov_Ip = box(guide * p, r) - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return box(a, r) * guide + box(b, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--cache_dir", default=r"E:\tmp\valcache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--out_txt", default=r"E:\桌面\人工智能技术大作业\outputs\guided_filter.txt")
    args = ap.parse_args()

    device = torch.device("cuda")
    n_img = len(os.listdir(os.path.join(args.cache_dir, "target")))
    fit_idx = set(range(1, n_img, 2))
    hold_idx = set(range(0, n_img, 2))

    tasks = [("base(无滤波)", 0, 0.0)]
    tasks += [(f"gf r={r} eps={eps:g}", r, eps) for r, eps in GRID]

    ds = VOCSegmentation(args.voc_root, args.split, train=False,
                         base_size=args.base_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        pin_memory=True)
    assert len(ds) == n_img, f"缓存 {n_img} 张与数据集 {len(ds)} 张不一致"

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"数据集 {n_img} 张；拟合（奇数）{len(fit_idx)} / 验证（偶数）{len(hold_idx)}")
    log(f"成员 {' + '.join(MEMBERS)}，权重 {WEIGHTS}")
    log(f"guidance = ImageNet 反归一化后的灰度图，缩放到半分辨率网格")

    # 两轮：先只在奇数半上评估全部配置，再在偶数半上验证 base + 奇数 top2
    fit_cms = [ConfusionMatrix() for _ in tasks]
    hold_cms = [ConfusionMatrix() for _ in range(3)]  # base + top2
    t0 = time.time()
    with torch.no_grad():
        for i, (x, target) in enumerate(loader):
            target = target.numpy()
            tshape = target.shape[-2:]
            acc = None
            for m, w in zip(MEMBERS, WEIGHTS):
                p = np.load(os.path.join(args.cache_dir, m, f"{i:04d}.npy")).astype(np.float32)
                p = p * w
                acc = p if acc is None else acc + p
            maps = torch.from_numpy(acc).unsqueeze(0).to(device, non_blocking=True)

            rgb = x[0].to(device, non_blocking=True) * STD.to(device) + MEAN.to(device)
            gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
            guide = F.interpolate(gray[None, None], size=maps.shape[-2:],
                                  mode="bilinear", align_corners=False)
            guide = guide.clamp(0.0, 1.0)

            is_fit, is_hold = i in fit_idx, i in hold_idx
            for ci, (name, r, eps) in enumerate(tasks):
                q = maps if r == 0 else guided_filter(guide, maps, r, eps)
                pred = q[0].argmax(0).to(torch.uint8).cpu().numpy()
                if pred.shape != tshape:
                    pred = F.interpolate(
                        torch.from_numpy(pred[None, None]).float(), size=tshape,
                        mode="nearest")[0, 0].numpy().astype(np.uint8)
                if is_fit:
                    fit_cms[ci].update(pred, target)
                if is_hold and ci < 3:
                    hold_cms[ci].update(pred, target)
            del maps, acc
            if (i + 1) % 200 == 0:
                log(f"  {i + 1}/{n_img}  {(time.time() - t0) / 60:.1f} min")

    log("\n=== 阶段 1：奇数号图像（724 张）上选参 ===")
    fit_res = [(cm.miou(True), cm.miou(False), cm.pixel_accuracy) for cm in fit_cms]
    order = sorted(range(len(tasks)), key=lambda k: -fit_res[k][0])
    for k in order:
        m21, m20, pa = fit_res[k]
        log(f"  {m21:.4f}  (mIoU20 {m20:.4f} / pixAcc {pa:.4f})  {tasks[k][0]}")

    top2 = [k for k in order if k != 0][:2]
    # 上面 hold_cms 只覆盖 ci<3，因此要求 base + top2 恰好是前三个任务
    assert tasks[0][0].startswith("base") and all(k < 3 for k in top2), \
        f"top2 索引 {top2} 超出 held-out 覆盖范围，需调整 GRID 长度"
    labels = [tasks[k][0] for k in [0] + top2]

    log("\n=== 阶段 2：偶数号图像（725 张，held-out）验证 ===")
    hold_res = [(cm.miou(True), cm.miou(False), cm.pixel_accuracy) for cm in hold_cms]
    base21 = hold_res[0][0]
    best_i, best21 = 0, base21
    for k, (lab, (m21, m20, pa)) in enumerate(zip(labels, hold_res)):
        log(f"  {m21:.4f}  (Δ{m21 - base21:+.4f} / mIoU20 {m20:.4f})  {lab}")
        if m21 > best21:
            best_i, best21 = k, m21

    gain = best21 - base21
    log(f"\nheld-out baseline = {base21:.4f}")
    log(f"held-out 最优 = {best21:.4f}（{labels[best_i]}），Δ{gain:+.4f}")
    log("⇒ 有实质收益，值得全精度复核" if gain >= 0.002 else "⇒ 无实质收益，不采用")

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存：{args.out_txt}", flush=True)


if __name__ == "__main__":
    main()
