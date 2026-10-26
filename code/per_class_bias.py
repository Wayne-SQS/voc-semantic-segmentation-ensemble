# -*- coding: utf-8 -*-
"""类别先验校正（per-class logit bias）+ A/B 半集交叉验证。

动机：mIoU 是 21 类等权平均，当前弱类 bicycle 0.433 / chair 0.516 / diningtable 0.648
把总分拖得很低。给每个类加一个 log 域偏置 b_c（等价于先验重加权 P_c*exp(b_c)），
可以提升弱类召回；但盲目调会过拟合 val。

严谨做法：奇数索引拟合 b_c，偶数索引评估（held-out），报告后者。
拟合在 1/4 分辨率上加速，最终验证回到统一的半分辨率口径。
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from metrics import ConfusionMatrix
from voc_dataset import NUM_CLASSES

CACHE = r"E:\tmp\valcache"
MEMBERS = ["zs_r50", "zs_r101", "sbd_lfca", "r101_e13"]
WEIGHTS = [1.0, 1.25, 1.0, 1.0]
CAND = (0.0, -0.15, 0.15, -0.3, 0.3, -0.6, 0.6, -1.0, 1.0)


def load_wsum(i):
    """4 成员加权概率之和，半分辨率（与 sweep 同口径）。"""
    acc = None
    for m, w in zip(MEMBERS, WEIGHTS):
        p = np.load(os.path.join(CACHE, m, f"{i:04d}.npy")).astype(np.float32) * w
        acc = p if acc is None else acc + p
    return acc


def load_target(i):
    t = np.load(os.path.join(CACHE, "target", f"{i:04d}.npy"))
    return t[0] if t.ndim == 3 else t


def ds(t, size):
    """双线性/最近邻降到 size=(h,w)。"""
    kw = dict(size=size, mode="bilinear", align_corners=False)
    x = torch.from_numpy(t)[None]
    if t.ndim == 2:
        return F.interpolate(x[:, None].float(), **kw)[0, 0].numpy()
    return F.interpolate(x.float(), **kw)[0].numpy()


def miou_of(preds, targets):
    cm = ConfusionMatrix(NUM_CLASSES, 255)
    for p, t in zip(preds, targets):
        cm.update(p, t)
    return cm.miou(True)


def main():
    n = len(os.listdir(os.path.join(CACHE, "target")))
    fit_idx = [i for i in range(n) if i % 2 == 1]
    val_idx = [i for i in range(n) if i % 2 == 0]
    print(f"拟合 {len(fit_idx)} 张 / 验证 {len(val_idx)} 张", flush=True)

    # ---------- 拟合集：1/4 分辨率 ----------
    fit_p, fit_t = [], []
    for i in fit_idx:
        a = load_wsum(i)
        h, w = a.shape[-2] // 2, a.shape[-1] // 2
        a = ds(a, (h, w))
        t = load_target(i)
        t = torch.from_numpy(t)[None, None].float()
        t = F.interpolate(t, size=(h, w), mode="nearest")[0, 0].numpy().astype(np.int64)
        fit_p.append(np.log(np.clip(a, 1e-8, None)).astype(np.float32))
        fit_t.append(t)
    print("拟合集载入完成", flush=True)

    def score(bias):
        preds = [(p + bias[:, None, None]).argmax(0) for p in fit_p]
        return miou_of(preds, fit_t)

    bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    base_fit = score(bias)
    print(f"拟合集 baseline mIoU21 = {base_fit:.4f}", flush=True)

    for rnd in range(2):
        for c in range(NUM_CLASSES):
            best_b, best_m = bias[c], -1.0
            for b in CAND:
                bias[c] = b
                m = score(bias)
                if m > best_m:
                    best_b, best_m = b, m
            bias[c] = best_b
        print(f"  round {rnd + 1}: mIoU21 = {score(bias):.4f}", flush=True)

    print("拟合 bias:", " ".join(f"{c}:{b:+.2f}" for c, b in enumerate(bias)
                                if abs(b) > 1e-6), flush=True)

    # ---------- 验证集：统一半分辨率口径 ----------
    vb, vp = [], []
    for i in val_idx:
        a = load_wsum(i)
        vb.append(a)
        vp.append(load_target(i))

    def score_val(bias_use):
        preds = [(np.log(np.clip(a, 1e-8, None)) + bias_use[:, None, None]).argmax(0)
                 for a in vb]
        ups = []
        for p, t in zip(preds, vp):
            ts = t.shape[-2:]
            if p.shape != ts:
                p = F.interpolate(torch.from_numpy(p[None, None]).float(), size=ts,
                                  mode="nearest")[0, 0].numpy()
            ups.append(p.astype(np.uint8))
        return miou_of(ups, vp)

    m_base = score_val(np.zeros(NUM_CLASSES, dtype=np.float32))
    m_bias = score_val(bias)
    print(f"\n验证集（held-out）baseline = {m_base:.4f}", flush=True)
    print(f"验证集（held-out）+bias   = {m_bias:.4f}  Δ{(m_bias - m_base) * 100:+.2f}点",
          flush=True)
    if m_bias - m_base < 0.001:
        print("⇒ 校正无实质收益，不采用", flush=True)
    else:
        print("⇒ 有收益，下一步需全精度复核后再决定是否上线", flush=True)


if __name__ == "__main__":
    main()
