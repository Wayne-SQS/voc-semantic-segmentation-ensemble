# -*- coding: utf-8 -*-
"""在已有半分辨率概率缓存上比较**聚合方式**（而非成员或权重）。

已试过的都集中在"加谁/给多大权重"，这里换一个从未试过的维度——成员之间
用什么范数平均。基准是加权算术平均 softmax 概率；候选项：

    arith  加权算术平均（基准，scale-invariant，等价 sum(w*p)）
    geom   加权几何平均（等价平均 log 概率，比算术更"锐"）
    quad   加权二次平均（mean(p^2) 开方前不看，argmax 等价于 sum(w*p^2)）
    max    加权最大值（完全信任最自信的成员）
    median 加权中位数（对单个成员的离谱预测鲁棒）
    harm   加权调和平均（最"保守"，任一成员否决即否决）

为什么值得单独测：温度校准 p^g 是"先锐化单成员再算术平均"，与在成员间
换范数是两件不同的事，不能互相替代。

防自欺：奇数号图像只用于挑，偶数号图像只用于验；geom 再做一轮权重坐标上升，
但只在奇数半上搜，搜出来的结果仍需偶数半确认。

口径：与 sweep_val_cache.py / calib_search.py 一致（半分辨率 argmax + 最近邻上采样），
      仅用于筛选排序，最终采纳仍需 predict_ensemble.py 全精度复核。
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from metrics import ConfusionMatrix

MEMBERS = ["zs_r50", "zs_r101", "sbd_lfca", "r101_e13"]
WEIGHTS = [1.0, 1.25, 1.0, 1.0]
MODES = ("arith", "geom", "quad", "max", "median", "harm")
EPS = 1e-5          # float16 量化后可能有精确 0，log 前必须截断
HARM_EPS = 1e-3     # 调和平均对小数极敏感，单独用更粗的截断


def aggregate(maps, w, mode):
    """maps: list of [21,h,w] float32；w: 已归一化的权重。返回 argmax 图。"""
    if mode == "arith":
        acc = None
        for m, ww in zip(maps, w):
            p = m * ww
            acc = p if acc is None else acc + p
        return acc.argmax(0)
    if mode == "geom":
        acc = None
        for m, ww in zip(maps, w):
            lp = np.log(np.maximum(m, EPS)) * ww
            acc = lp if acc is None else acc + lp
        return acc.argmax(0)
    if mode == "quad":
        acc = None
        for m, ww in zip(maps, w):
            p = (m * m) * ww
            acc = p if acc is None else acc + p
        return acc.argmax(0)
    if mode == "max":
        return np.max(np.stack([m * ww for m, ww in zip(maps, w)]), axis=0).argmax(0)
    if mode == "median":
        return np.median(np.stack([m * ww for m, ww in zip(maps, w)]), axis=0).argmax(0)
    if mode == "harm":
        acc = None
        for m, ww in zip(maps, w):
            q = ww / np.maximum(m, HARM_EPS)
            acc = q if acc is None else acc + q
        return acc.argmin(0)
    raise ValueError(mode)


def norm_weights(w):
    w = np.asarray(w, dtype=np.float64)
    return (w / w.sum()).astype(np.float32)


def upsample(pred, shape):
    if pred.shape != shape:
        pred = F.interpolate(torch.from_numpy(pred[None, None]).float(),
                             size=shape, mode="nearest")[0, 0].numpy().astype(np.uint8)
    return pred


def evaluate(cache_dir, idxs, tasks, tag, log):
    """tasks: list of (说明, 权重list, mode)。"""
    cms = [ConfusionMatrix() for _ in tasks]
    tdir = os.path.join(cache_dir, "target")
    t0 = time.time()
    for k, i in enumerate(idxs):
        target = np.load(os.path.join(tdir, f"{i:04d}.npy"))
        tshape = target.shape[-2:]
        maps = [np.load(os.path.join(cache_dir, m, f"{i:04d}.npy")).astype(np.float32)
                for m in MEMBERS]
        for ci, (_, w, mode) in enumerate(tasks):
            pred = aggregate(maps, norm_weights(w), mode)
            cms[ci].update(upsample(pred.astype(np.uint8), tshape), target)
        del maps
        if (k + 1) % 200 == 0:
            log(f"  [{tag}] {k + 1}/{len(idxs)}  {(time.time() - t0) / 60:.1f} min")
    return [(cm.miou(True), cm.miou(False), cm.pixel_accuracy) for cm in cms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=r"E:\tmp\valcache")
    ap.add_argument("--out_txt", default=r"E:\桌面\人工智能技术大作业\outputs\agg_compare.txt")
    args = ap.parse_args()

    n_img = len(os.listdir(os.path.join(args.cache_dir, "target")))
    fit_idx = list(range(1, n_img, 2))
    hold_idx = list(range(0, n_img, 2))
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"缓存 {n_img} 张；拟合 {len(fit_idx)} / held-out {len(hold_idx)}")
    log(f"成员 {' + '.join(MEMBERS)}，基准权重 {WEIGHTS}")

    tasks = [(m, list(WEIGHTS), m) for m in MODES]
    log("\n=== 阶段 1：奇数号图像上比较六种聚合方式 ===")
    fit = evaluate(args.cache_dir, fit_idx, tasks, "fit", log)
    for (m21, m20, pa), (name, _, _) in sorted(zip(fit, tasks), key=lambda x: -x[0][0]):
        log(f"  {m21:.4f}  (mIoU20 {m20:.4f} / pixAcc {pa:.4f})  {name}")

    # 几何平均上做权重坐标上升（只在奇数半搜）
    log("\n=== 阶段 2：几何平均 + 权重坐标上升（奇数半）===")
    best_w, cur = list(WEIGHTS), None
    for (m21, _, _), (name, _, _) in zip(fit, tasks):
        if name == "geom":
            cur = m21
    log(f"  geom 起点 {cur:.4f}  w={best_w}")
    for rnd in range(2):
        cands = [("keep", list(best_w))]
        for j in range(len(best_w)):
            for mult in (0.75, 1.25):
                w2 = list(best_w)
                w2[j] *= mult
                cands.append((f"w{j}×{mult:g}", w2))
        vals = evaluate(args.cache_dir, fit_idx,
                        [(nm, w, "geom") for nm, w in cands], f"g{rnd + 1}", log)
        order = int(np.argmax([v[0] for v in vals]))
        if vals[order][0] <= cur + 1e-5:
            log(f"  第 {rnd + 1} 轮无改进，停止")
            break
        cur = vals[order][0]
        best_w = cands[order][1]
        log(f"  第 {rnd + 1} 轮：{cur:.4f}  {cands[order][0]}  w={best_w}")

    # 偶数半验证：基准 + 非 arith 里奇数半前两名 + geom 最优权重
    top_modes = [name for (m21, _, _), (name, _, _) in
                 sorted(zip(fit, tasks), key=lambda x: -x[0][0]) if name != "arith"][:2]
    final = [("arith base", list(WEIGHTS), "arith")]
    for m in top_modes:
        final.append((f"{m} base-w", list(WEIGHTS), m))
    final.append((f"geom best-w", list(best_w), "geom"))

    log("\n=== 阶段 3：偶数号图像（held-out）验证 ===")
    hold = evaluate(args.cache_dir, hold_idx, final, "hold", log)
    base21 = hold[0][0]
    best_i, best21 = 0, base21
    for k, ((m21, m20, pa), (name, _, _)) in enumerate(zip(hold, final)):
        log(f"  {m21:.4f}  (Δ{m21 - base21:+.4f} / mIoU20 {m20:.4f})  {name}")
        if m21 > best21:
            best_i, best21 = k, m21

    gain = best21 - base21
    log(f"\nheld-out baseline(arith) = {base21:.4f}")
    log(f"held-out 最优 = {best21:.4f}（{final[best_i][0]}），Δ{gain:+.4f}")
    log("⇒ 有实质收益，值得全精度复核" if gain >= 0.002 else "⇒ 无实质收益，不采用")

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存：{args.out_txt}", flush=True)


if __name__ == "__main__":
    main()
