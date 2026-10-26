# -*- coding: utf-8 -*-
"""在 cache_val_probs.py 产出的概率缓存上，瞬时扫描集成配置。
图像主循环：每张图只载入 5 个成员的半分辨率概率图，配置在内层批量评估，
内存恒定（约 10MB/张），无需把 14.5GB 缓存整个装进 RAM。

1) 全部 2^5-1=31 个非空成员子集（等权）——含 hybrid4 / hybrid5
2) 对排名靠前的子集做权重坐标上升（逐成员乘子 ×{0.75,1.25,1.5}，两轮）

指标口径与 predict_ensemble.py 一致（同一 ConfusionMatrix 类）；唯一近似是
概率图降过半分辨率、argmax 后最近邻上采样——用于配置排序，最终采纳的配置
会用全精度 predict_ensemble.py 实测复核。
"""
import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from metrics import ConfusionMatrix

DEFAULT_MEMBERS = ["zs_r50", "zs_r101", "sbd_lfca", "lfca_res", "r101_e13"]
MULTIPLIERS = (0.75, 1.25, 1.5)


def _pred_from_maps(maps, weights):
    """maps: list of [21,ph,pw] float32（已按 members 顺序）；返回上采样前的 argmax。"""
    acc = None
    for m, w in zip(maps, weights):
        p = m * w
        acc = p if acc is None else acc + p
    return acc.argmax(axis=0).astype(np.uint8)


def _upsample(pred, shape):
    ph, pw = pred.shape
    if (ph, pw) != shape:
        pred = F.interpolate(torch.from_numpy(pred[None, None]).float(),
                             size=shape, mode="nearest")[0, 0].numpy().astype(np.uint8)
    return pred


def _metrics(cms):
    out = []
    for cm in cms:
        out.append((cm.miou(True), cm.miou(False), cm.pixel_accuracy))
    return out


def evaluate_configs(cache_dir, configs, n_img):
    """configs: list of (members, weights)。返回指标列表。"""
    cms = [ConfusionMatrix() for _ in configs]
    tdir = os.path.join(cache_dir, "target")
    for i in range(n_img):
        target = np.load(os.path.join(tdir, f"{i:04d}.npy"))
        tshape = target.shape[-2:]  # target 可能是 (H,W) 或 (1,H,W)
        maps = {}
        for cfg_idx, (members, _) in enumerate(configs):
            for m in members:
                if m not in maps:
                    maps[m] = np.load(os.path.join(cache_dir, m, f"{i:04d}.npy")).astype(np.float32)
        for cfg_idx, (members, weights) in enumerate(configs):
            pred = _pred_from_maps([maps[m] for m in members], weights)
            cms[cfg_idx].update(_upsample(pred, tshape), target)
        del maps
    return _metrics(cms)


def fmt(members, weights):
    return " + ".join(f"{m}×{w:g}" for m, w in zip(members, weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=r"E:\tmp\valcache")
    ap.add_argument("--out_txt", default=r"E:\桌面\人工智能技术大作业\outputs\ensemble_sweep.txt")
    ap.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS,
                    help="缓存目录内的成员名（须与 %s 下子目录同名）" % r"E:\tmp\valcache")
    args = ap.parse_args()
    members_all = args.members

    n_img = len(os.listdir(os.path.join(args.cache_dir, "target")))
    print(f"缓存：{n_img} 张 × {len(members_all)} 成员", flush=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    # ---------- 1) 全部等权子集（一遍扫描） ----------
    subsets = []
    for r in range(1, len(members_all) + 1):
        subsets.extend(itertools.combinations(members_all, r))
    configs = [(list(c), [1.0] * len(c)) for c in subsets]
    log(f"=== 等权子集扫描（{len(configs)} 个配置，半分辨率近似，全量 val）===")
    res = evaluate_configs(args.cache_dir, configs, n_img)
    ranked = sorted(zip(res, subsets), key=lambda x: -x[0][0])
    for (m21, m20, pa), combo in ranked:
        log(f"  {m21:.4f}  (mIoU20 {m20:.4f} / pixAcc {pa:.4f})  {' + '.join(combo)}")

    # ---------- 2) top5 子集权重坐标上升 ----------
    log("\n=== 权重坐标上升（top5 子集，乘子 ×{0.75,1.25,1.5}，两轮）===")
    best_cfg = (ranked[0][0][0], list(ranked[0][1]), [1.0] * len(ranked[0][1]))
    for (m21, _, _), combo in ranked[:5]:
        weights = [1.0] * len(combo)
        cur = m21
        for rnd in range(2):
            cands = [(list(combo), list(weights))]
            for j in range(len(combo)):
                for mult in MULTIPLIERS:
                    w2 = list(weights)
                    w2[j] = w2[j] * mult
                    cands.append((list(combo), w2))
            vals = evaluate_configs(args.cache_dir, cands, n_img)
            order = np.argsort([-v[0] for v in vals])
            if vals[order[0]][0] <= cur + 1e-5:
                break
            cur = vals[order[0]][0]
            weights = cands[order[0]][1]
        m20, pa = evaluate_configs(args.cache_dir, [(list(combo), weights)], n_img)[0][1:]
        log(f"  {cur:.4f}  (mIoU20 {m20:.4f} / pixAcc {pa:.4f})  {fmt(combo, weights)}")
        if cur > best_cfg[0]:
            best_cfg = (cur, list(combo), weights)

    log(f"\n最优（半分辨率口径）: {best_cfg[0]:.4f}  {fmt(best_cfg[1], best_cfg[2])}")

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存：{args.out_txt}", flush=True)


if __name__ == "__main__":
    main()
