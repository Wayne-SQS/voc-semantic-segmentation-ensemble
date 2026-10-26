# -*- coding: utf-8 -*-
"""在已有半分辨率概率缓存上做两类"零训练成本"搜索：

    1) 成员一进一出：基准 4 成员里任意一个换成异构零样本候选
    2) 单成员温度（幂）校准：把某成员的 softmax 概率取 p**g（g=0.75 平滑 / 1.5 锐化），
       再参与加权求和——只调置信度分布，不改模型、不重训

为什么必须切两半：
    这些超参（谁进谁出、g 取多少）如果全在同一个 val 上挑，分数一定虚高。
    所以奇数号图像只用来"挑"，偶数号图像只用来"验"，且偶数号图像全程不参与选择。

口径：与 sweep_val_cache.py 一致（半分辨率 argmax 后最近邻上采样），
      只用于排序和筛选，最终候选仍需 predict_ensemble.py 全精度复核。
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from metrics import ConfusionMatrix

BASE = ["zs_r50", "zs_r101", "sbd_lfca", "r101_e13"]
BW = [1.0, 1.25, 1.0, 1.0]
CAND = ["zs_fcn_r101", "zs_fcn_r50", "zs_dlv3mnv3", "zs_lraspp"]
GAMMAS = (0.75, 1.5)


def build_configs():
    """返回 [(说明, 成员列表, 权重列表, gamma 说明或 None)]。"""
    cfgs = [("base", list(BASE), list(BW), None)]
    for i, m in enumerate(BASE):
        for c in CAND:
            mem, w = list(BASE), list(BW)
            mem[i] = c
            cfgs.append((f"swap {m}->{c}", mem, w, None))
    for i, m in enumerate(BASE):
        for g in GAMMAS:
            cfgs.append((f"gamma {m}^{g:g}", list(BASE), list(BW), (i, g)))
    for c in CAND:
        cfgs.append((f"add {c}", list(BASE) + [c], list(BW) + [1.0], None))
    return cfgs


def upsample(pred, shape):
    if pred.shape != shape:
        pred = F.interpolate(torch.from_numpy(pred[None, None]).float(),
                             size=shape, mode="nearest")[0, 0].numpy().astype(np.uint8)
    return pred


def evaluate(cache_dir, idxs, cfgs, tag, log):
    n = len(idxs)
    need = sorted({m for _, mem, _, _ in cfgs for m in mem})
    cms = [ConfusionMatrix() for _ in cfgs]
    tdir = os.path.join(cache_dir, "target")
    t0 = time.time()
    for k, i in enumerate(idxs):
        target = np.load(os.path.join(tdir, f"{i:04d}.npy"))
        tshape = target.shape[-2:]
        maps = {m: np.load(os.path.join(cache_dir, m, f"{i:04d}.npy")).astype(np.float32)
                for m in need}
        gcache = {}
        for ci, (_, mem, w, g) in enumerate(cfgs):
            acc = None
            for j, (m, ww) in enumerate(zip(mem, w)):
                if g is not None and j == g[0]:
                    if g not in gcache:
                        gcache[g] = maps[m] ** g[1]
                    p = gcache[g] * ww
                else:
                    p = maps[m] * ww
                acc = p if acc is None else acc + p
            cms[ci].update(upsample(acc.argmax(0).astype(np.uint8), tshape), target)
        if (k + 1) % 100 == 0:
            log(f"  [{tag}] {k + 1}/{n}  {(time.time() - t0) / 60:.1f} min")
    del maps
    return [(cm.miou(True), cm.miou(False), cm.pixel_accuracy) for cm in cms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=r"E:\tmp\valcache")
    ap.add_argument("--out_txt", default=r"E:\桌面\人工智能技术大作业\outputs\calib_search.txt")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    n_img = len(os.listdir(os.path.join(args.cache_dir, "target")))
    fit_idx = list(range(1, n_img, 2))    # 奇数号：只用于挑
    hold_idx = list(range(0, n_img, 2))   # 偶数号：只用于验
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    cfgs = build_configs()
    log(f"缓存 {n_img} 张；拟合 {len(fit_idx)} / held-out {len(hold_idx)}；配置 {len(cfgs)} 个")

    log("\n=== 阶段 1：奇数号图像上筛选 ===")
    fit = evaluate(args.cache_dir, fit_idx, cfgs, "fit", log)
    ranked = sorted(zip(fit, cfgs), key=lambda x: -x[0][0])
    for (m21, m20, pa), (name, _, _, _) in ranked:
        log(f"  {m21:.4f}  (mIoU20 {m20:.4f} / pixAcc {pa:.4f})  {name}")

    log(f"\n=== 阶段 2：偶数号图像（held-out）验证 baseline + top{args.topk} ===")
    base_cfg = [c for c in cfgs if c[0] == "base"]
    picked = [cfg for _, cfg in ranked[:args.topk]]
    # base 可能已在 topk 里，去重
    final = base_cfg + [c for c in picked if c[0] != "base"]
    hold = evaluate(args.cache_dir, hold_idx, final, "hold", log)
    base21 = hold[0][0]
    best_i, best21 = 0, base21
    for k, ((m21, m20, pa), (name, _, _, _)) in enumerate(zip(hold, final)):
        log(f"  {m21:.4f}  (Δ{m21 - base21:+.4f} / mIoU20 {m20:.4f})  {name}")
        if m21 > best21:
            best_i, best21 = k, m21

    gain = best21 - base21
    log(f"\nheld-out baseline = {base21:.4f}")
    log(f"held-out 最优 = {best21:.4f}（{final[best_i][0]}），Δ{gain:+.4f}")
    log("⇒ 采纳" if gain >= 0.002 else "⇒ 无实质收益，不采用")

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存：{args.out_txt}", flush=True)


if __name__ == "__main__":
    main()
