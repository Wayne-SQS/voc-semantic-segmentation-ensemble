# -*- coding: utf-8 -*-
"""在 val 概率缓存上评估"异构零样本候选"的边际价值（只测必要配置）。

为什么不直接用 sweep_val_cache.py：9 个成员的全子集是 511 个配置，纯属浪费。
这里固定当前提交的 4 模型基准，只回答两个问题：
    1) 每个候选单独有多强（单模型分）
    2) 把它加进基准后，集成是涨还是跌（分别试 ×1.0 / ×1.25 两种权重）

口径与 sweep_val_cache.py 完全一致（半分辨率近似 + argmax + 最近邻上采样），
最终采纳的组合仍会用全精度 predict_ensemble.py 复核。
"""
import argparse
import os

from sweep_val_cache import evaluate_configs, fmt

BASE = ["zs_r50", "zs_r101", "sbd_lfca", "r101_e13"]
BASE_W = [1.0, 1.25, 1.0, 1.0]
CANDS = ["zs_fcn_r101", "zs_fcn_r50", "zs_dlv3mnv3", "zs_lraspp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=r"E:\tmp\valcache")
    ap.add_argument("--out_txt",
                    default=r"E:\桌面\人工智能技术大作业\outputs\hetero_sweep.txt")
    ap.add_argument("--cands", nargs="+", default=CANDS)
    args = ap.parse_args()

    n_img = len(os.listdir(os.path.join(args.cache_dir, "target")))
    configs, labels = [([*BASE], list(BASE_W))], ["基准 4 模型（当前提交）"]
    for c in args.cands:
        configs.append(([c], [1.0]))
        labels.append(f"{c} 单模型")
    for w in (1.0, 1.25):
        for c in args.cands:
            configs.append(([*BASE, c], [*BASE_W, w]))
            labels.append(f"基准 + {c}×{w:g}")

    print(f"缓存 {n_img} 张，评估 {len(configs)} 个配置", flush=True)
    res = evaluate_configs(args.cache_dir, configs, n_img)
    base = res[0][0]

    lines = [f"cache={args.cache_dir} n={n_img}", "（半分辨率近似口径，用于排序）", ""]
    rows = sorted(zip(labels, configs, res), key=lambda x: -x[2][0])
    for lab, cfg, (m21, m20, pa) in rows:
        delta = (m21 - base) * 100
        lines.append(f"{m21:.4f}  Δ{delta:+.2f}点  mIoU20={m20:.4f} pixAcc={pa:.4f}"
                     f"  {lab}")
    lines.append("")
    lines.append("基准权重：" + fmt(BASE, BASE_W))

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\n已保存 {args.out_txt}", flush=True)


if __name__ == "__main__":
    main()
