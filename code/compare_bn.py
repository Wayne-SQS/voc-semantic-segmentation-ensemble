# -*- coding: utf-8 -*-
"""对比两个 checkpoint 的 BN 统计量，判断训练中 BN 到底有没有被更新。

原理（不需要预训练权重就能判定）：
    如果 backbone 的 BN 在训练中被冻结（running_mean/var 不更新），那么**任何两个
    用不同数据、不同初始化训出来的模型，其 backbone BN 统计量必然逐位相同**——因为
    它们都还是预训练权重里那一份。
    反之，只要训练更新过，两个模型的这批统计量就会有明显差异。

用法：
    python compare_bn.py --a ..\runs\base_r50_e30\ckpt\best.pt --b ..\runs\sbd_lfca_e30\ckpt\best.pt
"""

import argparse
from collections import OrderedDict

import torch


def group_of(key):
    if key.startswith("backbone."):
        return "backbone"
    if key.startswith("decoder."):
        return "decoder"
    if key.startswith("aspp."):
        return "aspp"
    if key.startswith("mid."):
        return "mid"
    if key.startswith("aux_classifier."):
        return "aux_classifier"
    return "other"


def load_sd(path):
    ck = torch.load(path, map_location="cpu")
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    sa, sb = load_sd(args.a), load_sd(args.b)
    common = [k for k in sa if k in sb and k.endswith("running_mean")]

    groups = OrderedDict()
    for k in common:
        groups.setdefault(group_of(k), []).append(k)

    print(f"A = {args.a}")
    print(f"B = {args.b}\n")
    print(f"{'group':<16s} {'#BN':>4s} {'完全相同':>8s} {'最大绝对差':>12s}  判定")
    print("-" * 68)
    for g, keys in groups.items():
        same, maxdiff = 0, 0.0
        for k in keys:
            d = (sa[k].float() - sb[k].float()).abs().max().item()
            maxdiff = max(maxdiff, d)
            if d == 0.0:
                same += 1
        if same == len(keys):
            verdict = "完全冻结（=预训练值）"
        elif maxdiff > 1e-3:
            verdict = "训练中更新过"
        else:
            verdict = "几乎没变"
        print(f"{g:<16s} {len(keys):>4d} {same:>8d} {maxdiff:>12.6f}  {verdict}")

    print("\n判读：")
    print("  backbone 一栏若显示『完全冻结』，说明两个不同训练的模型用的是同一套预训练")
    print("  BN 统计量 —— 本次训练从未更新过骨干的 running_mean/var，")
    print("  这正是 AdaBN（用目标域无标签图像重估 BN）能带来收益的根因。")


if __name__ == "__main__":
    main()
