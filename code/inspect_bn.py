# -*- coding: utf-8 -*-
"""检查 checkpoint 里各 BN 层的 num_batches_tracked，用来验证两个假设：

    1. backbone 的 BN 是否在训练中真的被冻结了（=统计量仍是预训练那套）
    2. 自研解码器的 BN 是否正常累积了统计量

num_batches_tracked 是 BN 层从 train() 模式下前向过的批次数。
如果 training 时被判为 eval()，这个计数不会增长，running_mean/var 也不会变。
所以这个数就是"这层到底有没有被训练更新过"的直接证据。

用法：
    python inspect_bn.py --ckpt ..\runs\sbd_lfca_e30\ckpt\best.pt
"""

import argparse
from collections import Counter, OrderedDict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    groups = OrderedDict()
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            # 归组：backbone / aspp / mid / decoder / aux
            if k.startswith("backbone."):
                g = "backbone"
            elif k.startswith("decoder."):
                g = "decoder"
            elif k.startswith("aspp."):
                g = "aspp"
            elif k.startswith("mid."):
                g = "mid"
            elif k.startswith("aux_classifier."):
                g = "aux_classifier"
            else:
                g = "other"
            groups.setdefault(g, []).append(int(v.item()))

    print(f"\ncheckpoint: {args.ckpt}")
    print(f"{'group':<16s} {'#BN':>5s} {'min':>10s} {'max':>10s}  分布")
    print("-" * 62)
    for g, vals in groups.items():
        c = Counter(vals)
        top = ", ".join(f"{k}:{v}" for k, v in c.most_common(3))
        print(f"{g:<16s} {len(vals):>5d} {min(vals):>10d} {max(vals):>10d}  {top}")

    print()
    bb = groups.get("backbone", [])
    dec = groups.get("decoder", [])
    if bb:
        if max(bb) <= 1 and len(bb) > 10:
            print("[结论] backbone 的 BN 全部停在同一预训练值 → 训练中被完全冻结，"
                  "统计量与 VOC/SBD 数据分布不匹配")
        elif max(bb) == 0:
            print("[结论] backbone 的 BN 从未在 train 模式下前向过 → 统计量 100% 来自预训练权重")
        else:
            print("[结论] backbone BN 有更新记录")
    if dec:
        print(f"[参考] decoder 的 BN 累积了 {min(dec)}~{max(dec)} 个 batch 的统计量（正常应为大数）")


if __name__ == "__main__":
    main()
