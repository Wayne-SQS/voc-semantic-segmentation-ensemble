# -*- coding: utf-8 -*-
"""读取 history.csv 画训练曲线，产出可直接贴进报告的 PNG。

作业 P5 要求报告包含 "Figures for the accuracy curves and loss curves"，
TensorBoard 截图之外，这里再出一套干净的静态图（中文标签，本机 SimHei 字体）。

用法：
    python plot_history.py --csv ..\runs\base_r50_e30\history.csv --out ..\outputs\figures --tag base
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境必须
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

# 注册本机黑体，避免中文缺字（豆腐块）
_SIMHEI = r"C:\Windows\Fonts\simhei.ttf"
if os.path.isfile(_SIMHEI):
    fm.fontManager.addfont(_SIMHEI)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_SIMHEI).get_name()
plt.rcParams["axes.unicode_minus"] = False


def read_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("epoch"):
                continue
            rows.append({k: (float(v) if v not in ("", None) else float("nan"))
                         for k, v in r.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"E:\桌面\人工智能技术大作业\runs\base_r50_e30\history.csv")
    ap.add_argument("--out", default=r"E:\桌面\人工智能技术大作业\outputs\figures")
    ap.add_argument("--tag", default="", help="输出文件名后缀，如 base / lfca / sbd")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = read_history(args.csv)
    # 续训会产生重复 epoch 行（如同名 epoch 两次），去重保留最后一次
    seen = {}
    for r in rows:
        seen[int(r["epoch"])] = r
    rows = [seen[k] for k in sorted(seen)]
    if not rows:
        print("history.csv 还没有数据行")
        return
    ep = [int(r["epoch"]) for r in rows]
    suffix = f"_{args.tag}" if args.tag else ""

    # 1) loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [r["train_loss"] for r in rows], marker="o", label="训练损失")
    plt.xlabel("训练轮次 (Epoch)")
    plt.ylabel("交叉熵损失 (Loss)")
    plt.title("训练损失曲线")
    plt.grid(alpha=0.3)
    plt.legend()
    p1 = os.path.join(args.out, f"loss_curve{suffix}.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.close()

    # 2) accuracy 曲线：mIoU(21) / mIoU(20 前景) / 像素准确率
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [r["val_miou21"] for r in rows], marker="s", label="验证集 mIoU（21 类含背景）")
    plt.plot(ep, [r["val_miou20"] for r in rows], marker="^", label="验证集 mIoU（20 类前景）")
    plt.plot(ep, [r["val_pixel_acc"] for r in rows], marker="d", label="验证集像素准确率")
    plt.xlabel("训练轮次 (Epoch)")
    plt.ylabel("得分")
    plt.title("验证集准确率曲线（每轮全量验证）")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8.5)
    p2 = os.path.join(args.out, f"accuracy_curve{suffix}.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.close()

    import math
    valid = [r for r in rows if not math.isnan(r["val_miou21"])]
    if valid:
        best = max(valid, key=lambda r: r["val_miou21"])
        print(f"已输出：{p1}\n         {p2}")
        print(f"最佳 epoch {int(best['epoch'])}: mIoU21={best['val_miou21']:.4f} "
              f"mIoU20={best['val_miou20']:.4f} pixAcc={best['val_pixel_acc']:.4f}")
    else:
        print(f"已输出：{p1}\n         {p2}（该 run 无验证指标，仅训练损失）")


if __name__ == "__main__":
    main()
