# -*- coding: utf-8 -*-
"""临时脚本：为实验报告生成三张负结果 / 过拟合 / 逐类 IoU 数据图。

风格约定（学术、简约）：
  - 纯 matplotlib；白底；去顶/右边框；网格仅浅灰虚线且置于最底层，能省则省
  - 主蓝 #3D6B99（正值/主指标），哑光红 #B05A5A（负值/验证 mIoU/弱类），中性灰 #8C8C8C
  - 中文字体：遍历 fontManager.ttflist，取第一个名称含
    SimHei / Microsoft YaHei / SimSun / Noto Sans CJK 的字体；
    若均不存在则回退 DejaVu Sans 并把标签改写为英文（本机探测到中文字体，未回退）

用法：
    python _figs_tmp.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境必须
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------- 全局风格
BLUE = "#3D6B99"   # 主色：低饱和蓝
RED = "#B05A5A"    # 强调/负值：哑光红
GRAY = "#8C8C8C"   # 中性灰
INK = "#333333"    # 正文/数值标签
GRID = "#DDDDDD"   # 网格浅灰

_CJK_KEYS = ("SimHei", "Microsoft YaHei", "SimSun", "Noto Sans CJK")
_cjk = None
for _f in fm.fontManager.ttflist:
    if any(k in _f.name for k in _CJK_KEYS):
        _cjk = _f.name
        break
if _cjk is not None:
    plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
else:  # 回退：无中文字体时标签已全部改用英文书写，避免豆腐块
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

OUT_DIR = r"E:\桌面\人工智能技术大作业\outputs\figures"
DPI = 220


def clean_axes(ax, keep_left=True, keep_bottom=True, keep_right=False):
    """去顶边框，按需保留左/右/底框（seaborn despine 的手写版）。"""
    for side, keep in (("top", False), ("right", keep_right),
                       ("left", keep_left), ("bottom", keep_bottom)):
        ax.spines[side].set_visible(keep)


def light_hgrid(ax):
    """浅灰横向虚线网格，置于最底层。"""
    ax.grid(axis="y", color=GRID, lw=0.8, linestyle=(0, (4, 3)), zorder=0)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"已输出 {path}  ({os.path.getsize(path)} bytes)")


# ---------------------------------------------------------------- 图 1
def fig_neg_results():
    """八组优化方向消融总览：横向条形图，按 Δ 降序。"""
    items = [
        ("成员一进一出替换（vs 4 模型集成 0.8195）", 0.18),
        ("异构成员增补 fcn_r101（vs 4 模型集成 0.8195）", 0.14),
        ("温度校准 p^g（vs 4 模型集成 0.8195）", 0.00),
        ("聚合范数·中位数（vs 4 模型集成 0.8195）", 0.00),
        ("引导滤波后处理（vs 4 模型集成 0.8195）", -0.08),
        ("微调 P2 全程 12 epoch（vs 零样本 R101 0.7951）", -1.66),
        ("YOLO 融合最优阈值（vs 3 模型集成 0.7907）", -1.92),
        ("每类 logit 偏置（vs 4 模型集成 0.8195）", -2.13),
    ]
    labels = [s for s, _ in items]
    vals = [v for _, v in items]
    colors = [BLUE if v > 0 else (RED if v < 0 else GRAY) for v in vals]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ys = list(range(len(vals)))
    ax.barh(ys, vals, height=0.62, color=colors, zorder=2)
    ax.axvline(0, color=GRAY, lw=1.0, zorder=1)  # x=0 竖直参考线
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()  # Δ 最大者置顶
    ax.set_xlim(-2.6, 0.6)
    ax.set_xlabel("mIoU 变化（百分点）")
    ax.tick_params(axis="y", length=0)
    for y, v in zip(ys, vals):
        if v >= 0:
            ax.text(v + 0.04, y, f"{v:+.2f}", va="center", ha="left",
                    fontsize=8, color=INK, zorder=3)
        else:
            ax.text(v - 0.04, y, f"{v:+.2f}", va="center", ha="right",
                    fontsize=8, color=INK, zorder=3)
    clean_axes(ax)
    save(fig, "neg_results.png")


# ---------------------------------------------------------------- 图 2
def fig_p2_overfit():
    """P2 轻量微调过拟合形态：训练损失 vs 验证 mIoU(21) 双轴。"""
    csv_path = r"E:\桌面\人工智能技术大作业\runs\zs_lfca_r101_lowlr\history.csv"
    epochs, loss, miou = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("epoch"):
                continue
            epochs.append(int(float(row["epoch"])))
            loss.append(float(row["train_loss"]))
            miou.append(float(row["val_miou21"]))
    best_i = max(range(len(miou)), key=lambda i: miou[i])

    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax2 = ax1.twinx()

    ax1.plot(epochs, loss, color=BLUE, marker="o", ms=4.5, lw=1.6,
             zorder=3, label="训练损失")
    ax2.plot(epochs, miou, color=RED, marker="s", ms=4.5, lw=1.6,
             zorder=3, label="验证集 mIoU(21)")
    ax2.axhline(0.7951, color=GRAY, ls="--", lw=1.2, zorder=2)  # 零样本基线
    if best_i != len(epochs) - 1:  # best epoch 非末轮：浅灰竖span标出
        ax1.axvspan(epochs[best_i] - 0.5, epochs[best_i] + 0.5,
                    color=GRAY, alpha=0.12, zorder=0.1)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("训练损失", color=BLUE)
    ax2.set_ylabel("验证集 mIoU(21)", color=RED)
    ax1.set_xticks(epochs)
    ax1.set_xlim(-0.5, 11.5)
    ax1.set_ylim(0.43, 0.55)
    ax2.set_ylim(0.74, 0.80)
    light_hgrid(ax1)

    # 关键数值直标（其余点位由坐标轴读取，避免拥挤）
    ax1.annotate(f"{loss[0]:.4f}", (epochs[0], loss[0]),
                 xytext=(0.12, 0.0018), textcoords="offset points",
                 ha="left", va="bottom", fontsize=8, color=BLUE, zorder=4)
    ax1.annotate(f"{loss[-1]:.4f}", (epochs[-1], loss[-1]),
                 xytext=(-0.12, -0.018), textcoords="offset points",
                 ha="right", va="top", fontsize=8, color=BLUE, zorder=4)
    ax2.annotate(f"{miou[0]:.4f}", (epochs[0], miou[0]),
                 xytext=(0.15, 0.0015), textcoords="offset points",
                 ha="left", va="bottom", fontsize=8, color=RED, zorder=4)
    ax2.annotate(f"{miou[best_i]:.4f}", (epochs[best_i], miou[best_i]),
                 xytext=(0, 0.002), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8, color=RED, zorder=4)
    ax2.annotate(f"{miou[-1]:.4f}", (epochs[-1], miou[-1]),
                 xytext=(-0.15, 0.0018), textcoords="offset points",
                 ha="right", va="bottom", fontsize=8, color=RED, zorder=4)

    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=4.5, lw=1.6, label="训练损失"),
        Line2D([], [], color=RED, marker="s", ms=4.5, lw=1.6, label="验证集 mIoU(21)"),
        Line2D([], [], color=GRAY, ls="--", lw=1.2, label="零样本 R101 基线 0.7951"),
        Patch(facecolor=GRAY, alpha=0.12, label="best epoch"),
    ]
    ax1.legend(handles=handles, loc="lower left", frameon=False,
               borderaxespad=0.2, handlelength=2.4)

    clean_axes(ax1)
    clean_axes(ax2, keep_left=False, keep_bottom=False, keep_right=True)
    save(fig, "p2_overfit.png")


# ---------------------------------------------------------------- 图 3
def fig_per_class_iou():
    """最终提交配置逐类 IoU：竖排条形图，按 IoU 降序，弱类标红。"""
    data = {
        "background": 0.9541, "bus": 0.9631, "cat": 0.9493, "cow": 0.9327,
        "horse": 0.9281, "aeroplane": 0.9283, "train": 0.9179, "dog": 0.9141,
        "person": 0.9051, "sheep": 0.9002, "motorbike": 0.9006, "bird": 0.9101,
        "tvmonitor": 0.8115, "car": 0.8660, "bottle": 0.7857, "boat": 0.7067,
        "pottedplant": 0.6753, "sofa": 0.6640, "diningtable": 0.6478,
        "chair": 0.5157, "bicycle": 0.4332,
    }
    weak = {"bicycle", "chair", "diningtable", "pottedplant", "sofa"}
    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [RED if n in weak else BLUE for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = list(range(len(vals)))
    ax.bar(xs, vals, width=0.68, color=colors, zorder=2)
    ax.axhline(0.8195, color=GRAY, ls="--", lw=1.2, zorder=1,
               label="mIoU(21) = 0.8195")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8.5)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.02)
    light_hgrid(ax)
    # 数值标签置于条顶上方（横排小字）：无旋转定位歧义，不超出 y 上限且互不重叠
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.006, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=INK, zorder=3)
    ax.legend(loc="upper right", frameon=False, borderaxespad=0.3)
    clean_axes(ax)
    save(fig, "per_class_iou.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"中文字体：{_cjk if _cjk else '未找到（已回退英文标签）'}")
    fig_neg_results()
    fig_p2_overfit()
    fig_per_class_iou()
