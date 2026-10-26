# -*- coding: utf-8 -*-
"""绘制推理侧配置对比图（图 5-3）：全量 VOC val 上的 21 类 mIoU 阶梯。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm  # noqa: E402

_SIMHEI = r"C:\Windows\Fonts\simhei.ttf"
if os.path.exists(_SIMHEI):
    fm.fontManager.addfont(_SIMHEI)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_SIMHEI).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\桌面\人工智能技术大作业\outputs\figures\inference_ladder.png"

# (标签, mIoU21, 颜色)
rows = [
    ("零样本 R50（单尺度）", 0.7768, "#7f9fc4"),
    ("零样本 R101（单尺度）", 0.7951, "#4a72b8"),
    ("训练三模型集成 + 5尺度TTA（旧提交）", 0.7907, "#9a9a9a"),
    ("零样本 R101 + 5尺度TTA", 0.8024, "#2f8f83"),
    ("零样本 R50+R101 + 5尺度TTA", 0.8049, "#1f6f64"),
    ("3 模型混合集成 + 5尺度TTA", 0.8143, "#14524a"),
    ("2 模型（zs_r101×1.25 + sbd_lfca）", 0.8167, "#c9a227"),
    ("4 模型混合集成（zs_r101×1.25）+ 5尺度TTA（最终）", 0.8195, "#c0392b"),
]
rows.sort(key=lambda r: r[1])  # 最优在顶部

labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
colors = [r[2] for r in rows]

# 纵轴不再贴长配置名（会把绘图区挤成一条线），改为把「颜色 ↔ 配置」的
# 图例放在图表下方，绘图区可用宽度大幅增加（用户要求，2026-10-24）
fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
bars = ax.barh(list(range(len(vals))), vals, color=colors, height=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax.text(v + 0.0012, b.get_y() + b.get_height() / 2, f"{v:.4f}",
            va="center", ha="left", fontsize=10, color="#222222")
ax.set_yticks([])

ax.set_xlim(0.75, 0.825)
ax.set_xlabel("VOC val mIoU(21)，全量 1449 张（横轴自 0.75 起）", fontsize=10.5)
ax.set_title("推理侧配置对比：零样本与混合集成（全量 val，21 类 mIoU）", fontsize=12.5, pad=10)
# 不再画虚线网格：每根条末端已标数值，网格只会变成一排小黑点（用户要求，2026-10-24）
ax.tick_params(axis="x", labelsize=9.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

from matplotlib.patches import Patch  # noqa: E402
handles = [Patch(facecolor=c, edgecolor="none", label=t) for t, _, c in rows]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=2, fontsize=9, frameon=False, handlelength=1.1, handleheight=0.9,
          columnspacing=1.4, labelspacing=0.55)
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print("saved:", OUT)
