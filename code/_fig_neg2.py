# -*- coding: utf-8 -*-
"""重新生成图 5-6：各优化方向在拟合半 / held-out 半上的 mIoU 增量对比。

成组水平条形图。左淡右深：淡色 = 奇数号 724 张（拟合/选择半），
深色 = 偶数号 725 张（held-out，全程不参与选择）。
视觉主张：拟合半的增益换到 held-out 上一律缩水或反向。
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------- 全局风格
FIT = "#A8BFD6"    # 拟合半：浅蓝
HOLD = "#3D6B99"   # held-out 半：主蓝
RED = "#B05A5A"    # 负值强调：哑光红
GRAY = "#8C8C8C"   # 中性灰
INK = "#333333"
GRID = "#DDDDDD"

_CJK_KEYS = ("SimHei", "Microsoft YaHei", "SimSun", "Noto Sans CJK")
_cjk = None
for _f in fm.fontManager.ttflist:
    if any(k in _f.name for k in _CJK_KEYS):
        _cjk = _f.name
        break
plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"] if _cjk else ["DejaVu Sans"]
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

# (方向, 拟合半 Δ, held-out Δ) —— 单位：mIoU 百分点，相对同半基准
ITEMS = [
    ("① 异构零样本成员替换", 0.54, 0.18),
    ("② 异构零样本成员增补", 0.00, 0.00),
    ("③ 单成员温度校准", 0.00, None),
    ("④ 聚合范数·中位数", 0.13, 0.00),
    ("⑤ 引导滤波后处理 r=2", -0.11, -0.07),
    ("⑥ 逐类 logit 偏置", 1.50, -2.13),
]

labels = [s for s, _, _ in ITEMS]
fit = [v for _, v, _ in ITEMS]
hold = [v for _, _, v in ITEMS]

fig, ax = plt.subplots(figsize=(7.4, 3.9))
n = len(ITEMS)
ys = list(range(n))
h = 0.34

# held-out 为 None（未复核）时该条不画，仅留占位说明
for y, f, hd in zip(ys, fit, hold):
    ax.barh(y + h / 2 + 0.02, f, height=h, color=FIT, zorder=2)
    if hd is None:
        ax.text(0.06, y - h / 2 - 0.02, "未进入复核", va="center", ha="left",
                fontsize=7.5, color=GRAY, style="italic", zorder=3)
    else:
        c = HOLD if hd >= 0 else RED
        ax.barh(y - h / 2 - 0.02, hd, height=h, color=c, zorder=2)

ax.axvline(0, color="#666666", lw=1.0, zorder=1)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-2.75, 1.95)
ax.set_ylim(n - 0.45, -0.55)
ax.set_xlabel("mIoU 增量（百分点）")
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8, linestyle=(0, (4, 3)), zorder=0)

for y, f, hd in zip(ys, fit, hold):
    if f >= 0:
        ax.text(f + 0.05, y + h / 2 + 0.02, f"{f:+.2f}", va="center", ha="left",
                fontsize=7.8, color=INK, zorder=3)
    else:
        ax.text(f - 0.05, y + h / 2 + 0.02, f"{f:+.2f}", va="center", ha="right",
                fontsize=7.8, color=INK, zorder=3)
    if hd is not None:
        if hd >= 0:
            ax.text(hd + 0.05, y - h / 2 - 0.02, f"{hd:+.2f}", va="center",
                    ha="left", fontsize=7.8, color=INK, zorder=3)
        else:
            ax.text(hd - 0.05, y - h / 2 - 0.02, f"{hd:+.2f}", va="center",
                    ha="right", fontsize=7.8, color=INK, zorder=3)

for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

leg = ax.legend(handles=[Patch(facecolor=FIT, label="拟合半（奇数号 724 张，用于选参）"),
                         Patch(facecolor=HOLD, label="held-out 半（偶数号 725 张，不参与选参）")],
                loc="upper left", bbox_to_anchor=(0.005, 1.005), frameon=False,
                handlelength=1.4, handleheight=0.9, fontsize=9, labelcolor=INK)
leg.set_zorder(10)
for txt in leg.get_texts():
    txt.set_color(INK)

path = os.path.join(OUT_DIR, "neg_results.png")
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"已输出 {path}  ({os.path.getsize(path)} bytes)")
