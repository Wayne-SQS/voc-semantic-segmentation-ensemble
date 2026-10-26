# -*- coding: utf-8 -*-
"""生成 LFCA-DeepLabV3 网络结构图（作业要求"网络结构图"）。

严格按 code/model_v2.py 的 LFCADeepLabV3.forward 绘制：
    输入 → ResNet-50(空洞, OS=8) → layer1(/4, 256) 与 layer4(/8, 2048)
    layer4 → ASPP(2048→256) → classifier[1..3] 3×3 细化(256)
      ├─ 基线路径：classifier[4] 1×1 (256→21) → 上采样 → 基线 logits
      └─ 解码器：上采样到 /4 → 与 layer1(256→48) 拼接(304)
                 → 3×3(304→256) → SE(256) → 3×3(256)+Dropout → 1×1 残差头(零初始化)
                 → 上采样 4× → 残差 logits
    logits = 基线 + 残差 → (argmax) 原图尺寸 mask
    aux：layer3(/8) → 辅助分类头 → 上采样 → aux loss（仅训练期，权重 0.4）

输出 outputs/figures/network_diagram.png。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SIMHEI = r"C:\Windows\Fonts\simhei.ttf"
if os.path.isfile(SIMHEI):
    fm.fontManager.addfont(SIMHEI)
    SIMHEI_FP = fm.FontProperties(fname=SIMHEI)
    plt.rcParams["font.family"] = SIMHEI_FP.get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\桌面\人工智能技术大作业\outputs/figures"
os.makedirs(OUT, exist_ok=True)

BLUE = "#2E5A87"; GREEN = "#3C7A4E"; ORANGE = "#B5651D"
PURPLE = "#6A4C93"; RED = "#A33A3A"; GREY = "#6B6B6B"

fig, ax = plt.subplots(figsize=(11, 11.9))
ax.set_xlim(0, 12); ax.set_ylim(0, 13.15); ax.axis("off")


def box(x, y, w, text, fc, fs=12.0, h=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc=fc, ec="#333333", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="white", weight="bold", linespacing=1.35)


def seg(x1, y1, x2, y2, color="#333333", ls="-", head=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>" if head else "-",
                                 mutation_scale=12, color=color, lw=1.5,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def flow(points, color="#333333", ls="-"):
    for i in range(len(points) - 2):
        seg(*points[i], *points[i + 1], color=color, ls=ls, head=False)
    seg(*points[-2], *points[-1], color=color, ls=ls, head=True)


# ---------------------------------------------------------------- 骨干与三路输出
box(4.2, 12.45, 3.6, "输入 513×513×3", BLUE)
box(4.2, 11.40, 3.6, "ResNet-50 骨干\n（空洞卷积，OS=8）", BLUE)
seg(6.0, 12.45, 6.0, 12.23)

box(0.35, 10.35, 3.35, "Layer1 浅层\n256 通道 @ /4", GREEN)
box(4.2, 10.35, 3.6, "Layer4 高层\n2048 通道 @ /8", BLUE)
box(8.0, 10.35, 3.6, "辅助分类头 1024→256→21\n（取自 layer3 @ /8，仅训练期）", GREY, fs=10.6)

flow([(4.2, 11.79), (2.02, 11.79), (2.02, 11.18)], color=GREEN)          # 骨干 → layer1
seg(6.0, 11.40, 6.0, 11.18, color=BLUE)                                   # 骨干 → layer4
flow([(7.8, 11.79), (9.8, 11.79), (9.8, 11.18)], color=GREY, ls="--")     # 骨干 → aux

# ---------------------------------------------------------------- 基线路径
box(0.35, 9.30, 3.35, "1×1 卷积 256→48\n+BN/ReLU", GREEN)
box(4.2, 9.30, 3.6, "ASPP 多尺度池化\n2048→256", ORANGE)
ax.text(8.05, 9.80, "↑ 上采样回原图\n+ aux loss（训练期权重 0.4）",
        fontsize=10.2, color=GREY, va="top")

box(4.2, 8.25, 3.6, "3×3 细化卷积\n256 通道 + BN/ReLU", ORANGE)
box(8.0, 7.20, 3.6, "1×1 分类头 256→21\n（复用预训练权重）", ORANGE, fs=11.2)
box(8.0, 6.15, 3.6, "双线性上采样 8×\n→ 基线 logits", ORANGE, fs=11.2)

seg(6.0, 10.35, 6.0, 10.13)
seg(6.0, 9.30, 6.0, 9.08)
flow([(7.8, 8.65), (9.8, 8.65), (9.8, 8.03)], color=ORANGE)               # 细化 → 分类头
seg(9.8, 7.20, 9.8, 6.98, color=ORANGE)

# ---------------------------------------------------------------- 解码器
box(4.2, 7.20, 3.6, "双线性上采样到 /4\n（与浅层对齐）", PURPLE, fs=11.2)
box(2.6, 6.15, 4.8, "拼接：ASPP 256 + 浅层 48\n= 304 通道", PURPLE, fs=11.5)
box(2.6, 5.10, 4.8, "3×3 卷积 304→256\n+BN/ReLU", PURPLE, fs=11.5)
box(2.6, 4.05, 4.8, "SE 通道注意力（256）\n通道权重 × 特征", PURPLE, fs=11.5)
box(2.6, 3.00, 4.8, "3×3 卷积 256\n+BN/ReLU + Dropout 0.1", PURPLE, fs=11.5)
box(2.6, 1.95, 4.8, "1×1 残差头 256→21（零初始化）\n+ 双线性上采样 4× → 残差 logits",
    PURPLE, fs=11.0)

seg(6.0, 8.25, 6.0, 8.03, color=PURPLE)                                    # 细化 → 上采样/4
seg(6.0, 7.20, 6.0, 6.98, color=PURPLE)                                    # → 拼接
flow([(2.02, 9.30), (2.02, 6.55), (2.6, 6.55)], color=GREEN)               # 降维 → 拼接
for y1, y2 in ((6.15, 5.93), (5.10, 4.88), (4.05, 3.83), (3.00, 2.78)):
    seg(5.0, y1, 5.0, y2, color=PURPLE)

# ---------------------------------------------------------------- 相加与输出
box(2.6, 0.90, 4.8, "相加：基线 logits + 残差 logits\n→ argmax → 21 类 mask（原图尺寸）",
    RED, fs=11.0, h=0.85)
flow([(9.8, 6.15), (9.8, 1.32), (7.4, 1.32)], color=GREY)                  # 基线 logits → 相加
seg(5.0, 1.95, 5.0, 1.75, color=PURPLE)

# ---------------------------------------------------------------- 图例
ax.text(0.35, 0.35, "图例：蓝 = 骨干/输出   绿 = 浅层特征   橙 = ASPP 与基线路径   "
                    "紫 = 自研解码器   红 = 残差融合   灰虚线 = 辅助监督（仅训练期）",
        fontsize=9.8, color=GREY)
ax.text(0.35, 0.05, "关键：残差头 1×1 卷积零初始化，训练第 0 步与基线严格等价（实测差异 0.0），最差退化为基线，"
                    "不破坏 COCO 预训练权重。",
        fontsize=9.8, color="#333333")

plt.tight_layout()
p = os.path.join(OUT, "network_diagram.png")
plt.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
print("saved", p, os.path.getsize(p), "bytes")
