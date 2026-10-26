# -*- coding: utf-8 -*-
"""自研模块：LFCA-DeepLabV3（低层特征融合 + 通道注意力解码器）

对应作业"Designing a new network model is encouraged"这一条。

设计动机（可直接写进报告）：
    DeepLabV3 的 ASPP 输出位于 stride=8 的低分辨率特征图上，虽然多尺度感受野强，
    但物体边界（细长结构、小目标轮廓）在上采样回原图时会被抹平。
    标准做法（DeepLabV3+）是引入 backbone 的低层特征（stride=4，含丰富空间细节）来修复边界。
    本模块在此基础上再加一个**通道注意力（SE）**，让网络自己决定"哪些通道的语义特征
    与低层空间细节更值得融合"，而不是无差别拼接。

结构（在 torchvision DeepLabV3-ResNet50 之上构建，尽可能复用预训练权重）：
    ResNet-50(dilation) ── layer1(低级,256ch@/4) ──────────┐
                        └─ layer4 ─ ASPP(256ch@/8) ─ 卷积细化 ─┤
                                                              ↓
                        低层 1×1 降维 256→48  ←────────────────┘
                        拼接 [低层48 ‖ ASPP上采样256] = 304
                        → 3×3 卷积 → SE 通道注意力 → 3×3 卷积 → Dropout
                        → 复用预训练的 1×1 分类头(256→21) → 上采样回原图

复用预训练权重的部分（重要，避免随机初始化破坏已有能力）：
    - backbone（ResNet-50，COCO with VOC labels 预训练）
    - ASPP 及后续 256 通道细化卷积（classifier[0..3]）
    - 最终分类卷积 classifier[4]（256→21，因为解码器输出刻意保持 256 通道）
    新增参数只有：低层降维 1×1、两个 3×3 卷积、SE 的两个全连接层（约 0.9 M）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
)

from voc_dataset import NUM_CLASSES

BACKBONES = {
    "resnet50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    "resnet101": (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1),
}


class SEBlock(nn.Module):
    """通道注意力（Squeeze-and-Excitation）。先压缩到 1/r 再恢复，用 sigmoid 生成通道权重。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class LFCADecoder(nn.Module):
    """低层特征融合 + 通道注意力解码器。输出通道固定 256，以便复用预训练分类头。"""

    def __init__(self, low_channels=256, aspp_channels=256, out_channels=256,
                 low_out=48, use_attention=True, dropout=0.1):
        super().__init__()
        self.low_reduce = nn.Sequential(
            nn.Conv2d(low_channels, low_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_out),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(aspp_channels + low_out, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = SEBlock(out_channels) if use_attention else nn.Identity()
        self.fuse2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, low_feat, aspp_feat):
        # aspp 位于 stride=8，低层特征位于 stride=4，先把 aspp 上采样对齐
        aspp_up = F.interpolate(aspp_feat, size=low_feat.shape[-2:],
                                mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, self.low_reduce(low_feat)], dim=1)
        x = self.fuse1(x)
        x = self.attention(x)      # 通道注意力：让网络自己挑该融合哪些语义通道
        return self.fuse2(x)


class LFCADeepLabV3(nn.Module):
    """带自研解码器的 DeepLabV3。前向返回 {"out": logits, "aux": aux_logits}。

    重要设计（v2 残差式）：
        第一版把随机初始化的解码器直接接在预训练分类头之前，实测前 8 个 epoch
        全面落后基线 1~3 个点 —— 新模块毁掉了预训练权重带来的好起点。
        因此改成**残差修正**：
            logits = 原 DeepLabV3 输出（完整预训练路径） + 解码器输出（零初始化卷积）
        这样训练第 0 步时模型严格等价于基线，解码器只需学习"增量修正"，
        最差情况退回基线，不会更糟。
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, aux_loss=True,
                 use_attention=True, residual=True, backbone="resnet50"):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"未知骨干 {backbone}，可选 {list(BACKBONES)}")
        ctor, w_enum = BACKBONES[backbone]
        weights = w_enum if pretrained else None
        base = ctor(weights=weights, aux_loss=aux_loss)
        assert base.classifier[-1].out_channels == num_classes, "预训练头类别数应为 21"

        self.backbone = base.backbone           # ResNet（空洞卷积版），保留全部预训练权重
        self.aspp = base.classifier[0]          # ASPP：2048 → 256 @ stride8
        self.mid = nn.Sequential(*[base.classifier[i] for i in (1, 2, 3)])  # 256 通道细化卷积
        self.head = base.classifier[4]          # 复用预训练分类头 256 → 21（基线路径）
        self.aux_classifier = base.aux_classifier if aux_loss else None
        self.decoder = LFCADecoder(use_attention=use_attention)
        self.residual = residual
        # 残差头：零初始化，保证初始状态与基线完全一致
        self.res_head = nn.Conv2d(256, num_classes, kernel_size=1)
        if residual:
            nn.init.zeros_(self.res_head.weight)
            nn.init.zeros_(self.res_head.bias)
        self.aux_loss = aux_loss

    def forward(self, x):
        input_size = x.shape[-2:]
        bb = self.backbone
        f = bb.relu(bb.bn1(bb.conv1(x)))
        f = bb.maxpool(f)
        low = bb.layer1(f)          # stride 4，256 通道，空间细节丰富
        f2 = bb.layer2(low)
        f3 = bb.layer3(f2)
        f4 = bb.layer4(f3)          # stride 8，2048 通道

        aspp_feat = self.mid(self.aspp(f4))

        # 基线路径：与原始 DeepLabV3 完全一致的预训练通路
        base_logits = F.interpolate(self.head(aspp_feat), size=input_size,
                                    mode="bilinear", align_corners=False)
        if self.residual:
            # 残差路径：低层特征 + 通道注意力解码，只学增量
            dec_feat = self.decoder(low, aspp_feat)
            res = F.interpolate(self.res_head(dec_feat), size=input_size,
                                mode="bilinear", align_corners=False)
            logits = base_logits + res
        else:
            # 第一版（直接替换式），保留以便做消融对照
            logits = F.interpolate(self.head(self.decoder(low, aspp_feat)),
                                   size=input_size, mode="bilinear", align_corners=False)

        out = {"out": logits}
        if self.aux_loss and self.aux_classifier is not None:
            # aux 分支位于 stride=8，必须插值回输入尺寸才能和标签算 loss
            aux = self.aux_classifier(f3)
            aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
            out["aux"] = aux
        return out


def build_model_v2(num_classes=NUM_CLASSES, pretrained=True, aux_loss=True,
                   use_attention=True, residual=True, backbone="resnet50"):
    return LFCADeepLabV3(num_classes=num_classes, pretrained=pretrained,
                         aux_loss=aux_loss, use_attention=use_attention,
                         residual=residual, backbone=backbone)


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    m = build_model_v2()
    n_new = count_params(m.decoder)
    print("总可训练参数: %.3f M" % (count_params(m) / 1e6))
    print("其中自研解码器新增: %.3f M" % (n_new / 1e6))
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 257, 257))
    print("out:", tuple(y["out"].shape), "aux:", tuple(y["aux"].shape))
    # 冒烟必须断言尺寸：aux 若没插值回输入尺寸，训练时会直接报错
    assert y["out"].shape[-2:] == (257, 257), "主输出尺寸不对"
    assert y["aux"].shape[-2:] == (257, 257), "aux 输出未插值回输入尺寸"
    print("尺寸断言通过：out / aux 均与输入同分辨率")

    # 关键验证：残差头零初始化 ⇒ 初始输出应与原版 DeepLabV3 完全一致
    from model import build_model

    ref = build_model().eval()
    xx = torch.randn(2, 3, 257, 257)
    with torch.no_grad():
        diff = (m(xx)["out"] - ref(xx)["out"]).abs().max().item()
    print("与基线 DeepLabV3 的初始输出最大差异:", diff)
    assert diff < 1e-5, "残差初始化失败：初始输出应与基线一致"
    print("残差初始化验证通过：训练第 0 步等价于基线")
