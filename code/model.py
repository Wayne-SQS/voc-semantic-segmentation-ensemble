# -*- coding: utf-8 -*-
"""模型构建（对应作业任务点 2：Construct the semantic segmentation model）。

关于预训练权重的处理（这里刻意保守，避免踩坑）：
    torchvision 的 DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 这组权重
    **本身就是 VOC 标签体系（20 前景类 + background = 21 类输出）**，且在 COCO 上训练过。
    因此：
        1. 直接用 weights=... 实例化，不要传 num_classes，让它保留自带的 21 类分类头；
        2. 加载后断言输出通道数 == 21，确认无误后**不替换分类头**；
        3. 只有当类别数确实不匹配时才重建头部，并明确知道这部分参数是随机初始化的。
    否则会白白丢掉预训练分类头的参数。
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet101_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    fcn_resnet101,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)

from voc_dataset import NUM_CLASSES

# 支持的骨干：名字 -> (构造器, 预训练权重枚举)
BACKBONES = {
    "resnet50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    "resnet101": (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1),
}

# 异构零样本候选：架构与 DeepLabV3 不同，用来给集成提供"错误模式不同"的成员。
# 名字 -> (构造器, 权重枚举, 是否支持 aux_loss 参数)。全部是 COCO+VOC 21 类标签，无需改头。
HETERO_MODELS = {
    "fcn_resnet50": (fcn_resnet50, FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, True),
    "fcn_resnet101": (fcn_resnet101, FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, True),
    "deeplabv3_mobilenet_v3_large": (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1, True),
    "lraspp_mobilenet_v3_large": (
        lraspp_mobilenet_v3_large,
        LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1, False),
}


def build_model(num_classes=NUM_CLASSES, pretrained=True, aux_loss=True,
                backbone="resnet50"):
    if backbone not in BACKBONES:
        raise ValueError(f"未知骨干 {backbone}，可选 {list(BACKBONES)}")
    ctor, w_enum = BACKBONES[backbone]
    weights = w_enum if pretrained else None
    model = ctor(weights=weights, aux_loss=aux_loss)

    out_channels = model.classifier[-1].out_channels
    if out_channels != num_classes:
        # 只有确实不匹配时才替换，替换掉的头部参数是随机初始化（会在训练中重新学）
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    assert model.classifier[-1].out_channels == num_classes
    return model


def build_hetero(arch, pretrained=True, aux_loss=True):
    """构建异构零样本模型（FCN / MobileNetV3-DeepLabV3 / LRASPP）。

    这些权重都是 COCO_WITH_VOC_LABELS_V1，输出天然为 21 类，**不需要替换分类头**。
    与 build_model 分开是因为它们的 classifier 结构不同（LRASPP 没有 aux 分支、
    也没有 classifier[-1] 这种 Sequential），混在一起判断容易出错。
    """
    if arch not in HETERO_MODELS:
        raise ValueError(f"未知异构架构 {arch}，可选 {list(HETERO_MODELS)}")
    ctor, w_enum, supports_aux = HETERO_MODELS[arch]
    kwargs = {"weights": w_enum if pretrained else None}
    if supports_aux:
        kwargs["aux_loss"] = aux_loss
    model = ctor(**kwargs)
    if arch.startswith("lraspp"):
        out_ch = model.classifier.low_classifier.out_channels
    else:
        out_ch = model.classifier[-1].out_channels
    assert out_ch == NUM_CLASSES, f"{arch} 预训练头应为 {NUM_CLASSES} 类，实际 {out_ch}"
    return model


def freeze_backbone_bn(model):
    """小 batch 微调时的常用技巧：冻结骨干网络 BN 的统计量（不更新 running_mean/var）。

    8 GB 显存下 batch size 只有 4，BN 统计量噪声较大，冻结后通常更稳。
    注意：BN 的 affine 参数仍然可学习，只是不更新滑动统计。
    """
    n = 0
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            n += 1
    return n


def attach_low_level_hook(model):
    """给 W2 的自研解码器预留：抓取 ResNet layer1 的低级特征（stride=4, 256 通道）。

    用法：
        attach_low_level_hook(model)
        out = model(x)["out"]
        low = model._low_level        # [B, 256, H/4, W/4]
    """
    model._low_level = None

    def _hook(module, inp, out):
        model._low_level = out

    handle = model.backbone.layer1.register_forward_hook(_hook)
    model._low_level_handle = handle
    return model


def load_for_infer(ckpt_path, device, arch=None):
    """按 checkpoint 里记录的架构加载模型（评估/推理统一入口）。

    为什么要这个函数：不同类型模型的 state_dict 键名完全不同，若用 strict=False
    强行加载，PyTorch 会**静默跳过所有不匹配的键**，模型变成随机初始化，
    评估指标接近 0、推理出全背景的垃圾 mask —— 而且不报任何错。
    这里统一用 strict=True，加载不完整直接抛异常。
    """
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("args", {}) if isinstance(ck, dict) else {}
    arch = arch or cfg.get("model", "deeplabv3")
    bb = cfg.get("backbone", "resnet50")
    if arch in HETERO_MODELS:
        model = build_hetero(arch, pretrained=False, aux_loss=True)
    elif arch == "lfca":
        from model_v2 import build_model_v2
        model = build_model_v2(pretrained=False, aux_loss=True,
                               use_attention=not cfg.get("no_attention", False),
                               residual=not cfg.get("no_residual", False),
                               backbone=bb)
    else:
        model = build_model(pretrained=False, aux_loss=True, backbone=bb)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"权重加载不完整 missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval(), ck


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = build_model()
    print("输出通道数:", m.classifier[-1].out_channels)
    print("可训练参数:", count_params(m) / 1e6, "M")
    # 注意：ASPP 含全局池化分支 + BatchNorm，batch=1 时必须用 eval() 推理模式
    m.eval()
    x = torch.randn(2, 3, 513, 513)
    with torch.no_grad():
        y = m(x)
    print("out:", tuple(y["out"].shape), "aux:", tuple(y["aux"].shape))
