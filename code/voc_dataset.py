# -*- coding: utf-8 -*-
"""VOC2012 语义分割数据加载（对应作业任务点 1：Load segmentation data）。

目录约定（已实测确认）：
    data/VOCdevkit/VOC2012/
        JPEGImages/*.jpg             17125 张原图
        SegmentationClass/*.png      2913 张像素级标签（0-20 类别，255 = ignore）
        ImageSets/Segmentation/{train,val,trainval}.txt
    data/test_images/*.jpg           400 张 COCO test2014 图像（无标签，作为最终测试集）

标签规则说明（重要）：
    VOC 官方 mask 为调色板 PNG，直接读出的索引值即类别编号：
    0 = background，1..20 = 20 个前景类，255 = 边界/难例（ignore）。
    训练与评估统一使用 ignore_index=255。
"""

import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

NUM_CLASSES = 21
IGNORE_INDEX = 255

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 类别调色板（仅用于可视化，提交结果必须是单通道灰度图，不用这个）
PALETTE = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
]


def read_split(voc_root, split):
    """读取 ImageSets/Segmentation/<split>.txt，返回图像 id 列表（不含扩展名）。"""
    path = os.path.join(voc_root, "ImageSets", "Segmentation", split + ".txt")
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip().split()[0] for line in f if line.strip()]
    return ids


class VOCSegmentation(Dataset):
    """VOC2012 语义分割数据集。train=True 时做随机缩放/裁剪/翻转/颜色抖动。"""

    def __init__(self, voc_root, split="train", train=True, crop_size=513,
                 base_size=513, scale_range=(0.5, 2.0), flip_prob=0.5, jitter=True,
                 sbd_img_dir=None, sbd_mask_dir=None):
        super().__init__()
        self.voc_root = voc_root
        self.split = split
        self.train = train
        self.crop_size = crop_size
        self.base_size = base_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.jitter = jitter
        self.ids = read_split(voc_root, split)
        self.img_dir = os.path.join(voc_root, "JPEGImages")
        self.mask_dir = os.path.join(voc_root, "SegmentationClass")
        # SBD 增强集回退路径：train_aug 中 SBD 独有图像/掩码不在 VOC 目录，需从这里取
        self.sbd_img_dir = sbd_img_dir or os.path.normpath(os.path.join(
            voc_root, "..", "..", "sbd", "benchmark_RELEASE", "dataset", "img"))
        self.sbd_mask_dir = sbd_mask_dir or r"E:\桌面\人工智能技术大作业\data\sbd_masks"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        img_path = os.path.join(self.img_dir, idx + ".jpg")
        if not os.path.isfile(img_path) and self.sbd_img_dir:
            img_path = os.path.join(self.sbd_img_dir, idx + ".jpg")
        mask_path = os.path.join(self.mask_dir, idx + ".png")
        if not os.path.isfile(mask_path) and self.sbd_mask_dir:
            mask_path = os.path.join(self.sbd_mask_dir, idx + ".png")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.train:
            img, mask = self._random_scale(img, mask)
            img, mask = self._pad_and_crop(img, mask)
            if random.random() < self.flip_prob:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if self.jitter:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
                img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))
        else:
            img, mask = self._resize_keep_ratio(img, mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        target = np.array(mask, dtype=np.int64)  # 值域 {0..20, 255}
        return img, torch.from_numpy(target)

    # ---------- 变换工具 ----------
    def _random_scale(self, img, mask):
        s = random.uniform(*self.scale_range)
        w, h = img.size
        nw, nh = max(int(round(w * s)), 1), max(int(round(h * s)), 1)
        img = img.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        return img, mask

    def _pad_and_crop(self, img, mask):
        w, h = img.size
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        if pad_h or pad_w:
            # 图像用 0 填充、标签用 ignore(255) 填充，避免凭空制造监督信号
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
            mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)
        i, j, ch, cw = RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        return TF.crop(img, i, j, ch, cw), TF.crop(mask, i, j, ch, cw)

    def _resize_keep_ratio(self, img, mask):
        w, h = img.size
        scale = self.base_size / float(max(h, w))
        nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
        img = img.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        # 输出步长 16，补齐到 16 的倍数，避免尺寸不兼容
        pad_h = (16 - nh % 16) % 16
        pad_w = (16 - nw % 16) % 16
        if pad_h or pad_w:
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
            mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)
        return img, mask


class VOCTestDataset(Dataset):
    """400 张无标签测试图（COCO test2014）。只做 resize + 归一化，返回原图尺寸用于还原。"""

    def __init__(self, img_dir, base_size=513, limit=None):
        super().__init__()
        self.img_dir = img_dir
        self.base_size = base_size
        names = sorted([n for n in os.listdir(img_dir)
                        if n.lower().endswith((".jpg", ".jpeg", ".png"))])
        self.names = names[:limit] if limit else names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        orig_w, orig_h = img.size
        scale = self.base_size / float(max(orig_h, orig_w))
        nw, nh = max(int(round(orig_w * scale)), 1), max(int(round(orig_h * scale)), 1)
        img_r = img.resize((nw, nh), Image.BILINEAR)
        pad_h = (16 - nh % 16) % 16
        pad_w = (16 - nw % 16) % 16
        if pad_h or pad_w:
            img_r = TF.pad(img_r, (0, 0, pad_w, pad_h), fill=0)
        x = TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD)
        return x, (orig_h, orig_w), name
