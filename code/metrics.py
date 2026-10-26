# -*- coding: utf-8 -*-
"""分割指标：数据集级混淆矩阵 + IoU / mIoU / 像素准确率。

口径（与老师给的参考工程一致：全量像素累加，不逐图平均）：
    - 真值中 255（ignore）的像素直接排除，不参与任何统计
    - 21 类 mIoU：含 background（0 类），标准 VOC 口径，本报告主指标
    - 20 类前景 mIoU：排除 background，作为辅助指标
    - 某类在本次评估中一个像素都没出现 / 一个都没预测到时，记为 None（不虚构为 0）
"""

import numpy as np

from voc_dataset import IGNORE_INDEX, NUM_CLASSES, VOC_CLASSES


class ConfusionMatrix:
    """num_classes × num_classes 混淆矩阵，行=真值，列=预测。"""

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.float64)

    def reset(self):
        self.mat[:] = 0.0

    def update(self, pred, target):
        """pred/target 为同形状的 numpy 数组或张量（H×W 或 N×H×W）。"""
        if not isinstance(pred, np.ndarray):
            pred = pred.detach().cpu().numpy()
        if not isinstance(target, np.ndarray):
            target = target.detach().cpu().numpy()
        pred = pred.reshape(-1).astype(np.int64)
        target = target.reshape(-1).astype(np.int64)

        keep = (target >= 0) & (target < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        if self.ignore_index is not None:
            keep &= (target != self.ignore_index)
        if not keep.any():
            return
        inds = self.num_classes * target[keep] + pred[keep]
        self.mat += np.bincount(inds, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    # ---------- 指标 ----------
    @property
    def iou(self):
        """逐类 IoU，缺席类返回 nan。"""
        diag = np.diag(self.mat)
        row = self.mat.sum(axis=1)   # 真值像素数
        col = self.mat.sum(axis=0)   # 预测像素数
        union = row + col - diag
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, diag / np.maximum(union, 1e-12), np.nan)
        return iou

    @property
    def pixel_accuracy(self):
        total = self.mat.sum()
        if total == 0:
            return float("nan")
        return float(np.diag(self.mat).sum() / total)

    @property
    def mean_class_accuracy(self):
        row = self.mat.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(row > 0, np.diag(self.mat) / np.maximum(row, 1e-12), np.nan)
        return float(np.nanmean(acc)) if np.isfinite(acc).any() else float("nan")

    def miou(self, include_background=True):
        iou = self.iou
        if not include_background:
            iou = iou[1:]
        return float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan")

    def per_class_table(self):
        """返回逐类 IoU 列表，缺席类用 None 表示（与参考工程一致，不虚构数据）。"""
        iou = self.iou
        row = self.mat.sum(axis=1)
        out = []
        for i in range(self.num_classes):
            out.append({
                "id": i,
                "name": VOC_CLASSES[i],
                "gt_pixels": int(row[i]),
                "iou": None if not np.isfinite(iou[i]) else float(iou[i]),
            })
        return out


def format_table(rows):
    """把逐类 IoU 打成可直接贴进报告的文本表。"""
    lines = ["| id | class | IoU |", "|---:|---|---:|"]
    for r in rows:
        v = "未出现" if r["iou"] is None else "%.4f" % r["iou"]
        lines.append("| %d | %s | %s |" % (r["id"], r["name"], v))
    return "\n".join(lines)
