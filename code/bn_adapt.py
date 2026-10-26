# -*- coding: utf-8 -*-
"""BN 统计量域自适应（AdaBN / BNN 风格），纯无监督，不需要任何标签。

要解决的问题
------------
BN 层在推理时用的是训练阶段累积的 running_mean / running_var。如果目标域
（这里指测试集 COCO test2014）的激活分布和源域（VOC2012 + SBD）不一致，
这些统计量就是错的，特征会整体偏移，逐层放大。

更糟的是本项目的训练设置：`freeze_backbone_bn()` 在训练时把 ResNet 骨干的 BN
**全部设为 eval 并冻结**（因为 8 GB 显存只够 batch=4，BN 统计噪声太大）。
后果是——**骨干的 BN 统计量从始至终都是 torchvision COCO 预训练那套，
一次都没在 VOC 上更新过**。这是现成的、明确不匹配的一层，AdaBN 正好补这个洞。

做法
----
    1. model.eval()  —— 先关掉 Dropout 等训练行为
    2. 把每个 BN 单独切回 train()，并 reset_running_stats()
    3. momentum=None → PyTorch 用累积移动平均，样本越多估计越稳
    4. 无梯度前向遍历目标域图像若干轮，running stats 自然被重估
    5. 切回 eval()，保存权重

只改 running_mean / running_var / num_batches_tracked，**不动任何可学习参数**
（weight/bias 保持原样），所以这是一个"零风险可回退"的操作：效果不好就直接用原权重。

合法性说明（报告里要写清）
------------------------
作业只要求测试集做推理、输出 mask，没有禁止测试时自适应。本方法：
    - 不使用任何标签（COCO 测试集本来也没标签）
    - 不改变模型参数，只重新估计归一化统计量
    - 不针对单个样本输出，是标准的 transductive BN adaptation（AdaBN, Li et al. 2016）
属于"提升域鲁棒性"的合法手段，与"用测试标签调参"有本质区别。

用法
----
    # 探针：在 VOC val 上自适应再评估（量化"同分布下的收益基线"）
    python bn_adapt.py --ckpt ..\runs\sbd_lfca_e30\ckpt\best.pt \
        --adapt_voc_split val --eval_split val --iterations 3

    # 正式：用 400 张 COCO 测试图自适应（提交用这个）
    python bn_adapt.py --ckpt ..\runs\sbd_lfca_e30\ckpt\best.pt \
        --adapt_test_dir ..\data\test_images --iterations 5 \
        --save_ckpt ..\runs\sbd_lfca_e30\ckpt\best_adabn.pt
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop

from metrics import ConfusionMatrix, format_table
from model import load_for_infer
from train import validate
from voc_dataset import (IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, VOCSegmentation,
                         read_split)


class AdaptCrops(Dataset):
    """把图像随机缩放 + 裁剪成固定尺寸，专供 BN 自适应使用。

    为什么不用 VOCSegmentation 的 val 模式：那一版保持长宽比 resize，各图尺寸不同，
    DataLoader 无法拼 batch（BN 统计必须靠 batch 才有意义）。
    而 BN 自适应关心的是"目标域的激活分布"，用与训练一致的随机裁剪取样即可，
    长宽比和原图完整性都不重要。
    """

    def __init__(self, files, crop_size=513, scale_range=(0.5, 2.0), flip_prob=0.5):
        super().__init__()
        self.files = list(files)
        if not self.files:
            raise ValueError("自适应图像列表为空")
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        s = random.uniform(*self.scale_range)
        w, h = img.size
        img = img.resize((max(int(round(w * s)), 1), max(int(round(h * s)), 1)),
                         Image.BILINEAR)
        w, h = img.size
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        if pad_h or pad_w:
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
        i, j, ch, cw = RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, ch, cw)
        if random.random() < self.flip_prob:
            img = TF.hflip(img)
        return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)


def collect_files(voc_root="", voc_split="", test_dir=""):
    """收集自适应用的图像路径。VOC 走 split txt，COCO 走目录枚举。"""
    if test_dir:
        return [os.path.join(test_dir, n) for n in sorted(os.listdir(test_dir))
                if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    ids = read_split(voc_root, voc_split)
    img_dir = os.path.join(voc_root, "JPEGImages")
    files = [os.path.join(img_dir, i + ".jpg") for i in ids]
    return [f for f in files if os.path.isfile(f)]


def list_bn(model):
    return [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))]


def bn_snapshot(model):
    """保存所有 BN 的 running stats，用于对比"改了多少"。"""
    snap = {}
    for i, m in enumerate(list_bn(model)):
        snap[i] = (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
    return snap


def bn_shift(snap_before, snap_after):
    """统计统计量的相对变化幅度，确认自适应真的起作用了（而不是静默没跑）。"""
    rel = []
    for k in snap_before:
        m0, v0, n0 = snap_before[k]
        m1, v1, n1 = snap_after[k]
        if m0 is None or m1 is None:
            continue
        dm = (m1 - m0).abs().mean().item()
        norm = m0.abs().mean().item() + 1e-8
        rel.append(dm / norm)
    return (sum(rel) / len(rel)) if rel else 0.0


@torch.no_grad()
def adapt_bn(model, loader, device, iterations=3, momentum=None, reset=True,
             verbose=True):
    """用无标签目标域图像重估 BN 统计量。返回 (前快照, 后快照, 处理批次数)。

    reset 的含义（这是个关键选择，两种策略差别很大）：
        reset=True   清空原有统计量，完全用目标域重新估计。
                     激进，但会丢掉源域训练时累积的稳定统计（本项目实测为负收益）。
        reset=False  保留原统计量，用较小的 momentum 让它们朝目标域缓慢偏移。
                     温和，既不丢历史稳定性，又能修正域偏移——推荐。
    """
    before = bn_snapshot(model)

    # 顺序关键：先整体 eval() 关掉训练行为，再把 BN 单独切回 train()
    model.eval()
    for m in list_bn(model):
        if reset:
            m.reset_running_stats()
        m.momentum = momentum
        m.train()

    n_batches = 0
    t0 = time.time()
    for it in range(iterations):
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            model(x)                       # no_grad：只更新 running stats，不算梯度
            n_batches += 1
        if verbose:
            print(f"  pass {it + 1}/{iterations}  batches={n_batches}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    model.eval()
    after = bn_snapshot(model)
    return before, after, n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapt_voc_split", default="", help="用 VOC 的某个 split 做自适应")
    ap.add_argument("--adapt_test_dir", default="", help="用测试图目录做自适应（正式用）")
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--eval_split", default="", help="自适应后在哪个 VOC split 上评估")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--crop_size", type=int, default=513,
                    help="自适应时的随机裁剪尺寸，应等于训练时的 crop_size")
    ap.add_argument("--limit", type=int, default=0, help="自适应图像只用前 N 张")
    ap.add_argument("--momentum", type=float, default=-1,
                    help="<0 表示用累积移动平均（仅 reset=True 时合适）；>=0 用固定 momentum")
    ap.add_argument("--no_reset", action="store_true",
                    help="保留原 BN 统计量，只用目标域小步修正（配合 --momentum 0.02~0.1）")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_ckpt", default="")
    ap.add_argument("--save_txt", default="")
    args = ap.parse_args()

    if not args.adapt_voc_split and not args.adapt_test_dir:
        raise SystemExit("必须指定 --adapt_voc_split 或 --adapt_test_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    momentum = None if args.momentum < 0 else args.momentum

    model, ck = load_for_infer(args.ckpt, device)
    print(f"权重：{args.ckpt}", flush=True)
    print(f"BN 层数：{len(list_bn(model))}", flush=True)

    # ---- 自适应前先在评估集上测一遍，作为对照 ----
    if args.eval_split:
        ds = VOCSegmentation(args.voc_root, args.eval_split, train=False,
                             base_size=args.base_size)
        loader_eval = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)
        cm0 = validate(model, loader_eval, device, num_classes=NUM_CLASSES,
                       use_amp=(device.type == "cuda"))
        print(f"\n[自适应前] mIoU21={cm0.miou(True):.4f}  mIoU20={cm0.miou(False):.4f}  "
              f"pixAcc={cm0.pixel_accuracy:.4f}", flush=True)

    # ---- 构造自适应数据（固定尺寸随机裁剪，保证能拼 batch） ----
    files = collect_files(args.voc_root, args.adapt_voc_split, args.adapt_test_dir)
    if args.limit:
        files = files[:args.limit]
    adapt_ds = AdaptCrops(files, crop_size=args.crop_size)
    adapt_loader = DataLoader(adapt_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"\n自适应数据：{len(adapt_ds)} 张，crop={args.crop_size}，batch={args.batch_size}，"
          f"轮数={args.iterations}，momentum={momentum}", flush=True)

    # ---- 执行自适应 ----
    before, after, nb = adapt_bn(model, adapt_loader, device, iterations=args.iterations,
                                 momentum=momentum, reset=not args.no_reset)
    shift = bn_shift(before, after)
    print(f"自适应完成：累计前向 {nb} 个 batch，BN 统计量平均相对变化 {shift * 100:.2f}%",
          flush=True)
    if nb == 0:
        raise RuntimeError("没有前向任何 batch，自适应未生效")
    if shift < 1e-4:
        raise RuntimeError("BN 统计量几乎没变化，自适应可能未生效（检查 momentum / train 模式）")

    # ---- 自适应后评估 ----
    if args.eval_split:
        cm1 = validate(model, loader_eval, device, num_classes=NUM_CLASSES,
                       use_amp=(device.type == "cuda"))
        d = cm1.miou(True) - cm0.miou(True)
        print(f"[自适应后] mIoU21={cm1.miou(True):.4f}  mIoU20={cm1.miou(False):.4f}  "
              f"pixAcc={cm1.pixel_accuracy:.4f}", flush=True)
        print(f"[差值]     mIoU21 {d:+.4f}", flush=True)
        print("\n逐类 IoU：\n" + format_table(cm1.per_class_table()), flush=True)
        if args.save_txt:
            with open(args.save_txt, "w", encoding="utf-8") as f:
                f.write(f"ckpt={args.ckpt}\nadapt_test={args.adapt_test_dir or args.adapt_voc_split}\n")
                f.write(f"eval_split={args.eval_split} iterations={args.iterations} "
                        f"batch={args.batch_size}\n")
                f.write(f"before mIoU21={cm0.miou(True):.4f}\n")
                f.write(f"after  mIoU21={cm1.miou(True):.4f}\n")
                f.write(f"delta={d:+.4f}\n\n")
                f.write(format_table(cm1.per_class_table()) + "\n")
            print(f"已保存：{args.save_txt}", flush=True)

    # ---- 保存 ----
    if args.save_ckpt:
        ck = ck if isinstance(ck, dict) else {"model": ck}
        ck["model"] = model.state_dict()
        ck.setdefault("args", {})
        ck["args"]["adabn"] = {"source": args.adapt_test_dir or args.adapt_voc_split,
                               "iterations": args.iterations,
                               "batch_size": args.batch_size,
                               "bn_shift": round(shift, 6)}
        os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt)), exist_ok=True)
        torch.save(ck, args.save_ckpt)
        print(f"已保存自适应权重：{args.save_ckpt}", flush=True)


if __name__ == "__main__":
    main()
