# -*- coding: utf-8 -*-
"""训练 + 每 epoch 验证 + TensorBoard 可视化（作业任务点 4、5、7）。

保守起步配置（8 GB 显存 / RTX 4060 Laptop）：
    DeepLabV3-ResNet50，crop 513×513，batch=4 + 梯度累积 4（等效 16），
    AMP 混合精度，输出步长 16，SGD(momentum=0.9) + poly 学习率衰减。
    先用 --epochs 5 跑通流程，确认收敛与指标正常后再拉长训练。

TensorBoard：
    tensorboard --logdir runs
"""

import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from metrics import ConfusionMatrix
from model import build_model, freeze_backbone_bn
from voc_dataset import IGNORE_INDEX, NUM_CLASSES, PALETTE, VOCSegmentation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def colorize(mask_np):
    """把类别索引图转成彩色 RGB（仅用于 TensorBoard 可视化）。"""
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(PALETTE):
        out[mask_np == cid] = color
    return out


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler,
                    device, epoch, accum_steps, writer, global_step, use_amp,
                    freeze_bn=False):
    model.train()
    if freeze_bn:
        # model.train() 会把**所有** BN 重新切回训练模式，包括之前被冻结的那些。
        # 所以冻结操作必须每个 epoch 都做一次，否则只在训练开始前调用一次等于无效
        # ——这正是本项目最初 30 轮训练里 freeze_bn 形同虚设的原因（详见报告 5.9 节）。
        freeze_backbone_bn(model)
    total_loss, seen = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(images)
            loss = criterion(out["out"], targets)
            if "aux" in out and out["aux"] is not None:
                loss = loss + 0.4 * criterion(out["aux"], targets)

        scaler.scale(loss / accum_steps).backward()

        if (it + 1) % accum_steps == 0 or (it + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if writer is not None and global_step % 20 == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        total_loss += loss.item() * images.size(0)
        seen += images.size(0)

        if it % 50 == 0:
            print(f"  epoch {epoch} iter {it}/{len(loader)} loss {loss.item():.4f} "
                  f"{(time.time() - t0):.0f}s", flush=True)

    return total_loss / max(seen, 1), global_step


@torch.no_grad()
def validate(model, loader, device, num_classes=NUM_CLASSES, use_amp=True,
             log_images=None, writer=None, epoch=None):
    model.eval()
    cm = ConfusionMatrix(num_classes, IGNORE_INDEX)
    vis = []
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)["out"]
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear",
                               align_corners=False)
        pred = logits.argmax(dim=1).cpu()
        cm.update(pred, targets)

        if log_images is not None and i < log_images:
            rgb = (images[0].cpu().float() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                   + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)).clamp(0, 1).numpy()
            rgb = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
            # 上下拼接：原图 | 预测彩色图
            vis.append(np.concatenate([rgb, colorize(pred[0].numpy())], axis=0))
    if writer is not None and vis and epoch is not None:
        # 验证集保持原始长宽比，各图尺寸不同，必须逐张单独写入
        for j, v in enumerate(vis):
            writer.add_image(f"val/pred_{j}", v, epoch, dataformats="HWC")
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--accum_steps", type=int, default=4, help="梯度累积，等效 batch=16")
    ap.add_argument("--crop_size", type=int, default=513)
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--lr", type=float, default=0.01, help="分类头学习率")
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1, help="骨干学习率倍数")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--model", default="deeplabv3", choices=["deeplabv3", "lfca"],
                    help="deeplabv3=基线；lfca=自研低层特征融合+通道注意力解码器")
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"],
                    help="骨干网络；resnet101 容量更大但显存占用和耗时也更高")
    ap.add_argument("--no_attention", action="store_true",
                    help="仅 lfca 生效：去掉通道注意力，用于消融对比")
    ap.add_argument("--no_residual", action="store_true",
                    help="仅 lfca 生效：解码器直接替换原分类头（第一版，效果较差，仅作对照）")
    ap.add_argument("--skip_val", action="store_true",
                    help="最终在 trainval 上训练时用：跳过验证（此时 val 已被包含进训练集，"
                         "验证指标无意义且白白耗时）")
    ap.add_argument("--freeze_bn", action="store_true", help="小 batch 下冻结骨干 BN 统计量")
    ap.add_argument("--out_dir", default=r"E:\桌面\人工智能技术大作业\runs")
    ap.add_argument("--run_name", default="baseline_deeplabv3_r50")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| torch:", torch.__version__, flush=True)

    # ---------- 数据 ----------
    train_set = VOCSegmentation(args.voc_root, args.train_split, train=True,
                                crop_size=args.crop_size)
    val_set = VOCSegmentation(args.voc_root, args.val_split, train=False,
                              base_size=args.base_size)
    print(f"train={len(train_set)} val={len(val_set)}", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---------- 模型 ----------
    if args.model == "lfca":
        from model_v2 import build_model_v2
        model = build_model_v2(pretrained=not args.no_pretrain, aux_loss=True,
                               use_attention=not args.no_attention,
                               residual=not args.no_residual,
                               backbone=args.backbone).to(device)
        print(f"模型: LFCA-DeepLabV3/{args.backbone} (通道注意力={not args.no_attention}, "
              f"残差式={not args.no_residual})", flush=True)
    else:
        model = build_model(pretrained=not args.no_pretrain, aux_loss=True,
                            backbone=args.backbone).to(device)
        print(f"模型: DeepLabV3-{args.backbone} (基线)", flush=True)
    if args.freeze_bn:
        print("frozen BN:", freeze_backbone_bn(model), flush=True)

    backbone_params, head_params = [], []
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    for p in model.parameters():
        if not p.requires_grad:      # 冻结的 BN 参数不进优化器
            continue
        (backbone_params if id(p) in backbone_ids else head_params).append(p)
    optimizer = torch.optim.SGD([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # 注意：len(loader) 不一定被 accum_steps 整除，末尾不足一批时仍会走一次 optimizer.step，
    # 所以要用 ceil 计算真实步数；同时 clamp 到非负，否则 (负数)**0.9 会得到复数 lr 直接崩。
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    max_steps = max(steps_per_epoch * args.epochs, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / float(max_steps)) ** 0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    start_epoch, best_miou = 0, -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_miou", -1.0)
        print(f"resume from {args.resume}, epoch {start_epoch}", flush=True)

    log_dir = os.path.join(args.out_dir, args.run_name)
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_dir = os.path.join(log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    hist_path = os.path.join(log_dir, "history.csv")
    if not (args.resume and os.path.isfile(hist_path)):
        # 续训时不要覆盖已有记录
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_miou21", "val_miou20",
                                    "val_pixel_acc", "val_mean_acc", "lr", "minutes"])

    if start_epoch > 0:
        # 续训时让学习率接着之前的进度衰减，否则会从 base lr 重新开始（相当于 lr 突增）
        scheduler.last_epoch = start_epoch * steps_per_epoch

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device,
            epoch, args.accum_steps, writer, global_step, use_amp,
            freeze_bn=args.freeze_bn)

        writer.add_scalar("train/loss_epoch", train_loss, epoch)

        if args.skip_val:
            miou21 = miou20 = pacc = macc = float("nan")
        else:
            cm = validate(model, val_loader, device, use_amp=use_amp,
                          log_images=2, writer=writer, epoch=epoch)
            miou21, miou20 = cm.miou(True), cm.miou(False)
            pacc, macc = cm.pixel_accuracy, cm.mean_class_accuracy
            writer.add_scalar("val/mIoU_21", miou21, epoch)
            writer.add_scalar("val/mIoU_20", miou20, epoch)
            writer.add_scalar("val/pixel_acc", pacc, epoch)
            writer.add_scalar("val/mean_class_acc", macc, epoch)

        minutes = (time.time() - t0) / 60.0
        print(f"[epoch {epoch}] loss={train_loss:.4f} mIoU21={miou21:.4f} "
              f"mIoU20={miou20:.4f} pixAcc={pacc:.4f} ({minutes:.1f} min)", flush=True)

        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{miou21:.6f}",
                                    f"{miou20:.6f}", f"{pacc:.6f}", f"{macc:.6f}",
                                    f"{optimizer.param_groups[0]['lr']:.8f}",
                                    f"{minutes:.2f}"])

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_miou": best_miou, "args": vars(args)}
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if miou21 > best_miou:
            best_miou = miou21
            state["best_miou"] = best_miou
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))

    # 最后一次评估的逐类 IoU，写进日志方便报告取数（跳过验证时无 cm）
    if not args.skip_val:
        rows = cm.per_class_table()
        with open(os.path.join(log_dir, "per_class_iou.txt"), "w", encoding="utf-8") as f:
            from metrics import format_table
            f.write(format_table(rows) + "\n")
            f.write(f"\nmIoU(21)={cm.miou(True):.4f}  mIoU(20)={cm.miou(False):.4f}\n")
    writer.close()
    print(f"best mIoU21 = {best_miou:.4f} | logs: {log_dir}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
