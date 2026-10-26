# -*- coding: utf-8 -*-
"""检测辅助的语义分割融合（YOLOv8-seg + 自训语义模型）。

动机（可写进报告）：
    测试集来自 COCO，而我们的分割模型只在 VOC（1464 张）上微调过，存在明显域偏移：
    人群被合并成一整块、diningtable 大面积误报。YOLOv8 是在 COCO 上训练的检测/实例分割模型，
    天然适应测试域，且能区分**相邻的不同实例**（这正是纯语义分割模型的短板）。
    因此考虑用 YOLO 的实例级结果去修正语义模型的像素级输出。

融合规则（阈值在验证集上网格搜索确定，不靠拍脑袋）：
    规则 A（补漏）：YOLO 高置信判定为某 VOC 类，而语义模型不确定或判为背景 → 采用 YOLO 类
    规则 B（去伪）：语义模型判为前景但 YOLO 未检出任何目标且置信度低 → 改为背景

用法（先在验证集子集上量化增益）：
    python yolo_fusion.py --ckpt ..\runs\lfca_res_e30\ckpt\best.pt --num 200
"""

import argparse
import time

import numpy as np
import torch

from metrics import ConfusionMatrix
from model import load_for_infer
from voc_dataset import IGNORE_INDEX, NUM_CLASSES, VOC_CLASSES

# COCO 80 类中能对应到 VOC 20 类的部分（COCO 类别名 → VOC 类别下标）
COCO_TO_VOC = {
    "person": 15, "bicycle": 2, "car": 7, "motorcycle": 14, "airplane": 1,
    "bus": 6, "train": 19, "boat": 4, "bird": 3, "cat": 8, "dog": 12,
    "horse": 13, "sheep": 17, "cow": 10, "bottle": 5, "chair": 9,
    "couch": 18, "dining table": 11, "potted plant": 16, "tv": 20,
}


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
COCO_NAMES_TO_ID = {i: COCO_TO_VOC[n] for i, n in enumerate(COCO_NAMES) if n in COCO_TO_VOC}


class YoloSeg:
    """YOLOv8-seg 包装：输出与原图同尺寸的实例类别图（-1 表示无目标）。"""

    def __init__(self, weights="yolov8n-seg.pt", conf=0.35, device="cuda", imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    @torch.no_grad()
    def __call__(self, rgb_u8, hw):
        """rgb_u8: (H,W,3) uint8 RGB；返回 label(H,W) int（-1 无目标）与 conf(H,W) float。"""
        h, w = hw
        res = self.model.predict(rgb_u8, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, retina_masks=True,
                                 verbose=False)[0]
        label = np.full((h, w), -1, dtype=np.int64)
        conf = np.zeros((h, w), dtype=np.float32)
        if res.masks is None:
            return label, conf
        masks = res.masks.data.detach().cpu().numpy()          # (N, Hm, Wm)
        cls = res.boxes.cls.detach().cpu().numpy().astype(int)
        cf = res.boxes.conf.detach().cpu().numpy()
        for i in range(masks.shape[0]):
            voc_id = COCO_NAMES_TO_ID.get(int(cls[i]), None)
            if voc_id is None:
                continue
            m = masks[i]
            if m.shape != (h, w):
                from PIL import Image
                m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
                    (w, h), Image.NEAREST)) > 127
            else:
                m = m > 0.5
            m = m.astype(bool)
            # 只覆盖尚未被更高置信实例占据的像素
            take = m & (conf < cf[i])
            label[take] = voc_id
            conf[take] = cf[i]
        return label, conf


def fuse(sem_prob, yolo_label, yolo_conf, tau_sem, tau_yolo=0.35,
         rule_b_tau=None):
    """融合。sem_prob: (C,H,W) float；返回 (H,W) 类别索引。"""
    pred = sem_prob.argmax(axis=0)
    maxp = sem_prob.max(axis=0)
    out = pred.copy()

    if yolo_label is not None:
        hit = (yolo_label >= 0) & (yolo_conf >= tau_yolo)
        # 规则 A：YOLO 有判定，且语义模型不确定 / 判成背景 → 采用 YOLO 类
        need = hit & ((maxp < tau_sem) | (pred == 0))
        out[need] = yolo_label[need]

    if rule_b_tau is not None:
        # 规则 B：语义模型判前景但置信低、且 YOLO 完全没检出 → 改为背景
        no_obj = (yolo_label < 0) if yolo_label is not None else np.ones_like(pred, bool)
        wrong = (pred != 0) & (maxp < rule_b_tau) & no_obj
        out[wrong] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--voc_root", default=r"E:\桌面\人工智能技术大作业\data\VOCdevkit\VOC2012")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--base_size", type=int, default=513)
    ap.add_argument("--yolo", default="yolov8n-seg.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.35)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0],
                    help="语义模型的 TTA 尺度；传 0.5 0.75 1.0 1.25 1.5 即开启多尺度")
    ap.add_argument("--no_flip", action="store_true", help="TTA 关闭水平翻转")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_for_infer(args.ckpt, device)
    yolo = YoloSeg(args.yolo, conf=args.yolo_conf, device=str(device))

    from torch.utils.data import DataLoader, Subset
    import torch.nn.functional as F
    from voc_dataset import VOCSegmentation

    ds = VOCSegmentation(args.voc_root, "val", train=False, base_size=args.base_size)
    sub = Subset(ds, list(range(min(args.num, len(ds)))))
    loader = DataLoader(sub, batch_size=1, shuffle=False, num_workers=0)

    print(f"在验证集前 {len(sub)} 张上做 YOLO 融合消融（YOLO 权重 {args.yolo}）", flush=True)
    t0 = time.time()
    sem_probs, yolo_labels, yolo_confs, targets = [], [], [], []
    for images, tgt in loader:
        x = images.to(device)
        h, w = tgt.shape[-2:]
        # 语义模型：多尺度 + 翻转 TTA，softmax 概率平均（scales=[1.0] 时等价单尺度）
        probs_avg = torch.zeros((NUM_CLASSES, h, w), device=device, dtype=torch.float32)
        n_views = 0
        for s in args.scales:
            if s != 1.0:
                xs = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
            else:
                xs = x
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xs)["out"]
            logits = F.interpolate(logits.float(), size=(h, w), mode="bilinear",
                                   align_corners=False)[0]
            probs_avg += logits.softmax(dim=0)
            n_views += 1
            if not args.no_flip:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits_f = model(torch.flip(xs, dims=[-1]))["out"]
                logits_f = F.interpolate(logits_f.float(), size=(h, w), mode="bilinear",
                                         align_corners=False)[0]
                probs_avg += torch.flip(logits_f, dims=[-1]).softmax(dim=0)
                n_views += 1
        probs_avg /= n_views
        sem_probs.append(probs_avg.cpu().numpy())

        # 用同一分辨率的原图喂给 YOLO
        rgb = (images[0].cpu().float() *
               torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) +
               torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)).clamp(0, 1).numpy()
        rgb = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
        lab, cf = yolo(rgb, tgt.shape[-2:])
        yolo_labels.append(lab)
        yolo_confs.append(cf)
        targets.append(tgt[0].numpy())
    print(f"前向完成 {time.time() - t0:.0f}s", flush=True)

    def miou_of(preds):
        cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
        for p, t in zip(preds, targets):
            cm.update(p, t)
        return cm.miou(True), cm.miou(False), cm.pixel_accuracy

    base = [sp.argmax(axis=0) for sp in sem_probs]
    base_m21, base_m20, base_pa = miou_of(base)
    print(f"{'仅语义模型（对照）':<26} mIoU21={base_m21:.4f} "
          f"mIoU20={base_m20:.4f} pixAcc={base_pa:.4f}", flush=True)

    best = (base_m21, "仅语义模型（对照）", None, None)
    for tau_sem in (0.4, 0.6, 0.8):
        for rule_b in (None, 0.5, 0.7):
            preds = [fuse(sp, lab, cf, tau_sem, args.yolo_conf, rule_b)
                     for sp, lab, cf in zip(sem_probs, yolo_labels, yolo_confs)]
            m21, m20, pa = miou_of(preds)
            tag = f"τ_sem={tau_sem}" + (f"+规则B={rule_b}" if rule_b else "")
            print(f"{tag:<26} mIoU21={m21:.4f} mIoU20={m20:.4f} pixAcc={pa:.4f} "
                  f"Δ={(m21 - base_m21) * 100:+.2f}点", flush=True)
            if m21 > best[0]:
                best = (m21, tag, tau_sem, rule_b)

    print(f"\n最佳：{best[1]}  mIoU21={best[0]:.4f}  "
          f"相比对照 {base_m21:.4f} 提升 {(best[0] - base_m21) * 100:+.2f} 点", flush=True)
    if best[0] - base_m21 < 0.005:
        print("⇒ 增益 <0.5 点，建议**不要**采用融合（收益不抵引入的复杂度与风险）", flush=True)
    else:
        print(f"⇒ 建议采用：tau_sem={best[2]}, rule_b={best[3]}, yolo_conf={args.yolo_conf}", flush=True)


if __name__ == "__main__":
    main()
