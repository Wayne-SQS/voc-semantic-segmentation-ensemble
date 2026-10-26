# 基于零样本预训练集成的 PASCAL VOC 语义分割

东北大学人工智能技术大作业 —— 语义分割任务 + 智能体技术现状调查报告。

## 作者

孙秋实　软件 2406　学号 20245715

## 任务说明

1. **语义分割**：在 PASCAL VOC 2012 验证集（1449 张）上完成 21 类语义分割，最终验证集 mIoU = **0.8195**。
2. **智能体调查报告**：智能体技术现状调研，11 页、4 图 2 表、17 条参考文献。

## 核心方法

- **零样本路线**：直接使用 torchvision 提供的 COCO+VOC 预训练权重，不做微调，避免小样本过拟合。
- **异构集成**：4 个成员（DeepLabV3-ResNet50、DeepLabV3-ResNet101、FCN-ResNet50、LR-ASPP-MobileNetV3-Large）加权平均，利用成员间误差不重叠提升精度。
- **TTA**：5 尺度 + 水平翻转的测试时增强。
- **已验证无效的优化方向**（详见报告第 5 章）：微调、per-class 偏置、温度标定、聚合范数、CRF/引导滤波后处理等，均在独立 held-out 集上验证为无收益或负收益。

## 目录结构

```
.
├── code/                         # 核心代码（40 个 .py）
│   ├── voc_dataset.py            # VOC / SBD 数据加载
│   ├── model.py / model_v2.py    # 模型定义
│   ├── train.py                  # 训练脚本（最终未采用微调路线）
│   ├── evaluate.py               # mIoU 评估
│   ├── predict_final.py          # 最优集成 + TTA 推理
│   ├── predict_ensemble.py       # 多模型集成推理
│   ├── metrics.py                # 指标计算
│   └── ...
├── 提交/
│   ├── 软件2406_孙秋实_20245715_语义分割.zip      # 三件套（报告 + 代码 + 400 张预测 mask）
│   ├── 软件2406_孙秋实_20245715_智能体调查报告.docx
│   ├── 软件2406_孙秋实_20245715_智能体调查报告.pdf
│   └── 提交清单.md
├── 智能体调查报告/
│   ├── 软件2406_孙秋实_20245715_智能体调查报告.docx
│   └── 软件2406_孙秋实_20245715_智能体调查报告.pdf
├── report_孙秋实_20245715.docx    # 语义分割实验报告（Word 版）
├── report_孙秋实_20245715.pdf     # 语义分割实验报告（PDF 版）
└── 大作业.pdf                     # 课程作业要求
```

## 数据集

本仓库不包含数据集，请自行下载：

- **PASCAL VOC 2012**：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
  解压为 `data/VOCdevkit/VOC2012/`
- **SBD（Semantic Boundaries Dataset）**：http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
  解压为 `data/sbd/`，并运行 `code/sbd_prepare.py` 生成 `data/sbd_masks/`

## 环境

```
python >= 3.9
torch
torchvision
numpy
pillow
tqdm
```

## 复现最优结果

```bash
# 1. 准备数据集（见上）
# 2. 运行最优集成推理（4 模型加权 + 5 尺度翻转 TTA）
python code/predict_final.py
# 3. 评估
python code/evaluate.py
```

## 最终结果

| 指标 | 值 |
|---|---|
| VOC val mIoU | 0.8195 |
| SBD 独立验证集 mIoU | 0.7730 |
| 测试集预测 mask | 400 张（见提交包内 `predict_*.zip`） |

## 参考文献

详见报告末尾参考文献（GB/T 7714 格式）。
