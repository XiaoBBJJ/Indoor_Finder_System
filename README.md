以下是训练说明文件（Markdown格式），请保存为 `D:\Indoor_Finder_Project\train_readme.md`：

```markdown
# 室内寻物系统 YOLOv5s 模型训练说明

## 1. 环境配置

- **操作系统**：Windows 10
- **GPU**：NVIDIA GeForce GTX 1050 (3GB 显存)
- **CUDA**：11.3
- **Python**：3.8.20
- **PyTorch**：1.12.1+cu113
- **YOLOv5 版本**：官方代码库（2025-12-22）
- **虚拟环境**：Conda（环境名 `yolo_gpu`）

## 2. 数据集

- **类别**：钥匙 (key)、水杯 (cup)
- **总图片数**：360 张（各 180 张）
- **划分**：训练集 300 张，验证集 60 张（比例 5:1）
- **标注工具**：LabelImg，YOLO 格式（.txt 文件）
- **数据增强**：训练时自动应用 YOLOv5 内置的 Mosaic、随机翻转、色彩调整等
- **目录结构**：
  ```
  D:\Indoor_Finder_Project\indoor_dataset\
  ├── images/
  │   ├── train/   # 训练图片（300张）
  │   └── val/     # 验证图片（60张）
  └── labels/
      ├── train/   # 训练标签（YOLO格式）
      └── val/     # 验证标签
  ```

## 3. 训练参数

| 参数 | 设置值 |
|------|--------|
| 基础模型 | YOLOv5s |
| 预训练权重 | COCO 预训练（官方提供） |
| 输入图像尺寸 | 320×320 |
| 批次大小 (batch size) | 16（受 GPU 显存限制） |
| 优化器 | SGD |
| 初始学习率 | 0.01 |
| 动量 (momentum) | 0.937 |
| 权重衰减 | 0.0005 |
| 学习率调度 | 余弦退火 |
| 训练轮次 (epochs) | 100（早停触发，实际约 50 轮） |
| 早停耐心值 | 10 epochs |
| 数据增强 | Mosaic, 随机翻转, HSV 增强等（YOLOv5 默认） |
| 损失函数权重 | 默认（分类:定位:置信度 ≈ 0.5:0.05:1） |

## 4. 训练命令

在 YOLOv5 目录下执行以下命令（已在 `yolo_gpu` 环境中）：

```bash
python train.py --data data/indoor_finder.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --imgsz 320 --device 0 --name final_train_v12 --exist-ok --hyp data/hyps/hyp.scratch-low.yaml
```

说明：
- `indoor_finder.yaml` 是数据集配置文件，指向训练/验证图片路径和类别定义。
- 训练日志和权重保存至 `runs/train/final_train_v12/`。

## 5. 训练结果

- **最佳权重**：`runs/train/final_train_v12/weights/best.pt`
- **验证集性能**（mAP@0.5）：
  - 整体：96.6%
  - 钥匙 (key)：93.7%
  - 水杯 (cup)：99.5%
- **单帧推理时间**：7.2 ms（预处理 0.2ms，推理 4.7ms，NMS 2.3ms）
- **理论帧率**：约 138 FPS

详细验证结果可通过以下命令复现：

```bash
python val.py --weights runs/train/final_train_v12/weights/best.pt --data data/indoor_finder.yaml --img 320 --name final_evaluation --workers 0
```

## 6. 关键说明

- **迁移学习**：利用 COCO 预训练权重加速收敛，适应小数据集。
- **轻量化设计**：选择 YOLOv5s 模型，输入尺寸 320×320，确保在 GTX 1050 上实时运行。
- **超参数优化**：主要采用 YOLOv5 默认参数，仅调整批次大小以适配显存。
- **早停机制**：当验证集 mAP 连续 10 轮未提升时停止训练，避免过拟合。

## 7. 训练日志与监控

- **TensorBoard**：训练过程中可通过 TensorBoard 查看损失曲线和 mAP 变化。
  ```bash
  tensorboard --logdir runs/train
  ```
- **结果图表**：训练完成后，在 `runs/train/final_train_v12/` 下会生成 `results.png`，包含损失、精度、召回率等曲线。

---

*生成日期：2026年3月16日*
```

此文件包含了完整的训练说明，与其他项目文件放在同一目录下。如果需要调整内容或补充细节，请随时告知。