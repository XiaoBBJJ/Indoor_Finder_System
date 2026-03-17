
```
# Indoor Finder System YOLOv5s Model Training Instructions

## 1. Environment Configuration

- **Operating System**: Windows 10
- **GPU**: NVIDIA GeForce GTX 1050 (3GB VRAM)
- **CUDA**: 11.3
- **Python**: 3.8.20
- **PyTorch**: 1.12.1+cu113
- **YOLOv5 Version**: Official codebase (2025-12-22)
- **Virtual Environment**: Conda (environment name: `yolo_gpu`)

## 2. Dataset

- **Classes**: key, cup
- **Total Images**: 360 (180 each)
- **Split**: Training set 300 images, validation set 60 images (ratio 5:1)
- **Annotation Tool**: LabelImg, YOLO format (.txt files)
- **Data Augmentation**: Automatically applied during training using YOLOv5's built-in Mosaic, random flip, color adjustment, etc.
- **Directory Structure**:
  ```
  D:\Indoor_Finder_Project\indoor_dataset\
  ├── images/
  │   ├── train/   # Training images (300)
  │   └── val/     # Validation images (60)
  └── labels/
      ├── train/   # Training labels (YOLO format)
      └── val/     # Validation labels
  ```

## 3. Training Parameters

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv5s |
| Pretrained Weights | COCO pretrained (official) |
| Input Image Size | 320×320 |
| Batch Size | 16 (limited by GPU memory) |
| Optimizer | SGD |
| Initial Learning Rate | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Learning Rate Schedule | Cosine annealing |
| Epochs | 100 (early stopping triggered, actual ~50 epochs) |
| Early Stopping Patience | 10 epochs |
| Data Augmentation | Mosaic, random flip, HSV enhancement, etc. (YOLOv5 default) |
| Loss Function Weights | Default (classification:localization:confidence ≈ 0.5:0.05:1) |

## 4. Training Command

Run the following command in the YOLOv5 directory (under `yolo_gpu` environment):

```bash
python train.py --data data/indoor_finder.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --imgsz 320 --device 0 --name final_train_v12 --exist-ok --hyp data/hyps/hyp.scratch-low.yaml
```

Explanation:
- `indoor_finder.yaml` is the dataset configuration file pointing to training/validation image paths and class definitions.
- Training logs and weights are saved to `runs/train/final_train_v12/`.

## 5. Training Results

- **Best Weights**: `runs/train/final_train_v12/weights/best.pt`
- **Validation Performance** (mAP@0.5):
  - Overall: 96.6%
  - Key: 93.7%
  - Cup: 99.5%
- **Inference Time per Frame**: 7.2 ms (preprocessing 0.2ms, inference 4.7ms, NMS 2.3ms)
- **Theoretical Frame Rate**: ~138 FPS

Detailed validation results can be reproduced with the following command:

```bash
python val.py --weights runs/train/final_train_v12/weights/best.pt --data data/indoor_finder.yaml --img 320 --name final_evaluation --workers 0
```

## 6. Key Notes

- **Transfer Learning**: COCO pretrained weights accelerate convergence and adapt to small datasets.
- **Lightweight Design**: Choosing YOLOv5s with 320×320 input ensures real-time performance on GTX 1050.
- **Hyperparameter Optimization**: Mainly uses YOLOv5 default parameters, only batch size adjusted to fit GPU memory.
- **Early Stopping**: Training stops when validation mAP does not improve for 10 consecutive epochs, preventing overfitting.

## 7. Training Logs and Monitoring

- **TensorBoard**: During training, you can monitor loss curves and mAP changes via TensorBoard.
  ```bash
  tensorboard --logdir runs/train
  ```
- **Result Charts**: After training, `results.png` is generated in `runs/train/final_train_v12/`, containing loss, precision, recall curves, etc.

---

*Generated: March 16, 2026*
```

