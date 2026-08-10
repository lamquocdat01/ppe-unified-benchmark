# Comparison report: CNN vs RNN vs Transformer – Safety-PPE detection, 12 classes

## 1. Dataset & protocol
- PPE-Unified v1 (three CC BY 4.0 sources merged, leakage removed by source image + MD5, normalised to 640×640), 12 classes = 6 compliant/violation pairs, 17,951 images.
- Fixed 70/20/10 split grouped by source image (group-aware, seed 42), shared by all three branches; TEST is used exactly once. Public: kaggle.com/datasets/quocdatlam/ppe-unified-v1
- Imbalance handling: repeat-factor oversampling (train), crop undersampling, class weights, model selection by macro-F1.

## 2. DETECTION results (TEST)
| model                   | branch      | cfg                                |   mAP50_val |   mAP50_test |   mAP50_95_test |   precision_test |   recall_test |   infer_ms |   train_min |
|:------------------------|:------------|:-----------------------------------|------------:|-------------:|----------------:|-----------------:|--------------:|-----------:|------------:|
| RT-DETR-L (Transformer) | Transformer | {'lr0': 0.0001}                    |      0.8161 |       0.7959 |          0.4884 |           0.7693 |        0.7682 |      37.5  |        61.8 |
| YOLO26s (CNN)           | CNN         | {'lr0': 0.01, 'optimizer': 'auto'} |      0.811  |       0.789  |          0.4851 |           0.7675 |        0.7377 |       9.78 |       165.4 |
| Faster R-CNN (CNN)      | CNN         | lr=0.005,SGD,8ep                   |      0.7247 |       0.7207 |          0.387  |         nan      |      nan      |      86.29 |       396.5 |

## 3. CLASSIFICATION results, 12 classes (TEST)
| model                  | branch      | cfg                                                                             |   accuracy |   macro_f1 |   worst_class_f1 |   train_min |
|:-----------------------|:------------|:--------------------------------------------------------------------------------|-----------:|-----------:|-----------------:|------------:|
| ViT-B/16 (Transformer) | Transformer | vit_base_patch16_224, {'lr': 5e-05, 'wd': 0.1}                                  |     0.9521 |     0.8711 |           0.7967 |        19   |
| ViT-S/16 (Transformer) | Transformer | vit_small_patch16_224, {'lr': 5e-05, 'wd': 0.1}                                 |     0.9453 |     0.8429 |           0.8193 |         7   |
| ResNet50 (CNN)         | CNN         | {'backbone': 'ResNet50', 'lr': 0.001, 'dropout': 0.5}+unfreeze conv5            |     0.9014 |     0.8072 |           0.7353 |         9.3 |
| VGG16 (CNN)            | CNN         | {'backbone': 'VGG16', 'lr': 0.001, 'dropout': 0.3}+unfreeze block5              |     0.8966 |     0.7894 |           0.7167 |        16.4 |
| GRU-ref (RNN)          | RNN         | GRU, hidden 256                                                                 |     0.8023 |     0.6796 |           0.5126 |         2   |
| GRU-256x1 (RNN)        | RNN         | {'cell': 'GRU', 'hidden': 256, 'dropout': 0.2, 'opt': 'rmsprop', 'n_layers': 1} |     0.8075 |     0.678  |           0.5333 |         2   |
| BiLSTM-ref (RNN)       | RNN         | BiLSTM, hidden 256                                                              |     0.7879 |     0.6533 |           0.4512 |         3.6 |
| LSTM-ref (RNN)         | RNN         | LSTM, hidden 256                                                                |     0.7302 |     0.5942 |           0.4211 |         2.2 |

## 4. Selected models
- Detection: **RT-DETR-L (Transformer)** | Classification: **ViT-B/16 (Transformer)**
- Deployment pipeline: RT-DETR-L (Transformer) + RNN temporal smoothing over the camera frame sequence.

## 5. Comparison against GitHub baselines, 2023–2026
| Source                                                        | Dataset                                 |    mAP50 | Notes                                                          |
|:--------------------------------------------------------------|:----------------------------------------|---------:|:----------------------------------------------------------------|
| This work – RT-DETR-L (Transformer)                           | PPE-Unified v1, 12 classes (10% test)   |   0.7959 | Full HPO, fixed split seed 42, reproducible                    |
| snehilsanyal/Construction-Site-Safety-PPE-Detection (YOLOv8n) | Roboflow CSS, 2,801 images, 10 classes  | nan      | No metrics published; only curves/confusion matrix             |
| jomarkow/Safety-Helmet-Detection (YOLOv8n)                    | Kaggle Hard Hat 5k (different dataset)  | nan      | No metrics published; 10-epoch proof of concept                |
| lyuwenyu/RT-DETR (R18 official)                               | COCO val2017 (80 classes)               | nan      | Official 46.5 mAP50-95 on COCO – reference only, different dataset |

## 6. Appendix: hyperparameter screening tables

### HPO – CNN_cls

| backbone   |     lr |   dropout |   macro_f1_val |   min |
|:-----------|-------:|----------:|---------------:|------:|
| ResNet50   | 0.001  |       0.5 |         0.7894 |   3.8 |
| ResNet50   | 0.001  |       0.3 |         0.7851 |   3.9 |
| ResNet50   | 0.0001 |       0.5 |         0.7842 |   4   |
| ResNet50   | 0.0001 |       0.3 |         0.7823 |   4   |
| VGG16      | 0.001  |       0.3 |         0.7649 |   7   |
| VGG16      | 0.001  |       0.5 |         0.7572 |   6.4 |
| VGG16      | 0.0001 |       0.3 |         0.7054 |   6.4 |
| VGG16      | 0.0001 |       0.5 |         0.672  |   6.4 |

### HPO – YOLO26s

|   lr0 | optimizer   |   mAP50_val |   min |
|------:|:------------|------------:|------:|
| 0.01  | auto        |      0.7564 |  20.9 |
| 0.01  | SGD         |      0.7416 |  19.3 |
| 0.001 | AdamW       |      0.7339 |  19.8 |
| 0.005 | SGD         |      0.7224 |  19.3 |

### HPO – RT-DETR

|    lr0 |   mAP50_val |   min |
|-------:|------------:|------:|
| 0.0001 |      0.6119 |  79.5 |
| 5e-05  |      0.496  |  79.2 |

### HPO – ViT

|    lr |   wd |   macro_f1_val |   min |
|------:|-----:|---------------:|------:|
| 5e-05 | 0.1  |         0.8735 |   7.9 |
| 1e-05 | 0.1  |         0.8727 |   7.9 |
| 5e-05 | 0.01 |         0.8637 |   8   |
| 1e-05 | 0.01 |         0.86   |   7.9 |

### HPO – RNN_round1_cell_hidden

| cell   |   hidden |   macro_f1_val |   min |
|:-------|---------:|---------------:|------:|
| GRU    |      256 |         0.5978 |   1   |
| BiGRU  |      128 |         0.5558 |   1.2 |
| GRU    |      128 |         0.5373 |   0.9 |
| BiGRU  |      256 |         0.5327 |   1.6 |
| BiLSTM |      128 |         0.4936 |   1.2 |
| BiLSTM |      256 |         0.4713 |   1.7 |
| LSTM   |      256 |         0.3925 |   1.2 |
| LSTM   |      128 |         0.3793 |   0.9 |

### HPO – RNN_round2_dropout_opt

| cell   |   hidden |   dropout | opt     |   macro_f1_val |   min |
|:-------|---------:|----------:|:--------|---------------:|------:|
| GRU    |      256 |       0.2 | rmsprop |         0.6036 |   1   |
| GRU    |      256 |       0.4 | rmsprop |         0.5861 |   1.1 |
| GRU    |      256 |       0.2 | adam    |         0.5523 |   1.1 |
| GRU    |      256 |       0.4 | adam    |         0.5427 |   1.1 |
