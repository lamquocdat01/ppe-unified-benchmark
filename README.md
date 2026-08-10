# PPE-Unified Benchmark — leakage-free Safety-PPE detection

A controlled comparison of **CNN – RNN – Transformer** architectures for personal protective
equipment (PPE) detection on **PPE-Unified v1** — a benchmark of 17,951 images / 51,179 boxes /
12 classes (6 compliant-violation pairs), merged from three public sources and
**split by source-image group so that train/test leakage is eliminated at the root**.

## Main finding

**Two of the four most widely used public PPE datasets leak between train and test** — the hosting
platform augments images *before* splitting, so variants of the same source image land in both the
training and the test set. The resulting score inflation is measured directly with a
leaky-vs-clean experiment across **2 architectures × 2 sources**:

| Source | Model | Original split (leaky) | Clean split | mAP@50 inflation |
|---|---|---|---|---|
| B (aug ~×3) | YOLO26s | 0.5628 | 0.5436 | **+1.92 points** |
| B (aug ~×3) | RT-DETR-L | 0.5278 | 0.5076 | **+2.02 points** |
| C (aug ×3–×16) | YOLO26s | 0.9542 | 0.9136 | **+4.06 points** |
| C (aug ×3–×16) | RT-DETR-L | 0.9476 | 0.9131 | **+3.45 points** |

The effect scales with augmentation intensity and reproduces fully across architectures.

## Headline results (PPE-Unified v1 test split, seed 42)

**Detection:** RT-DETR-L **0.7959** mAP@50 (37.5 ms) · YOLO26s **0.7890** (9.78 ms — recommended for
edge deployment) · Faster R-CNN R50-FPN 0.7207 (86.3 ms, reference baseline).

**Classification (crops, 12 classes):** ViT-B/16 **0.9521 acc / 0.8711 macro-F1** · ViT-S/16
0.9453/0.8429 (best worst-class F1 at 0.8193) · ResNet50 0.9014/0.8072 · VGG16 0.8966/0.7894 ·
GRU-256 (best RNN) 0.8075/0.6780.

**Eight verification experiments:** multi-seed stability ±0.24 points (E4) · repeat-factor sampling
+0.77 overall / +4.6 on rare classes (E5) · zero-shot SH17 drops −70…−88 points, so fine-tuning is
mandatory under domain shift (E3) · 640-upscale normalisation is harmless (E6) · complete per-class
breakdowns for all 8 models (E8/E9).

## Repository layout

```
notebooks/
  01_CNN_YOLO26_FRCNN_VGG_ResNet.ipynb   # YOLO26s + Faster R-CNN + VGG16/ResNet50 (HPO + final)
  02_RNN_GRU_LSTM_BiLSTM.ipynb           # 12 RNN HPO configs + 4 final runs
  03_Transformer_RTDETR_ViT.ipynb        # RT-DETR-L + ViT-B/16 + ViT-S/16
  04_Summary_Comparison.ipynb            # aggregates results, generates tables/figures
  E2_LeakyVsClean_YOLO26s.ipynb          # measures leakage-driven inflation (YOLO26s)
  E2b_LeakyVsClean_RTDETR.ipynb          # reproduces it on a second architecture
  E3_CrossDataset_SH17.ipynb             # out-of-domain zero-shot on SH17
  E4_MultiSeed_YOLO26s.ipynb             # seed stability (42/0/1/2)
  E5_Ablation_RFS.ipynb                  # repeat-factor sampling ablation
  E6_Ablation_Resolution_416v640.ipynb   # source-C resolution ablation
  E7_LeaveOneSourceOut.ipynb             # between-source domain gap (not yet run — future work)
  E8_PerClass_Classifier_Eval.ipynb      # per-class F1 for 5 classifiers (inference only)
  E9_FasterRCNN_DetailedEval.ipynb       # Faster R-CNN per-class AP + operating-point P/R
results/          # main result CSVs + auto-generated comparison_report.md
results/extended/ # CSVs for E2–E9
requirements.txt
```

## Reproducing the results

1. Dataset (public): https://www.kaggle.com/datasets/quocdatlam/ppe-unified-v1 (CC BY 4.0)
2. Trained weights: Kaggle dataset `quocdatlam/ppe-trained-weights`
3. Every notebook runs standalone on a Kaggle T4 GPU: Add Input the dataset from step 1 → Save & Run All.
   Shared conventions: `QUICK_TEST` (a few-minute smoke run), seed 42, and checkpoint-resume through
   `state.json` (if a session hits the 12 h limit, upload `state.json` + `last.pt` as a separate
   dataset and Add Input — see the notes inside each notebook).
4. E2/E2b download the original sources from Roboflow: supply your own `ROBOFLOW_API_KEY`
   (use **Kaggle Secrets**; never hard-code a key into a notebook).

Every number reported in the paper can be regenerated from the files in `results/`,
which the notebooks emit.

## Source data & attribution

- Source A — [Hard Hat Detection v2](https://universe.roboflow.com/computer-vision-filqz/hard-hat-detection-62rrp/dataset/2) (CC BY 4.0)
- Source B — [PPE detection v1, testcasque](https://universe.roboflow.com/testcasque/ppe-detection-qlq3d/dataset/1) (CC BY 4.0)
- Source C — [PPE Detection v5, mohamed-traore](https://universe.roboflow.com/mohamed-traore-2ekkp/ppe-detection-l80fg/dataset/5) (CC BY 4.0)
- Out-of-domain test set — [SH17](https://github.com/ahmadmughees/SH17dataset) (non-commercial, not redistributed)

## License

Code: MIT (see `LICENSE`) · PPE-Unified v1 dataset: CC BY 4.0.
