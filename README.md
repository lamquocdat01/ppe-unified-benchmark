# PPE-Unified Benchmark — two-tier leakage audit, PPE-Unified v1.1 and a schema-driven validator

Code and results for the manuscript *How leaky are public PPE detection benchmarks? A two-tier
audit, a verified multi-source benchmark, and a schema-driven validation protocol* (v5, tag
`v5.0-eswa`).

- `audit_v2/` — two-tier leakage audit: tier 1 (identifier + MD5), tier 2a (copy-level: DINOv2
  cosine + perceptual hash, calibrated on augmentation groups), tier 2b (adjacent video frames);
  calibration, near-duplicate-aware group re-split (`resplit.py`, `stratify_bg.py`), clean2 arms
  of sources B/C and the leave-one-sequence-out folds of C.
- `validator/` — eight-point validator, configured only by a schema file
  (`python -m validator <schema.yaml> --out report.json`); schemas in `results/generality/`.
- `scripts/` — result aggregation (`t4_partial.py`, `t4_2b_numbers.py`, `e2pp_leaked_vs_clean.py`)
  and `make_numbers_tex.py`, which writes every number of the manuscript from
  `paper/generated/numbers.json`.
- `paper/latex_source/` — LaTeX source of the manuscript (`main.tex`; `main_anon.tex` builds the
  anonymised version) with `generated/` tables/figures and `numbers.tex`.
- `results/t4/` — v1-fix vs v1.1 retraining (YOLO26s seeds 0-1, RT-DETR-L seed 0), per-run state and logs.
- `kaggle/` — generators of the Kaggle notebooks for the retraining runs (v1-fix vs v1.1, source B arms).
- `SHA256SUMS` — hashes of the files listed in it, at this tag.

**Erratum to the earlier version.** PPE-Unified v1 passed identifier and MD5 checks but still holds
copy-level and sequence-level near-duplicates across its splits; it is **not** leakage-free, and
the "clean" re-splits of sources B/C used in the earlier study still leak at the perceptual tier.
PPE-Unified v1.1 (same corrected images, near-duplicate-aware split) replaces it; see the dataset
card and ERRATA of the Kaggle dataset `ppe-unified-v1fix-v11`.

Dataset paths resolve under the environment variable `THS_DATASETS`; feature caches under
`$THS_DATASETS/_derived/ppe_unified`.

## Earlier version (v4.0, PPE-Unified v1) — superseded

The section below describes the earlier version and is kept for provenance. Its "clean" splits are
lower bounds on inflation (see the erratum above).

### Main finding (v4.0)

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

### Headline results (v4.0, PPE-Unified v1 test split, seed 42)

**Detection:** RT-DETR-L **0.7959** mAP@50 (37.5 ms) · YOLO26s **0.7890** (9.78 ms — recommended for
edge deployment) · Faster R-CNN R50-FPN 0.7207 (86.3 ms, reference baseline).

**Classification (crops, 12 classes):** ViT-B/16 **0.9521 acc / 0.8711 macro-F1** · ViT-S/16
0.9453/0.8429 (best worst-class F1 at 0.8193) · ResNet50 0.9014/0.8072 · VGG16 0.8966/0.7894 ·
GRU-256 (best RNN) 0.8075/0.6780.

**Eight verification experiments:** multi-seed stability ±0.24 points (E4) · repeat-factor sampling
+0.77 overall / +4.6 on rare classes (E5) · zero-shot SH17 drops −70…−88 points, so fine-tuning is
mandatory under domain shift (E3) · 640-upscale normalisation is harmless (E6) · complete per-class
breakdowns for all 8 models (E8/E9).

### Repository layout (v4.0 notebooks)

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

### Reproducing the v4.0 results

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
