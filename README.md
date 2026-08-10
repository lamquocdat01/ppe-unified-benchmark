# PPE-Unified Benchmark — Safety-PPE Detection sạch rò rỉ dữ liệu

So sánh có kiểm soát **CNN – RNN – Transformer** cho bài toán nhận diện phương tiện bảo vệ cá nhân (PPE)
trên **PPE-Unified v1** — benchmark 17.951 ảnh / 51.179 box / 12 lớp (6 cặp tuân thủ/vi phạm),
hợp nhất 3 nguồn công khai và **chia tập theo nhóm ảnh gốc để chặn rò rỉ train/test tận gốc**.

> Đồ án môn Deep Learning (AIN501) — FSB, Đại học FPT · Học viên: Lâm Quốc Đạt · GVHD: TS. Lê Thanh Tùng

## Phát hiện chính

**2/4 dataset PPE công khai phổ biến nhất bị rò rỉ train/test** (nền tảng augment ảnh *trước* khi chia tập,
khiến biến thể của cùng một ảnh gốc rơi vào cả train lẫn test). Mức thổi phồng điểm số được đo trực tiếp
bằng thí nghiệm leaky-vs-clean trên **2 kiến trúc × 2 nguồn**:

| Nguồn | Mô hình | Split gốc (leaky) | Split sạch (clean) | Thổi phồng mAP@50 |
|---|---|---|---|---|
| B (aug ~×3) | YOLO26s | 0.5628 | 0.5436 | **+1.92 điểm** |
| B (aug ~×3) | RT-DETR-L | 0.5278 | 0.5076 | **+2.02 điểm** |
| C (aug ×3–×16) | YOLO26s | 0.9542 | 0.9136 | **+4.06 điểm** |
| C (aug ×3–×16) | RT-DETR-L | 0.9476 | 0.9131 | **+3.45 điểm** |

Hiệu ứng tỷ lệ thuận với cường độ augment và tái lập trọn vẹn qua kiến trúc.

## Kết quả chính (tập test PPE-Unified v1, seed 42)

**Detection:** RT-DETR-L **0.7959** mAP@50 (37.5 ms) · YOLO26s **0.7890** (9.78 ms — khuyến nghị cho biên) · Faster R-CNN R50-FPN 0.7207 (86.3 ms, mốc đối chứng).

**Classification (crop, 12 lớp):** ViT-B/16 **0.9521 acc / 0.8711 macro-F1** · ViT-S/16 0.9453/0.8429 (worst-F1 cao nhất 0.8193) · ResNet50 0.9014/0.8072 · VGG16 0.8966/0.7894 · GRU-256 (RNN tốt nhất) 0.8075/0.6780.

**8 thí nghiệm kiểm chứng:** đa seed ±0.24 điểm (E4) · repeat-factor sampling +0.77 tổng/+4.6 lớp hiếm (E5) · zero-shot SH17 sụt −70…−88 điểm — bắt buộc fine-tune khi đổi miền (E3) · chuẩn hóa 640-upscale vô hại (E6) · per-class đầy đủ cho cả 8 mô hình (E8/E9).

## Cấu trúc repo

```
notebooks/
  01_CNN_YOLO26_FRCNN_VGG_ResNet.ipynb   # YOLO26s + Faster R-CNN + VGG16/ResNet50 (HPO + final)
  02_RNN_GRU_LSTM_BiLSTM.ipynb           # 12 cấu hình HPO RNN + 4 lượt cuối
  03_Transformer_RTDETR_ViT.ipynb        # RT-DETR-L + ViT-B/16 + ViT-S/16
  04_TongHop_BaoCao_SoSanh.ipynb         # tổng hợp kết quả, sinh bảng/hình
  E2_LeakyVsClean_YOLO26s.ipynb          # đo thổi phồng do rò rỉ (YOLO26s)
  E2b_LeakyVsClean_RTDETR.ipynb          # tái lập bằng kiến trúc thứ hai
  E3_CrossDataset_SH17.ipynb             # zero-shot ngoài miền trên SH17
  E4_MultiSeed_YOLO26s.ipynb             # ổn định theo seed (42/0/1/2)
  E5_Ablation_RFS.ipynb                  # ablation repeat-factor sampling
  E6_Ablation_Resolution_416v640.ipynb   # ablation độ phân giải nguồn C
  E7_LeaveOneSourceOut.ipynb             # domain gap giữa nguồn (chưa chạy — hướng phát triển)
  E8_PerClass_Classifier_Eval.ipynb      # F1 per-class 5 classifier (inference-only)
  E9_FasterRCNN_DetailedEval.ipynb       # AP per-class + P/R vận hành Faster R-CNN
results/          # csv kết quả chính + BaoCao_SoSanh.md tự sinh
results/extended/ # csv của E2–E9
requirements.txt
```

## Tái lập

1. Dataset (public): https://www.kaggle.com/datasets/quocdatlam/ppe-unified-v1 (CC BY 4.0)
2. Trọng số đã train: Kaggle dataset `quocdatlam/ppe-trained-weights`
3. Mỗi notebook chạy độc lập trên Kaggle GPU T4: Add Input dataset ở bước 1 → Save & Run All.
   Convention chung: `QUICK_TEST` (chạy thử ~phút), seed 42, checkpoint-resume qua `state.json`
   (nếu phiên chạm trần 12h: upload `state.json` + `last.pt` thành dataset riêng rồi Add Input — xem ghi chú trong notebook).
4. Notebook E2/E2b tải nguồn gốc từ Roboflow: điền `ROBOFLOW_API_KEY` của bạn
   (khuyến nghị dùng **Kaggle Secrets**, không hard-code vào notebook).

Mọi bảng số trong báo cáo tạo lại được từ các file trong `results/` do notebook xuất.

## Dữ liệu nguồn & ghi công

- Nguồn A — [Hard Hat Detection v2](https://universe.roboflow.com/computer-vision-filqz/hard-hat-detection-62rrp/dataset/2) (CC BY 4.0)
- Nguồn B — [PPE detection v1, testcasque](https://universe.roboflow.com/testcasque/ppe-detection-qlq3d/dataset/1) (CC BY 4.0)
- Nguồn C — [PPE Detection v5, mohamed-traore](https://universe.roboflow.com/mohamed-traore-2ekkp/ppe-detection-l80fg/dataset/5) (CC BY 4.0)
- Tập kiểm tra ngoài miền — [SH17](https://github.com/ahmadmughees/SH17dataset) (non-commercial, không phân phối lại)

## Giấy phép

Mã nguồn: MIT (xem `LICENSE`) · Dataset PPE-Unified v1: CC BY 4.0.
