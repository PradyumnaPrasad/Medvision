# MEDVISION evaluation report

Generated: 2026-09-18T13:35:13.972189+00:00

Dataset: keremberke/chest-xray-classification (HF parquet mirror of the Kermany/Mooney 'Chest X-Ray Images (Pneumonia)' dataset), full/validation split, N=1165 images (861 pneumonia, 304 normal). This is the split HELD OUT from calibration fitting (scripts/fit_calibration.py fits on full/test) — nothing reported here was seen during calibration.

## Classification metrics (raw vs. calibrated)

| Label | AUROC | F1 (raw) | F1 (calibrated) | ECE (raw) | ECE (calibrated) |
|---|---|---|---|---|---|
| F_PNEUMONIA_OPACITY | 0.709 | 0.339 | 0.339 | 0.571 | 0.571 |
| F_CONSOLIDATION | 0.947 | 0.901 | 0.901 | 0.308 | 0.308 |
| F_INFILTRATION | 0.731 | 0.845 | 0.845 | 0.168 | 0.063 |
| F_LUNG_OPACITY | 0.946 | 0.918 | 0.918 | 0.144 | 0.047 |

## Differential-diagnosis ranking (knowledge-graph reasoning)

Disease evaluated: **D_PNEUMONIA** (861 true positive cases, 304 true negative cases)

- Recall@1 (ranked #1 when truly present): **86.4%**
- Recall@3 (ranked in top 3 when truly present): **96.7%**
- Recall@any (appears anywhere in the ranking when truly present): **96.7%**
- False-alarm@1 (wrongly ranked #1 on a true negative case): **37.8%**
