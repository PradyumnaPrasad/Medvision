# Image <-> knowledge-graph alignment: evaluation

Generated: 2026-11-21T04:26:06+00:00

Data: NIH ChestX-ray14 sample (Sohaibsoussi/NIH-Chest-X-ray-dataset-small): 4326 train / 1730 validation / 1279 test images. Training uses train, early stopping uses validation, **every number below is on the test split**, and the retrieval index for evaluation is the train split only.

**Caveat that affects every row.** The frozen image encoder is torchxrayvision's DenseNet-121 `all` checkpoint, which was itself trained on NIH ChestX-ray14. These features have therefore already seen NIH-style labels, and absolute numbers are optimistic compared with an encoder that had not. The comparisons between rows (same features, same split) remain fair; the absolute level does not carry over to other hospitals.

## 1. Image -> KG finding node: per-finding AUROC

Score = cosine similarity to the node (aligned model, seed 0), the pretrained classifier's own output, or a logistic-regression probe fitted on the same train split. Findings with fewer than 20 test positives or negatives are skipped.

| Finding | test positives | Aligned | Classifier head | Linear probe |
|---|---|---|---|---|
| No Finding | 541 | 0.555 | n/a | 0.651 |
| Atelectasis | 160 | 0.573 | 0.697 | 0.662 |
| Consolidation | 77 | 0.604 | 0.747 | 0.612 |
| Infiltration | 284 | 0.582 | 0.669 | 0.610 |
| Pneumothorax | 146 | 0.650 | 0.658 | 0.707 |
| Pulmonary Edema | 31 | 0.535 | 0.707 | 0.694 |
| Emphysema | 60 | 0.701 | 0.658 | 0.714 |
| Pulmonary Fibrosis | 26 | 0.620 | 0.676 | 0.756 |
| Pleural Effusion | 213 | 0.688 | 0.789 | 0.748 |
| Pneumonia-pattern Opacity | 23 | 0.411 | 0.597 | 0.561 |
| Pleural Thickening | 55 | 0.633 | 0.713 | 0.658 |
| Cardiomegaly | 42 | 0.616 | 0.809 | 0.772 |
| Pulmonary Nodule | 73 | 0.608 | 0.665 | 0.585 |
| Pulmonary Mass | 96 | 0.663 | 0.738 | 0.743 |

Mean over the 13 findings all three methods cover: aligned **0.606**, classifier head **0.702**, linear probe **0.679**.

## 2. Image -> KG node retrieval (hit@k)

Share of test images whose top-k retrieved finding nodes contain at least one true finding (seed 0). Normal images count `No Finding` as their true node.

| k | hit@k |
|---|---|
| 1 | 0.360 |
| 3 | 0.685 |
| 5 | 0.830 |

## 3. Similar-case retrieval (precision@k)

A retrieved train case is relevant if it shares at least one finding with the query. 'Abnormal' keeps only queries with a finding other than `No Finding`, and relevance then also ignores `No Finding`, so it is the harder and more meaningful column.

| Space | P@1 all | P@5 all | P@10 all | P@1 abnormal | P@5 abnormal | P@10 abnormal |
|---|---|---|---|---|---|---|
| Aligned (seed 0) | 0.418 | 0.433 | 0.424 | 0.308 | 0.314 | 0.303 |
| Raw DenseNet features, cosine | 0.385 | 0.391 | 0.385 | 0.286 | 0.294 | 0.286 |
| Random ranking | 0.327 | 0.324 | 0.321 | 0.123 | 0.140 | 0.137 |

## 4. Ablations (mean ± std over seeds [0, 1, 2])

| Variant | mean AUROC | hit@1 | P@5 abnormal |
|---|---|---|---|
| full (graph + set loss + node loss) | 0.610 ± 0.008 | 0.371 ± 0.010 | 0.310 ± 0.006 |
| no node-level loss | 0.612 ± 0.009 | 0.263 ± 0.010 | 0.303 ± 0.001 |
| no graph edges (text + id only) | 0.616 ± 0.015 | 0.389 ± 0.014 | 0.303 ± 0.023 |

## Reading these results

This file is generated, so numbers are never edited by hand. The author's interpretation lives in `reports/alignment_notes.md`.
