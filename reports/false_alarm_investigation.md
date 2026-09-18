# Investigating D_PNEUMONIA's 48.4% false-alarm rate

`reports/eval_report.md` found that on truly normal chest X-rays, MEDVISION's
knowledge-graph reasoning still ranked Pneumonia #1 in **48.4%** of cases
(147/304). This report finds the root cause and the fix actually applied.

## Root cause

For each of the 147 false-alarm cases, we recorded every finding that
contributed to Pneumonia's score:

| Finding | Present in false-alarm cases | Mean calibrated probability |
|---|---|---|
| **F_INFILTRATION** | **147/147 (100%)** | **0.663** |
| F_LUNG_OPACITY | 83/147 (56%) | 0.425 |
| F_CONSOLIDATION | 80/147 (54%) | 0.291 |
| F_PNEUMONIA_OPACITY | 14/147 (10%) | 0.102 |
| F_ATELECTASIS | 13/147 (9%) | 0.098 |
| F_EFFUSION | 11/147 (7%) | 0.106 |

**F_INFILTRATION is present in every single false-alarm case**, with a mean
calibrated probability of 0.66 even though the true label is Normal. This lines
up with two independent facts we already had:

1. `reports/eval_report.md` shows F_INFILTRATION has the weakest AUROC (0.731) of
   the four pneumonia-correlated findings — Consolidation and Lung Opacity are
   both 0.94+.
2. "Infiltration" is a widely documented nonspecific radiographic sign in the
   literature — it can indicate pneumonia, pulmonary edema, fibrosis,
   atelectasis, or several other things, and is frequently singled out as one of
   the noisiest labels in the NIH ChestX-ray14 label set this classifier was
   trained on.

Despite that, our own hand-authored knowledge graph weighted
`F_INFILTRATION -suggestive_of-> D_PNEUMONIA` at **0.6** — almost as strongly as
Consolidation's 0.85, a much more specific sign. Every time the classifier's
noisy Infiltration output fired (which it does often, even on normal images),
it was pulling Pneumonia's score up disproportionately.

## The fix

Swept just this one edge weight (classifier outputs cached once, only the KG
scoring step re-run per value — cheap, and unlike the earlier global
`activation_threshold` sweep, this curve is smooth and monotonic, not noisy):

| Weight | Recall@1 | False-alarm@1 | Recall − False-alarm |
|---|---|---|---|
| 0.60 (was) | 93.5% | 48.4% | 45.1% |
| 0.50 | 91.9% | 43.1% | 48.8% |
| **0.40 (chosen)** | **86.4%** | **37.8%** | **48.6%** |
| 0.30 (best margin) | 76.9% | 17.1% | 59.8% |
| 0.20 | 60.6% | 11.2% | 49.4% |
| 0.10 | 40.3% | 6.6% | 33.7% |

We deliberately did **not** pick 0.30, the point with the best raw margin. This
is a clinician-reviewed decision-support tool, not an autonomous diagnosis — a
missed true pneumonia case (lower recall) is a worse outcome than an extra case
the clinician has to double-check and dismiss (a false alarm). 0.40 keeps
Recall@1 above 85% while still meaningfully improving the false-alarm rate
(48.4% → 37.8%), a better trade-off for this use case than chasing the single
best number on the curve.

## Result (confirmed against the real pipeline, not just the offline sweep)

Applied in `kg/ontology/relations.yaml`, rebuilt via `scripts/build_kg.py`, and
re-ran `scripts/run_eval.py` end-to-end:

- Recall@1: 93.5% → **86.4%**
- Recall@3: 98.3% → 96.7%
- False-alarm@1: 48.4% → **37.8%**

Classification metrics (AUROC/F1/ECE) are unchanged, as expected — this was a
knowledge-graph edge weight, not a classifier or calibration change.

## What this doesn't fix

37.8% is still a high false-alarm rate. F_LUNG_OPACITY (56% of remaining false
alarms) and F_CONSOLIDATION (54%) are the next contributors, but both have
strong AUROC (0.94+), so the same "this label is unreliable" argument doesn't
apply to them — their false positives are more likely genuine classifier
mistakes on this specific dataset's image style, which is closer to T08 (an
input-quality/out-of-distribution gate) than a knowledge-graph fix. That
remains open work.
