# Appendix — Expert validation of the affinity pipeline (run `20260726_134607_5`)

**Scope.** One prediction run, `20260726_134607_5`, scored against two independent expert sets, labelled
**Expert 1** and **Expert 2**. The two expert sets are never pooled: each column below is an independent
case. The analysis reproduces `notebooks/11_affinity_diagnostisc_metrics_pipeline.ipynb` against
`src/analysis/affinity_metrics.py`.

**Prediction run.** 5-model panel (`claude-haiku-4.5`, `deepseek-v4-flash-cloud`, `gemini-3.1-flash-lite`,
`gemma4-31b-cloud`, `gpt-5.4-mini`); 5040 pairs = 105 abstracts × 48 RA2025 questions;
`few_shot_max_examples_per_prompt: 4`, `top_k: 48`, `ra_retrieval_mode: cosine`, `score_scale_version: 2`;
commit `cf93f62` (dirty). Both adjudication agreement thresholds are **0.0**, so no pair was sent to the
LLM judge: `Final_Method` is `ensemble_mean` for all 5040 pairs and `Final_Affinity` equals
`Affinity_Mean` exactly. **The system evaluated here is therefore the plain 5-model ensemble mean, with no
LLM adjudication.** 5025 pairs carry all five model scores; 15 carry four.

**Labels.** Raw scores (0–120) are binned Low / Moderate / High at 40 and 80 with
`pd.cut(bins=[-inf, 40, 80, inf])` (project convention: a score of exactly 40 is Low, exactly 80 is
Moderate). Ordinal codes Low = 0, Moderate = 1, High = 2. QWK is quadratically weighted Cohen's κ; mean
signed error (bias) is in ordinal steps, positive = pipeline scores above the expert.

## I. Headline metrics

| Metric | Expert 1 × `134607_5` | Expert 2 × `134607_5` |
|---|---|---|
| N pairs / abstracts / questions | 602 / 91 / 48 | 722 / 99 / 39 |
| Accuracy | 65.0% | **75.3%** |
| Quadratic weighted kappa | **0.423** | 0.366 |
| Mean signed error (ordinal steps) | +0.178 | +0.168 |
| Mean absolute error | 0.367 | 0.259 |
| Expert label mix (Low / Mod / High) | 438 / 137 / 27 | 617 / 105 / **0** |
| Predicted label mix (Low / Mod / High) | 336 / 234 / 32 | 525 / 168 / 29 |
| Mean raw score, expert / pipeline | 24.9 / 38.8 | 17.3 / 29.9 |

Accuracy and QWK rank the two cases in opposite directions. Expert 2's set is 85.5% Low against Expert 1's
72.8%, so a Low-leaning predictor scores higher on accuracy while scoring lower on chance-corrected
ordinal agreement. **QWK is the metric to quote**; accuracy here is a base-rate statistic.

The pipeline over-scores against both experts by the same amount, +0.17 ordinal steps (≈ +14 and +13 raw
points). Expert 2 assigned **no High label at all** in 722 pairs, so his set cannot evaluate the High class
and his QWK is effectively computed on a two-level scale.

## II. Per-class metrics

| Case | Class | Support | Precision | Recall | One-vs-rest acc. |
|---|---|---|---|---|---|
| Expert 1 | Low | 438 | 88.1% | 67.6% | 69.8% |
| Expert 1 | Moderate | 137 | 36.3% | 62.0% | 66.6% |
| Expert 1 | High | 27 | 31.3% | 37.0% | 93.5% |
| Expert 2 | Low | 617 | 93.7% | 79.7% | 78.1% |
| Expert 2 | Moderate | 105 | 31.0% | 49.5% | 76.6% |
| Expert 2 | High | 0 | 0.0% (29 predicted) | — | 96.0% |

Confusion matrices (rows = expert, columns = predicted):

```
Expert 1                        Expert 2
        Low  Mod  High                  Low  Mod  High
Low     296  136     6          Low     492  116     9
Mod      36   85    16          Mod      33   52    20
High      4   13    10          High      0    0     0
```

**Moderate over-prediction is the dominant systematic error in both cases**: precision 31–36% against
recall 50–62%, driven by 136 and 116 expert-Low pairs predicted Moderate. Low recall (67.6% / 79.7%) is
the mirror of the same effect. Two-step misses are rare: 6 and 9 expert-Low pairs predicted High, and 4 of
Expert 1's 27 High pairs predicted Low.

## III. Bootstrap confidence intervals

Cluster bootstrap by abstract (resample abstracts with replacement, all their pairs kept together), 500
resamples, seed 42, 2.5–97.5 percentile interval.

| Case | QWK | Accuracy | Mean signed error | Recall Low | Recall Moderate | Recall High |
|---|---|---|---|---|---|---|
| Expert 1 | 0.423 [0.351, 0.494] | 0.650 [0.599, 0.698] | +0.178 [+0.109, +0.247] | 0.676 [0.611, 0.736] | 0.620 [0.544, 0.702] | 0.370 [0.200, 0.569] |
| Expert 2 | 0.366 [0.284, 0.450] | 0.753 [0.704, 0.811] | +0.168 [+0.109, +0.227] | 0.797 [0.743, 0.852] | 0.495 [0.395, 0.589] | n/a\* |

\* Expert 2 has zero High pairs, so every resample has zero support.

The two QWK intervals overlap over [0.351, 0.450]: the pipeline's ordinal agreement is not measurably
different between the two experts. Both bias intervals exclude zero, so the +0.17-step over-scoring is a
real effect and not sampling noise. Expert 1's High recall interval is 37pp wide at n = 27 and is the
weakest estimate in the table.

## IV. Does the pipeline agree with each expert more or less than the experts agree with each other?

The two experts co-rated **212 pairs over 39 abstracts and 39 questions**. All three raters are compared on
exactly this common subset, so the comparison is like-for-like.

| Comparison (212 shared pairs) | Exact agreement | QWK | Mean signed diff. |
|---|---|---|---|
| Expert 1 vs. Expert 2 (human–human ceiling) | **71.7%** | 0.346 | −0.189 (E2 below E1) |
| Pipeline vs. Expert 1 | 61.8% | **0.462** | +0.226 |
| Pipeline vs. Expert 2 | 55.2% | 0.280 | +0.415 |

Paired cluster bootstrap over the 39 shared abstracts (2000 draws, seed 0), each draw recomputing all
three agreements on the same resample:

| Difference | Median | 95% CI | Draws > 0 |
|---|---|---|---|
| ΔQWK: (pipeline, E1) − (E1, E2) | **+0.117** | [−0.064, +0.299] | 89% |
| ΔQWK: (pipeline, E2) − (E1, E2) | −0.065 | [−0.198, +0.084] | 20% |
| Δexact: (pipeline, E1) − (E1, E2) | −0.098 | [−0.204, +0.011] | 4% |
| Δexact: (pipeline, E2) − (E1, E2) | **−0.163** | [−0.261, −0.063] | 0% |

**Answer.** It depends on the metric, and the two metrics disagree for a reason that is itself the finding.

- On **chance-corrected ordinal agreement (QWK)**, the pipeline agrees with Expert 1 *more* than the two
  experts agree with each other (0.462 vs. 0.346) and with Expert 2 *less* (0.280 vs. 0.346). Neither gap
  is resolved at 39 abstracts — both intervals straddle zero. The defensible statement is that
  pipeline–expert agreement (0.28–0.46 on the shared subset, 0.366–0.423 on the full sets) sits **inside
  the same band as expert–expert agreement (0.346)**, not below it.
- On **exact agreement**, the two humans match each other more often than the pipeline matches either
  (71.7% vs. 61.8% and 55.2%), significantly so for Expert 2. This is a base-rate and offset artifact, not
  independent evidence: both experts are Low-dominated, so they share a high floor of trivially matching
  Low–Low cells, and the pipeline is uniformly more generous than either — mean ordinal level on the shared
  pairs is **Expert 2 = 0.18 < Expert 1 = 0.37 < pipeline = 0.59**. An offset of that kind is removable by
  re-fitting the two cut points; a chance-corrected rank disagreement is not.

Where the humans themselves diverge (rows = Expert 1, columns = Expert 2):

```
        Low  Mod  High
Low     133   12     0
Mod      37   19     0
High      4    7     0
```

The two experts match exactly on 152 of 212 pairs, and their disagreement is concentrated at the
Moderate/High boundary: 37 pairs Expert 1 called Moderate and 4 he called High are Low to Expert 2, and
**none** of Expert 1's 11 High pairs in this subset is High to Expert 2, who used no High label anywhere.
Only 4 pairs span the full Low-to-High range. **The Moderate/High boundary of the taxonomy is not shared
between the two annotators**, which caps what any predictor can score against a single one of them.

## Summary

1. Against Expert 1 the ensemble scores QWK **0.423** [0.351, 0.494] at 65.0% accuracy; against Expert 2,
   QWK **0.366** [0.284, 0.450] at 75.3% accuracy. The accuracy gap is a label-mix effect; the QWK
   intervals overlap.
2. The pipeline over-scores both experts by **+0.17 ordinal steps** (both bootstrap intervals exclude
   zero), and the dominant error is **Moderate over-prediction** (precision 31–36%).
3. The High class is barely measurable on this evidence: Expert 1 supplies 27 High pairs (recall 37.0%,
   CI [0.200, 0.569]) and Expert 2 supplies none.
4. On the 212 co-rated pairs the human–human ceiling is **QWK 0.346 / 71.7% exact**. The pipeline clears
   that ceiling on QWK against Expert 1 (0.462) and falls under it against Expert 2 (0.280), with both
   differences inside the noise at 39 shared abstracts. **Pipeline–expert disagreement is of the same
   order as expert–expert disagreement**, and the ranking of the three raters by severity
   (Expert 2 < Expert 1 < pipeline) means the residual gap is largely a threshold offset.

---

*Reproducibility.* Prediction file
`data/results/affinity_benchmark/20260726_134607_5/adjudicated_ra_affinities.csv`; expert files
`data/processed/affinity_calibration/20260728_abstracts_evaluation_template_mark_om_v1/` (Expert 1) and
`.../20260728_abstracts_evaluation_template_charlie_s_v1/` (Expert 2), column
`Evaluator_Affinity_Level`, merged on `(Abstract_Index, RA2025_ID)`. Boundary convention is immaterial
here: only 1–2 pairs per case land exactly on 40 or 80, and reading both boundaries as Moderate
(`40 ≤ s ≤ 80`, the rule used in `11_adjudication_readiness_analysis_v3.md`) shifts QWK by ≤ 0.005
(Expert 1 0.421, Expert 2 0.370) and accuracy by ≤ 0.2pp.
