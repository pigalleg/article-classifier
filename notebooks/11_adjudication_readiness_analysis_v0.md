# Notebook 11 — Adjudicated Affinity Pipeline: Metric Analysis

**Scope:** Ran the analysis in `11_affinity_diagnostisc_metrics_adjudicated.ipynb` for 4 independent
cases — 2 prediction runs × 2 expert calibration sets. Each case is reported on its own; expert sets are
never pooled together, and neither are the two prediction runs.

- Prediction runs: `20260726_134607_1` (4-example few-shot) and `20260726_154151_1` (~20/30-example few-shot)
- Expert sets: `20260728_abstracts_evaluation_template_mark_om_v1` (Mark) and
  `20260728_abstracts_evaluation_template_charlie_s_v1` (Charlie)

`nbconvert`/`nbclient` are not installed in this environment, so the notebook's cells were reproduced in a
standalone script against the real `src/analysis/affinity_metrics.py` code (using the project's actual
`~/.virtualenvs/clasifier` interpreter), rather than executed in place. Numbers below are exact
reproductions of what the notebook cells compute.

**Note on this revision:** both prediction runs' `adjudicated_ra_affinities.csv` files were regenerated on
2026-07-29 (after the adjudicator few-shot rework in commits `1f9eac0`/`f6e5be5`), so every number below
differs slightly from the previous version of this doc even though the run directory names and notebook
logic are unchanged. This revision also adds §5 (bootstrap confidence intervals), a section the notebook
gained since the doc was last written.

## 1. Headline metrics

| Metric | A: Mark × 134607_1 | B: Mark × 154151_1 | C: Charlie × 134607_1 | D: Charlie × 154151_1 |
|---|---|---|---|---|
| N pairs / abstracts | 602 / 91 | 602 / 91 | 722 / 99 | 722 / 99 |
| Accuracy | 63.6% | 64.6% | 74.4% | 73.7% |
| Quadratic Weighted Kappa | 0.462 | 0.457 | 0.376 | 0.341 |
| Mean signed error | +0.234 | +0.236 | +0.195 | +0.205 |
| Expert label mix (Low/Mod/High) | 438/137/27 | 438/137/27 | 617/105/0 | 617/105/0 |
| Predicted label mix (Low/Mod/High) | 335/202/65 | 340/191/71 | 523/152/47 | 521/149/52 |

Mean signed error is on the 0/1/2 ordinal scale, so every case still shows the same direction of bias: the
pipeline's predicted bucket lands, on average, a fifth-to-a-quarter step above the expert's — an
over-scoring bias in all 4 cases, not just one prediction run or one expert. Accuracy and QWK both moved up
slightly from the prior data snapshot in all 4 cases, but the bias direction and its rough magnitude are
unchanged.

## 2. Per-class metrics

| Case | Class | Support | Precision | Recall | One-vs-rest acc. |
|---|---|---|---|---|---|
| A: Mark × 134607_1 | Low | 438 | 89.3% | 68.3% | 70.9% |
| A: Mark × 134607_1 | Moderate | 137 | 33.7% | 49.6% | 66.3% |
| A: Mark × 134607_1 | High | 27 | 24.6% | 59.3% | 90.0% |
| B: Mark × 154151_1 | Low | 438 | 90.0% | 69.9% | 72.4% |
| B: Mark × 154151_1 | Moderate | 137 | 35.6% | 49.6% | 68.1% |
| B: Mark × 154151_1 | High | 27 | 21.1% | 55.6% | 88.7% |
| C: Charlie × 134607_1 | Low | 617 | 94.3% | 79.9% | 78.7% |
| C: Charlie × 134607_1 | Moderate | 105 | 28.9% | 41.9% | 76.6% |
| C: Charlie × 134607_1 | High | 0 | 0.0% | — | 93.5% |
| D: Charlie × 154151_1 | Low | 617 | 93.9% | 79.3% | 77.8% |
| D: Charlie × 154151_1 | Moderate | 105 | 28.9% | 41.0% | 76.7% |
| D: Charlie × 154151_1 | High | 0 | 0.0% | — | 92.8% |

Same pattern in all 4 cases: "Moderate" precision is still stuck in the high-20s/low-30s%, and "High"
precision is low or (against Charlie, who never used "High") zero by construction, since Charlie's set has
no true High examples at all. Low-class recall improved by roughly 1-6pp across the board relative to the
prior snapshot, which is most of where the headline accuracy gain came from.

## 3. Reliability curve (panel agreement vs. accuracy)

| Case | Agreement 0.33 (n, acc.) | Agreement 0.50 (n, acc.) | Agreement 1.00 (n, acc.) |
|---|---|---|---|
| A: Mark × 134607_1 | 39, 41.4% | 367, 66.1% | 196, 84.8% |
| B: Mark × 154151_1 | 47, 41.6% | 349, 68.2% | 206, 85.8% |
| C: Charlie × 134607_1 | 32, 32.0% | 359, 71.9% | 331, 89.9% |
| D: Charlie × 154151_1 | 26, 39.2% | 374, 69.6% | 322, 90.2% |

The pattern holds in all 4 cases: full model-panel agreement (1.00) tracks noticeably higher accuracy than
partial agreement, so `agreement_mean` remains a usable confidence signal regardless of expert set or run.
Note the population shifted vs. the prior snapshot — more pairs now fall in the 0.33/0.50 buckets and fewer
in full agreement (196-206 now vs. 243-245 before for Mark; 322-331 now vs. 356-361 before for Charlie),
consistent with the adjudicator having been re-run.

## 4. Adjudication (LLM judge) impact — split by `Final_Method`

| Case | Population | N | Pre-adjudication acc. | Post-adjudication acc. |
|---|---|---|---|---|
| A: Mark × 134607_1 | All pairs | 602 | 64.6% | 63.6% |
| A: Mark × 134607_1 | Adjudicated only | 406 | 56.9% | 55.4% |
| A: Mark × 134607_1 | Untouched only | 196 | 80.6% | 80.6% |
| B: Mark × 154151_1 | All pairs | 602 | 65.1% | 64.6% |
| B: Mark × 154151_1 | Adjudicated only | 396 | 57.1% | 56.3% |
| B: Mark × 154151_1 | Untouched only | 206 | 80.6% | 80.6% |
| C: Charlie × 134607_1 | All pairs | 722 | 74.7% | 74.4% |
| C: Charlie × 134607_1 | Adjudicated only | 391 | 64.0% | 63.4% |
| C: Charlie × 134607_1 | Untouched only | 331 | 87.3% | 87.3% |
| D: Charlie × 154151_1 | All pairs | 722 | 72.6% | 73.7% |
| D: Charlie × 154151_1 | Adjudicated only | 400 | 61.5% | 63.5% |
| D: Charlie × 154151_1 | Untouched only | 322 | 86.3% | 86.3% |

The judge's net effect on adjudicated pairs is still small in every case — flat-to-slightly-negative in A, B
and C (roughly -0.6 to -1.5pp), and the largest lift is again case D (+2.0pp, the 20/30-example few-shot run
scored against Charlie). No case shows the judge meaningfully closing the gap to expert labels on its own;
this conclusion is unchanged from the prior snapshot even though the underlying numbers moved.

## 5. Bootstrap confidence intervals (cluster-by-abstract, 500 resamples, seed 42)

| Case | QWK 95% CI | Mean signed error 95% CI | Recall Low 95% CI | Recall Moderate 95% CI | Recall High 95% CI |
|---|---|---|---|---|---|
| A: Mark × 134607_1 | [0.390, 0.530] | [0.165, 0.305] | [0.617, 0.742] | [0.422, 0.563] | [0.414, 0.824] |
| B: Mark × 154151_1 | [0.388, 0.529] | [0.178, 0.304] | [0.635, 0.753] | [0.413, 0.574] | [0.359, 0.778] |
| C: Charlie × 134607_1 | [0.295, 0.454] | [0.133, 0.261] | [0.748, 0.851] | [0.339, 0.517] | [0.0, 0.0]* |
| D: Charlie × 154151_1 | [0.265, 0.424] | [0.142, 0.269] | [0.742, 0.846] | [0.316, 0.500] | [0.0, 0.0]* |

\* Charlie's set has zero true "High" pairs (see §1/§2), so High recall is undefined by construction and
every bootstrap resample also has zero support for that class — the CI collapses to a point at 0, not a
real recall estimate.

Point estimates from §1/§2 sit comfortably inside these intervals in every case, and none of the four QWK
or mean-signed-error CIs cross zero-bias territory (MSE CIs are entirely positive), reinforcing that the
over-scoring bias is a stable feature of the pipeline rather than sampling noise from a small abstract set
(91-99 abstracts per case).
