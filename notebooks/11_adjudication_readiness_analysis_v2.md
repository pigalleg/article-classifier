# Notebook 11 — Adjudication Readiness: Runs `20260726_134607_4` + `20260726_154151_4`

**Scope:** the analysis in `11_affinity_diagnostisc_metrics_pipeline_v1.ipynb` run for **two** prediction
runs against two independent expert sets — 4 independent cases. Neither the expert sets nor the
prediction runs are ever pooled.

- Prediction runs, both 5 models (`claude-haiku-4.5`, `deepseek-v4-flash-cloud`,
  `gemini-3.1-flash-lite`, `gpt-5.4-mini`, `mistral-small-2603`), both 5040 adjudicated pairs over the
  same 105 abstracts:
  - `20260726_134607_4` — `few_shot_max_examples_per_prompt: 4`
  - `20260726_154151_4` — `few_shot_max_examples_per_prompt: 30`
- Expert sets: `20260728_abstracts_evaluation_template_mark_om_v1` (Mark) and
  `20260728_abstracts_evaluation_template_charlie_s_v1` (Charlie)

Per `run_metadata.json`, the few-shot budget is the **only** settings difference between the two runs
(the model roster, `top_k: 48`, `ra_retrieval_mode: cosine`, `score_scale_version: 2` and both
adjudication agreement thresholds at 1.0 are identical). The runs were produced at different commits
(`109e57e` vs `96be6ef`), both dirty.

`nbconvert`/`nbclient` are not installed here, so the notebook's cells were reproduced verbatim in a
standalone script against the real `src/analysis/affinity_metrics.py`, using the project's
`~/.virtualenvs/classifier` interpreter. Notebook 11's `normalize_level` maps a score of exactly 80 to
**Moderate** (`40 <= score <= 80`), unlike notebook 10 — reproduced as-is. Every `134607_4` number below
reproduces `11_adjudication_readiness_analysis_v1.md` exactly; that document is left untouched.

§8 is new relative to v1: with two runs over identical abstracts, the run-to-run difference can be
tested directly rather than eyeballed across columns, and that test changes how §1 should be read.

## 1. Headline metrics

| Metric | A: Mark × 134607_4 | B: Mark × 154151_4 | C: Charlie × 134607_4 | D: Charlie × 154151_4 |
|---|---|---|---|---|
| Few-shot examples | 4 | 30 | 4 | 30 |
| N pairs / abstracts | 602 / 91 | 602 / 91 | 722 / 99 | 722 / 99 |
| Accuracy | 65.3% | **66.8%** | **76.5%** | 75.4% |
| Quadratic Weighted Kappa | 0.435 | **0.458** | **0.389** | 0.369 |
| Mean signed error | +0.158 | +0.186 | +0.155 | +0.177 |
| Mean absolute error | 0.367 | 0.359 | 0.252 | 0.269 |
| Expert label mix (Low/Mod/High) | 438 / 137 / 27 | 438 / 137 / 27 | 617 / 105 / **0** | 617 / 105 / **0** |
| Predicted label mix (Low/Mod/High) | 358 / 202 / 42 | 350 / 201 / 51 | 544 / 139 / 39 | 533 / 145 / 44 |
| Share of pairs sent to the LLM judge | 429 / 602 = 71.3% | 422 / 602 = 70.1% | 392 / 722 = 54.3% | 388 / 722 = 53.7% |

The 30-example run looks better against Mark (+1.5pp accuracy, +0.023 QWK) and worse against Charlie
(−1.1pp, −0.019 QWK) — i.e. the two experts disagree on which run is better, which is the first sign
that the difference is not real. §8 tests it properly and confirms it is not.

What *is* consistent across all four cases is the over-scoring bias of **+0.16 to +0.19 of an ordinal
step**, and it is slightly *larger* in the 30-example run against both experts (+0.158 → +0.186 for
Mark, +0.155 → +0.177 for Charlie). More few-shot examples pushed scores up, not toward the experts.

Charlie's higher accuracy is a base-rate effect in both runs: 85% of Charlie's labels are Low (vs. 73%
for Mark), so a Low-leaning predictor scores better on accuracy while scoring *worse* on QWK.

The judge-touched share is far above the run-wide rate in every case (40.5% of all 5040 pairs in
`134607_4`, 40.1% in `154151_4`): the calibration abstracts are more model-divergent than the benchmark
set as a whole, so adjudication cost per abstract is higher here than a full-corpus estimate would
suggest. Raising the few-shot budget did **not** reduce panel divergence — the adjudicated share is
essentially unchanged (2039 vs. 2022 pairs).

## 2. Per-class metrics

| Case | Class | Support | Precision | Recall | One-vs-rest acc. |
|---|---|---|---|---|---|
| A: Mark × 134607_4 | Low | 438 | 86.6% | 70.8% | 70.8% |
| A: Mark × 134607_4 | Moderate | 137 | 35.1% | 51.8% | 67.3% |
| A: Mark × 134607_4 | High | 27 | 28.6% | 44.4% | 92.5% |
| B: Mark × 154151_4 | Low | 438 | 88.9% | 71.0% | 72.4% |
| B: Mark × 154151_4 | Moderate | 137 | 38.3% | 56.2% | 69.4% |
| B: Mark × 154151_4 | High | 27 | 27.5% | 51.9% | 91.7% |
| C: Charlie × 134607_4 | Low | 617 | 93.6% | 82.5% | 80.2% |
| C: Charlie × 134607_4 | Moderate | 105 | 30.9% | 41.0% | 78.1% |
| C: Charlie × 134607_4 | High | 0 | 0.0% | — | 94.6% |
| D: Charlie × 154151_4 | Low | 617 | 93.8% | 81.0% | 79.2% |
| D: Charlie × 154151_4 | Moderate | 105 | 30.3% | 41.9% | 77.6% |
| D: Charlie × 154151_4 | High | 0 | 0.0% | — | 93.9% |

Moderate precision remains the weak point in all four cases (30-38%), unchanged in character from the
earlier runs. The 30-example run buys Mark's Moderate class ~3pp of precision and ~4pp of recall and
Mark's High recall 7pp (44.4% → 51.9%, on n=27), but buys Charlie nothing — Charlie's Moderate
precision is flat at ~30% and Low recall drops 1.5pp.

Charlie assigned **no** High labels at all, so all 39 (run `134607_4`) / 44 (run `154151_4`) High
predictions in cases C and D are false positives by construction, and High precision/recall are
undefined rather than measured.

Confusion matrices (rows = expert, columns = predicted):

```
A: Mark × 134607_4             B: Mark × 154151_4
        Low  Mod  High                 Low  Mod  High
Low     310  120     8         Low     311  115    12
Mod      44   71    22         Mod      35   77    25
High      4   11    12         High      4    9    14

C: Charlie × 134607_4          D: Charlie × 154151_4
        Low  Mod  High                 Low  Mod  High
Low     509   96    12         Low     500  101    16
Mod      35   43    27         Mod      33   44    28
High      0    0     0         High      0    0     0
```

## 3. Reliability curve (`agreement_mean` vs. accuracy)

| Case | Agreement 0.33 (n, acc.) | Agreement 0.50 (n, acc.) | Agreement 1.00 (n, acc.) |
|---|---|---|---|
| A: Mark × 134607_4 | 46, 36.1% | 383, 68.5% | 173, 85.4% |
| B: Mark × 154151_4 | 51, 46.2% | 372, 68.0% | 179, 89.8% |
| C: Charlie × 134607_4 | 41, 39.0% | 351, 73.5% | 330, 91.7% |
| D: Charlie × 154151_4 | 36, 47.2% | 352, 67.8% | 334, 94.2% |

Accuracy is monotone in panel agreement in all four cases, so `agreement_mean` remains a usable
confidence signal overall. **But it only works for the Low class, and it inverts for Moderate — in
both runs:**

| Case | Class | acc. @ 0.33 | acc. @ 0.50 | acc. @ 1.00 |
|---|---|---|---|---|
| A: Mark × 134607_4 | Low | 7.4% (n=27) | 66.5% (n=260) | 89.4% (n=151) |
| A: Mark × 134607_4 | Moderate | 73.7% (n=19) | 52.5% (n=99) | **26.3%** (n=19) |
| B: Mark × 154151_4 | Low | 15.6% (n=32) | 64.3% (n=252) | 93.5% (n=154) |
| B: Mark × 154151_4 | Moderate | 88.2% (n=17) | 55.4% (n=101) | **31.6%** (n=19) |
| C: Charlie × 134607_4 | Low | 17.2% (n=29) | 75.9% (n=282) | 94.8% (n=306) |
| C: Charlie × 134607_4 | Moderate | 83.3% (n=12) | 37.7% (n=69) | **29.2%** (n=24) |
| D: Charlie × 154151_4 | Low | 29.6% (n=27) | 68.2% (n=280) | 97.1% (n=310) |
| D: Charlie × 154151_4 | Moderate | 88.9% (n=9) | 40.3% (n=72) | **29.2%** (n=24) |

The inversion is stable at 26-32% accuracy on unanimous-panel Moderate pairs in all four cases, and the
larger few-shot budget does not fix it. A unanimous panel on a pair the expert called Moderate is
usually a panel unanimously agreeing on *Low*. High `agreement_mean` should not be read as "safe to
auto-accept" for anything the panel did not call Low — confident consensus is exactly where the
Moderate class fails.

## 4. Adjudication (LLM judge) impact — split by `Final_Method`

| Case | Population | N | Pre-adjudication acc. | Post-adjudication acc. | Δ |
|---|---|---|---|---|---|
| A: Mark × 134607_4 | All pairs | 602 | 65.5% | 65.3% | −0.2pp |
| A: Mark × 134607_4 | Adjudicated only | 429 | 58.7% | 58.5% | −0.2pp |
| A: Mark × 134607_4 | Untouched only | 173 | 82.1% | 82.1% | 0.0 |
| B: Mark × 154151_4 | All pairs | 602 | 66.6% | 66.8% | +0.2pp |
| B: Mark × 154151_4 | Adjudicated only | 422 | 58.3% | 58.5% | +0.2pp |
| B: Mark × 154151_4 | Untouched only | 180 | 86.1% | 86.1% | 0.0 |
| C: Charlie × 134607_4 | All pairs | 722 | 75.9% | 76.5% | +0.6pp |
| C: Charlie × 134607_4 | Adjudicated only | 392 | 64.0% | 65.1% | +1.0pp |
| C: Charlie × 134607_4 | Untouched only | 330 | 90.0% | 90.0% | 0.0 |
| D: Charlie × 154151_4 | All pairs | 722 | 76.3% | 75.4% | −1.0pp |
| D: Charlie × 154151_4 | Adjudicated only | 388 | 62.6% | 60.8% | −1.8pp |
| D: Charlie × 154151_4 | Untouched only | 334 | 92.2% | 92.2% | 0.0 |

Adding the second run makes the sign instability worse, not better. On QWK the four cases split two
against two: Mark × 134607_4 0.445 → **0.435** (judge hurts), Mark × 154151_4 0.439 → **0.458** (helps),
Charlie × 134607_4 0.375 → **0.389** (helps), Charlie × 154151_4 0.381 → **0.369** (hurts). The judge's
sign flips with the annotator *and* with the few-shot budget.

Per-pair outcomes on the pairs the judge actually touched:

| Case | Improved | Worsened | Unchanged |
|---|---|---|---|
| A: Mark × 134607_4 | 16 (3.7%) | 17 (4.0%) | 396 (92.3%) |
| B: Mark × 154151_4 | 14 (3.3%) | 13 (3.1%) | 395 (93.6%) |
| C: Charlie × 134607_4 | 15 (3.8%) | 11 (2.8%) | 366 (93.4%) |
| D: Charlie × 154151_4 | 9 (2.3%) | 16 (4.1%) | 363 (93.6%) |

Measured run-wide (no expert labels needed):

| Run | Pairs sent to judge | Level changed | Raised | Lowered | Mean raw shift | Mean abs shift |
|---|---|---|---|---|---|---|
| `20260726_134607_4` | 2039 | 175 (8.6%) | 72 | 103 | −2.7 | 4.8 |
| `20260726_154151_4` | 2022 | 126 (6.2%) | 81 | 45 | −1.4 | 4.1 |

Every adjudicated pair has `agreement_mean <= 0.5` by construction, so the judge is being invoked
precisely where the panel is split, and it mostly ratifies the ensemble mean rather than overriding it.
With 30 few-shot examples the judge is *even more* of a no-op: it moves 6.2% of levels instead of 8.6%,
and its net direction flips from lowering (103 down / 72 up) to raising (81 up / 45 down).

**Read together with §6 and §8, the honest conclusion is that the judge's effect (±1-2pp, sign flipping
by both annotator and few-shot budget) is smaller than the disagreement between two human experts, so
this data cannot establish whether adjudication helps.** It costs ~2000 LLM calls per run to move
126-175 labels.

## 5. Bootstrap confidence intervals (cluster-by-abstract, 500 resamples, seed 42)

| Case | QWK | Accuracy | Mean signed error | Recall Low | Recall Moderate | Recall High |
|---|---|---|---|---|---|---|
| A: Mark × 134607_4 | [0.355, 0.512] | [0.598, 0.705] | [0.096, 0.228] | [0.642, 0.771] | [0.447, 0.599] | [0.245, 0.680] |
| B: Mark × 154151_4 | [0.379, 0.533] | [0.619, 0.716] | [0.125, 0.255] | [0.651, 0.765] | [0.491, 0.638] | [0.321, 0.739] |
| C: Charlie × 134607_4 | [0.305, 0.467] | [0.716, 0.815] | [0.096, 0.218] | [0.776, 0.873] | [0.314, 0.507] | [0.0, 0.0]* |
| D: Charlie × 154151_4 | [0.293, 0.452] | [0.704, 0.809] | [0.116, 0.240] | [0.762, 0.863] | [0.330, 0.506] | [0.0, 0.0]* |

\* Charlie has zero true High pairs, so every resample also has zero support — the interval is a
degenerate point at 0, not a recall estimate.

All four mean-signed-error intervals are entirely positive, confirming the over-scoring bias is
structural rather than sampling noise. The QWK intervals overlap almost completely between the two runs
for the same expert (A vs. B: [0.355, 0.512] vs. [0.379, 0.533]; C vs. D: [0.305, 0.467] vs.
[0.293, 0.452]) and also across experts, so the pipeline is **not** measurably better in either
configuration or against either expert. High recall for Mark is too wide at n=27 in both runs to
support any claim about that class.

These are unpaired intervals; §8 does the paired version, which is the correct test for a run-vs-run
comparison on identical abstracts.

## 6. Inter-expert ceiling (Mark vs. Charlie)

Mark and Charlie both labelled **212 pairs across 39 abstracts**. This section is run-independent for
the human-human row; the pipeline rows are reported for both runs on that same overlap.

| Comparison | Exact agreement | QWK |
|---|---|---|
| Mark vs. Charlie | 71.7% | **0.346** |
| Pipeline (`134607_4`) vs. Mark | 61.8% | **0.470** |
| Pipeline (`154151_4`) vs. Mark | 64.2% | **0.496** |
| Pipeline (`134607_4`) vs. Charlie | 58.0% | 0.286 |
| Pipeline (`154151_4`) vs. Charlie | 56.1% | 0.284 |

```
Mark \ Charlie   Low  Mod  High
Low              133   12     0
Moderate          37   19     0
High               4    7     0
```

**Both pipeline runs agree with Mark (QWK 0.470 / 0.496) better than Charlie agrees with Mark (0.346).**
Mark scores systematically higher than Charlie — mean difference **+0.189** ordinal steps, comparable in
magnitude and identical in direction to the pipeline's own +0.16 to +0.19 bias against both experts
(pipeline-vs-Mark bias on the overlap is +0.193 / +0.250; pipeline-vs-Charlie is +0.382 / +0.439). The
disagreement is concentrated in the Moderate/High boundary: 37 pairs Mark called Moderate and Charlie
called Low, and all 11 of Mark's High pairs in the overlap were Low or Moderate to Charlie.

This reframes §1-§4. A QWK of 0.37-0.46 against a single annotator is not obviously below the
human-human ceiling on this taxonomy, and the "over-scoring bias" the pipeline is being penalised for is
the same axis on which the two experts disagree with each other. Chasing bias reduction against one
expert may just be fitting that expert's threshold.

## 7. Cut-point cross-check

Notebook 10 found that refitting the Low/Moderate/High boundaries was the largest single gain on the
un-adjudicated ensemble. On the *adjudicated* output the result is inconsistent for the 4-example run
and clearly positive for the 30-example run (grouped 5-fold by abstract, cut points fitted on train
abstracts only, grid 10-120 step 5, selected by QWK):

| Case | QWK @ 40/80 | QWK refit (held out) | Bias | Accuracy | All-data cuts | Lower cut per fold |
|---|---|---|---|---|---|---|
| A: Mark × 134607_4 | 0.435 | 0.420 (**−0.015**) | +0.158 → +0.012 | 0.653 → 0.701 | 65/80 | 55-70 |
| B: Mark × 154151_4 | 0.458 | 0.495 (**+0.037**) | +0.186 → +0.042 | 0.663 → 0.724 | 55/80 | 55 (all folds) |
| C: Charlie × 134607_4 | 0.389 | 0.407 (**+0.018**) | +0.155 → +0.021 | 0.765 → 0.837 | 55/95 | 55-60 |
| D: Charlie × 154151_4 | 0.369 | 0.422 (**+0.053**) | +0.177 → +0.029 | 0.754 → 0.838 | 55/95 | 55-60 |

Recalibration reliably removes the bias (+0.16/+0.19 → ~0.01-0.04) and adds 5-8pp of exact accuracy in
all four cases. QWK now improves in three of four — the only loss is case A, the one case v1 already
reported. The 30-example run is the more recalibratable of the two: it gains QWK against both experts
and its per-fold lower cut is perfectly stable at 55 against Mark.

The fitted cuts still differ substantially between the two experts (65/80 or 55/80 for Mark vs. 55/95
for Charlie), so: refit cut points if exact accuracy and unbiasedness are the goal, expect a QWK gain
only on the 30-example run, and do not hard-code one expert's cuts.

(The "QWK @ 40/80" column uses notebook 11's `normalize_level`, where a score of exactly 80 is Moderate.
Using notebook 10's strict `score < 80` rule instead shifts this column by at most 0.004 — A 0.431,
B 0.460, C 0.389, D 0.366 — which does not change any conclusion above.)

## 8. Run vs. run — is the 30-example few-shot budget actually better? (new)

The two runs cover identical `(Abstract_Index, RA2025_ID)` pairs, so they can be compared directly.

Over all 5040 pairs, with no expert labels involved, the two runs are near-duplicates: **95.0% exact
level agreement, QWK 0.854**, mean raw score shift **+0.39 points** on the 0-120 scale, mean level shift
−0.004. The disagreements are almost entirely Low↔Moderate churn:

```
134607_4 \ 154151_4   Low   Mod  High
Low                  4213    94     0
Moderate              117   504    21
High                    0    20    71
```

Paired cluster bootstrap over abstracts (2000 draws, seed 0), `154151_4` minus `134607_4`:

| Expert | Δ QWK | Δ Accuracy | Δ Bias |
|---|---|---|---|
| Mark (602 pairs, 91 abstracts) | +0.023 [−0.014, +0.063] | +0.015 [−0.009, +0.039] | **+0.028 [+0.003, +0.054]** |
| Charlie (722 pairs, 99 abstracts) | −0.019 [−0.052, +0.015] | −0.011 [−0.031, +0.009] | **+0.022 [+0.003, +0.043]** |

**Every QWK and accuracy interval straddles zero, and the two experts put the point estimate on opposite
sides of it.** Raising the few-shot budget from 4 to 30 examples produced no measurable accuracy or
agreement gain. The one effect that *is* measurable — both intervals exclude zero, and in the same
direction for both experts — is that it made the **over-scoring bias worse** by about +0.02-0.03 of an
ordinal step.

Caveat on `154151_4`: 241 of its 5040 pairs were scored by only 4 models and 2 by 3 models (missing LLM
scores: `mistral-small-2603` 200, `gemini-3.1-flash-lite` 36, `claude-haiku-4.5` 5, `gpt-5.4-mini` 4),
against 27 four-model pairs in `134607_4`. That is ~4.8% of pairs on a thinner panel, which affects both
`Affinity_Mean` and `agreement_mean` for those pairs. It is too small to explain the §8 result but should
be fixed before the run is used as a baseline.

## Summary

1. **The 4→30 few-shot increase is not an improvement.** The runs agree on 95% of pairs; paired
   bootstrap ΔQWK and Δaccuracy both straddle zero with opposite signs per expert. The only significant
   effect is *more* over-scoring bias (+0.02-0.03 ordinal steps, both experts).
2. **The judge is effectively a no-op in both runs**: 8.6% (`134607_4`) and 6.2% (`154151_4`) of
   adjudicated levels changed, net effect ±1-2pp with the sign depending on which expert *and* which run
   you score against. ~2000 LLM calls per run to move 126-175 labels.
3. **`agreement_mean` is a confidence signal for Low only** — it inverts for Moderate in all four cases,
   where unanimous panels are wrong 68-74% of the time, and the larger few-shot budget does not fix it.
4. **The human-human ceiling (QWK 0.346) sits below the pipeline's agreement with Mark (0.470 / 0.496)
   in both runs.** Before further tuning, the experts' Moderate/High boundary needs reconciling —
   otherwise the target itself is the limiting factor.
5. **Cut-point refitting is the one intervention that reliably pays**, removing the bias and adding
   5-8pp of exact accuracy in all four cases; it now also gains QWK in three of four, and gains most on
   the 30-example run.
6. **Charlie's set cannot evaluate the High class** (zero High labels) and Mark's barely can (n=27). Any
   High-class conclusion here is unsupported.
