# Notebook 11 — Adjudication Readiness: Run `20260726_134607_4`

**Scope:** the analysis in `11_affinity_diagnostisc_metrics_pipeline.ipynb` run for one prediction run
against two independent expert sets. The expert sets are never pooled.

- Prediction run: `20260726_134607_4` — 5 models (`claude-haiku-4.5`, `deepseek-v4-flash-cloud`,
  `gemini-3.1-flash-lite`, `gpt-5.4-mini`, `mistral-small-2603`), 5040 adjudicated pairs
- Expert sets: `20260728_abstracts_evaluation_template_mark_om_v1` (Mark) and
  `20260728_abstracts_evaluation_template_charlie_s_v1` (Charlie)

`nbconvert`/`nbclient` are not installed here, so the notebook's cells were reproduced verbatim in a
standalone script against the real `src/analysis/affinity_metrics.py`, using the project's
`~/.virtualenvs/classifier` interpreter. Note that notebook 11's `normalize_level` maps a score of
exactly 80 to **Moderate** (`40 <= score <= 80`), unlike notebook 10 — reproduced as-is.

This is a new document; the earlier `11_adjudication_readiness_analysis.md` (runs `134607_1` /
`154151_1`) is left untouched. §6 is new: it measures the human-human ceiling, which turns out to
change how §1-§4 should be read.

## 1. Headline metrics

| Metric | A: Mark × 134607_4 | B: Charlie × 134607_4 |
|---|---|---|
| N pairs / abstracts | 602 / 91 | 722 / 99 |
| Accuracy | 65.3% | 76.5% |
| Quadratic Weighted Kappa | 0.435 | 0.389 |
| Mean signed error | +0.158 | +0.155 |
| Mean absolute error | 0.367 | 0.252 |
| Expert label mix (Low/Mod/High) | 438 / 137 / 27 | 617 / 105 / **0** |
| Predicted label mix (Low/Mod/High) | 358 / 202 / 42 | 544 / 139 / 39 |
| Share of pairs sent to the LLM judge | 429 / 602 = 71.3% | 392 / 722 = 54.3% |

Both cases show the same over-scoring bias of about **+0.16 of an ordinal step**, and it is almost
identical across the two experts — a property of the pipeline, not of one annotator. Charlie's higher
accuracy is a base-rate effect: 85% of Charlie's labels are Low (vs. 73% for Mark), so a
Low-leaning predictor scores better on accuracy while scoring *worse* on QWK.

The judge-touched share is far above the run-wide rate (40.5% of all 5040 pairs): the calibration
abstracts are more model-divergent than the benchmark set as a whole, so adjudication cost per
abstract is higher here than a full-corpus estimate would suggest.

## 2. Per-class metrics

| Case | Class | Support | Precision | Recall | One-vs-rest acc. |
|---|---|---|---|---|---|
| A: Mark | Low | 438 | 86.6% | 70.8% | 70.8% |
| A: Mark | Moderate | 137 | 35.1% | 51.8% | 67.3% |
| A: Mark | High | 27 | 28.6% | 44.4% | 92.5% |
| B: Charlie | Low | 617 | 93.6% | 82.5% | 80.2% |
| B: Charlie | Moderate | 105 | 30.9% | 41.0% | 78.1% |
| B: Charlie | High | 0 | 0.0% | — | 94.6% |

Moderate precision remains the weak point in both cases (31-35%), unchanged in character from the
earlier runs. Charlie assigned **no** High labels at all, so all 39 of the pipeline's High
predictions in case B are false positives by construction and High precision/recall are undefined
rather than measured.

Confusion matrices (rows = expert):

```
A: Mark                        B: Charlie
        Low  Mod  High                 Low  Mod  High
Low     310  120     8         Low     509   96    12
Mod      44   71    22         Mod      35   43    27
High      4   11    12         High      0    0     0
```

## 3. Reliability curve (`agreement_mean` vs. accuracy)

| Case | Agreement 0.33 (n, acc.) | Agreement 0.50 (n, acc.) | Agreement 1.00 (n, acc.) |
|---|---|---|---|
| A: Mark | 46, 36.1% | 383, 68.5% | 173, 85.4% |
| B: Charlie | 41, 39.0% | 351, 73.5% | 330, 91.7% |

Accuracy is monotone in panel agreement in both cases, so `agreement_mean` remains a usable
confidence signal overall. **But it only works for the Low class**, and it inverts for Moderate:

| Case | Class | acc. @ 0.33 | acc. @ 0.50 | acc. @ 1.00 |
|---|---|---|---|---|
| A: Mark | Low | 7.4% (n=27) | 66.5% (n=260) | 89.4% (n=151) |
| A: Mark | Moderate | 73.7% (n=19) | 52.5% (n=99) | **26.3%** (n=19) |
| B: Charlie | Low | 17.2% (n=29) | 75.9% (n=282) | 94.8% (n=306) |
| B: Charlie | Moderate | 83.3% (n=12) | 37.7% (n=69) | **29.2%** (n=24) |

A unanimous panel on a pair the expert called Moderate is usually a panel unanimously agreeing on
*Low*. So high `agreement_mean` should not be read as "safe to auto-accept" for anything the panel
did not call Low — confident consensus is exactly where the Moderate class fails.

## 4. Adjudication (LLM judge) impact — split by `Final_Method`

| Case | Population | N | Pre-adjudication acc. | Post-adjudication acc. | Δ |
|---|---|---|---|---|---|
| A: Mark | All pairs | 602 | 65.5% | 65.3% | −0.2pp |
| A: Mark | Adjudicated only | 429 | 58.7% | 58.5% | −0.2pp |
| A: Mark | Untouched only | 173 | 82.1% | 82.1% | 0.0 |
| B: Charlie | All pairs | 722 | 75.9% | 76.5% | +0.6pp |
| B: Charlie | Adjudicated only | 392 | 64.0% | 65.1% | +1.0pp |
| B: Charlie | Untouched only | 330 | 90.0% | 90.0% | 0.0 |

On QWK the two cases also disagree in sign: Mark 0.445 → **0.435** (judge hurts), Charlie 0.375 →
**0.389** (judge helps). Per-pair outcomes on the pairs the judge actually touched:

| Case | Improved | Worsened | Unchanged |
|---|---|---|---|
| A: Mark | 16 (3.7%) | 17 (4.0%) | 429−33 = 92.3% |
| B: Charlie | 15 (3.8%) | 11 (2.8%) | 392−26 = 93.4% |

Measured run-wide (no expert labels needed), the judge is close to a no-op: of **2039** pairs sent to
it, it changed the discretised level on **175 (8.6%)** — 72 raised, 103 lowered — with a mean raw
score shift of **−2.7 points** on a 0-120 scale (mean absolute shift 4.8). Every adjudicated pair has
`agreement_mean <= 0.5` by construction, so the judge is being invoked precisely where the panel is
split, and it mostly ratifies the ensemble mean rather than overriding it.

**Read together with §6, the honest conclusion is that the judge's effect (±1pp, sign flipping by
annotator) is smaller than the disagreement between two human experts, so this data cannot establish
whether adjudication helps.** It costs 2039 LLM calls to move 175 labels.

## 5. Bootstrap confidence intervals (cluster-by-abstract, 500 resamples, seed 42)

| Case | QWK | Accuracy | Mean signed error | Recall Low | Recall Moderate | Recall High |
|---|---|---|---|---|---|---|
| A: Mark | [0.355, 0.512] | [0.598, 0.705] | [0.096, 0.228] | [0.642, 0.771] | [0.447, 0.599] | [0.245, 0.680] |
| B: Charlie | [0.305, 0.467] | [0.716, 0.815] | [0.096, 0.218] | [0.776, 0.873] | [0.314, 0.507] | [0.0, 0.0]* |

\* Charlie has zero true High pairs, so every resample also has zero support — the interval is a
degenerate point at 0, not a recall estimate.

Both mean-signed-error intervals are entirely positive, confirming the over-scoring bias is
structural rather than sampling noise. The two QWK intervals overlap heavily ([0.355, 0.512] vs.
[0.305, 0.467]), so the pipeline is **not** measurably better against one expert than the other —
and High recall for Mark ([0.245, 0.680]) is too wide at n=27 to support any claim about that class.

## 6. Inter-expert ceiling (Mark vs. Charlie) — new

Mark and Charlie both labelled **212 pairs across 39 abstracts**. On that overlap:

| Comparison | Exact agreement | QWK |
|---|---|---|
| Mark vs. Charlie | 71.7% | **0.346** |
| Pipeline vs. Mark | 61.8% | **0.470** |
| Pipeline vs. Charlie | 58.0% | 0.286 |

```
Mark \ Charlie   Low  Mod  High
Low              133   12     0
Moderate          37   19     0
High               4    7     0
```

**The pipeline agrees with Mark (QWK 0.470) better than Charlie agrees with Mark (0.346).** Mark
scores systematically higher than Charlie — mean difference **+0.189** ordinal steps, comparable in
magnitude and identical in direction to the pipeline's own +0.16 bias against both experts. The
disagreement is concentrated in the Moderate/High boundary: 37 pairs Mark called Moderate and
Charlie called Low, and all 11 of Mark's High pairs in the overlap were Low or Moderate to Charlie.

This reframes §1-§4. A QWK of 0.39-0.44 against a single annotator is not obviously below the
human-human ceiling on this taxonomy, and the "over-scoring bias" the pipeline is being penalised for
is the same axis on which the two experts disagree with each other. Chasing bias reduction against
one expert may just be fitting that expert's threshold.

## 7. Cut-point cross-check

Notebook 10 found that refitting the Low/Moderate/High boundaries was the largest single gain on the
un-adjudicated ensemble. On the *adjudicated* output the result is weaker and inconsistent
(grouped 5-fold by abstract, cut points fitted on train abstracts only):

| Case | QWK @ 40/80 | QWK refit (held out) | Bias | Accuracy | All-data cuts | Lower cut per fold |
|---|---|---|---|---|---|---|
| A: Mark | 0.435 | 0.421 (**−0.014**) | +0.158 → +0.005 | 0.653 → 0.701 | 65/80 | 55-70 |
| B: Charlie | 0.389 | 0.407 (**+0.018**) | +0.155 → +0.021 | 0.765 → 0.837 | 55/95 | 55-60 |

Recalibration reliably removes the bias (+0.16 → ~0.01) and adds 5-7pp of exact accuracy in both
cases, but QWK moves in opposite directions and the fitted cuts differ substantially between the two
experts (65/80 vs. 55/95). Refit cut points if exact accuracy and unbiasedness are the goal; do not
expect a QWK gain, and do not hard-code one expert's cuts.

## Summary

1. **The judge is effectively a no-op on this data**: 8.6% of adjudicated levels changed, net effect
   ±1pp with the sign depending on which expert you score against.
2. **`agreement_mean` is a confidence signal for Low only** — it inverts for Moderate, where
   unanimous panels are wrong 70-74% of the time.
3. **The human-human ceiling (QWK 0.346) sits below the pipeline's agreement with Mark (0.470).**
   Before further tuning, the experts' Moderate/High boundary needs reconciling — otherwise the
   target itself is the limiting factor.
4. **Charlie's set cannot evaluate the High class** (zero High labels) and Mark's barely can (n=27).
   Any High-class conclusion here is unsupported.
