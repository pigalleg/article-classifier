# Notebook 11 — Adjudication Readiness: Runs `20260726_134607_4` + `20260726_154151_4` × 3 experts

**Scope:** the analysis in `11_affinity_diagnostisc_metrics_pipeline_v1.ipynb` run for **two** prediction
runs against **three** independent expert sets — 6 independent cases. Neither the expert sets nor the
prediction runs are ever pooled.

- Prediction runs, both 5 models (`claude-haiku-4.5`, `deepseek-v4-flash-cloud`,
  `gemini-3.1-flash-lite`, `gpt-5.4-mini`, `mistral-small-2603`), both 5040 adjudicated pairs over the
  same 105 abstracts:
  - `20260726_134607_4` — `few_shot_max_examples_per_prompt: 4`
  - `20260726_154151_4` — `few_shot_max_examples_per_prompt: 30`
- Expert sets: `20260728_abstracts_evaluation_template_mark_om_v1` (Mark),
  `20260728_abstracts_evaluation_template_charlie_s_v1` (Charlie) and
  `20260811_abstracts_evaluation_template_janusz_v1` (Janusz, **new in v3**)

Per `run_metadata.json`, the few-shot budget is the **only** settings difference between the two runs
(the model roster, `top_k: 48`, `ra_retrieval_mode: cosine`, `score_scale_version: 2` and both
adjudication agreement thresholds at 1.0 are identical). The runs were produced at different commits
(`109e57e` vs `96be6ef`), both dirty.

`nbconvert`/`nbclient` are not installed here, so the notebook's cells were reproduced verbatim in a
standalone script against the real `src/analysis/affinity_metrics.py`, using the project's
`~/.virtualenvs/classifier` interpreter. Notebook 11's `normalize_level` maps a score of exactly 80 to
**Moderate** (`40 <= score <= 80`), unlike notebook 10 — reproduced as-is. Every Mark and Charlie number
in §1-§6 and §8 reproduces `11_adjudication_readiness_analysis_v2.md` exactly; that document is left
untouched. Two deltas against v2, both flagged in place: §7's cross-validated cut-point refit is
recomputed and two of its cells move (fold-assignment sensitivity, quantified in §7), and case D's
accuracy is 75.3% (75.35%, shown as 75.4% in v2).

§9 is new: Janusz's set is built on a different annotation convention from Mark's and Charlie's, and that
has to be read before §1 or §1 will mislead.

## 0. Plain-language summary

- Three power-system experts labelled the same kind of abstract-question pairs, and they **disagree with
  each other more than the pipeline disagrees with any one of them.** Charlie is the strictest rater,
  Mark is in the middle, Janusz is the most generous; the pipeline sits between Mark and Janusz.
- So "the pipeline over-scores" was never a property of the pipeline. Against Mark and Charlie it scores
  high; against Janusz it scores *low*. It is a fixed rater being compared to a moving target.
- Raising the few-shot budget from 4 to 30 examples did essentially nothing, confirmed on all three
  experts now.
- The LLM judge is still a near no-op: it changes 6-9% of the labels it is asked to arbitrate, and
  whether that helps depends on which expert you score against.
- The one intervention that reliably pays is **re-fitting the Low/Moderate/High cut points**. v3 tests
  this 20 different ways instead of once, and it now improves ordinal agreement in every one of the six
  cases (v2's single "it hurt case A" result was a fold-assignment artifact).
- Janusz's sheet only contains the pairs he thought were relevant — the "not relevant" cells are blank
  rather than zero. That is why his raw numbers look so different, and §9 quantifies it.

## The Janusz set is not the same kind of sample

`generate_affinity_calibration_from_excel.py` exports a cell when the score is **> 0** *or* when the
question's programme matches the abstract's row-3 programme selection. The three workbooks were filled in
three different ways:

| Expert | Row 3 filled | Explicit `0`s typed in the matrix | Exported rows | Share of the 5040-cell grid |
|---|---|---|---|---|
| Mark | 105 / 105 | 4871 | 602 (167 positive + 435 subset zeros) | 11.9% |
| Charlie | 99 / 105 | 590 | 722 (132 positive + 590 subset zeros) | 14.3% |
| Janusz | **0 / 105** | **0** | **212 (all positive, 0 subset zeros)** | 4.1% |

Janusz left the programme-selection row blank and left irrelevant cells **blank rather than 0**, so his
export contains only pairs he actively rated ≥ 1. Consequences:

- His 212 rows cover 93 of 105 abstracts and only 34 of 48 questions, median 2 scored questions per
  abstract (mean 2.28, max 7).
- 6 of the 212 rows come from a **write-in question row he added** to the sheet —
  `other: optimal dispatch, congestion management` — which has no `RA2025_ID`, so those rows cannot be
  matched to the taxonomy and drop out of the merge. **All 6 are High** (three at 110, three at 90). That
  is an expert-flagged gap in the 48-question RA2025 set, and it is the only finding in this document that
  is about the taxonomy rather than the pipeline.
- The usable set is **206 pairs over 91 abstracts**, and its label mix is 22 Low / 109 Moderate / 75
  High — the mirror image of Mark's and Charlie's Low-dominated sets.

**Cases E and F below are therefore evaluated on a positives-enriched sample.** Accuracy and precision
are not comparable to A-D, and the mean signed error necessarily comes out negative. What *is* directly
comparable is QWK and the Moderate/High recalls, because those depend only on the rows Janusz explicitly
scored. §9 re-runs E and F under the assumption that blank means Low, which puts them back on A-D's
footing.

## 1. Headline metrics

| Metric | A: Mark × 134607_4 | B: Mark × 154151_4 | C: Charlie × 134607_4 | D: Charlie × 154151_4 | E: Janusz × 134607_4 | F: Janusz × 154151_4 |
|---|---|---|---|---|---|---|
| Few-shot examples | 4 | 30 | 4 | 30 | 4 | 30 |
| N pairs / abstracts | 602 / 91 | 602 / 91 | 722 / 99 | 722 / 99 | 206 / 91 | 206 / 91 |
| Accuracy | 65.3% | **66.8%** | **76.5%** | 75.3% | 45.1% | **46.1%** |
| Quadratic Weighted Kappa | 0.435 | **0.458** | **0.389** | 0.369 | 0.387 | **0.393** |
| Mean signed error | +0.158 | +0.186 | +0.155 | +0.177 | **−0.393** | **−0.384** |
| Mean absolute error | 0.367 | 0.359 | 0.252 | 0.269 | 0.597 | 0.587 |
| Expert label mix (Low/Mod/High) | 438 / 137 / 27 | 438 / 137 / 27 | 617 / 105 / **0** | 617 / 105 / **0** | 22 / 109 / **75** | 22 / 109 / **75** |
| Predicted label mix (Low/Mod/High) | 358 / 202 / 42 | 350 / 201 / 51 | 544 / 139 / 39 | 533 / 145 / 44 | 75 / 84 / 47 | 74 / 84 / 48 |
| Share of pairs sent to the LLM judge | 429 / 602 = 71.3% | 422 / 602 = 70.1% | 392 / 722 = 54.3% | 388 / 722 = 53.7% | 155 / 206 = 75.2% | 163 / 206 = 79.1% |

**QWK is the one metric that survives the change of sample: 0.369-0.458 across all six cases.** Accuracy
does not (45% to 77%) and neither does bias, and both are driven by the expert's label mix rather than by
the pipeline. Charlie's 85%-Low set makes a Low-leaning predictor look accurate; Janusz's Moderate/High
set makes the same predictor look inaccurate on the same pipeline output.

The 30-example run looks better against Mark (+1.5pp accuracy, +0.023 QWK) and Janusz (+1.0pp, +0.006)
and worse against Charlie (−1.1pp, −0.019). With three experts the vote is 2-1 rather than 1-1, but §8's
paired test shows all three intervals straddle zero, so this is still noise.

The over-scoring bias of **+0.16 to +0.19 of an ordinal step** against Mark and Charlie becomes **−0.38
to −0.39 against Janusz**. Both directions are real and both are artifacts of the target, not the
predictor: see §6. What *is* consistent is the direction of the few-shot effect — scores moved up against
all three experts (+0.158→+0.186, +0.155→+0.177, −0.393→−0.384).

The judge-touched share is far above the run-wide rate in every case (40.5% of all 5040 pairs in
`134607_4`, 40.1% in `154151_4`), and it is highest on Janusz's pairs (75-79%): the pairs an expert
thinks are worth scoring are exactly the pairs the model panel splits on. Adjudication cost per abstract
is higher on calibration abstracts than a full-corpus estimate would suggest. Raising the few-shot budget
did **not** reduce panel divergence — the adjudicated share is essentially unchanged (2039 vs. 2022
pairs).

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
| E: Janusz × 134607_4 | Low | 22 | 22.7% | 77.3% | 69.4% |
| E: Janusz × 134607_4 | Moderate | 109 | 53.6% | 41.3% | 50.0% |
| E: Janusz × 134607_4 | High | **75** | 66.0%\* | **41.3%** | 70.9% |
| F: Janusz × 154151_4 | Low | 22 | 21.6% | 72.7% | 68.9% |
| F: Janusz × 154151_4 | Moderate | 109 | 54.8% | 42.2% | 51.0% |
| F: Janusz × 154151_4 | High | **75** | 68.8%\* | **44.0%** | 72.3% |

\* Janusz's Low rows are almost entirely absent by construction, so Moderate and High precision in E/F
are inflated and Low precision deflated. Recall is unaffected: every row where Janusz said Moderate or
High is explicitly in his file. §9 gives the deflated-precision version (High precision falls to
34.8% / 36.7% once blanks are read as Low; recall is unchanged at 41.3% / 44.0%).

**Janusz's set gives the first usable measurement of the High class** (n=75, against Mark's n=27 and
Charlie's n=0): the pipeline recovers **41-44% of expert-High pairs**, and the 30-example run is 2.7pp
better on that recall. Both figures are assumption-free.

Moderate recall is the most stable quantity in the whole document — 41-56% in all six cases, and 41-42%
for both Charlie and Janusz despite their opposite label mixes. Moderate precision against Mark and
Charlie stays at 30-38%.

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

E: Janusz × 134607_4           F: Janusz × 154151_4
        Low  Mod  High                 Low  Mod  High
Low      17    5     0         Low      16    6     0
Mod      48   45    16         Mod      48   46    15
High     10   34    31         High     10   32    33
```

Janusz's matrix shows the failure mode the other two sets cannot: **10 of his 75 High pairs are predicted
Low** (a 2-step ordinal miss) and 48 of his 109 Moderate pairs are predicted Low. The pipeline's errors
against Janusz are under-calls, not over-calls.

## 3. Reliability curve (`agreement_mean` vs. accuracy)

| Case | Agreement 0.33 (n, acc.) | Agreement 0.50 (n, acc.) | Agreement 1.00 (n, acc.) |
|---|---|---|---|
| A: Mark × 134607_4 | 46, 36.1% | 383, 68.5% | 173, 85.4% |
| B: Mark × 154151_4 | 51, 46.2% | 372, 68.0% | 179, 89.8% |
| C: Charlie × 134607_4 | 41, 39.0% | 351, 73.5% | 330, 91.7% |
| D: Charlie × 154151_4 | 36, 47.2% | 352, 67.8% | 334, 94.2% |
| E: Janusz × 134607_4 | 19, 52.6% | 136, 53.1% | 51, 74.2% |
| F: Janusz × 154151_4 | 17, 52.9% | 146, 57.0% | 43, 70.5% |

Accuracy is monotone in panel agreement in all six cases, so `agreement_mean` remains a usable confidence
signal overall — but the curve is much flatter on Janusz's set (53% → 74% instead of 36% → 85%), because
a unanimous panel is usually unanimous on *Low* and Janusz has almost no Low pairs. **The signal only
works for the Low class, and it inverts for Moderate — in all six cases:**

| Case | Class | acc. @ 0.33 | acc. @ 0.50 | acc. @ 1.00 |
|---|---|---|---|---|
| A: Mark × 134607_4 | Low | 7.4% (n=27) | 66.5% (n=260) | 89.4% (n=151) |
| A: Mark × 134607_4 | Moderate | 73.7% (n=19) | 52.5% (n=99) | **26.3%** (n=19) |
| A: Mark × 134607_4 | High | — | 41.7% (n=24) | 66.7% (n=3) |
| B: Mark × 154151_4 | Low | 15.6% (n=32) | 64.3% (n=252) | 93.5% (n=154) |
| B: Mark × 154151_4 | Moderate | 88.2% (n=17) | 55.4% (n=101) | **31.6%** (n=19) |
| B: Mark × 154151_4 | High | 0.0% (n=2) | 52.6% (n=19) | 66.7% (n=6) |
| C: Charlie × 134607_4 | Low | 17.2% (n=29) | 75.9% (n=282) | 94.8% (n=306) |
| C: Charlie × 134607_4 | Moderate | 83.3% (n=12) | 37.7% (n=69) | **29.2%** (n=24) |
| D: Charlie × 154151_4 | Low | 29.6% (n=27) | 68.2% (n=280) | 97.1% (n=310) |
| D: Charlie × 154151_4 | Moderate | 88.9% (n=9) | 40.3% (n=72) | **29.2%** (n=24) |
| E: Janusz × 134607_4 | Low | — | 55.6% (n=9) | 92.3% (n=13) |
| E: Janusz × 134607_4 | Moderate | 88.9% (n=9) | 37.5% (n=80) | **35.0%** (n=20) |
| E: Janusz × 134607_4 | High | 10.0% (n=10) | 38.3% (n=47) | 66.7% (n=18) |
| F: Janusz × 154151_4 | Low | — | 66.7% (n=12) | 80.0% (n=10) |
| F: Janusz × 154151_4 | Moderate | 88.9% (n=9) | 38.1% (n=84) | **37.5%** (n=16) |
| F: Janusz × 154151_4 | High | 12.5% (n=8) | 46.0% (n=50) | 52.9% (n=17) |

The inversion is stable at 26-38% accuracy on unanimous-panel Moderate pairs in all six cases, and the
larger few-shot budget does not fix it. High is the one class where the curve is monotone on a decent
sample (Janusz: 10% → 38% → 67% and 13% → 46% → 53%), so unanimity is informative for High as well as
Low — the Moderate band is the one place where confident consensus predicts *failure*.

Practical reading unchanged from v2: high `agreement_mean` is not "safe to auto-accept" for anything the
panel did not call Low.

## 4. Adjudication (LLM judge) impact — split by `Final_Method`

| Case | Population | N | Pre-adjudication acc. | Post-adjudication acc. | Δ |
|---|---|---|---|---|---|
| A: Mark × 134607_4 | All pairs | 602 | 65.4% | 65.3% | −0.2pp |
| A: Mark × 134607_4 | Adjudicated only | 429 | 58.7% | 58.5% | −0.2pp |
| A: Mark × 134607_4 | Untouched only | 173 | 82.1% | 82.1% | 0.0 |
| B: Mark × 154151_4 | All pairs | 602 | 66.6% | 66.8% | +0.2pp |
| B: Mark × 154151_4 | Adjudicated only | 422 | 58.3% | 58.5% | +0.2pp |
| B: Mark × 154151_4 | Untouched only | 180 | 86.1% | 86.1% | 0.0 |
| C: Charlie × 134607_4 | All pairs | 722 | 75.9% | 76.5% | +0.6pp |
| C: Charlie × 134607_4 | Adjudicated only | 392 | 64.0% | 65.1% | +1.0pp |
| C: Charlie × 134607_4 | Untouched only | 330 | 90.0% | 90.0% | 0.0 |
| D: Charlie × 154151_4 | All pairs | 722 | 76.3% | 75.3% | −1.0pp |
| D: Charlie × 154151_4 | Adjudicated only | 388 | 62.6% | 60.8% | −1.8pp |
| D: Charlie × 154151_4 | Untouched only | 334 | 92.2% | 92.2% | 0.0 |
| E: Janusz × 134607_4 | All pairs | 206 | 46.6% | 45.1% | −1.5pp |
| E: Janusz × 134607_4 | Adjudicated only | 155 | 41.9% | 40.0% | −1.9pp |
| E: Janusz × 134607_4 | Untouched only | 51 | 60.8% | 60.8% | 0.0 |
| F: Janusz × 154151_4 | All pairs | 206 | 42.7% | 46.1% | +3.4pp |
| F: Janusz × 154151_4 | Adjudicated only | 163 | 39.9% | 44.2% | +4.3pp |
| F: Janusz × 154151_4 | Untouched only | 43 | 53.5% | 53.5% | 0.0 |

Accuracy still flips sign with the annotator and the few-shot budget (three of six negative), and
Janusz's set gives the largest swing in both directions (−1.5pp, +3.4pp) simply because it is the
smallest.

On QWK the third expert breaks the 2-2 tie in the judge's favour:

| Case | QWK pre-judge | QWK post-judge | Verdict |
|---|---|---|---|
| A: Mark × 134607_4 | 0.445 | **0.435** | hurts |
| B: Mark × 154151_4 | 0.439 | **0.458** | helps |
| C: Charlie × 134607_4 | 0.375 | **0.389** | helps |
| D: Charlie × 154151_4 | 0.381 | **0.369** | hurts |
| E: Janusz × 134607_4 | 0.351 | **0.387** | helps |
| F: Janusz × 154151_4 | 0.343 | **0.393** | helps |

Four of six now improve, and the two largest gains (+0.036, +0.050) are both on Janusz. That is
consistent with the judge doing its useful work at the *top* of the scale — the region Mark barely covers
and Charlie does not cover at all. It is one expert on 206 pairs, so it is a hypothesis to test, not a
result: it needs an expert set with real High coverage *and* real Low coverage before it can be believed.

Per-pair outcomes on the pairs the judge actually touched:

| Case | Improved | Worsened | Unchanged |
|---|---|---|---|
| A: Mark × 134607_4 | 16 (3.7%) | 17 (4.0%) | 396 (92.3%) |
| B: Mark × 154151_4 | 14 (3.3%) | 13 (3.1%) | 395 (93.6%) |
| C: Charlie × 134607_4 | 15 (3.8%) | 11 (2.8%) | 366 (93.4%) |
| D: Charlie × 154151_4 | 9 (2.3%) | 16 (4.1%) | 363 (93.6%) |
| E: Janusz × 134607_4 | 7 (4.5%) | 10 (6.5%) | 138 (89.0%) |
| F: Janusz × 154151_4 | 13 (8.0%) | 6 (3.7%) | 144 (88.3%) |

Measured run-wide (no expert labels needed):

| Run | Pairs sent to judge | Level changed | Raised | Lowered | Mean raw shift | Mean abs shift |
|---|---|---|---|---|---|---|
| `20260726_134607_4` | 2039 | 175 (8.6%) | 72 | 103 | −2.7 | 4.8 |
| `20260726_154151_4` | 2022 | 126 (6.2%) | 81 | 45 | −1.4 | 4.1 |

Every adjudicated pair has `agreement_mean <= 0.5` by construction, so the judge is being invoked
precisely where the panel is split, and it mostly ratifies the ensemble mean rather than overriding it.
With 30 few-shot examples the judge is *even more* of a no-op: it moves 6.2% of levels instead of 8.6%,
and its net direction flips from lowering (103 down / 72 up) to raising (81 up / 45 down).

**Read together with §6 and §8: the judge's effect (−1.8pp to +4.3pp, sign flipping by both annotator and
few-shot budget) is smaller than the disagreement between any two of the three human experts, so this
data still cannot establish whether adjudication helps.** It costs ~2000 LLM calls per run to move
126-175 labels. The QWK evidence now leans positive (4/6), which is a reason to keep the judge while a
better-covered expert set is assembled, not a reason to declare it validated.

## 5. Bootstrap confidence intervals (cluster-by-abstract, 500 resamples, seed 42)

| Case | QWK | Accuracy | Mean signed error | Recall Low | Recall Moderate | Recall High |
|---|---|---|---|---|---|---|
| A: Mark × 134607_4 | [0.355, 0.512] | [0.598, 0.705] | [0.096, 0.228] | [0.642, 0.771] | [0.447, 0.599] | [0.245, 0.680] |
| B: Mark × 154151_4 | [0.379, 0.533] | [0.619, 0.716] | [0.125, 0.255] | [0.651, 0.765] | [0.491, 0.638] | [0.321, 0.739] |
| C: Charlie × 134607_4 | [0.305, 0.467] | [0.716, 0.815] | [0.096, 0.218] | [0.776, 0.873] | [0.314, 0.507] | [0.0, 0.0]\* |
| D: Charlie × 154151_4 | [0.293, 0.452] | [0.704, 0.809] | [0.116, 0.240] | [0.762, 0.863] | [0.330, 0.506] | [0.0, 0.0]\* |
| E: Janusz × 134607_4 | [0.283, 0.481] | [0.379, 0.524] | [**−0.509, −0.283**] | [0.571, 0.933] | [0.312, 0.508] | [0.295, 0.520] |
| F: Janusz × 154151_4 | [0.287, 0.492] | [0.390, 0.536] | [**−0.487, −0.273**] | [0.523, 0.897] | [0.328, 0.517] | [0.331, 0.551] |

\* Charlie has zero true High pairs, so every resample also has zero support — the interval is a
degenerate point at 0, not a recall estimate.

**All six QWK intervals share a common region** — they span [0.283, 0.533] end to end and their mutual
intersection is [0.379, 0.452], so no pair of them is separated. The pipeline is not measurably better in
either configuration or against any particular expert; its ordinal agreement is a stable ~0.28-0.53
whoever holds the pen.

The mean-signed-error intervals are entirely positive for Mark and Charlie and entirely negative for
Janusz. Both exclude zero, which rules out sampling noise and leaves only one explanation: the two groups
of intervals are measuring the *annotator's* threshold, not the pipeline's.

Janusz's High-recall interval [0.295, 0.520] is the first non-degenerate High interval in this series
(Mark's is too wide at n=27, Charlie's does not exist). It is still 22pp wide, but it does place the
pipeline's High recall firmly below 55%.

## 6. Inter-expert ceiling

Each expert pair overlaps on a different subset, so the three comparisons are independent. Pipeline rows
are reported on each pair's own overlap.

| Comparison | Overlap (pairs / abstracts) | Exact agreement | QWK |
|---|---|---|---|
| Mark vs. Charlie | 212 / 39 | 71.7% | **0.346** |
| Mark vs. Janusz | 92 / 56 | 38.0% | **0.173** |
| Charlie vs. Janusz | 66 / 40 | 31.8% | **0.123** |

```
Mark \ Charlie   Low  Mod  High     Mark \ Janusz   Low  Mod  High
Low              133   12     0     Low               0   17     7
Moderate          37   19     0     Moderate          2   25    28
High               4    7     0     High              0    3    10

Charlie \ Janusz   Low  Mod  High
Low                  4   17    10
Moderate             0   17    18
High                 0    0     0
```

The three experts sit on a **severity ladder covering a full ordinal step**. Mean pairwise differences:
Mark − Charlie = **+0.189**, Janusz − Mark = **+0.587**, Janusz − Charlie = **+0.833**. On the 43 pairs
all three labelled (29 abstracts), mean level is Charlie **0.535** < Mark **0.907** < Janusz **1.512**,
the pipeline lands at **1.186 / 1.279** — third of four, between Mark and Janusz. All three agree exactly
on only **6 of 43** pairs, and **12 of 43** span the full Low-to-High range.

Pipeline agreement on each overlap:

| Overlap | Pipeline vs. | `134607_4` exact / QWK / bias | `154151_4` exact / QWK / bias |
|---|---|---|---|
| Mark ∩ Charlie (212) | Mark | 61.8% / **0.470** / +0.193 | 64.2% / **0.496** / +0.250 |
| Mark ∩ Charlie (212) | Charlie | 58.0% / 0.286 / +0.382 | 56.1% / 0.284 / +0.439 |
| Mark ∩ Janusz (92) | Mark | 45.7% / 0.239 / +0.228 | 48.9% / 0.214 / +0.272 |
| Mark ∩ Janusz (92) | Janusz | 51.1% / **0.319** / −0.359 | 50.0% / **0.345** / −0.315 |
| Charlie ∩ Janusz (66) | Charlie | 36.4% / 0.177 / +0.591 | 36.4% / 0.215 / +0.652 |
| Charlie ∩ Janusz (66) | Janusz | 56.1% / **0.408** / −0.242 | 51.5% / 0.380 / −0.182 |

**On every overlap, the pipeline agrees with at least one of the two humans better than the two humans
agree with each other**, and on the Mark ∩ Janusz and Charlie ∩ Janusz overlaps it beats the human-human
QWK against *both* experts (0.214-0.345 and 0.177-0.408 vs. ceilings of 0.173 and 0.123).

This is the central result of v3 and it reframes §1-§5 completely. The taxonomy's Moderate/High boundary
is not shared between these three experts.
A QWK of 0.37-0.46 against a single annotator is above, not below, the human-human ceiling on
this taxonomy, and the "bias" the pipeline is penalised for is the axis the humans themselves disagree
on. Reducing bias against one expert is fitting that expert's threshold, and it necessarily increases
bias against the other two.

Where the boundary actually breaks: 37 pairs Mark called Moderate were Low to Charlie and none of Mark's
11 High pairs in that overlap were High to Charlie; 28 pairs Mark called Moderate were High to Janusz; and
all 35 of Charlie's Moderate pairs in his Janusz overlap were Moderate (17) or High (18) to Janusz, with
10 of his Low pairs High to Janusz.

## 7. Cut-point cross-check

Notebook 10 found that refitting the Low/Moderate/High boundaries was the largest single gain on the
un-adjudicated ensemble. On the *adjudicated* output, with cut points fitted on train abstracts only
(grouped 5-fold by abstract, grid 10-120 step 5, selected by QWK, held-out predictions pooled):

| Case | QWK @ 40/80 | QWK refit (held out) | Bias | Accuracy | All-data cuts | Lower cut per fold |
|---|---|---|---|---|---|---|
| A: Mark × 134607_4 | 0.435 | 0.421 (**−0.014**) | +0.158 → +0.005 | 0.653 → 0.701 | 65/80 | 55-70 |
| B: Mark × 154151_4 | 0.458 | 0.488 (**+0.030**) | +0.186 → +0.018 | 0.668 → 0.738 | 55/80 | 55 (all folds) |
| C: Charlie × 134607_4 | 0.389 | 0.407 (**+0.018**) | +0.155 → +0.021 | 0.765 → 0.837 | 55/95 | 55-60 |
| D: Charlie × 154151_4 | 0.369 | 0.411 (**+0.042**) | +0.177 → +0.033 | 0.753 → 0.837 | 55/95 | 55-60 |
| E: Janusz × 134607_4 | 0.387 | 0.420 (**+0.033**) | −0.393 → −0.107 | 0.452 → 0.524 | 15/70 | 15-35 |
| F: Janusz × 154151_4 | 0.393 | 0.406 (**+0.012**) | −0.384 → −0.010 | 0.461 → 0.563 | 20/65 | 20-25 |

(v2 reported 0.495 for B and 0.422 for D from the same procedure. This recomputation uses a deterministic
`GroupKFold(5)`; the difference is entirely which abstracts land in which fold — the shuffled-split ranges
below are [0.451, 0.489] for B and [0.390, 0.431] for D, which brackets v2's D figure and comes within
0.006 of its B figure. A, C, E and F reproduce v2 exactly where v2 covered them. Grid tie-breaking is not
the cause: no cut pair in any fold of any case ties for best train QWK.)

Recalibration removes the bias in all six cases (+0.16/+0.19 → +0.01/+0.03 for Mark and Charlie, −0.39 →
−0.10/−0.01 for Janusz) and adds 4.8-10.2pp of exact accuracy. QWK improves in five of six, the only loss
being case A — the one case v1 and v2 also reported.

**That single loss does not survive testing.** Repeating the same procedure with 20 different shuffled
grouped 5-fold splits:

| Case | Base QWK | Refit QWK median | Refit QWK range | Δ median | Splits where refit wins | Acc. median | Bias median |
|---|---|---|---|---|---|---|---|
| A: Mark × 134607_4 | 0.435 | 0.452 | [0.422, 0.473] | **+0.017** | 16 / 20 | 0.729 | −0.026 |
| B: Mark × 154151_4 | 0.458 | 0.479 | [0.451, 0.489] | **+0.020** | 17 / 20 | 0.735 | +0.010 |
| C: Charlie × 134607_4 | 0.389 | 0.393 | [0.340, 0.410] | **+0.004** | 13 / 20 | 0.831 | +0.028 |
| D: Charlie × 154151_4 | 0.369 | 0.410 | [0.390, 0.431] | **+0.041** | 20 / 20 | 0.834 | +0.040 |
| E: Janusz × 134607_4 | 0.387 | 0.418 | [0.374, 0.460] | **+0.031** | 19 / 20 | 0.563 | −0.044 |
| F: Janusz × 154151_4 | 0.393 | 0.434 | [0.390, 0.467] | **+0.040** | 19 / 20 | 0.573 | −0.012 |

Every median delta is positive and refitting wins in 13-20 of 20 splits in every case, including A. The
"recalibration hurts case A" finding in v1 and v2 was an artifact of a single fold assignment on 91
abstracts. **Cut-point refitting improves QWK, removes bias, and adds accuracy in all six cases.**

The fitted cuts differ sharply between experts, and now by more than v2 could show: **65/80 or 55/80 for
Mark, 55/95 for Charlie, 15/70 or 20/65 for Janusz**. Janusz's lower cut is a quarter to a third of
Charlie's. Refit cut points if exact accuracy and unbiasedness are the goal, but do not hard-code any one
expert's cuts — the cut points *are* the disagreement documented in §6.

(The "QWK @ 40/80" column uses notebook 11's `normalize_level`, where a score of exactly 80 is Moderate.
Using notebook 10's strict `score < 80` rule instead shifts this column by at most 0.004 — A 0.431,
B 0.460, C 0.389, D 0.366, E 0.387, F 0.393 — which does not change any conclusion above. Only 0-3 pairs
per case sit exactly on the boundary; E and F have none.)

## 8. Run vs run — is the 30-example few-shot budget actually better?

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
| Janusz (206 pairs, 91 abstracts) | +0.006 [−0.035, +0.049] | +0.010 [−0.041, +0.061] | +0.010 [−0.043, +0.066] |

**All six QWK and accuracy intervals straddle zero.** Raising the few-shot budget from 4 to 30 examples
produced no measurable accuracy or agreement gain against any of the three experts.

The Δ Bias point estimate is positive for all three (+0.028, +0.022, +0.010) — the 30-example run scores
higher, full stop — and excludes zero for Mark and Charlie. Janusz's interval includes zero, which at 206
pairs is a power limit rather than a contradiction: the sign and rough magnitude agree with the other two.
So the only measurable effect of the extra 26 examples is a **worse over-scoring bias against the two
experts whose sets can detect it**.

Caveat on `154151_4`: 241 of its 5040 pairs were scored by only 4 models and 2 by 3 models (missing LLM
scores: `mistral-small-2603` 200, `gemini-3.1-flash-lite` 36, `claude-haiku-4.5` 5, `gpt-5.4-mini` 4),
against 27 four-model pairs in `134607_4` (`gemini-3.1-flash-lite` 13, `claude-haiku-4.5` 6,
`mistral-small-2603` 6, `gpt-5.4-mini` 2). That is ~4.8% of pairs on a thinner panel, which affects both
`Affinity_Mean` and `agreement_mean` for those pairs. It is too small to explain the §8 result but should
be fixed before the run is used as a baseline.

## 9. Sensitivity: reading Janusz's blank cells as Low (new)

Cases E and F evaluate 206 pairs Janusz chose to score. This section re-runs them on the full 48-question
grid for the **91 abstracts where he scored at least one RA2025 question**, treating every blank cell as
raw score 0 → 10 → Low, which is what Mark's 4871 explicit zeros mean in his workbook. The 14 abstracts
left out are the 12 he never marked at all plus the 2 whose only mark was the write-in row; for those,
blank and unreviewed are indistinguishable. **This is an assumption about intent, not data in the file**
— a blank could equally mean "not reviewed". It is reported because it is the only way to compare
Janusz's set with Mark's and Charlie's on the same footing, and because the two readings bracket the
truth.

Grid: 4368 pairs over 91 abstracts (91 × 48), mix 4184 Low / 109 Moderate / 75 High.

| Metric | E′: Janusz(blanks=Low) × 134607_4 | F′: Janusz(blanks=Low) × 154151_4 |
|---|---|---|
| N pairs / abstracts | 4368 / 91 | 4368 / 91 |
| Accuracy | 84.5% | **85.1%** |
| Quadratic Weighted Kappa | 0.348 | **0.360** |
| Mean signed error | **+0.121** | **+0.115** |
| Mean absolute error | 0.167 | 0.161 |
| Predicted label mix (Low/Mod/High) | 3671 / 608 / 89 | 3696 / 582 / 90 |
| Share sent to the LLM judge | 1827 / 4368 = 41.8% | 1842 / 4368 = 42.2% |
| Low precision / recall | 98.4% / 86.4% | 98.4% / 87.0% |
| Moderate precision / recall | **7.4%** / 41.3% | **7.9%** / 42.2% |
| High precision / recall | 34.8% / 41.3% | 36.7% / 44.0% |
| QWK pre-judge → post-judge | 0.323 → **0.348** | 0.341 → **0.360** |

```
E′: Janusz(blanks=Low) × 134607_4      F′: Janusz(blanks=Low) × 154151_4
         Low   Mod  High                        Low   Mod  High
Low     3613   529    42                Low    3638   504    42
Mod       48    45    16                Mod      48    46    15
High      10    34    31                High     10    32    33
```

Under this reading Janusz stops being an outlier and joins the pattern:

- **Bias flips to +0.12**, the same direction as Mark (+0.16) and Charlie (+0.16), just smaller. The
  −0.39 in §1 was entirely a sampling artifact of the positives-only export.
- **Accuracy jumps to 84.5-85.1%** — above Charlie's — purely from the 96% Low base rate, which is the
  clearest demonstration in this document that accuracy is uninformative on this task.
- **QWK barely moves: 0.348-0.360 against 0.387-0.393 in E/F.** The pipeline's ordinal agreement with
  Janusz is ~0.35-0.39 under either reading, and it sits inside the same 0.28-0.53 band as every other
  case in §5. QWK is the metric to report.
- **Moderate precision collapses to 7-8%**: the pipeline predicts 582-608 Moderate pairs where Janusz
  marked 109. Combined with A-D's 30-38%, Moderate over-prediction is the pipeline's single largest
  systematic error, and it is worse the more sparsely the expert uses the middle band.
- The judge helps QWK in both runs here too (+0.025, +0.019), consistent with E/F.
- Moderate and High **recall are identical to E/F by construction** (41.3%/42.2% and 41.3%/44.0%) — those
  are the assumption-free numbers.

The recommendation this points to is operational, not analytical: **re-issue Janusz's workbook asking for
explicit 0s (or fill row 3), and re-export.** Until then, quote his QWK and his Moderate/High recalls, and
quote nothing else without stating which reading is in use.

## Summary

1. **The pipeline agrees with each expert better than the experts agree with each other.** Human-human
   QWK is 0.346 (Mark-Charlie), 0.173 (Mark-Janusz), 0.123 (Charlie-Janusz); the pipeline scores
   0.369-0.458 against the three full sets, clears each overlap's ceiling against at least one of its two
   experts, and clears it against *both* on the two Janusz overlaps. The taxonomy's Moderate/High boundary
   is the limiting factor, not the model.
2. **"Over-scoring bias" is a property of the annotator, not the pipeline.** The same output is +0.16
   against Mark, +0.16 against Charlie and −0.38 against Janusz; the three experts span a full ordinal
   step of severity (Charlie 0.54 < Mark 0.91 < Janusz 1.51 on their 43 shared pairs) and the pipeline
   lands third of four. Bias against one expert cannot be reduced without raising it against the others.
3. **The 4→30 few-shot increase is not an improvement**, now confirmed on three experts. The runs agree
   on 95% of pairs, and all six paired-bootstrap ΔQWK and Δaccuracy intervals straddle zero. The only
   measurable effect is *more* over-scoring bias (+0.02-0.03 ordinal steps for Mark and Charlie, same
   sign for Janusz).
4. **The judge is still close to a no-op** (8.6% and 6.2% of adjudicated levels changed, ~2000 LLM calls
   per run to move 126-175 labels), but the QWK verdict now leans positive at 4 of 6, with the two
   biggest gains on Janusz's High-rich set. Keep it; do not claim it is validated.
5. **Cut-point refitting is the one intervention that reliably pays** — and v3 upgrades this from "three
   of four" to **all six cases**. Across 20 shuffled grouped-CV splits every case has a positive median
   ΔQWK and refitting wins in 13-20 of 20 splits, while removing the bias and adding 5-10pp of exact
   accuracy (+4.8pp in the weakest case). v2's single "case A loses QWK" result was a fold-assignment
   artifact. Do not hard-code cuts: the fitted lower cut ranges from 15 (Janusz) to 65 (Mark).
6. **`agreement_mean` is a confidence signal for Low and High, never for Moderate** — unanimous panels on
   expert-Moderate pairs are wrong 62-74% of the time in all six cases. Janusz's set adds the missing
   piece: for High the curve *is* monotone (10% → 38% → 67%).
7. **The High class is finally measurable, and the answer is 41-44% recall** (Janusz, n=75, assumption
   free; precision 35-37% under the blanks-as-Low reading). Charlie's set cannot evaluate High at all
   (zero labels) and Mark's barely can (n=27).
8. **Moderate over-prediction is the pipeline's largest systematic error**: precision 30-38% against Mark
   and Charlie, and 7-8% against Janusz's full grid, where 582-608 predicted Moderates face 109 expert
   ones.
9. **Fix the annotation templates before the next round.** Janusz's blank-vs-zero convention left 95% of
   his grid unusable and forced §9; his 6 write-in `other: optimal dispatch, congestion management` rows are an
   expert-flagged gap in the 48-question RA2025 set. Neither is a modelling problem, and both are
   cheaper to fix than anything else in this document.
