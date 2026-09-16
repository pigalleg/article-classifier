# LLM Ensemble Optimizer

A diagnostic-only Julia optimizer for learning non-negative LLM ensemble weights that sum to one. Two solvers are available: an exact Gurobi linear/quadratic program, and a projected gradient descent solver that also supports objectives no linear program can express.

## Requirements

Use a current stable Julia release. The exact solver needs the Gurobi 12 installation at `/opt/gurobi1201`; `--solver gradient` runs in pure Julia and needs no licence. The project launcher sets `GUROBI_HOME=/opt/gurobi1201/linux64` and `GRB_LICENSE_FILE=/opt/gurobi1201/gurobi.lic`. Instantiate the isolated environment:

```bash
julia --project=optimization -e 'using Pkg; Pkg.instantiate()'
```

## Run

The defaults use the Mark calibration set and its corresponding benchmark run:

```bash
bash optimization/run_optimizer.sh
```

Use a different expert/benchmark run pair without editing code:

```bash
bash optimization/run_optimizer.sh \
  --expert-run 20260728_abstracts_evaluation_template_mark_om_v1 \
  --benchmark-run 20260726_134607_5
```

## Options

| Option | Values / default | Purpose |
|---|---|---|
| `--expert-run` | calibration run ID; default `20260618_Mark_batch1_v4` | Uses `data/processed/affinity_calibration/<run ID>/calibration_abstract_question_affinities.csv` as expert labels. |
| `--benchmark-run` | benchmark run ID; default `20260724_155639` | Uses `data/results/affinity_benchmark/<run ID>/merged_ra_affinities.csv` as model scores. |
| `--objective` | `weighted_mae` (default), `weighted_mse`, `soft_qwk` | Selects absolute score error, squared score error, or the smoothed quadratic weighted kappa. `soft_qwk` requires `--solver gradient`. |
| `--solver` | `exact` (default), `gradient` | Solves with Gurobi, or with projected gradient descent on the weight simplex. |
| `--learning-rate` | positive; default `0.05` | Largest weight change any single model may take in one gradient iteration, before backtracking. |
| `--max-iterations` | positive integer; default `5000` | Gradient iteration budget per starting point. |
| `--patience` | positive integer; default `200` | Consecutive iterations with relative improvement below `--tolerance` before the descent stops. |
| `--tolerance` | positive; default `1e-10` | Relative improvement that still counts as progress. |
| `--level-temperature` | positive; default `5.0` | Affinity-score width over which the `soft_qwk` band membership transitions from one level to the next. |
| `--restarts` | positive integer; default `1` | Number of starting points: uniform weights, then the single-model vertices ranked by their own objective. |
| `--weight-scope` | `global` (default), `programme` | Learns one model vector globally or one vector per RA programme. |
| `--calibration-scope` | `none` (default), `programme` | Applies no score calibration or an affine post-ensemble calibration per programme. |
| `--low-multiplier` | non-negative; default `1.0` | Relative training importance of expert Low examples. |
| `--moderate-multiplier` | non-negative; default `2.0` | Relative training importance of expert Moderate examples. |
| `--high-multiplier` | non-negative; default `2.0` | Relative training importance of expert High examples. |
| `--folds` | integer at least `2`; default `5` | Number of cross-validation folds, grouped by `Abstract_Index` and stratified by `RA2025_ID`. |
| `--fit-all` | flag; off by default | Fits final deployable weights using all retained labelled pairs. |
| `--output-dir` | result directory | Destination for reports, predictions, metadata, and optional deployable parameters. |

At least one level multiplier must be positive. A multiplier of `0` excludes that expert level from the fitting objective but does not remove it from evaluation.

Set separate target-level multipliers to prioritise affinity bands. The default `1 / 2 / 2` gives Low examples a weight of 1 and Moderate/High examples a weight of 2.

```bash
bash optimization/run_optimizer.sh \
  --expert-run 20260728_abstracts_evaluation_template_mark_om_v1 \
  --benchmark-run 20260726_134607_5 \
  --low-multiplier 1.5 \
  --moderate-multiplier 1 \
  --high-multiplier 4 \
  --folds 5 \
  --fit-all \
  --output-dir data/results/ensemble_weight_optimizer/my_run
```

`--fit-all` additionally trains a final deployable vector on all labelled pairs. Use cross-validation metrics to estimate generalisation; the all-data fit is for deployment.

`--weight-scope programme` learns a separate model-weight vector for each canonical RA programme (`Grouping` in `data/processed/ra_grouping_ra2025.csv`). It remains an unregularized fit, so use its grouped cross-validation results rather than in-sample performance to judge whether the added flexibility generalizes.

`--calibration-scope programme` fits an affine calibration $a_g + b_g \hat{s}$ after each fold's ensemble weights, separately for each programme $g$, with $b_g \geq 0$. Calibration uses only the corresponding outer-fold training rows and is then applied to held-out predictions. `Weighted_Affinity` contains the calibrated score; `Raw_Weighted_Affinity` preserves the score before calibration. With `--fit-all`, the all-data calibration coefficients are written to `final_calibration.csv`; fold-specific coefficients are retained in `fold_calibrations.csv`.

## Programme Weights And Two-Stage Calibration

Global weighting uses one convex combination for every abstract-question pair:

$$
\hat{s}_i = \sum_m w_m s_{im}
$$

Programme weighting instead selects the combination from the RA programme $g_i$ of pair $i$:

$$
\hat{s}_i = \sum_m w_{g_i,m}s_{im}
$$

For each programme, weights are non-negative and sum to one:

$$
w_{g,m} \geq 0, \qquad \sum_m w_{g,m}=1
$$

This lets the ensemble prefer models that are comparatively reliable for a programme, while fitting only one vector per programme rather than one per RA question. The six programme vectors introduce $6M$ parameters for $M$ models, compared with $M$ global parameters or $48M$ question-specific parameters.

Cross-validation assigns every pair belonging to one `Abstract_Index` to the same fold. It greedily balances the count of each `RA2025_ID` across folds, then uses total pair count as a secondary balancing criterion. This preserves the no-leakage abstract grouping while making each fold more representative of the question mix.

The optional two-stage approach first fits these ensemble weights, then corrects remaining programme-specific score bias:

$$
	ilde{s}_i = a_{g_i} + b_{g_i}\hat{s}_i, \qquad b_{g_i} \geq 0
$$

The non-negative slope preserves within-programme ordering. The intercept can correct systematic under- or over-scoring. Both stages are fit inside each outer `Abstract_Index`-grouped training fold; the held-out fold is scored only with that fold's fitted weights and calibration coefficients. This keeps the reported out-of-fold metrics free of label leakage.

Calibration optimizes continuous score fit, not the discrete Low/Moderate/High decision directly. It can reduce MAE while compressing high scores below the fixed High threshold of 80, harming High recall and QWK. Judge this option using the reported per-level precision and recall, QWK, and score errors together; do not select it on accuracy or MAE alone.

## Examples

Run the default global-weight experiment:

```bash
bash optimization/run_optimizer.sh
```

Learn one unregularized model-weight vector for each RA programme:

```bash
bash optimization/run_optimizer.sh \
  --expert-run 20260728_abstracts_evaluation_template_mark_om_v1 \
  --benchmark-run 20260726_134607_5 \
  --weight-scope programme \
  --low-multiplier 1 \
  --moderate-multiplier 1 \
  --high-multiplier 1 \
  --folds 5 \
  --fit-all \
  --output-dir data/results/ensemble_weight_optimizer/programme_equal
```

Add fold-local programme calibration to a programme-weighted model:

```bash
bash optimization/run_optimizer.sh \
  --expert-run 20260728_abstracts_evaluation_template_mark_om_v1 \
  --benchmark-run 20260726_134607_5 \
  --weight-scope programme \
  --calibration-scope programme \
  --low-multiplier 1 \
  --moderate-multiplier 1 \
  --high-multiplier 1 \
  --folds 5 \
  --fit-all \
  --output-dir data/results/ensemble_weight_optimizer/programme_calibrated
```

## K-Fold Sweep

Run an independent programme-by-model optimization for every integer $k$ in an inclusive range, then compare their pooled out-of-fold metrics and retain the deployment weights from the best observed evaluation run:

```bash
~/.virtualenvs/classifier/bin/python scripts/run_k_fold_optimizer_sweep.py \
  --expert-run 20260728_abstracts_evaluation_template_mark_om_v1 \
  --benchmark-run 20260726_134607_6 \
  --weight-scope programme \
  --low-multiplier 1 \
  --moderate-multiplier 1 \
  --high-multiplier 1 \
  --k-min 3 \
  --k-max 10
```

The sweep accepts `--solver` and, when it is `gradient`, forwards `--learning-rate`, `--max-iterations`, `--patience`, `--tolerance`, `--level-temperature` and `--restarts` to every child; those settings become part of the resume compatibility check. The sweep always supplies `--fit-all` to its children. Its output directory is named `<timestamp>_kfold_sweep_<benchmark>_k<min>-<max>/` and contains `k3/`, `k4/`, and so on. Each child retains the normal optimizer report, metadata, out-of-fold predictions, and `final_weights.csv`.

At the sweep level, `k_fold_comparison.xlsx` contains `Overall Metrics by K`, `Per-Level Metrics by K`, `Fold Weight Stability by K`, `Run Metadata`, and `Selection`. The selected run has the highest optimized pooled OOF `Quadratic_Weighted_Kappa`; exact ties select the lower $k$. `selected_programme_model_weights.csv` is copied from that run's `final_weights.csv`; programme calibration, when selected, is copied to `selected_programme_calibration.csv`. `selection.toml` records the metric, winning $k$, tie-breaker, and source child directory.

To resume an interrupted sweep, pass its exact directory and `--resume`:

```bash
~/.virtualenvs/classifier/bin/python scripts/run_k_fold_optimizer_sweep.py \
  --expert-run 20260728_abstracts_evaluation_template_mark_om_v1 \
  --benchmark-run 20260726_134607_6 \
  --weight-scope programme \
  --low-multiplier 1 \
  --moderate-multiplier 1 \
  --high-multiplier 1 \
  --k-min 3 \
  --k-max 10 \
  --output-dir data/results/ensemble_weight_optimizer/<sweep-directory> \
  --resume
```

Resume skips only child directories with a complete report, `final_weights.csv`, and matching metadata; partial or incompatible children fail clearly. The comparison workbook can also be rebuilt from finished child reports without fitting or recalculating metrics:

```bash
~/.virtualenvs/classifier/bin/python scripts/util/aggregate_k_fold_optimizer_sweep.py \
  --sweep-dir data/results/ensemble_weight_optimizer/<sweep-directory>
```

$k$ changes the grouped, stratified out-of-fold split used to estimate QWK. Since the selected `final_weights.csv` is refit on all labelled pairs with the same objective and configuration, it should be the same for every $k$ apart from numerically equivalent optimizer solutions. Use the sweep to judge the stability of the OOF estimate; do not treat its winning $k$ as an independent model hyperparameter.

## Objective

For LLM score $s_{im}$, expert score $y_i$, and model weight $w_m$:

$$
\hat{s}_i = \sum_m w_m s_{im}
$$

$$
\min \sum_i c_i |\hat{s}_i-y_i|
$$

where $c_i$ is selected by the expert affinity level: `--low-multiplier` for Low ($y_i \leq 40$), `--moderate-multiplier` for Moderate ($40 < y_i \leq 80$), and `--high-multiplier` for High ($y_i > 80$). The absolute error is linearized in JuMP, making this a linear program. Weights satisfy $w_m \geq 0$ and $\sum_m w_m=1$.

Use `--objective weighted_mse` for the convex quadratic alternative:

$$
\min \sum_i c_i (\hat{s}_i-y_i)^2
$$

It penalizes large score errors more strongly while retaining the same fixed 40 and 80 class boundaries for evaluation.

Level multipliers must be non-negative, with at least one positive multiplier. Set two multipliers to `0` to optimize exclusively for the remaining expert level.

## Gradient Descent Solver

`--solver gradient` replaces the Gurobi call with projected gradient descent, run identically inside each cross-validation fold and for `--fit-all`. Each iteration computes the analytic gradient $\nabla_w L$, rescales it so its largest entry moves a weight by `--learning-rate`, takes a step, and projects the result back onto the probability simplex $\{w \geq 0, \sum_m w_m = 1\}$ by the exact Euclidean projection. A backtracking line search halves the step until the objective actually decreases, which makes every accepted iteration monotone and removes step-size tuning as a practical concern. Programme weighting projects each programme's row separately, so every programme keeps its own convex combination. Weights below `1e-4` are snapped to zero and the row renormalized whenever that does not increase the objective, so the reported weights stay as sparse and readable as the linear program's.

The descent is deterministic: it starts from uniform weights, and `--restarts` adds the single-model vertices in order of their own objective value, keeping the best result. There is no random initialization and no seed to record.

On `weighted_mae` and `weighted_mse` the two solvers optimize the same convex problem, so the gradient solver is a check on the exact one rather than an alternative to it. On the Mark calibration set against benchmark `20260726_134607_5` with programme weights, it reproduced the Gurobi optimum to within 0.03 percent of the objective (10647.85 against 10644.88) and returned the same weights. Prefer `--solver exact` for these two objectives; use the gradient solver when Gurobi is unavailable, or to fit an objective Gurobi cannot express.

### Soft Quadratic Weighted Kappa

`--objective soft_qwk` optimizes the metric the k-fold sweep selects on, instead of a score-error stand-in for it. Hard level assignment is a step function of the ensemble score and has zero gradient almost everywhere, so band membership is smoothed with logistic gates at the fixed 40 and 80 cuts, using the width $\tau$ from `--level-temperature`:

$$
p_{i,\text{High}} = \sigma\!\left(\frac{\hat{s}_i - 80}{\tau}\right), \qquad
p_{i,\text{Moderate}} = \sigma\!\left(\frac{\hat{s}_i - 40}{\tau}\right) - p_{i,\text{High}}, \qquad
p_{i,\text{Low}} = 1 - \sigma\!\left(\frac{\hat{s}_i - 40}{\tau}\right)
$$

These soft memberships replace the predicted column of the confusion matrix, each pair contributing its level multiplier $c_i$ rather than a count. With quadratic disagreement weights $W_{kl} = (k-l)^2/(K-1)^2$, expert level $t_i$, expert totals $n_k = \sum_{i: t_i = k} c_i$, predicted totals $m_l = \sum_i c_i p_{i,l}$ and $N = \sum_i c_i$, the solver minimizes

$$
1 - \kappa_{\text{soft}} = \frac{\sum_i c_i \sum_l W_{t_i,l}\, p_{i,l}}{\frac{1}{N}\sum_{k,l} W_{kl}\, n_k m_l}
$$

which is exactly $1 - \kappa$ once $\tau \to 0$. `Objective_Value` holds this quantity, so lower is better and `0` is perfect agreement; it is not comparable with the score-error objective values. Smaller $\tau$ tracks the hard metric more closely but flattens the gradient away from the two cuts; the `5.0` default transitions over roughly $\pm 15$ affinity points.

This objective fits the metric harder, not necessarily better. On the Mark calibration set with programme weights and equal level multipliers it raised in-sample quadratic weighted kappa from 0.468 to 0.522 at $\tau = 2.5$, while pooled out-of-fold kappa fell from 0.472 to 0.432; every temperature between 1 and 20 landed below the `weighted_mae` fit. With 602 labelled pairs, 27 of them High, and $6 \times 8$ free weights, directly fitting a discrete agreement metric overfits where score-error fitting does not. It did lift out-of-fold High recall (0.259 to 0.333 at $\tau = 5$), so it is worth trying when High recall is the priority, but judge it on out-of-fold metrics only.

```bash
bash optimization/run_optimizer.sh \
  --expert-run 20260728_abstracts_evaluation_template_mark_om_v1 \
  --benchmark-run 20260726_134607_5 \
  --weight-scope programme \
  --objective soft_qwk \
  --solver gradient \
  --level-temperature 5 \
  --restarts 3 \
  --low-multiplier 1 \
  --moderate-multiplier 1 \
  --high-multiplier 1 \
  --folds 5 \
  --fit-all \
  --output-dir data/results/ensemble_weight_optimizer/programme_soft_qwk
```

`fold_weights.csv` and the report's weight-stability table record `Solver_Status` as `GRADIENT_CONVERGED` or `GRADIENT_ITERATION_LIMIT`; an iteration-limit status means `--max-iterations` was exhausted before the line search stalled.

## Outputs

The output directory contains `optimization_report.xlsx`, `out_of_fold_predictions.csv`, and `run_metadata.toml`. With `--fit-all`, it also contains `final_weights.csv`.

`optimization_report.xlsx` contains pooled out-of-fold overall and per-level Optimized-versus-Equal-Weight comparisons, plus `Overall Metrics by Fold` and `Per-Level Metrics by Fold` sheets. Each fold sheet includes one row per held-out fold and an `Average` row containing the unweighted mean across folds. Julia calculates both metric tables from the out-of-fold predictions; the Python post-processor only formats them into the workbook. The report also contains the minimum, maximum, and mean weight assigned to every model across the grouped cross-validation folds. Weights are fit separately in each fold, so every out-of-fold prediction comes from weights trained without labels from that abstract. The detailed out-of-fold predictions remain available as CSV for audit and follow-up analysis; the intermediate `metrics_summary.csv`, `fold_metrics_summary.csv`, and `fold_weights.csv` files are removed after the workbook is written.
