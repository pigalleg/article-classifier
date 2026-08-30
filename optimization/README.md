# LLM Ensemble Optimizer

A diagnostic-only Julia linear program for learning non-negative LLM ensemble weights that sum to one.

## Requirements

Use a current stable Julia release and the Gurobi 12 installation at `/opt/gurobi1201`. The project launcher sets `GUROBI_HOME=/opt/gurobi1201/linux64` and `GRB_LICENSE_FILE=/opt/gurobi1201/gurobi.lic`. Instantiate the isolated environment:

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
| `--objective` | `weighted_mae` (default), `weighted_mse` | Selects absolute or squared score error for fitting ensemble weights. |
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

The sweep always supplies `--fit-all` to its children. Its output directory is named `<timestamp>_kfold_sweep_<benchmark>_k<min>-<max>/` and contains `k3/`, `k4/`, and so on. Each child retains the normal optimizer report, metadata, out-of-fold predictions, and `final_weights.csv`.

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

## Outputs

The output directory contains `optimization_report.xlsx`, `out_of_fold_predictions.csv`, and `run_metadata.toml`. With `--fit-all`, it also contains `final_weights.csv`.

`optimization_report.xlsx` contains pooled out-of-fold overall and per-level Optimized-versus-Equal-Weight comparisons, plus `Overall Metrics by Fold` and `Per-Level Metrics by Fold` sheets. Each fold sheet includes one row per held-out fold and an `Average` row containing the unweighted mean across folds. Julia calculates both metric tables from the out-of-fold predictions; the Python post-processor only formats them into the workbook. The report also contains the minimum, maximum, and mean weight assigned to every model across the grouped cross-validation folds. Weights are fit separately in each fold, so every out-of-fold prediction comes from weights trained without labels from that abstract. The detailed out-of-fold predictions remain available as CSV for audit and follow-up analysis; the intermediate `metrics_summary.csv`, `fold_metrics_summary.csv`, and `fold_weights.csv` files are removed after the workbook is written.
