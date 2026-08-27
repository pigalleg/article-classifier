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

Other options are `--objective weighted_mae`, `--low-multiplier`, `--moderate-multiplier`, `--high-multiplier`, `--folds 5`, `--fit-all`, and `--output-dir <path>`.

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

## Objective

For LLM score $s_{im}$, expert score $y_i$, and model weight $w_m$:

$$
\hat{s}_i = \sum_m w_m s_{im}
$$

$$
\min \sum_i c_i |\hat{s}_i-y_i|
$$

where $c_i$ is selected by the expert affinity level: `--low-multiplier` for Low ($y_i \leq 40$), `--moderate-multiplier` for Moderate ($40 < y_i \leq 80$), and `--high-multiplier` for High ($y_i > 80$). The absolute error is linearized in JuMP, making this a linear program. Weights satisfy $w_m \geq 0$ and $\sum_m w_m=1$.

## Outputs

The output directory contains `fold_weights.csv`, `out_of_fold_predictions.csv`, `metrics_summary.csv`, and `run_metadata.toml`. With `--fit-all`, it also contains `final_weights.csv`.

`metrics_summary.csv` reports weighted MAE, MAE, mean signed error, three-level accuracy, quadratic weighted kappa, and per-class support, precision, recall, one-vs-rest accuracy, and confusion-count components. Weights are fit separately in each grouped cross-validation fold, so every out-of-fold prediction comes from weights trained without labels from that abstract.
