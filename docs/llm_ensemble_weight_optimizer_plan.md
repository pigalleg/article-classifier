# Modular LLM Ensemble Weight Optimizer

## Purpose

Build a standalone Julia/JuMP/Gurobi experiment under `optimization/` that learns interpretable weights for the benchmarked LLMs. The first version is diagnostic-only and does not change the existing Python scoring or adjudication pipeline.

## Scope

- The latest stable Julia release, JuMP, Gurobi, CSV, and DataFrames.
- Modular objective interface; initial objective: `weighted_mae`.
- Raw expert target: `Evaluator_Affinity`.
- Expert scores above 40 receive a configurable weight, initially `2.0`.
- Deterministic five-fold cross-validation grouped by `Abstract_Index`.
- CLI-configured input runs and CSV/TOML result artifacts.

Excluded: false-positive/false-negative count objectives, integer variables, learned thresholds, Python-Julia bridging, and changes to the adjudication pipeline.

## Inputs

Expert reference:

`data/processed/affinity_calibration/<expert-run>/calibration_abstract_question_affinities.csv`

Required columns: `Abstract_Index`, `RA2025_ID`, `Evaluator_Affinity`.

Per-model scores:

`data/results/affinity_benchmark/<benchmark-run>/merged_ra_affinities.csv`

Required columns: `Abstract_Index`, `RA2025_ID`, `Model_Slug`, `LLM_Affinity`.

Pivot the long benchmark table to one row per `(Abstract_Index, RA2025_ID)` pair and one column per LLM. Then inner-join with the expert table on:

$$
(\texttt{Abstract\_Index}, \texttt{RA2025\_ID})
$$

Drop and report rows missing the expert score or a selected model score.

Initial defaults:

```text
--expert-run 20260618_Mark_batch1_v4
--benchmark-run 20260724_155639
```

Later inputs use the same CLI options:

```text
--expert-run 20260728_abstracts_evaluation_template_mark_om_v1
--benchmark-run 20260726_134607_5
```

`adjudicated_ra_affinities.csv` is not optimizer input because it has a final aggregate score rather than one score per LLM.

## Formulation

For calibration pair $i \in \{1,\ldots,N\}$ and model $m \in \{1,\ldots,M\}$:

| Symbol | Meaning |
|---|---|
| $s_{im}$ | Raw LLM $m$ score for pair $i$. |
| $y_i$ | Expert `Evaluator_Affinity` score. |
| $w_m$ | LLM weight decision variable. |
| $\hat{s}_i$ | Weighted ensemble score. |
| $e_i$ | Absolute error decision variable. |
| $c_i$ | Fixed sample-importance multiplier. |

Ensemble score:

$$
\hat{s}_i=\sum_{m=1}^{M}w_m s_{im}
$$

Convex-weight constraints:

$$
w_m\geq0 \qquad \forall m
$$

$$
\sum_{m=1}^{M}w_m=1
$$

Linear absolute-error constraints:

$$
e_i\geq\hat{s}_i-y_i
$$

$$
e_i\geq y_i-\hat{s}_i
$$

$$
e_i\geq0
$$

Set sample importance as:

$$
c_i=
\begin{cases}
2.0, & y_i>40 \\
1.0, & y_i\leq40
\end{cases}
$$

Initial objective:

$$
\boxed{\min_{w,e}\sum_{i=1}^{N}c_i e_i}
$$

Equivalently:

$$
\sum_{i:y_i\leq40}|\hat{s}_i-y_i|+2\sum_{i:y_i>40}|\hat{s}_i-y_i|
$$

This is a linear program. It gives an equally sized raw-score error twice the importance for an expert Moderate/High pair than for an expert Low pair. It does not minimize threshold-error counts.

## Objective Modularity

Keep data loading, cross-validation, reporting, and convex constraints shared. An objective module has a common interface:

```julia
build_objective!(model, ensemble_score_expressions, expert_scores, objective_config)
```

A small registry maps `--objective weighted_mae` to the initial loss. Future continuous objectives can be registered without changing the common model workflow.

## Cross-Validation and Fold Weights

Use grouped five-fold cross-validation to avoid evaluating weights on labels seen during fitting:

1. Partition unique `Abstract_Index` values deterministically into five folds.
2. For fold $k$, reserve all pairs belonging to its abstract IDs as test set $T_k$.
3. Fit the LP on pairs from all remaining abstracts, $R_k$.
4. Save learned weights $w^{(k)}$.
5. Score only held-out pairs using those weights.
6. Repeat until each retained pair receives exactly one out-of-fold prediction.

For held-out pair $i \in T_k$:

$$
\hat{s}^{\mathrm{OOF}}_i=\sum_{m=1}^{M}w_m^{(k)}s_{im}
$$

`fold_weights.csv` contains one weight per `(fold, model)`. There are five weight vectors because each fold trains on a different subset of abstracts. Stable weights support a robust model preference; substantial variation indicates limited or heterogeneous expert data.

The first version reports only fold weights and out-of-fold performance. A later `--fit-all` mode can create one deployable weight vector after stability review.

## Artifacts

- `fold_weights.csv`: per-fold model weights, train/test sizes, objective value, solver status.
- `out_of_fold_predictions.csv`: pair IDs, expert score, fold, LLM scores, weighted score, and fixed-band levels.
- `metrics_summary.csv`: weighted MAE, ordinary MAE, mean signed error, exact three-level accuracy, recall, QWK, and equal-weight baseline.
- `run_metadata.toml`: CLI values, input paths/run IDs, models, retained/dropped rows, objective settings, package versions, and solver status.

Use existing fixed levels: Low `<= 40`, Moderate `40 < score <= 80`, High `> 80`.

## Implementation Steps

1. Create `optimization/Project.toml`, following the Julia/JuMP/Gurobi/CSV/DataFrames conventions in `energy_reserve/model`, with a quiet Gurobi optimizer factory.
2. Implement CLI options: `--expert-run`, `--benchmark-run`, `--objective`, `--above-40-multiplier`, `--folds`, and `--output-dir`.
3. Implement input validation, pivoting, joining, and complete-case filtering.
4. Implement the generic JuMP model and result extraction.
5. Implement `weighted_mae` objective dispatch and constraints.
6. Implement deterministic abstract-grouped five-fold evaluation.
7. Write artifacts and compare with equal-weight predictions on the same rows.
8. Add Julia tests for constraints, objective behavior, group isolation, invalid inputs, and complete out-of-fold coverage.
9. Document setup and example runs in `optimization/README.md`.

## Verification

```bash
julia --project=optimization -e 'using Pkg; Pkg.test()'
```

Confirm the inputs join correctly; train/test abstract IDs are disjoint; all weights are non-negative and sum to one; every retained pair has one out-of-fold prediction; and optimized metrics are compared with the equal-weight baseline.

## Deferred Work

- Additional continuous objectives, such as asymmetric distance to 40.
- Full-data fitting after reviewing fold stability.
- Separate Mark and Charlie expert-set analysis or a pre-registered combination rule.
- Integration of validated weights into adjudication.
