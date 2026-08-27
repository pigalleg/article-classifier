# LLM Ensemble Level-Weight Analysis

Date: 2026-08-27

## Scope

This comparison used the Mark OM v1 calibration labels (`602` abstract-question pairs) and the `20260726_134607_5` benchmark. Each result is based on five-fold grouped cross-validation by `Abstract_Index`. The learned weights for a fold were fitted without that fold's abstracts.

The objective remained weighted mean absolute error. Only the target-level multiplier changed.

## Results

| Low / Moderate / High multiplier | Low precision | Low recall | High precision | High recall | MAE | Accuracy | QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 2 / 2 | 85.5% | 74.0% | 30.8% | 29.6% | 19.55 | 68.1% | 0.397 |
| 1.5 / 1 / 4 | 85.6% | 74.9% | 30.8% | 29.6% | 19.39 | 68.8% | 0.405 |
| 1.5 / 1 / 6 | 85.2% | 74.9% | 30.8% | 29.6% | 19.44 | 68.4% | 0.399 |

## Decision

Use the `1.5 / 1 / 4` configuration for the next experiment. It gives the best OOF MAE, accuracy, and quadratic weighted kappa in this comparison, while slightly improving Low precision and recall.

Increasing the High multiplier from `4` to `6` did not improve High precision or recall and reduced the other metrics. Neither High-focused configuration changed High classification performance from the baseline. This suggests that convex model weighting under raw-score MAE alone cannot create a different High-band trade-off for this dataset.

The High class has only `27` expert-labelled pairs, so any future High-specific method should be assessed with grouped OOF results and fold-level variation before adoption.

## Result Directories

- `data/results/ensemble_weight_optimizer/20260827_mark_om_v1_134607_5_level_weights_baseline`
- `data/results/ensemble_weight_optimizer/20260827_mark_om_v1_134607_5_level_weights_high_4`
- `data/results/ensemble_weight_optimizer/20260827_mark_om_v1_134607_5_level_weights_high_6`