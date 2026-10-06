# Few-Shot Prompting and Affinity Scoring Scale

Few-shot prompting and affinity-score instructions are shared by the affinity
evaluator and adjudicator. Both use the same reasoner configuration, with
separate example sets for RA questions and PRPs. The examples provide
calibration context to the model; they do not replace the benchmark's model
scores or the adjudicator's agreement thresholds.

## Current Configuration

The active settings are under `runtime.llm_reasoner` in
`src/config/settings.yaml`:

```yaml
runtime:
  llm_reasoner:
    enable_few_shot: true
    enable_affinity_reasons: true
    few_shot_ra_file: data/prompts/few_shot_ra.yaml
    few_shot_prp_file: data/prompts/few_shot_prp.yaml
    few_shot_max_examples_per_prompt: 4
    score_scale_version: 2
    score_scale_file: data/prompts/affinity_scoring_scale.yaml
```

Few-shot prompting and affinity reasons are currently enabled. Up to four
examples are included from the appropriate RA or PRP file for each prompt. The
active score scale is **version 2**, selected by `score_scale_version: 2` from
`data/prompts/affinity_scoring_scale.yaml`. That file defines scoring
instructions for both RA and PRP targets; its top-level version is also 2.

The score scale and few-shot examples serve different roles: the score scale
explains how the model should interpret the affinity range, while examples show
calibrated abstract/target/score/rationale patterns. Affinity scores use the
project's 0-120 scale.

## Example Format

Each file contains a YAML mapping with a `version` and an `examples` list. Use
`target_type: RA` in `few_shot_ra.yaml` and `target_type: PRP` in
`few_shot_prp.yaml`:

```yaml
version: 1
examples:
  - id: ra_high_1
    target_type: RA
    abstract: "Example abstract text"
    targets:
      - target_id: "42"
        score: 88
        rationale: "Why this abstract is relevant to the question"
      - target_id: "43"
        score: 61
        rationale: "Why this abstract has partial relevance"
```

Multiple targets per abstract are supported. The loader also accepts legacy
single-target examples with `target_id`, `score`, and `rationale` directly on
the example. Keep example scores on the 0-120 affinity scale. When
`enable_affinity_reasons` is true, evaluation records the model's reasons in
`LLM_Affinity_Reason`; adjudication can include a final reason for pairs sent
for LLM review.

## Overrides

For a single evaluation or benchmark run, these command-line switches take
precedence over the environment and settings defaults:

- `--enable-few-shot` / `--no-enable-few-shot`
- `--enable-affinity-reasons` / `--no-enable-affinity-reasons`

The adjudication CLI exposes the affinity-reasons switch, but not a separate
few-shot switch. For adjudication, set `AFFINITY_ENABLE_FEW_SHOT` in the
environment or change `enable_few_shot` in settings to control examples.

Environment overrides are:

- `AFFINITY_ENABLE_FEW_SHOT`
- `AFFINITY_ENABLE_AFFINITY_REASONS`
- `AFFINITY_FEW_SHOT_RA_FILE`
- `AFFINITY_FEW_SHOT_PRP_FILE`
- `AFFINITY_FEW_SHOT_MAX_EXAMPLES`
- `AFFINITY_SCORE_SCALE_FILE`

`score_scale_version` remains in `src/config/settings.yaml`; the environment
variable can select another score-scale file, while the configured version
selects the versioned prompt inside that file.

## Import Examples From Excel

Importing an evaluator workbook into the RA prompt file is a manual one-shot
preparation task, not part of the benchmark/evaluation/adjudication workflow.
See [Build RA Few-Shot Examples From Excel](one-shot-few-shot-examples.md) for
its expected spreadsheet columns and dry-run/write commands.
