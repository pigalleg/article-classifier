"""Append RA few-shot examples to few_shot_ra.yaml from evaluator Excel rows.

The script reads an Excel sheet (default: ``Rationale_clean``), extracts RA
examples, maps evaluator scores from a 0-5 scale into affinity scores,
combines the two rationale columns into one LLM-friendly rationale, and
appends new examples to the RA few-shot YAML file.

Launch from the repository root:

Dry run (no YAML changes):
    python scripts/util/populate_few_shot_ra_from_excel_evaluations.py --dry-run

Write mode (append examples):
    python scripts/util/populate_few_shot_ra_from_excel_evaluations.py

Explicit file/sheet/output:
    python scripts/util/populate_few_shot_ra_from_excel_evaluations.py \
      --input-xlsx "<path-to-xlsx>" \
      --sheet-name "Rationale_clean" \
      --output-yaml data/prompts/few_shot_ra.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_XLSX = (
    REPO_ROOT / "data" / "raw" / "Abstracts Evaluation Template - Batch 1.xlsx"
)
DEFAULT_SHEET_NAME = "Rationale_clean"
DEFAULT_OUTPUT_YAML = REPO_ROOT / "data" / "prompts" / "few_shot_ra.yaml"
SOURCE_MARKER = "# Added by populate_few_shot_ra_from_excel_evaluations.py"

SCORE_MAPPING = {
    0: 10,
    1: 30,
    2: 50,
    3: 70,
    4: 90,
    5: 110,
}


def _slugify(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "item"


def _to_scalar(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_text(value: object) -> str:
    if _is_missing(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _is_missing(value: object) -> bool:
    return bool(pd.isna(cast(Any, value)))


def _block_scalar(text: str, indent: int = 6) -> str:
    pad = " " * indent
    lines = text.rstrip().splitlines() or [""]
    return ">-\n" + "\n".join(f"{pad}{line}" if line else pad for line in lines)


def _canonical_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_column(df: pd.DataFrame, expected: str) -> str:
    expected_canonical = _canonical_col_name(expected)
    by_canonical = {_canonical_col_name(c): c for c in df.columns}
    if expected_canonical in by_canonical:
        return by_canonical[expected_canonical]
    raise ValueError(
        f"Could not find column '{expected}'. Available columns: {list(df.columns)}"
    )


def _map_score(raw_score: object) -> int:
    if _is_missing(raw_score):
        raise ValueError("Evaluator_Score is missing for a row")

    try:
        parsed = int(float(str(raw_score).strip()))
    except ValueError as exc:
        raise ValueError(f"Evaluator_Score '{raw_score}' is not numeric") from exc

    if parsed not in SCORE_MAPPING:
        raise ValueError(
            f"Evaluator_Score '{raw_score}' is out of expected range [0, 1, 2, 3, 4, 5]"
        )
    return SCORE_MAPPING[parsed]


def _merge_rationale(primary_reason: object, boundary_reason: object) -> str:
    primary = _normalize_text(primary_reason)
    boundary = _normalize_text(boundary_reason)

    parts: list[str] = []
    if primary:
        parts.append(f"Why this score was assigned:\n{primary}")
    if boundary:
        parts.append(
            "Why this abstract falls below the next relevance level:\n"
            f"{boundary}"
        )

    return "\n\n".join(parts).strip()


def _find_existing_ids(yaml_text: str) -> set[str]:
    return set(re.findall(r"^\s*-\s+id:\s+([^\s]+)\s*$", yaml_text, flags=re.MULTILINE))


def _format_example(example: dict[str, object]) -> str:
    abstract = _normalize_text(example["abstract"])
    rationale = _normalize_text(example["rationale"])
    target_id = _to_scalar(example["target_id"])
    example_id = _to_scalar(example["id"])
    score = _to_scalar(example["score"])
    abstract_id = _to_scalar(example["abstract_id"])
    question_id = _to_scalar(example["question_id"])

    lines = [
        SOURCE_MARKER,
        (
            "# Source: Excel sheet row "
            f"abstract_id={abstract_id} question_id={question_id}"
        ),
        f"  - id: {example_id}",
        "    target_type: RA",
        f"    abstract: {_block_scalar(abstract, indent=6)}",
        "    targets:",
        f"      - target_id: \"{target_id}\"",
        f"        score: {score}",
        "        rationale: >-",
    ]

    rationale_lines = rationale.rstrip().splitlines() or [""]
    for line in rationale_lines:
        lines.append(f"          {line}" if line else "          ")

    return "\n".join(lines)


def build_examples(input_xlsx: Path, sheet_name: str) -> list[dict[str, object]]:
    df = pd.read_excel(input_xlsx, sheet_name=sheet_name)

    abstract_index_col = _find_column(df, "Abstract_Index")
    abstract_col = _find_column(df, "Abstract")
    ra_id_col = _find_column(df, "RA2025_ID")
    evaluator_score_col = _find_column(df, "Evaluator_Score")
    rationale_primary_col = _find_column(
        df, "Why this score was assigned to the abstract?"
    )
    rationale_boundary_col = _find_column(
        df, "Why does this abstract fall below the next relevance level?"
    )

    examples: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    skipped_no_rationale: list[str] = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        abstract_id = _to_scalar(row[abstract_index_col]).rstrip(".0")
        question_id = _to_scalar(row[ra_id_col]).rstrip(".0")

        if not abstract_id or not question_id:
            raise ValueError(
                f"Missing Abstract_Index or RA2025_ID in row {row_number}"
            )

        rationale = _merge_rationale(
            row[rationale_primary_col],
            row[rationale_boundary_col],
        )
        if not rationale:
            skipped_no_rationale.append(
                f"abstract_id={abstract_id} question_id={question_id} (row {row_number})"
            )
            continue

        abstract_text = _normalize_text(row[abstract_col])
        mapped_score = _map_score(row[evaluator_score_col])

        base_id = f"id_excel_abs{_slugify(abstract_id)}_ra{_slugify(question_id)}"
        example_id = base_id
        counter = 2
        while example_id in seen_ids:
            example_id = f"{base_id}_{counter}"
            counter += 1
        seen_ids.add(example_id)

        examples.append(
            {
                "id": example_id,
                "target_type": "RA",
                "abstract": abstract_text,
                "target_id": question_id,
                "score": mapped_score,
                "rationale": rationale,
                "abstract_id": abstract_id,
                "question_id": question_id,
            }
        )

    if skipped_no_rationale:
        print(f"Skipped {len(skipped_no_rationale)} rows with no rationale:")
        for item in skipped_no_rationale:
            print(f"  - {item}")

    return examples


def append_examples_to_yaml(output_yaml: Path, examples: Iterable[dict[str, object]]) -> int:
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    existing_text = output_yaml.read_text(encoding="utf-8") if output_yaml.exists() else "version: 1\nexamples:\n"
    existing_ids = _find_existing_ids(existing_text)

    rendered_blocks: list[str] = []
    appended = 0
    for example in examples:
        if example["id"] in existing_ids:
            continue
        rendered_blocks.append(_format_example(example))
        appended += 1

    if not rendered_blocks:
        return 0

    if not existing_text.endswith("\n"):
        existing_text += "\n"

    if "examples:" not in existing_text:
        existing_text = "version: 1\nexamples:\n"

    if not existing_text.rstrip().endswith("examples:"):
        existing_text = existing_text.rstrip() + "\n"

    block_text = "\n\n".join(rendered_blocks) + "\n"
    output_yaml.write_text(existing_text + block_text, encoding="utf-8")
    return appended


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append RA few-shot examples to few_shot_ra.yaml from evaluator Excel rows"
        )
    )
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT_XLSX)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT_YAML)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    examples = build_examples(args.input_xlsx, args.sheet_name)

    if args.dry_run:
        print(
            f"Prepared {len(examples)} examples from {args.input_xlsx} "
            f"(sheet: {args.sheet_name})"
        )
        for example in examples:
            print(
                "- "
                f"{example['id']} | "
                f"target_id={example['target_id']} | "
                f"score={_to_scalar(example['score'])}"
            )
        return

    appended = append_examples_to_yaml(args.output_yaml, examples)
    print(f"Appended {appended} examples to {args.output_yaml}")


if __name__ == "__main__":
    main()
