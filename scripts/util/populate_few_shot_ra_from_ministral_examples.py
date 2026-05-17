"""Append Ministral-anchored RA few-shot examples to few_shot_ra.yaml.

The script reads the selected pair CSV, recovers the Ministral score and
reasoning from the adjudicated benchmark rows, resolves the abstract text from
`abstracts_cleaned.csv`, and appends new RA examples to the YAML file while
keeping the existing structure intact.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_CSV = (
    REPO_ROOT
    / "data"
    / "results"
    / "affinity_benchmark"
    / "20260422_212121"
    / "ministral_anchored_examples.csv"
)
DEFAULT_BENCHMARK_CSV = (
    REPO_ROOT
    / "data"
    / "results"
    / "affinity_benchmark"
    / "20260422_212121"
    / "adjudicated_ra_model_rows.csv"
)
DEFAULT_ABSTRACTS_CSV = REPO_ROOT / "data" / "processed" / "abstracts_cleaned.csv"
DEFAULT_QUESTIONS_CSV = REPO_ROOT / "data" / "processed" / "ra_questions_cleaned.csv"
DEFAULT_OUTPUT_YAML = REPO_ROOT / "data" / "prompts" / "few_shot_ra.yaml"
DEFAULT_ANCHOR_MODEL = "ministral-3-14b-cloud"
SOURCE_MARKER = "# Added by populate_few_shot_ra_from_ministral_examples.py"


def _slugify(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "item"


def _to_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _block_scalar(text: str, indent: int = 6) -> str:
    pad = " " * indent
    lines = text.rstrip().splitlines() or [""]
    return ">-\n" + "\n".join(f"{pad}{line}" if line else pad for line in lines)


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _load_abstract_lookup(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    id_col = "Source_Index" if "Source_Index" in df.columns else df.columns[0]
    text_col = "Abstract_Cleaned" if "Abstract_Cleaned" in df.columns else ("Abstract" if "Abstract" in df.columns else None)
    if text_col is None:
        raise ValueError(f"Could not locate an abstract text column in {path}")
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        lookup[_to_scalar(row[id_col]).rstrip(".0")] = _normalize_text(row[text_col])
    return lookup


def _load_question_lookup(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    id_col = "RA2025" if "RA2025" in df.columns else df.columns[0]
    text_col = "Question_Cleaned" if "Question_Cleaned" in df.columns else ("Questions - long" if "Questions - long" in df.columns else None)
    if text_col is None:
        raise ValueError(f"Could not locate a question text column in {path}")
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        lookup[_to_scalar(row[id_col]).rstrip(".0")] = _normalize_text(row[text_col])
    return lookup


def _load_benchmark_lookup(path: Path, anchor_model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Abstract_Index", "RA2025_ID", "Model_Slug"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Benchmark CSV {path} is missing columns: {sorted(missing)}")
    return df[df["Model_Slug"] == anchor_model].copy()


def _find_existing_ids(yaml_text: str) -> set[str]:
    return set(re.findall(r"^\s*-\s+id:\s+([^\s]+)\s*$", yaml_text, flags=re.MULTILINE))


def _format_example(example: dict[str, object]) -> str:
    abstract = _normalize_text(example["abstract"])
    rationale = _normalize_text(example["rationale"])
    target_id = _to_scalar(example["target_id"])
    example_id = _to_scalar(example["id"])
    score = example["score"]
    affinity_level = _to_scalar(example["affinity_level"])
    sub_band = _to_scalar(example["sub_band"])
    abstract_id = _to_scalar(example["abstract_id"])
    question_id = _to_scalar(example["question_id"])
    model = _to_scalar(example["model"])

    lines = [
        SOURCE_MARKER,
        f"# Source: ministral_anchored_examples.csv row abstract_id={abstract_id} question_id={question_id}",
        f"# Ministral affinity level: {affinity_level} ({score})",
        f"# Sub-band: {sub_band}; anchor model: {model}",
        f"  - id: {example_id}",
        "    target_type: RA",
        f"    abstract: {_block_scalar(abstract, indent=6)}",
        "    targets:",
        f"      - target_id: \"{target_id}\"",
        f"        score: {_to_scalar(score)}",
        "        rationale: >-",
    ]

    rationale_lines = rationale.rstrip().splitlines() or [""]
    for line in rationale_lines:
        lines.append(f"          {line}" if line else "          ")

    return "\n".join(lines)


def build_examples(
    selected_csv: Path,
    benchmark_csv: Path,
    abstracts_csv: Path,
    questions_csv: Path,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
) -> list[dict[str, object]]:
    selected = pd.read_csv(selected_csv)
    benchmark = _load_benchmark_lookup(benchmark_csv, anchor_model)
    abstracts_lookup = _load_abstract_lookup(abstracts_csv)
    questions_lookup = _load_question_lookup(questions_csv)

    merged = selected.merge(
        benchmark[["Abstract_Index", "RA2025_ID", "LLM_Affinity", "LLM_Affinity_Reason", "affinity_level", "Model_Slug"]],
        left_on=["abstract_id", "question_id"],
        right_on=["Abstract_Index", "RA2025_ID"],
        how="left",
        suffixes=("", "_bench"),
    )

    merged = merged.sort_values(
        by=[c for c in ["sub_band", "selection_rank", "selection_priority"] if c in merged.columns]
    )

    examples: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        abstract_id = _to_scalar(row.get("abstract_id", row.get("Abstract_Index"))).rstrip(".0")
        question_id = _to_scalar(row.get("question_id", row.get("RA2025_ID"))).rstrip(".0")
        score = row.get("score_ministral", row.get("LLM_Affinity"))
        rationale = row.get("LLM_Affinity_Reason", "")
        affinity_level = row.get("affinity_level", "")
        if pd.isna(score) and not pd.isna(row.get("LLM_Affinity")):
            score = row.get("LLM_Affinity")
        if not rationale or pd.isna(rationale):
            rationale = row.get("Final_Reason", "")
        abstract_text = abstracts_lookup.get(abstract_id, "")
        if not abstract_text:
            abstract_text = _normalize_text(row.get("abstract_text", ""))

        model_slug = _to_scalar(row.get("Model_Slug", anchor_model))
        # Use the abstract id as the example id (user requested), prefixed with 'id_'.
        # Keep it slugified to ensure it's YAML-safe and consistent.
        example_id = f"id_{_slugify(abstract_id)}"
        examples.append(
            {
                "id": example_id,
                "target_type": "RA",
                "abstract": abstract_text,
                "target_id": question_id,
                "score": score,
                "rationale": _normalize_text(rationale),
                "affinity_level": affinity_level,
                "sub_band": row.get("sub_band", ""),
                "abstract_id": abstract_id,
                "question_id": question_id,
                "model": model_slug,
            }
        )
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
        description="Append Ministral-anchored RA examples to few_shot_ra.yaml"
    )
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--abstracts-csv", type=Path, default=DEFAULT_ABSTRACTS_CSV)
    parser.add_argument("--questions-csv", type=Path, default=DEFAULT_QUESTIONS_CSV)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT_YAML)
    parser.add_argument("--anchor-model", default=DEFAULT_ANCHOR_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    examples = build_examples(
        args.selected_csv,
        args.benchmark_csv,
        args.abstracts_csv,
        args.questions_csv,
        anchor_model=args.anchor_model,
    )

    if args.dry_run:
        print(f"Prepared {len(examples)} examples from {args.selected_csv}")
        for example in examples:
            print(f"- {example['id']} | target_id={example['target_id']} | score={_to_scalar(example['score'])}")
        return

    appended = append_examples_to_yaml(args.output_yaml, examples)
    print(f"Appended {appended} examples to {args.output_yaml}")


if __name__ == "__main__":
    main()
