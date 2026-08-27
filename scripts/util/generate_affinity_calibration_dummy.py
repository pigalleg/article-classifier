"""Generate a synthetic calibration dataset from benchmark affinity outputs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.analysis.affinity_levels import affinity_level_fixed

DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "results" / "affinity_benchmark"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "affinity_calibration"
DEFAULT_RUN_IDS = ["20260422_212121_1", "20260422_212121_2", "20260422_212121_3"]
DEFAULT_OUTPUT_RUN_ID = "20260601_dummy_100_abstracts_v3"


@dataclass(frozen=True)
class SourceGroup:
    title: str
    run_id: str
    model_slug: str
    rows: pd.DataFrame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-ids", nargs="*", default=DEFAULT_RUN_IDS)
    parser.add_argument("--num-abstracts", type=int, default=100)
    parser.add_argument("--min-questions", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    return parser.parse_args()


def _load_source_groups(input_root: Path, run_ids: list[str]) -> list[SourceGroup]:
    groups: list[SourceGroup] = []
    for run_id in run_ids:
        run_dir = input_root / run_id
        ra_path = run_dir / "merged_ra_affinities.csv"
        if not ra_path.exists():
            raise FileNotFoundError(f"Missing benchmark file: {ra_path}")

        df = pd.read_csv(ra_path)
        if df.empty:
            continue

        required_columns = {"Abstract_Index", "Document Title", "RA2025_ID", "RA_Question", "Cosine_x100", "Model_Slug", "Run_ID"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"{ra_path} is missing columns: {sorted(missing)}")

        for title, title_df in df.groupby("Document Title", dropna=False):
            if not isinstance(title, str) or not title.strip():
                continue
            groups.append(
                SourceGroup(
                    title=title,
                    run_id=str(title_df["Run_ID"].iloc[0]),
                    model_slug=str(title_df["Model_Slug"].iloc[0]),
                    rows=title_df.reset_index(drop=True),
                )
            )
    return groups


def _sample_rows_for_abstract(rng: random.Random, group: SourceGroup, min_questions: int, max_questions: int) -> pd.DataFrame:
    available = len(group.rows)
    if available < min_questions:
        raise ValueError(f"Abstract '{group.title}' in run '{group.run_id}' has only {available} questions")

    question_count = rng.randint(min_questions, min(max_questions, available))
    sampled = group.rows.sample(n=question_count, random_state=rng.randint(0, 2**31 - 1), replace=False)
    sampled = sampled.copy().reset_index(drop=True)
    sampled["Evaluator_Affinity"] = [round(rng.uniform(0, 100), 1) for _ in range(question_count)]
    sampled["Evaluator_Affinity_Level"] = sampled["Evaluator_Affinity"].apply(affinity_level_fixed)
    sampled["Evaluator_Affinity_Reason"] = "Synthetic calibration sample with randomized affinity."
    sampled["Source_Evaluator"] = "synthetic-calibration-generator"
    return sampled


def _build_output_frame(groups: list[SourceGroup], num_abstracts: int, min_questions: int, max_questions: int, seed: int) -> pd.DataFrame:
    eligible_titles = sorted({group.title for group in groups if len(group.rows) >= min_questions})
    if len(eligible_titles) < num_abstracts:
        raise ValueError(f"Only {len(eligible_titles)} unique abstracts are available, but {num_abstracts} were requested")

    rng = random.Random(seed)
    sampled_titles = rng.sample(eligible_titles, k=num_abstracts)
    output_groups: list[pd.DataFrame] = []

    for title in sampled_titles:
        candidates = [group for group in groups if group.title == title and len(group.rows) >= min_questions]
        if not candidates:
            raise ValueError(f"No eligible source group found for abstract '{title}'")
        chosen_group = rng.choice(candidates)
        output_groups.append(_sample_rows_for_abstract(rng, chosen_group, min_questions, max_questions))

    output = pd.concat(output_groups, ignore_index=True)
    column_order = [
        "Abstract_Index",
        "Document Title",
        "RA2025_ID",
        "RA_Question",
        "Evaluator_Affinity",
        "Evaluator_Affinity_Level",
        "Evaluator_Affinity_Reason",
        "Source_Evaluator",
    ]
    return output[column_order]


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_dir = output_root / args.output_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _load_source_groups(input_root, list(args.run_ids))
    output_df = _build_output_frame(groups, args.num_abstracts, args.min_questions, args.max_questions, args.seed)

    output_csv = output_dir / "calibration_dummy_abstract_question_affinities.csv"
    output_df.to_csv(output_csv, index=False)

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "output_csv": output_csv.name,
        "run_ids": list(args.run_ids),
        "num_abstracts": args.num_abstracts,
        "min_questions": args.min_questions,
        "max_questions": args.max_questions,
        "seed": args.seed,
        "source_rows": int(sum(len(group.rows) for group in groups)),
        "generated_rows": int(len(output_df)),
        "unique_abstract_titles": int(output_df["Document Title"].nunique()),
    }

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {output_csv}")
    print(f"Rows: {len(output_df)} | Unique abstracts: {output_df['Document Title'].nunique()}")


if __name__ == "__main__":
    main()