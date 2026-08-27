#!/usr/bin/env python3
"""Retry missing RA affinity scores for one completed benchmark model.

The command updates only rows with a missing or nonnumeric ``LLM_Affinity``.
It preserves a one-time backup of the original CSV and adds repair provenance
columns so the normal postprocessing and adjudication pipeline can run unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_affinity_benchmark import _load_settings, _resolve_profiles, _slugify
from src.models.llm_affinity_reasoner import LLMAffinityReasoner


CFG_PATH = REPO_ROOT / "src" / "config" / "settings.yaml"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "results" / "affinity_benchmark"
REPAIR_METHOD = "retry_smaller_batch"


def _normalize_identifier(value: object) -> str:
    text = str(value).strip()
    try:
        numeric_value = float(text)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    return text


def _load_abstract_lookup(run_dir: Path) -> dict[str, str]:
    lookup_path = run_dir / "abstract_lookup_cleaned.csv"
    lookup = pd.read_csv(lookup_path)
    required_columns = {"Abstract_Index", "Abstract_Cleaned"}
    missing_columns = required_columns.difference(lookup.columns)
    if missing_columns:
        raise ValueError(f"{lookup_path} is missing columns: {sorted(missing_columns)}")
    return {
        _normalize_identifier(row.Abstract_Index): str(row.Abstract_Cleaned)
        for row in lookup.itertuples(index=False)
        if str(row.Abstract_Cleaned).strip()
    }


def repair_missing_ra_affinities(
    scores: pd.DataFrame,
    abstract_lookup: dict[str, str],
    reasoner: Any,
    repaired_at_utc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return repaired score rows and any pairs that did not receive a valid retry score."""
    required_columns = {"Abstract_Index", "RA2025_ID", "RA_Question", "LLM_Affinity"}
    missing_columns = required_columns.difference(scores.columns)
    if missing_columns:
        raise ValueError(f"RA affinity input is missing columns: {sorted(missing_columns)}")

    repaired = scores.copy()
    numeric_scores = pd.to_numeric(repaired["LLM_Affinity"], errors="coerce")
    missing_mask = numeric_scores.isna()
    repaired["LLM_Affinity"] = numeric_scores
    repaired["Was_Repaired"] = False
    repaired["Repair_Method"] = pd.NA
    repaired["Repair_Attempt"] = pd.NA
    repaired["Repaired_At_UTC"] = pd.NA

    unresolved_rows: list[dict[str, object]] = []
    pending = repaired.loc[missing_mask].copy()
    for abstract_index, rows in pending.groupby("Abstract_Index", sort=False):
        normalized_abstract_index = _normalize_identifier(abstract_index)
        abstract_text = abstract_lookup.get(normalized_abstract_index)
        if not abstract_text:
            for row_index, row in rows.iterrows():
                unresolved_rows.append({
                    "Row_Index": row_index,
                    "Abstract_Index": row["Abstract_Index"],
                    "RA2025_ID": row["RA2025_ID"],
                    "Reason": "abstract text not found in abstract lookup",
                })
            continue

        targets = [
            {"id": _normalize_identifier(row.RA2025_ID), "text": str(row.RA_Question)}
            for row in rows.itertuples()
        ]
        retry_scores = reasoner.rate_affinity_batch(abstract_text, targets, target_type="RA")
        for row_index, row in rows.iterrows():
            target_id = _normalize_identifier(row["RA2025_ID"])
            retry_value = retry_scores.get(target_id)
            if isinstance(retry_value, dict):
                retry_score = retry_value.get("score")
                retry_reason = retry_value.get("reason")
            else:
                retry_score = retry_value
                retry_reason = None
            retry_score = pd.to_numeric(pd.Series([retry_score]), errors="coerce").iloc[0]
            if pd.isna(retry_score):
                unresolved_rows.append({
                    "Row_Index": row_index,
                    "Abstract_Index": row["Abstract_Index"],
                    "RA2025_ID": row["RA2025_ID"],
                    "Reason": "retry did not return a numeric affinity score",
                })
                continue

            repaired.loc[row_index, "LLM_Affinity"] = float(retry_score)
            if "LLM_Affinity_Reason" in repaired.columns:
                repaired.loc[row_index, "LLM_Affinity_Reason"] = retry_reason
            repaired.loc[row_index, "Was_Repaired"] = True
            repaired.loc[row_index, "Repair_Method"] = REPAIR_METHOD
            repaired.loc[row_index, "Repair_Attempt"] = 1
            repaired.loc[row_index, "Repaired_At_UTC"] = repaired_at_utc

    return repaired, pd.DataFrame(unresolved_rows)


def _resolve_profile(model_slug: str) -> Any:
    profiles, _ = _resolve_profiles(_load_settings(CFG_PATH))
    for profile in profiles:
        if profile.name == model_slug or _slugify(profile.name) == model_slug:
            return profile
    raise ValueError(f"No enabled benchmark profile named {model_slug!r} in {CFG_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry missing RA affinity scores for one benchmark model")
    parser.add_argument("--run-id", required=True, help="Benchmark run id under --input-root")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Root containing benchmark runs")
    parser.add_argument("--model-slug", required=True, help="Model directory and enabled settings profile name")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Retry targets per LLM call (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Report missing rows without making LLM calls or writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (Path(args.input_root).expanduser() / args.run_id).resolve()
    model_dir = run_dir / args.model_slug
    scores_path = model_dir / "ra_affinities.csv"
    backup_path = model_dir / "ra_affinities_original.csv"
    unresolved_path = model_dir / "ra_affinities_repair_unresolved.csv"
    scores = pd.read_csv(scores_path)
    missing_count = int(pd.to_numeric(scores["LLM_Affinity"], errors="coerce").isna().sum())
    print(f"Missing or nonnumeric RA scores for {args.model_slug}: {missing_count}")
    if not missing_count or args.dry_run:
        return

    profile = _resolve_profile(args.model_slug)
    if backup_path.exists():
        print(f"Using existing original backup: {backup_path}")
    else:
        shutil.copy2(scores_path, backup_path)
        print(f"Saved original backup: {backup_path}")

    reasoner = LLMAffinityReasoner(
        model=profile.model,
        base_url=profile.base_url,
        api_key=profile.api_key,
        requests_per_minute=profile.requests_per_minute,
        micro_batch_size=args.micro_batch_size,
    )
    repaired_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repaired, unresolved = repair_missing_ra_affinities(
        scores=scores,
        abstract_lookup=_load_abstract_lookup(run_dir),
        reasoner=reasoner,
        repaired_at_utc=repaired_at_utc,
    )
    repaired.to_csv(scores_path, index=False)
    unresolved.to_csv(unresolved_path, index=False)
    repaired_count = missing_count - len(unresolved)
    print(f"Repaired scores: {repaired_count}")
    print(f"Unresolved scores: {len(unresolved)} -> {unresolved_path}")
    print(f"Updated model output: {scores_path}")


if __name__ == "__main__":
    main()