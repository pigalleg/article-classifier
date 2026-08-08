#!/usr/bin/env python3
"""Check whether all launched affinity benchmark models completed their outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "results" / "affinity_benchmark"
STAGE_CONFIG = {
    "ra": ("ra_affinities.csv", "RA2025_ID"),
    "prp": ("prp_affinities.csv", "PRP_Name"),
}


@dataclass(frozen=True)
class ModelRun:
    slug: str
    mode: str
    manifest_path: Path


@dataclass
class StageCoverage:
    model_slug: str
    stage: str
    rows: int = 0
    missing_affinities: int = 0
    pairs: set[tuple[str, str]] | None = None
    abstracts: set[str] | None = None
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.pairs is None:
            self.pairs = set()
        if self.abstracts is None:
            self.abstracts = set()
        if self.errors is None:
            self.errors = []


def _normalise_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _manifest_paths(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("*benchmark_manifest.csv") if path.is_file())


def _required_model_runs(run_dir: Path) -> tuple[list[ModelRun], list[str]]:
    errors: list[str] = []
    model_runs: dict[str, ModelRun] = {}
    manifest_paths = _manifest_paths(run_dir)
    if not manifest_paths:
        return [], [f"No benchmark manifest found in {run_dir}"]

    for manifest_path in manifest_paths:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("status") or "").strip().lower() not in {"success", "skipped"}:
                    continue
                output_dir = str(row.get("output_dir") or "").strip()
                if not output_dir:
                    errors.append(f"{manifest_path.name}: required model has no output_dir")
                    continue
                # Parallel launchers preserve child manifests, whose output paths
                # can belong to a different OS or temporary staging directory.
                slug = Path(output_dir.replace("\\", "/")).name
                if not slug:
                    errors.append(f"{manifest_path.name}: could not determine model directory from {output_dir!r}")
                    continue
                mode = str(row.get("mode") or "").strip().lower()
                existing = model_runs.get(slug)
                if existing and existing.mode != mode:
                    errors.append(f"{slug}: conflicting manifest modes ({existing.mode}, {mode})")
                    continue
                model_runs[slug] = ModelRun(slug=slug, mode=mode, manifest_path=manifest_path)

    if not model_runs:
        errors.append("No successful or skipped model entries found in benchmark manifests")
    return sorted(model_runs.values(), key=lambda item: item.slug), errors


def _stages_for_mode(mode: str) -> set[str]:
    if mode == "both":
        return {"ra", "prp"}
    if mode in STAGE_CONFIG:
        return {mode}
    return set()


def _expected_abstracts(run_dir: Path) -> tuple[set[str] | None, str | None]:
    lookup_path = run_dir / "abstract_lookup_cleaned.csv"
    if not lookup_path.exists() or lookup_path.stat().st_size == 0:
        return None, None
    lookup = pd.read_csv(lookup_path)
    if "Abstract_Index" not in lookup.columns:
        return None, f"{lookup_path.name} has no Abstract_Index column"
    return {_normalise_identifier(value) for value in lookup["Abstract_Index"] if _normalise_identifier(value)}, None


def _read_stage_coverage(run_dir: Path, model_run: ModelRun, stage: str) -> StageCoverage:
    filename, target_column = STAGE_CONFIG[stage]
    coverage = StageCoverage(model_slug=model_run.slug, stage=stage)
    csv_path = run_dir / model_run.slug / filename
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        coverage.errors.append(f"missing or empty {csv_path.relative_to(run_dir)}")
        return coverage

    dataframe = pd.read_csv(csv_path)
    required_columns = {"Abstract_Index", target_column, "LLM_Affinity"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        coverage.errors.append(f"{csv_path.relative_to(run_dir)} missing columns: {', '.join(sorted(missing_columns))}")
        return coverage

    coverage.rows = len(dataframe)
    coverage.missing_affinities = int(
        pd.to_numeric(dataframe["LLM_Affinity"], errors="coerce").isna().sum()
    )
    if coverage.missing_affinities:
        coverage.errors.append(
            f"{csv_path.relative_to(run_dir)} has {coverage.missing_affinities} missing or nonnumeric affinity scores"
        )
    for abstract, target in zip(dataframe["Abstract_Index"], dataframe[target_column]):
        abstract_id = _normalise_identifier(abstract)
        target_id = _normalise_identifier(target)
        if not abstract_id or not target_id:
            coverage.errors.append(f"{csv_path.relative_to(run_dir)} contains an empty abstract or target identifier")
            continue
        coverage.abstracts.add(abstract_id)
        coverage.pairs.add((abstract_id, target_id))

    duplicate_count = coverage.rows - len(coverage.pairs)
    if duplicate_count:
        coverage.errors.append(f"{csv_path.relative_to(run_dir)} has {duplicate_count} duplicate abstract-target rows")
    return coverage


def check_run_coverage(run_dir: Path, mode: str | None = None) -> list[str]:
    """Return human-readable coverage errors for a benchmark run directory."""
    model_runs, errors = _required_model_runs(run_dir)
    if not model_runs:
        return errors

    expected_abstracts, lookup_error = _expected_abstracts(run_dir)
    if lookup_error:
        errors.append(lookup_error)

    selected_stages = _stages_for_mode(mode) if mode else set().union(
        *(_stages_for_mode(model_run.mode) for model_run in model_runs)
    )
    if not selected_stages:
        return errors + ["Could not determine stages from manifests; pass --mode explicitly"]

    for stage in sorted(selected_stages):
        applicable_models = [
            model_run for model_run in model_runs if stage in _stages_for_mode(model_run.mode)
        ]
        if mode:
            applicable_models = model_runs
        coverages = [_read_stage_coverage(run_dir, model_run, stage) for model_run in applicable_models]
        for coverage in coverages:
            errors.extend(f"{coverage.model_slug} ({stage}): {error}" for error in coverage.errors)
            if expected_abstracts is not None:
                missing_abstracts = expected_abstracts.difference(coverage.abstracts)
                unexpected_abstracts = coverage.abstracts.difference(expected_abstracts)
                if missing_abstracts:
                    errors.append(
                        f"{coverage.model_slug} ({stage}): missing {len(missing_abstracts)} abstracts "
                        f"(for example: {', '.join(sorted(missing_abstracts)[:5])})"
                    )
                if unexpected_abstracts:
                    errors.append(
                        f"{coverage.model_slug} ({stage}): contains {len(unexpected_abstracts)} unexpected abstracts"
                    )

        valid_coverages = [coverage for coverage in coverages if not coverage.errors]
        if not valid_coverages:
            continue
        reference = max(valid_coverages, key=lambda coverage: len(coverage.pairs))
        for coverage in valid_coverages:
            missing_pairs = reference.pairs.difference(coverage.pairs)
            unexpected_pairs = coverage.pairs.difference(reference.pairs)
            if missing_pairs:
                errors.append(
                    f"{coverage.model_slug} ({stage}): missing {len(missing_pairs)} abstract-target pairs "
                    f"relative to {reference.model_slug}"
                )
            if unexpected_pairs:
                errors.append(
                    f"{coverage.model_slug} ({stage}): has {len(unexpected_pairs)} extra abstract-target pairs "
                    f"relative to {reference.model_slug}"
                )
    return errors


def _resolve_run_dir(run_dir: str | None, input_root: str, run_id: str | None) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    if not run_id:
        raise ValueError("Pass --run-dir or --run-id")
    return (Path(input_root).expanduser() / run_id).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check affinity benchmark model coverage")
    parser.add_argument("--run-dir", help="Benchmark result directory to check")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Root containing benchmark runs")
    parser.add_argument("--run-id", help="Benchmark run id under --input-root")
    parser.add_argument("--mode", choices=["ra", "prp", "both"], help="Stages to check; defaults to manifest modes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir, args.input_root, args.run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Benchmark run directory not found: {run_dir}")

    errors = check_run_coverage(run_dir, mode=args.mode)
    if errors:
        print(f"Coverage check failed: {run_dir}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Coverage check passed: {run_dir}")


if __name__ == "__main__":
    main()