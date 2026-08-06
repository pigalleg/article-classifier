#!/usr/bin/env python3
"""Assemble per-model affinity outputs into consolidated long-format files.

This script intentionally avoids producing derived summary artifacts.
Use the notebook to compute PRP/RA summaries on the fly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "results" / "affinity_benchmark"


def _discover_run_dir(input_root: Path, run_id: str | None) -> Path:
    if run_id:
        run_dir = input_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    candidates = [p for p in input_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {input_root}")
    return sorted(candidates)[-1]


def _collect_affinity_rows(run_dir: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    prp_frames: list[pd.DataFrame] = []
    ra_frames: list[pd.DataFrame] = []

    for model_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        model_slug = model_dir.name

        prp_path = model_dir / "prp_affinities.csv"
        if prp_path.exists() and prp_path.stat().st_size > 0:
            df = pd.read_csv(prp_path)
            df["Model_Slug"] = model_slug
            df["Run_ID"] = run_dir.name
            prp_frames.append(df)

        ra_path = model_dir / "ra_affinities.csv"
        if ra_path.exists() and ra_path.stat().st_size > 0:
            df = pd.read_csv(ra_path)
            df["Model_Slug"] = model_slug
            df["Run_ID"] = run_dir.name
            ra_frames.append(df)

    return prp_frames, ra_frames


def _write_assembled_outputs(run_dir: Path, prp_frames: list[pd.DataFrame], ra_frames: list[pd.DataFrame]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}

    if prp_frames:
        merged_prp = pd.concat(prp_frames, ignore_index=True)
        prp_out = run_dir / "merged_prp_affinities.csv"
        merged_prp.to_csv(prp_out, index=False)
        outputs["prp_path"] = str(prp_out)
        outputs["prp_rows"] = int(len(merged_prp))
    else:
        outputs["prp_path"] = None
        outputs["prp_rows"] = 0

    if ra_frames:
        merged_ra = pd.concat(ra_frames, ignore_index=True)
        ra_out = run_dir / "merged_ra_affinities.csv"
        merged_ra.to_csv(ra_out, index=False)
        outputs["ra_path"] = str(ra_out)
        outputs["ra_rows"] = int(len(merged_ra))
    else:
        outputs["ra_path"] = None
        outputs["ra_rows"] = 0

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble per-model affinity outputs for a benchmark run")
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root containing benchmark run directories",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Benchmark run id to assemble; defaults to latest run folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    run_dir = _discover_run_dir(input_root=input_root, run_id=args.run_id)
    prp_frames, ra_frames = _collect_affinity_rows(run_dir)
    outputs = _write_assembled_outputs(run_dir, prp_frames, ra_frames)

    print(f"Run directory: {run_dir}")
    print(f"Merged PRP rows: {outputs['prp_rows']}")
    print(f"Merged RA rows: {outputs['ra_rows']}")
    if outputs["prp_path"]:
        print(f"Saved merged PRP -> {outputs['prp_path']}")
    if outputs["ra_path"]:
        print(f"Saved merged RA -> {outputs['ra_path']}")


if __name__ == "__main__":
    main()
