#!/usr/bin/env python3
"""Run multiple affinity benchmark slices defined in a CSV manifest.

Expected manifest columns (minimal):
- run_id
- enabled
- year_start
- year_end
- journals
- output_tag
- notes

Only enabled rows are executed in parallel. For each row this launcher calls:
    scripts/run_affinity_benchmark_parallel.py
and forwards year/journal filters so each benchmark run records and uses
exact subset criteria.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "run_affinity_benchmark_parallel.py"


@dataclass
class ManifestRow:
    run_id: str
    enabled: bool
    year_start: int | None
    year_end: int | None
    journals: str | None
    output_tag: str | None
    notes: str | None


def _slugify(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_manifest(path: Path) -> list[ManifestRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")

        rows: list[ManifestRow] = []
        for index, raw in enumerate(reader, start=1):
            enabled = _as_bool(raw.get("enabled"), default=True)
            run_id = _as_optional_str(raw.get("run_id")) or f"{_timestamp_id()}_{index:03d}"
            rows.append(
                ManifestRow(
                    run_id=run_id,
                    enabled=enabled,
                    year_start=_as_optional_int(raw.get("year_start")),
                    year_end=_as_optional_int(raw.get("year_end")),
                    journals=_as_optional_str(raw.get("journals")),
                    output_tag=_as_optional_str(raw.get("output_tag")),
                    notes=_as_optional_str(raw.get("notes")),
                )
            )
        return rows


def _build_run_id(row: ManifestRow) -> str:
    if not row.output_tag:
        return row.run_id
    return f"{row.run_id}_{_slugify(row.output_tag)}"


def _build_command(row: ManifestRow, mode: str, output_root: str) -> list[str]:
    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--run-id",
        _build_run_id(row),
        "--mode",
        mode,
        "--output-root",
        output_root,
    ]

    if row.year_start is not None:
        cmd.extend(["--year-start", str(row.year_start)])
    if row.year_end is not None:
        cmd.extend(["--year-end", str(row.year_end)])
    if row.journals:
        cmd.extend(["--journals", row.journals])

    return cmd


def _run_row(cmd: list[str], run_id: str) -> tuple[str, str | None]:
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        return run_id, None
    except Exception as exc:
        return run_id, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel affinity benchmark rows from a manifest CSV")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to CSV manifest (resolved from current working directory, with repo-root fallback)",
    )
    parser.add_argument("--mode", default="ra", choices=["both", "prp", "ra"], help="Benchmark mode (default: ra)")
    parser.add_argument(
        "--output-root",
        default="data/results/affinity_benchmark",
        help="Output root passed to run_affinity_benchmark_parallel.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        help="Continue with remaining manifest rows if one run fails",
    )
    parser.add_argument(
        "--stop-on-error",
        dest="continue_on_error",
        action="store_false",
        help="Stop immediately on first failed run",
    )
    parser.set_defaults(continue_on_error=False)
    return parser.parse_args()


def _resolve_manifest_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    processed_candidate = (REPO_ROOT / "data" / "processed" / candidate).resolve()
    return processed_candidate


def main() -> None:
    args = parse_args()
    manifest_path = _resolve_manifest_path(args.manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    rows = _read_manifest(manifest_path)
    enabled_rows = [row for row in rows if row.enabled]
    if not enabled_rows:
        print(f"No enabled rows found in manifest: {manifest_path}")
        return

    print(f"Manifest: {manifest_path}")
    print(f"Enabled rows: {len(enabled_rows)}")

    planned_runs: list[tuple[str, list[str], str | None]] = []
    for index, row in enumerate(enabled_rows, start=1):
        run_id = _build_run_id(row)
        cmd = _build_command(row, mode=args.mode, output_root=args.output_root)
        printable = " ".join(cmd)
        print(f"[{index}/{len(enabled_rows)}] run_id={run_id}")
        if row.notes:
            print(f"  notes={row.notes}")
        print(f"  cmd={printable}")
        planned_runs.append((run_id, cmd, row.notes))

    if args.dry_run:
        print("Dry run complete")
        return

    failures: list[tuple[str, str]] = []
    max_workers = max(1, len(planned_runs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_row, cmd, run_id): run_id for run_id, cmd, _notes in planned_runs
        }
        for future in as_completed(futures):
            completed_run_id, error_text = future.result()
            if error_text is None:
                print(f"  COMPLETED: {completed_run_id}")
                continue

            failures.append((completed_run_id, error_text))
            print(f"  FAILED: {completed_run_id} -> {error_text}")
            if not args.continue_on_error:
                for pending in futures:
                    if not pending.done():
                        pending.cancel()
                break

    if failures:
        print("Failed runs:")
        for run_id, error_text in failures:
            print(f"- {run_id}: {error_text}")
        raise RuntimeError(f"{len(failures)} manifest run(s) failed")

    print("All enabled manifest runs completed")


if __name__ == "__main__":
    main()
