#!/usr/bin/env python3
"""Launch multiple affinity benchmark instances in parallel.

Usage guidelines:
- Use this launcher when benchmark models can be split into independent
  execution groups.
- Keep models that share a constrained resource in the same
  ``execution_group`` so they run sequentially inside one benchmark instance.
- Let different execution groups run in parallel.
- By default, models without an explicit ``execution_group`` are treated as
  independent groups, so they can run in parallel.

The launcher reuses the existing benchmark script for the actual evaluation.
It only decides which models run together and starts one benchmark process per
group. Each benchmark process first writes to a staging folder, and completed
group folders are moved into the final dated benchmark folder when the process
finishes.

Example configuration pattern in ``settings.yaml``:

.. code-block:: yaml

   models:
     llm:
       benchmark:
         models:
           - name: model-a
             backend: cloud
             model: model-a
             execution_group: cloud
           - name: model-b
             backend: cloud
             model: model-b
             execution_group: cloud
           - name: model-c
             backend: local
             model: model-c
             execution_group: local-gpu-1

Concrete split example:

.. code-block:: yaml

     execution groups:
         - cloud: model-a, model-b
         - local-gpu-1: model-c
         - local-gpu-2: model-d, model-e

This launcher starts one child benchmark process per group and passes only the
models from that group to ``run_affinity_benchmark.py``.

When a group finishes, its folder contents are moved into the final dated
benchmark folder. Colliding group-level metadata files are renamed with the
group prefix so they do not overwrite each other.

Final output layout:

.. code-block:: text

     data/results/affinity_benchmark/<run_id>/
         model outputs and merged artifacts
         <group_prefix>_benchmark_manifest.csv
         <group_prefix>_run_metadata.json

Examples:

.. code-block:: bash

   python scripts/run_affinity_benchmark_parallel.py --dry-run
    python scripts/run_affinity_benchmark_parallel.py --spawn-dry-run
   python scripts/run_affinity_benchmark_parallel.py --run-id 20260713_parallel
    python scripts/run_affinity_benchmark_parallel.py --mode ra
    python scripts/run_affinity_benchmark_parallel.py --mode both --ra-retrieval-mode prp_only

Notes:
- The launcher forwards all benchmark options it does not own directly to
  ``run_affinity_benchmark.py``.
- The launcher itself only owns orchestration options such as ``--run-id``,
    ``--output-root``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_affinity_benchmark import _as_bool, _load_settings, _slugify


DEFAULT_OUTPUT_ROOT = "data/results/affinity_benchmark"


@dataclass(frozen=True)
class LauncherModel:
    name: str
    backend: str
    execution_group: str


@dataclass(frozen=True)
class LaunchGroup:
    name: str
    backend: str
    dir_name: str
    models: tuple[LauncherModel, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch affinity benchmark groups in parallel"
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for benchmark outputs",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Outer run id for the parallel launcher (default: timestamp)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned group layout and exit",
    )
    parser.add_argument(
        "--spawn-dry-run",
        action="store_true",
        help="Launch the child benchmark processes in parallel, but forward --dry-run to them",
    )
    return parser


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_enabled_models(settings: dict[str, Any]) -> list[LauncherModel]:
    llm_cfg = settings.get("models", {}).get("llm", {})
    bench_cfg = llm_cfg.get("benchmark", {})
    default_backend = str(llm_cfg.get("default_backend") or "local").strip().lower()

    models: list[LauncherModel] = []
    for item in bench_cfg.get("models", []) or []:
        if not _as_bool(item.get("enabled"), True):
            continue

        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue

        backend = str(item.get("backend") or default_backend).strip().lower()
        execution_group = str(item.get("execution_group") or name).strip()

        models.append(
            LauncherModel(
                name=name,
                backend=backend,
                execution_group=execution_group,
            )
        )

    return models


def _group_models(models: list[LauncherModel]) -> list[LaunchGroup]:
    grouped: "OrderedDict[tuple[str, str], list[LauncherModel]]" = OrderedDict()
    group_names: dict[tuple[str, str], str] = {}

    for model in models:
        group_slug = _slugify(model.execution_group)
        key = (group_slug, model.backend)
        grouped.setdefault(key, []).append(model)
        group_names.setdefault(key, model.execution_group)

    launch_groups: list[LaunchGroup] = []
    for key, grouped_models in grouped.items():
        group_name = group_names[key]
        launch_groups.append(
            LaunchGroup(
                name=group_name,
                backend=key[1],
                dir_name=_slugify(f"{group_name}-{key[1]}"),
                models=tuple(grouped_models),
            )
        )

    return launch_groups


def _build_group_command(
    child_run_id: str,
    output_root: str,
    forwarded_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_affinity_benchmark.py"),
        "--run-id",
        child_run_id,
        "--output-root",
        output_root,
    ]
    command.extend(forwarded_args)
    return command


def _run_group(
    group: LaunchGroup,
    output_root: str,
    forwarded_args: list[str],
    child_dry_run: bool,
) -> tuple[str, str]:
    env = dict(os.environ)
    env["AFFINITY_BENCHMARK_MODELS"] = ",".join(model.name for model in group.models)
    env["AFFINITY_BENCHMARK_BACKEND"] = group.backend

    child_args = list(forwarded_args)
    if child_dry_run and "--dry-run" not in child_args:
        child_args.append("--dry-run")

    command = _build_group_command(group.dir_name, output_root, child_args)
    started = datetime.now(timezone.utc)
    print(f"[START] {group.name} at {started.isoformat()}")
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)
    finished = datetime.now(timezone.utc)
    print(f"[END] {group.name} at {finished.isoformat()}")
    return group.dir_name, group.name


def _move_directory_contents(source_dir: Path, destination_dir: Path, group_prefix: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        target_name = child.name
        if target_name in {"benchmark_manifest.csv", "run_metadata.json"}:
            target_name = f"{group_prefix}_{target_name}"
        target = destination_dir / target_name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    source_dir.rmdir()


def main() -> None:
    parser = _build_parser()
    args, forwarded_args = parser.parse_known_args()

    settings = _load_settings(REPO_ROOT / "src" / "config" / "settings.yaml")
    models = _load_enabled_models(settings)
    groups = _group_models(models)

    if not groups:
        raise RuntimeError("No enabled benchmark models found in settings.yaml")

    outer_run_id = args.run_id or _timestamp_run_id()
    outer_run_root = (REPO_ROOT / args.output_root).resolve() / outer_run_id
    staging_root = outer_run_root / "_staging"
    outer_run_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    print(f"Parallel benchmark root: {outer_run_root}")
    print(f"Parallel staging root: {staging_root}")
    print(f"Planned groups: {len(groups)}")

    for group in groups:
        model_list = ", ".join(model.name for model in group.models)
        print(f"- {group.name} (backend={group.backend}): {model_list}")

    if args.dry_run:
        return

    if args.spawn_dry_run:
        print("Spawn diagnostic mode: child benchmark processes will run with --dry-run")

    max_workers = max(1, len(groups))

    output_root = str(staging_root)
    failures: list[tuple[str, BaseException]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_group, group, output_root, forwarded_args, args.spawn_dry_run)
            for group in groups
        ]

        for future in as_completed(futures):
            try:
                group_dir_name, group_name = future.result()
                src_dir = staging_root / group_dir_name
                _move_directory_contents(src_dir, outer_run_root, group_dir_name)
                print(f"[DONE] {group_name}")
            except BaseException as exc:
                failures.append(("group", exc))
                print(f"[FAILED] {exc}")

    if failures:
        raise RuntimeError(f"{len(failures)} parallel benchmark group(s) failed")


if __name__ == "__main__":
    main()
