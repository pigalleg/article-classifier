from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv


@dataclass
class AffinityTimingLogger:
    """Append per-abstract RA/PRP timing events to a CSV log."""

    output_path: Path

    @classmethod
    def from_results_dir(cls, results_dir: str | Path) -> "AffinityTimingLogger":
        path = Path(results_dir) / "affinity_timing_log.csv"
        return cls(output_path=path)

    def _ensure_header(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            return
        with self.output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp_utc",
                    "stage",
                    "abstract_index",
                    "document_title",
                    "targets_evaluated",
                    "records_written",
                    "elapsed_seconds",
                    "ms_per_target",
                ],
            )
            writer.writeheader()

    def log_event(
        self,
        stage: str,
        abstract_index,
        document_title: str,
        targets_evaluated: int,
        records_written: int,
        elapsed_seconds: float,
    ) -> None:
        # Keep logging non-blocking for the main pipeline.
        try:
            self._ensure_header()
            ms_per_target = (elapsed_seconds * 1000.0 / targets_evaluated) if targets_evaluated > 0 else None
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "abstract_index": abstract_index,
                "document_title": document_title,
                "targets_evaluated": int(targets_evaluated),
                "records_written": int(records_written),
                "elapsed_seconds": round(float(elapsed_seconds), 6),
                "ms_per_target": None if ms_per_target is None else round(ms_per_target, 3),
            }
            with self.output_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writerow(row)
        except Exception:
            # Logging should never fail the affinity run.
            return
