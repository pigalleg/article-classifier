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
                    "s_per_target",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
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
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        # Keep logging non-blocking for the main pipeline.
        try:
            self._ensure_header()
            s_per_target = (elapsed_seconds / targets_evaluated) if targets_evaluated > 0 else None
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "abstract_index": abstract_index,
                "document_title": document_title,
                "targets_evaluated": int(targets_evaluated),
                "records_written": int(records_written),
                "elapsed_seconds": round(float(elapsed_seconds), 6),
                "s_per_target": None if s_per_target is None else round(s_per_target, 6),
                "input_tokens": None if input_tokens is None else int(input_tokens),
                "output_tokens": None if output_tokens is None else int(output_tokens),
                "total_tokens": None if total_tokens is None else int(total_tokens),
            }
            with self.output_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writerow(row)
        except Exception:
            # Logging should never fail the affinity run.
            return
