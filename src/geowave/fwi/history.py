from __future__ import annotations
from dataclasses import dataclass, asdict
import csv
import json
import time
from pathlib import Path

@dataclass(frozen=True)
class IterationRecord:
    """One row of the objective-function log (one per gradient evaluation)."""
    band_index: int
    band_lowpass_hz: float
    iteration: int
    objective: float
    step_length: float
    gradient_max_abs: float
    model_min: float
    model_max: float
    accepted: bool
    snapshot: str
    note: str = ''

class RunHistory:
    """CSV objective log + JSONL trial log + human-readable run.log."""

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._csv = open(self.workdir / 'objective_log.csv', 'w', newline='')
        self._writer = csv.DictWriter(self._csv, fieldnames=list(IterationRecord.__dataclass_fields__))
        self._writer.writeheader()
        self._trials = open(self.workdir / 'linesearch_trials.jsonl', 'w')
        self._log = open(self.workdir / 'run.log', 'w')
        self.records: list[IterationRecord] = []

    def log_record(self, record: IterationRecord) -> None:
        self._writer.writerow(asdict(record))
        self._csv.flush()
        self.records.append(record)
        self.message(f'band={record.band_index} iter={record.iteration} '
                     f'J={record.objective:.6e} step={record.step_length:.4g} '
                     f'accepted={record.accepted} {record.note}')

    def log_trial(self, info: dict) -> None:
        self._trials.write(json.dumps(info, default=float) + '\n')
        self._trials.flush()

    def message(self, text: str) -> None:
        line = f'[{time.strftime("%H:%M:%S")}] {text}'
        print(line)
        self._log.write(line + '\n')
        self._log.flush()

    def close(self) -> None:
        for fh in (self._csv, self._trials, self._log):
            fh.close()

def write_manifest(workdir: str | Path, manifest: dict) -> Path:
    path = Path(workdir) / 'run_manifest.json'
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
