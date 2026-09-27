from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np


class RunArchive:
    """On-disk record of an FWI run.

    Layout under ``root``::

        manifest.json            run configuration, status, abort reason
        history.jsonl            one line per completed iteration
        snapshots/iter_*.npz     velocity snapshot per iteration
    """

    def __init__(self, root: str | Path, config: dict | None = None):
        self.root = Path(root)
        self.snapshot_dir = self.root / 'snapshots'
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / 'history.jsonl'
        self.manifest_path = self.root / 'manifest.json'
        self.run_id = time.strftime('fwi-%Y%m%d-%H%M%S')
        manifest = {
            'run_id': self.run_id,
            'created_at': time.time(),
            'status': 'running',
            'config': config or {},
        }
        self._write_manifest(manifest)

    def _write_manifest(self, manifest: dict) -> None:
        tmp = self.manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        tmp.replace(self.manifest_path)

    def load_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def save_snapshot(self, iteration: int, band_label: str, velocity: np.ndarray,
                      extra: dict | None = None) -> Path:
        path = self.snapshot_dir / f'iter_{iteration:04d}.npz'
        np.savez_compressed(path, velocity=np.asarray(velocity, dtype=float),
                            band_label=np.array(band_label),
                            **{f'meta_{k}': np.asarray(v) for k, v in (extra or {}).items()})
        return path

    def load_snapshot(self, iteration: int) -> np.ndarray:
        path = self.snapshot_dir / f'iter_{iteration:04d}.npz'
        with np.load(path) as z:
            return z['velocity'].copy()

    def latest_snapshot(self) -> np.ndarray | None:
        files = sorted(self.snapshot_dir.glob('iter_*.npz'))
        if not files:
            return None
        with np.load(files[-1]) as z:
            return z['velocity'].copy()

    def append_history(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, allow_nan=False, default=_json_default)
        with self.history_path.open('a') as f:
            f.write(line + '\n')

    def load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in self.history_path.read_text().splitlines() if line.strip()]

    def finalize(self, status: str, reason: str = '', extra: dict | None = None) -> None:
        manifest = self.load_manifest()
        manifest.update({'status': status, 'reason': reason, 'finished_at': time.time()})
        if extra:
            manifest.update(extra)
        self._write_manifest(manifest)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'not JSON serializable: {type(obj)!r}')
