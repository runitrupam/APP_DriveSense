"""Run directory management, streaming writers and packaging."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.utils.hashing import stable_json_hash
from src.utils.run_id import make_run_id


def _json_default(value: Any) -> Any:
    """JSON fallback for numpy scalars and Paths."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write JSON via a temporary file and an atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a JSON Lines file, skipping blank lines."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
            if limit is not None and len(records) >= limit:
                break
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSON Lines file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


class JSONLWriter:
    """Append-only JSON Lines writer that keeps memory flat."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, record: Any) -> None:
        """Serialise one record to a single line."""
        self._handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
        self._handle.write("\n")
        self.count += 1

    def flush(self) -> None:
        """Flush buffered data to disk."""
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        """Flush and close the file."""
        if not self._handle.closed:
            self.flush()
            self._handle.close()

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RunStore:
    """Owns the persistent layout of a single pipeline execution."""

    SUBDIRS = (
        "input",
        "frames",
        "detection",
        "tracking",
        "motion",
        "temporal",
        "validation",
        "final",
        "metrics",
        "annotated",
        "logs",
    )

    def __init__(self, run_id: str, root: str | Path, config_hash: str | None = None) -> None:
        self.run_id = run_id
        self.root = Path(root)
        self.run_dir = self.root / run_id
        self.config_hash = config_hash
        self._open_writers: dict[Path, JSONLWriter] = {}
        self._create_layout()

    @classmethod
    def create(
        cls,
        root: str | Path,
        run_id: str | None = None,
        config_hash: str | None = None,
    ) -> "RunStore":
        """Create a new run directory using a fresh run id unless one is given."""
        return cls(run_id or make_run_id(), root, config_hash=config_hash)

    def _create_layout(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self.SUBDIRS:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def path(self, subdir: str, name: str | None = None) -> Path:
        """Absolute path inside a run subdirectory."""
        if subdir not in self.SUBDIRS:
            raise KeyError(f"unknown run subdirectory: {subdir}")
        base = self.run_dir / subdir
        return base / name if name else base

    def writer(self, subdir: str, name: str) -> JSONLWriter:
        """Return a long-lived JSONL writer for a file, creating it once."""
        target = self.path(subdir, name)
        if target not in self._open_writers:
            self._open_writers[target] = JSONLWriter(target)
        return self._open_writers[target]

    def write_json(self, subdir: str, name: str, payload: Any) -> Path:
        """Atomically write a JSON document inside a subdirectory."""
        return write_json_atomic(self.path(subdir, name), payload)

    def append_jsonl(self, subdir: str, name: str, record: Any) -> None:
        """Append one record using the managed writer."""
        self.writer(subdir, name).write(record)

    def write_text(self, subdir: str, name: str, text: str) -> Path:
        """Write a text file."""
        target = self.path(subdir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def copy_input(self, source: str | Path) -> Path:
        """Copy the input video into the run for provenance."""
        import shutil

        source_path = Path(source)
        target = self.path("input", "source" + source_path.suffix)
        shutil.copy2(source_path, target)
        return target

    def close(self) -> None:
        """Flush and close every managed writer."""
        for writer in self._open_writers.values():
            writer.close()
        self._open_writers.clear()

    def inventory(self) -> dict[str, Any]:
        """File counts and byte sizes per subdirectory."""
        inventory: dict[str, Any] = {}
        total_files = 0
        total_bytes = 0
        for subdir in self.SUBDIRS:
            directory = self.run_dir / subdir
            files = [p for p in directory.rglob("*") if p.is_file()]
            size = int(sum(p.stat().st_size for p in files))
            inventory[subdir] = {"files": len(files), "bytes": size}
            total_files += len(files)
            total_bytes += size
        inventory["total"] = {"files": total_files, "bytes": total_bytes}
        return inventory

    def list_files(self) -> list[Path]:
        """Every file currently in the run directory."""
        return sorted(p for p in self.run_dir.rglob("*") if p.is_file())

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        """Write ``run_manifest.json`` with an inventory snapshot merged in."""
        manifest = dict(payload)
        manifest.setdefault("run_id", self.run_id)
        manifest["run_dir"] = str(self.run_dir)
        if self.config_hash:
            manifest.setdefault("config_hash", self.config_hash)
        manifest["inventory"] = self.inventory()
        return write_json_atomic(self.run_dir / "run_manifest.json", manifest)

    def package_zip(
        self,
        destination: str | Path | None = None,
        exclude_dirs: Iterable[str] = ("frames",),
    ) -> Path:
        """Zip the run, skipping large directories by default."""
        target = Path(destination) if destination else self.run_dir / f"{self.run_id}.zip"
        skip = set(exclude_dirs)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in self.list_files():
                if file_path == target:
                    continue
                relative = file_path.relative_to(self.run_dir)
                if relative.parts and relative.parts[0] in skip:
                    continue
                archive.write(file_path, arcname=str(Path(self.run_id) / relative))
        return target

    @staticmethod
    def payload_hash(payload: Any) -> str:
        """Stable hash helper exposed for stages that need provenance."""
        return stable_json_hash(payload)
