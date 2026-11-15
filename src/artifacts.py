"""Validated file replacement, rollback, provenance and content fingerprints.

Writers stage complete outputs before publishing. Ordinary exceptions roll back
all replacements. A crash leaves a marker: readers fail closed rather than show
a mixed run. This is a local single-writer protocol, not a distributed transaction.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class ArtifactError(ValueError):
    """Missing, stale or inconsistent output bundle."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(paths):
    """Content-based cache key, including absence and resolved path identity."""
    return tuple((str(Path(p).resolve()), digest(p) if Path(p).is_file() else None)
                 for p in paths)


def assert_idle(directory):
    marker = Path(directory) / ".update-in-progress"
    if marker.exists():
        raise ArtifactError(f"Update in progress or interrupted: {marker}. "
                            "Do not read mixed outputs; inspect/recover the staged backup first.")


@contextmanager
def atomic_path(path, validator=None):
    """Write into a sibling temporary file; validate before replacing the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        yield temp
        if not temp.is_file() or temp.stat().st_size == 0:
            raise ArtifactError(f"Empty output: {path}")
        if validator:
            validator(temp)
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def manifest(stage, inputs, outputs, **metadata):
    packages = ["pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "pyarrow",
                "matplotlib", "plotly", "streamlit", "requests"]
    source_root = Path(__file__).resolve().parent.parent
    source_files = [*sorted((source_root / "src").glob("*.py")), source_root / "main.py"]
    return {
        "schema_version": 1, "stage": stage, "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in packages},
        "source_sha256": {str(p.relative_to(source_root)): digest(p) for p in source_files},
        "inputs": {str(name): digest(path) for name, path in inputs.items()},
        "outputs": {str(name): digest(path) for name, path in outputs.items()},
        **metadata,
    }


def verify_manifest(directory, name):
    directory = Path(directory)
    assert_idle(directory)
    path = directory / name
    if not path.exists():
        return None  # committed pre-modernization artifacts remain readable
    value = json.loads(path.read_text(encoding="utf-8"))
    for filename, expected in value["outputs"].items():
        output = directory / filename
        if not output.is_file() or digest(output) != expected:
            raise ArtifactError(f"Artifact does not match {name}: {filename}. Rebuild this stage.")
    return value


def publish_files(files, lock_dir):
    """Publish prevalidated staged files; restore ALL old files on any exception.

    Put the manifest last in the mapping. Backups survive process termination in
    a sibling directory named in the marker. Concurrent writers are rejected.
    """
    files = {Path(dst): Path(src) for dst, src in files.items()}
    for src in files.values():
        if not src.is_file() or not src.stat().st_size:
            raise ArtifactError(f"Missing/empty staged output: {src}")
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    marker = lock_dir / ".update-in-progress"
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        backup_dir = Path(tempfile.mkdtemp(prefix=".artifact-backup-", dir=lock_dir))
    except BaseException:
        os.close(fd)
        marker.unlink()
        raise
    backups, replaced = {}, []
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"backup": str(backup_dir), "files": [str(p) for p in files],
                       "existed": [p.exists() for p in files]}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        for i, dst in enumerate(files):
            dst.parent.mkdir(parents=True, exist_ok=True)
            backups[dst] = backup_dir / str(i) if dst.exists() else None
            if backups[dst] is not None:
                shutil.copy2(dst, backups[dst])
        # A durable flag distinguishes complete backups from an interrupted copy.
        (backup_dir / "ready").write_text("Backups complete; replacement may have started\n")
        for dst, src in files.items():
            # Copy into the destination filesystem before atomic replacement.
            with atomic_path(dst) as temp:
                shutil.copyfile(src, temp)
            replaced.append(dst)
    except BaseException:
        try:
            for dst in reversed(replaced):
                if backups[dst] is None:
                    dst.unlink(missing_ok=True)
                else:
                    os.replace(backups[dst], dst)
        except BaseException:
            # Keep recovery information if rollback itself fails.
            raise
        else:
            shutil.rmtree(backup_dir)
            marker.unlink()
        raise
    else:
        shutil.rmtree(backup_dir)
        marker.unlink()
