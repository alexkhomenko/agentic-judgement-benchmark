"""Immutable artifact files.

Writes go through a temporary file and a no-clobber hard link, so a published path can never be
overwritten. Published files are made read-only. Every read verifies the recorded hash.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..domain.common import ARTIFACT_SCHEMA_VERSION, sha256_bytes
from ..domain.execution import ArtifactRef
from ..secrets import GUARD

TEMP_MARKER = ".tmp-"
LOCK_NAME = ".lock"


class ArtifactExistsError(FileExistsError):
    pass


class ArtifactIntegrityError(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relpath: str) -> Path:
        candidate = (self.root / relpath).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"artifact path escapes the results root: {relpath}")
        return candidate

    def experiment_dir(self, experiment_id: str) -> Path:
        path = self.path(experiment_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, relpath: str, data: bytes | str, *, name: str | None = None) -> ArtifactRef:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        GUARD.check(payload, f"artifact {relpath}")
        final = self.path(relpath)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(f"{TEMP_MARKER}{final.name}.{uuid.uuid4().hex}")
        try:
            with open(tmp, "xb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            try:
                os.link(tmp, final)  # atomic and fails if the published path already exists
            except FileExistsError as exc:
                raise ArtifactExistsError(f"refusing to overwrite published artifact {relpath}") from exc
            _fsync_dir(final.parent)
        finally:
            tmp.unlink(missing_ok=True)
        return ArtifactRef(
            name=name or final.name,
            relpath=relpath,
            sha256=sha256_bytes(payload),
            size_bytes=len(payload),
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )

    def write_json(self, relpath: str, value: Any, *, name: str | None = None) -> ArtifactRef:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        return self.write(relpath, text, name=name)

    def adopt(self, relpath: str, *, name: str | None = None) -> ArtifactRef:
        """Describe an already-written file (used when reconciling interrupted attempts)."""
        data = self.path(relpath).read_bytes()
        return ArtifactRef(
            name=name or Path(relpath).name,
            relpath=relpath,
            sha256=sha256_bytes(data),
            size_bytes=len(data),
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )

    def read_verified(self, ref: ArtifactRef) -> bytes:
        try:
            data = self.path(ref.relpath).read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(f"artifact missing: {ref.relpath}") from exc
        if sha256_bytes(data) != ref.sha256:
            raise ArtifactIntegrityError(f"artifact hash mismatch: {ref.relpath}")
        return data

    def read_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.read_verified(ref))

    def sink(self, prefix: str) -> DirectorySink:
        return DirectorySink(self, prefix)

    @contextmanager
    def lock(self, experiment_id: str) -> Iterator[None]:
        """Exclusive controller lease for one experiment."""
        path = self.experiment_dir(experiment_id) / LOCK_NAME
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another controller holds the lock for {experiment_id}") from exc
            yield
        finally:
            os.close(fd)


class DirectorySink:
    """Collects artifacts for one owner (attempt, packet, report); refs are registered by the caller."""

    def __init__(self, store: ArtifactStore, prefix: str) -> None:
        self._store = store
        self._prefix = prefix.rstrip("/")
        self.refs: list[ArtifactRef] = []

    def _relpath(self, name: str) -> str:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"invalid artifact name: {name!r}")
        return f"{self._prefix}/{name}"

    def write(self, name: str, data: bytes | str) -> ArtifactRef:
        ref = self._store.write(self._relpath(name), data, name=name)
        self.refs.append(ref)
        return ref

    def write_json(self, name: str, value: Any) -> ArtifactRef:
        ref = self._store.write_json(self._relpath(name), value, name=name)
        self.refs.append(ref)
        return ref


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
