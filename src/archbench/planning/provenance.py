"""Source revision, dependency versions, and host metadata."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from ..domain.plan import SourceRevision


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout


def source_revision(repo: Path) -> SourceRevision:
    try:
        commit = _git(repo, "rev-parse", "HEAD").decode().strip()
        status = _git(repo, "status", "--porcelain", "--untracked-files=all")
        diff = _git(repo, "diff", "HEAD", "--binary")
    except (OSError, subprocess.CalledProcessError):
        return SourceRevision(available=False)
    dirty = bool(status.strip())
    return SourceRevision(
        available=True,
        commit=commit,
        dirty=dirty,
        dirty_state_sha256=hashlib.sha256(status + b"\0" + diff).hexdigest() if dirty else None,
    )


def dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for dist in ("archdecisionbench", "pydantic", "pydantic-core", "PyYAML", "anthropic", "httpx2"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = "not-installed"
    return versions


def host_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    if platform.system() == "Darwin":
        for key, name in (("chip", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
            try:
                value = subprocess.run(["sysctl", "-n", name], check=True, capture_output=True, text=True).stdout.strip()
                info[key] = int(value) if key == "memory_bytes" else value
            except (OSError, subprocess.CalledProcessError, ValueError):
                info[key] = None
        try:
            info["os_build"] = subprocess.run(
                ["sw_vers", "-buildVersion"], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            info["os_build"] = None
    return info
