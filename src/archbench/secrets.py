"""Local provider credentials, kept out of every piece of evidence.

Credentials live in a project-local YAML file (default `secrets.local.yaml`, see `secrets.example.yaml`).
The file is loaded only when it is safe:
- a regular file owned by the current user and readable by nobody else (mode 600 or stricter);
- inside a Git work tree, ignored by Git and not tracked.

Loaded values are handed directly to adapters (never exported to `os.environ`, so worker subprocesses do
not inherit them) and registered with a process-wide guard. Artifact writes, database records, and
frozen configuration files are checked against the guard and against key-shaped strings; a match aborts
the write instead of persisting the secret.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

DEFAULT_SECRETS_FILE = "secrets.local.yaml"
PROVIDERS = ("anthropic", "deepseek", "google", "mistral", "openai", "typesafe", "xai", "zhipu")
KEY_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),  # Google API keys
    re.compile(r"\bxai-[A-Za-z0-9]{20,}"),  # xAI API keys
)


class SecretsError(RuntimeError):
    """The secrets file exists but is unsafe or malformed. Nothing from it was loaded."""


class SecretLeakError(RuntimeError):
    """A write would have persisted a credential."""


class ProviderCredentials(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr = Field(min_length=8)


class LocalSecrets(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    anthropic: ProviderCredentials | None = None
    deepseek: ProviderCredentials | None = None
    google: ProviderCredentials | None = None
    mistral: ProviderCredentials | None = None
    openai: ProviderCredentials | None = None
    typesafe: ProviderCredentials | None = None
    xai: ProviderCredentials | None = None
    zhipu: ProviderCredentials | None = None
    source: str | None = Field(default=None, exclude=True)

    def api_key(self, provider: str) -> str | None:
        creds = getattr(self, provider, None)
        return creds.api_key.get_secret_value() if isinstance(creds, ProviderCredentials) else None


class SecretGuard:
    """Process-wide set of credential values that must never be persisted."""

    def __init__(self) -> None:
        self._values: set[str] = set()

    def register(self, value: str) -> None:
        if value:
            self._values.add(value)

    def clear(self) -> None:
        self._values.clear()

    def find(self, data: bytes | str) -> str | None:
        text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
        if any(v in text for v in self._values):
            return "a loaded credential value"
        if any(p.search(text) for p in KEY_PATTERNS):
            return "an API-key-shaped string"
        return None

    def check(self, data: bytes | str, where: str) -> None:
        found = self.find(data)
        if found is not None:
            raise SecretLeakError(f"refusing to persist {where}: it contains {found}")

    def scrub(self, text: str) -> str:
        for value in self._values:
            text = text.replace(value, "[REDACTED]")
        return text


GUARD = SecretGuard()


def _git(project: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(["git", "-C", str(project), *args], capture_output=True, check=False)
    except OSError:
        return None


def check_file_safety(path: Path, project: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise SecretsError(f"{path} is a symlink; use a regular file")
    if not stat.S_ISREG(info.st_mode):
        raise SecretsError(f"{path} is not a regular file")
    if info.st_uid != os.getuid():
        raise SecretsError(f"{path} is not owned by the current user")
    if info.st_mode & 0o077:
        raise SecretsError(f"{path} is readable or writable by others (mode {oct(info.st_mode & 0o777)}); "
                           f"run: chmod 600 {path}")
    inside = _git(project, "rev-parse", "--is-inside-work-tree")
    if inside is None or inside.returncode != 0:
        return  # not a Git work tree: nothing can commit it from here
    tracked = _git(project, "ls-files", "--error-unmatch", "--", str(path))
    if tracked is not None and tracked.returncode == 0:
        raise SecretsError(f"{path} is tracked by Git; remove it from the index (git rm --cached) and rotate the key")
    ignored = _git(project, "check-ignore", "-q", "--", str(path))
    if ignored is None or ignored.returncode != 0:
        raise SecretsError(f"{path} is not ignored by Git; add it to .gitignore before storing keys in it")


def load_local_secrets(project: Path, path: Path | None = None) -> LocalSecrets:
    """Load and register credentials. A missing file means no local credentials (SDK defaults apply)."""
    project = project.resolve()
    target = path if path is not None else project / DEFAULT_SECRETS_FILE
    target = target if target.is_absolute() else project / target
    if not target.exists() and not target.is_symlink():
        if path is not None:
            raise SecretsError(f"secrets file {target} does not exist")
        return LocalSecrets()
    check_file_safety(target, project)
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        secrets = LocalSecrets.model_validate({**raw, "source": str(target)})
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        # Never echo file content: validation errors can quote the offending value.
        raise SecretsError(f"{target} is malformed ({type(exc).__name__}); see secrets.example.yaml") from None
    for provider in PROVIDERS:
        if (value := secrets.api_key(provider)) is not None:
            GUARD.register(value)
    return secrets
