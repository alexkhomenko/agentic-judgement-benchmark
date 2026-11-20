"""Local credentials file: load only when safe, never persist a loaded key anywhere."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from archbench.adapters.registry import default_registry
from archbench.domain.records import Record, RecordKind
from archbench.planning.loader import freeze_file
from archbench.secrets import GUARD, SecretLeakError, SecretsError, load_local_secrets
from archbench.storage.artifacts import ArtifactStore
from archbench.storage.database import Database

KEY = "sk-ant-api03-test-0123456789abcdefghij"


def repo(tmp_path: Path, *, ignore: bool = True) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    if ignore:
        (project / ".gitignore").write_text("secrets.local.yaml\n")
    return project


def write_secrets(project: Path, content: str | None = None, mode: int = 0o600) -> Path:
    path = project / "secrets.local.yaml"
    path.write_text(content if content is not None else f'anthropic:\n  api_key: "{KEY}"\n')
    path.chmod(mode)
    return path


def test_missing_default_file_means_no_local_credentials(tmp_path: Path) -> None:
    secrets = load_local_secrets(repo(tmp_path))
    assert secrets.api_key("anthropic") is None
    with pytest.raises(SecretsError, match="does not exist"):
        load_local_secrets(tmp_path / "project", Path("elsewhere.yaml"))


def test_safe_file_loads_without_exposing_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    project = repo(tmp_path)
    write_secrets(project)
    secrets = load_local_secrets(project)
    assert secrets.api_key("anthropic") == KEY
    assert KEY not in repr(secrets) and KEY not in str(secrets)
    assert "ANTHROPIC_API_KEY" not in os.environ  # handed to adapters, never exported to subprocesses
    adapter = default_registry(secrets).get("anthropic")
    assert adapter._api_key == KEY  # type: ignore[attr-defined]
    assert adapter.preflight(  # type: ignore[attr-defined]
        __import__("archbench.domain.configuration", fromlist=["x"]).ModelConfiguration(
            configuration_id="j", revision=1, track="cloud", adapter="anthropic", provider="anthropic", model="m")
    ) is None


def test_documented_template_copy_loads(tmp_path: Path) -> None:
    """The README setup (copy secrets.example.yaml) must load: every provider in it needs a credentials field."""
    project = repo(tmp_path)
    write_secrets(project, (Path(__file__).parents[1] / "secrets.example.yaml").read_text())
    secrets = load_local_secrets(project)
    assert secrets.api_key("anthropic") == "<your Anthropic API key>"
    assert secrets.api_key("zhipu") == "<your Z.ai API key>"
    assert secrets.api_key("minimax") == "<your MiniMax API key>"
    assert secrets.api_key("cohere") == "<your Cohere API key>"
    assert secrets.api_key("bedrock") == "<your Amazon Bedrock API key>"
    assert secrets.api_key("meta") == "<your Meta Model API key>"
    assert secrets.api_key("microsoft") == "<your Microsoft Foundry key>"
    assert secrets.api_key("openrouter") == "<your OpenRouter API key>"


def test_readable_by_others_is_refused(tmp_path: Path) -> None:
    project = repo(tmp_path)
    write_secrets(project, mode=0o644)
    with pytest.raises(SecretsError, match="chmod 600"):
        load_local_secrets(project)
    assert GUARD.find(KEY) == "an API-key-shaped string"  # nothing registered; pattern check still applies


def test_not_ignored_or_tracked_is_refused(tmp_path: Path) -> None:
    project = repo(tmp_path, ignore=False)
    write_secrets(project)
    with pytest.raises(SecretsError, match="not ignored"):
        load_local_secrets(project)
    subprocess.run(["git", "-C", str(project), "add", "-f", "secrets.local.yaml"], check=True)
    (project / ".gitignore").write_text("secrets.local.yaml\n")
    with pytest.raises(SecretsError, match="tracked by Git"):
        load_local_secrets(project)


def test_symlink_is_refused(tmp_path: Path) -> None:
    project = repo(tmp_path)
    real = write_secrets(project)
    real.rename(project / "real.yaml")
    (project / "secrets.local.yaml").symlink_to(project / "real.yaml")
    with pytest.raises(SecretsError, match="symlink"):
        load_local_secrets(project)


def test_malformed_file_error_never_echoes_content(tmp_path: Path) -> None:
    project = repo(tmp_path)
    write_secrets(project, f'anthropic:\n  api_key: "{KEY}"\n  extra: "{KEY}"\n')
    with pytest.raises(SecretsError) as err:
        load_local_secrets(project)
    assert KEY not in str(err.value) and err.value.__cause__ is None


def test_loaded_key_can_never_be_persisted(tmp_path: Path) -> None:
    project = repo(tmp_path)
    write_secrets(project)
    load_local_secrets(project)
    store = ArtifactStore(tmp_path / "results")
    with pytest.raises(SecretLeakError):
        store.write("exp/leak.txt", f"error: auth failed for {KEY}")
    assert not (tmp_path / "results" / "exp" / "leak.txt").exists()
    db = Database(tmp_path / "results" / "index.sqlite")
    with pytest.raises(SecretLeakError):
        db.insert_record(Record.create(kind=RecordKind.VALIDATION, payload={"note": KEY}))
    assert db.list_records() == []
    db.close()
    from archbench.adapters.redaction import redact_text

    assert KEY not in redact_text(f"x-api-key {KEY}")


def test_config_containing_a_key_cannot_be_frozen(tmp_path: Path) -> None:
    config = tmp_path / "judge.yaml"
    config.write_text("model: m\napi_key: sk-ant-api03-pasted-by-mistake-000000\n")
    with pytest.raises(SecretLeakError, match="never in configs"):
        freeze_file(config, tmp_path)


def test_cli_refuses_unsafe_secrets_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from archbench.cli import main

    project = repo(tmp_path)
    write_secrets(project, mode=0o640)
    assert main(["--project", str(project), "--results", str(tmp_path / "r"), "status", "x"]) == 2
    err = capsys.readouterr().err
    assert "chmod 600" in err and KEY not in err


@pytest.mark.parametrize(("provider", "adapter_name", "env_var"), [
    ("typesafe", "typesafe", "TYPESAFE_API_KEY"),
    ("zhipu", "glm", "ZAI_API_KEY"),
    ("minimax", "minimax", "MINIMAX_API_KEY"),
    ("cohere", "cohere", "CO_API_KEY"),
    ("bedrock", "bedrock", "AWS_BEARER_TOKEN_BEDROCK"),
    ("meta", "meta", "MODEL_API_KEY"),
    ("microsoft", "mai", "AZURE_INFERENCE_CREDENTIAL"),
    ("openrouter", "openrouter", "OPENROUTER_API_KEY"),
])
def test_provider_key_from_file_or_environment_is_guarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, adapter_name: str, env_var: str,
) -> None:
    local_key = f"{provider}-local-file-key-123"
    environment_key = f"{provider}-environment-key-456"
    monkeypatch.setenv(env_var, environment_key)
    project = repo(tmp_path)
    write_secrets(project, f'{provider}:\n  api_key: "{local_key}"\n')
    secrets = load_local_secrets(project)
    assert secrets.api_key(provider) == local_key
    assert local_key not in repr(secrets) and local_key not in str(secrets)
    assert default_registry(secrets).get(adapter_name)._api_key == local_key  # type: ignore[attr-defined]
    assert GUARD.find(f"leak {local_key}") == "a loaded credential value"
    assert os.environ[env_var] == environment_key  # the file key was not exported
    GUARD.clear()
    assert default_registry().get(adapter_name)._api_key == environment_key  # type: ignore[attr-defined]
    assert GUARD.find(f"leak {environment_key}") == "a loaded credential value"
