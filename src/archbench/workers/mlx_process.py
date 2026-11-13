"""Controller side of the MLX worker: process lifecycle and the JSON-lines request channel.

The worker gets a minimal environment: no provider keys, Hugging Face offline mode, no telemetry. Its
stderr goes to a private temporary log whose tail is attached to failures.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

WORKER_SCRIPT = Path(__file__).with_name("mlx_worker.py")
PASSTHROUGH_ENV = ("PATH", "HOME", "TMPDIR", "LANG", "HF_HOME", "HF_HUB_CACHE")
WORKER_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
              "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
STDERR_TAIL_BYTES = 4000


class WorkerExited(Exception):
    def __init__(self, status: dict[str, Any]) -> None:
        super().__init__(f"worker exited: {status}")
        self.status = status


class WorkerTimeout(Exception):
    pass


class WorkerProcess(Protocol):
    @property
    def pid(self) -> int | None: ...

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]: ...

    def close(self) -> None: ...


class SubprocessWorker:
    """`python -I mlx_worker.py` in the pinned worker environment."""

    def __init__(self, python: str, *, script: Path = WORKER_SCRIPT) -> None:
        interpreter = Path(python).expanduser()
        if not interpreter.exists():
            raise FileNotFoundError(f"worker interpreter {interpreter} does not exist")
        env = {k: v for k in PASSTHROUGH_ENV if (v := os.environ.get(k)) is not None}
        env.update(WORKER_ENV)
        self._log = tempfile.TemporaryFile("w+b")
        self._proc = subprocess.Popen([str(interpreter), "-I", str(script)], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=self._log, env=env, close_fds=True)
        self._buffer = b""
        self.command = [str(interpreter), "-I", str(script)]

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc.poll() is None else None

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write((json.dumps(message) + "\n").encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise WorkerExited(self.exit_status()) from None
        deadline = time.monotonic() + timeout_s
        fd = self._proc.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.kill()
                    raise WorkerTimeout(f"no reply to {message.get('op')!r} within {timeout_s}s; worker killed")
                if not selector.select(remaining):
                    continue
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    raise WorkerExited(self.exit_status())
                self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        reply: dict[str, Any] = json.loads(line)
        return reply

    def exit_status(self) -> dict[str, Any]:
        try:
            code = self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = None
        status: dict[str, Any] = {"returncode": code, "stderr_tail": self.stderr_tail()}
        if code is not None and code < 0:
            status["signal"] = signal.Signals(-code).name
        return status

    def stderr_tail(self) -> str:
        self._log.flush()
        size = self._log.seek(0, os.SEEK_END)
        self._log.seek(max(0, size - STDERR_TAIL_BYTES))
        return self._log.read().decode(errors="replace")

    def kill(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=10)

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self.request({"op": "shutdown"}, timeout_s=10)
            except (WorkerExited, WorkerTimeout, OSError, ValueError):
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill()
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                stream.close()
        self._log.close()
