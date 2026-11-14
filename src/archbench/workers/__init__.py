"""Local inference workers (architecture.md section 4): process lifecycle, warm-up, model reuse, and
resource monitoring.

- `mlx_worker.py` runs inside the pinned MLX environment (`.venv-mlx`); it never imports archbench.
- `mlx_process.py` starts it and carries requests over JSON lines.
- `host_memory.py` samples host memory pressure, swap, and the worker's footprint from the controller,
  so evidence survives a worker that dies.
"""
