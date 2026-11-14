"""Local adapters (implementation step 2).

Implemented:
- Ollama (`ollama_chat`). The Ollama server is the isolated inference worker; the adapter pins the
  checkpoint by digest and the engine by version, checks the digest before every call, and records engine
  timing and model memory. Ollama's chat API does not expose the rendered prompt or the input token IDs,
  so neither is captured for this adapter.
- MLX-LM (`mlx_lm`). A worker subprocess in a pinned environment (`archbench.workers`) runs mlx-lm
  directly. The adapter pins the checkpoint by repository, commit, and per-file sha256 and pins the engine by
  package versions. It records the rendered prompt, input token IDs, KV-cache size, and MLX memory, and it
  samples host memory pressure and swap from the controller.
"""
