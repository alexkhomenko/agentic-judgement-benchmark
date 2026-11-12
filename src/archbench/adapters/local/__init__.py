"""Local adapters (implementation step 2).

Implemented: Ollama (`ollama_chat`). The Ollama server is the isolated inference worker; the adapter pins
the checkpoint by digest and the engine by version, checks the digest before every call, and records
engine timing and model memory.

Known gaps against architecture.md: Ollama's chat API does not expose the rendered prompt or the input
token IDs, so neither is captured (the prompt-template renderer is identified by the checkpoint digest
and engine version), and there is no host memory-pressure or swap sampler yet. A direct MLX worker would
close both.
"""
