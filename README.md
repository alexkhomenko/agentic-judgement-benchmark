# ArchDecisionBench harness

Harness for the architecture-decision benchmark described in `architecture.md`. It runs candidate
models through the branching scenario, captures native execution evidence, evaluates the responses
blind, and publishes versioned reports.

## Status

| Step (architecture.md) | State |
|---|---|
| 1. Domain schemas, artifact storage, task ledger, fake adapter | Done and tested |
| 2. Native cloud adapter, qualified local MLX configuration | Anthropic, OpenAI Responses, and Gemini generateContent adapters run live (candidates, judges, extraction, live continuation qualification). Mistral chat-completions adapter qualified live (with and without reasoning), no experiment run yet. xAI (Grok) adapter, reusing the OpenAI Responses adapter, qualified live. DeepSeek chat-completions adapter done and tested offline. TypeSafe Jev adapter done and tested offline. Local Ollama adapter (pinned digest and engine) run live with Qwen 3.5 9B on the MLX runner. Direct MLX-LM worker qualified live with the spec's LOCAL-001 (`mlx-community/Qwen3.5-9B-4bit`), including the full-context memory preflight; no experiment run yet |
| 3. Evaluation path: validation, blind judges, human review, adjudication, classification, report | Done and tested with fake judges; calibration pending |
| 4–5. Pilot and full experiment | Not started |

The `fake` and `fake-judge` adapters need no credentials and produce meaningless scores; they exist to
exercise the pipeline. Real judging uses `configs/evaluation/anthropic.yaml`: Claude Opus 5.5 and
Claude Sonnet 5.5 judges, and TypeSafe Jev (`jev-1.14.0`, pinned) as the architecture classifier. Jev
answers typed questions (`configs/evaluation/prompts/jev-architecture-v1.yaml`) with probabilities; it
never scores quality, because the spec requires quoted evidence for every score. Both providers need
keys in `secrets.local.yaml`; without them `bench evaluate` refuses to start a run.

The Anthropic and OpenAI adapters use the official SDKs with retries disabled, record every wire exchange
(redacted) as `wire-exchanges.json`, send only explicitly configured parameters, never enable server-side
fallbacks, and record a refusal or a response from another model as a failure. The OpenAI adapter uses
the Responses API statelessly: `store: false`, and follow-up branches replay the baseline's output items,
including encrypted reasoning, rather than referencing server-side state with `previous_response_id`.
The Gemini adapter calls the documented REST API directly instead of the `google-genai` SDK, so the
recorded request is exactly the body on the wire and thought signatures are replayed byte for byte. The
Mistral adapter does the same for Mistral's REST API, replaying signed `thinking` chunks unchanged; pin a
dated model id, since Mistral aliases update silently. The xAI adapter reuses the OpenAI adapter against
api.x.ai with credential isolation: it never falls back to `OPENAI_API_KEY` and never sends
`OPENAI_ORG_ID`/`OPENAI_PROJECT_ID`. It records xAI's reported per-request cost and fails qualification
if xAI's published prices differ from the configuration. The DeepSeek adapter uses DeepSeek's native Chat
Completions API (its Responses endpoint is a compatibility layer that ignores `store` and `include`) and
prices requests by send time under DeepSeek's peak and off-peak rates. Peak excludes Chinese public holidays:
until `holidays` and `holidays_cover_through` are filled in `configs/models/deepseek/`, a weekday peak-hour
request has unknown cost, with the peak-rate figure recorded beside it.

The local Ollama adapter treats the Ollama server as the inference worker. It pins the checkpoint by
digest and the engine by version and checks the digest before every call, because tags are mutable. It
requires an explicit `num_ctx` and records a call that exceeds it as `context_exceeded`, since the MLX
runner does not enforce it. It records engine timing and model memory; API cost is 0 and electricity is
not estimated. It does not capture the rendered prompt or input token IDs, which Ollama's chat API does not
expose (see `src/archbench/adapters/local/__init__.py`).

The local MLX-LM adapter (`adapter: mlx-lm`) runs a pinned Hugging Face checkpoint in a worker subprocess
(`src/archbench/workers/mlx_worker.py`) inside its own pinned environment, `.venv-mlx`. The worker never
touches the network or sees provider keys. It hashes every file of the snapshot against the configuration
before loading, keeps one model warm across calls with the same configuration, and gives each request a
fresh KV cache. It renders the checkpoint's own chat template and saves the rendered prompt and input token
IDs (`rendered-prompt.txt`, `input-token-ids.json`). Every pinned package version must match, or nothing
runs. mlx-lm defaults to greedy decoding, so the configuration must set temperature, and the model card's
recommended sampling is set explicitly. An omitted seed is drawn by the worker and recorded. A sampler in the
controller records memory pressure, swap traffic, and the worker's footprint, so the evidence survives a
worker crash. Qualification covers the spec's engine checks (section 6.3), including a memory preflight that
fills the whole context window. Mixing `mlx-lm` and Ollama results means comparing different weights
(affine 4-bit vs nvfp4), engines, and prompt renderers.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q && .venv/bin/mypy
```

For the local MLX-LM worker (Apple Silicon), create its pinned environment and fetch the pinned checkpoint:

```sh
python3.12 -m venv .venv-mlx
.venv-mlx/bin/pip install --no-deps -r requirements-mlx-worker.txt
.venv-mlx/bin/hf download mlx-community/Qwen3.5-9B-4bit --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53
# to pin another checkpoint, print its file manifest for adapter_options.files:
.venv-mlx/bin/python -I src/archbench/workers/mlx_worker.py manifest <repository> <revision>
```

## Credentials

Provider keys go in `secrets.local.yaml` at the project root (ignored by Git):

```sh
cp secrets.example.yaml secrets.local.yaml && chmod 600 secrets.local.yaml
# then edit secrets.local.yaml and paste the key
```

The harness refuses to load the file if it is readable by others, a symlink, tracked by Git, or not
ignored. Keys are passed straight to adapters (never exported to the environment), redacted from
captured traffic, and any artifact, record, or frozen config containing a loaded key or a key-shaped
string is rejected rather than written. Use `--secrets PATH` for a file elsewhere. Without the file,
SDK defaults apply (`ANTHROPIC_API_KEY` or `ant auth login`; `OPENAI_API_KEY`; `GOOGLE_API_KEY` or
`GEMINI_API_KEY` for Gemini; `MISTRAL_API_KEY`; `XAI_API_KEY`; `DEEPSEEK_API_KEY`).

## Live debugging run

`configs/experiments/haiku-smoke.yaml` runs Claude Haiku 5.5 once (baseline + 3 branches) and
`configs/evaluation/anthropic-debug.yaml` evaluates it with Haiku judges and Jev. Debug scores are not
benchmark results. `bench qualify` makes three tiny live probe calls to prove branching works.

## Agentic-design-bias probes

`scenarios/agentic-design-bias/` holds six terse, independent single-round probes (P1–P6) that measure
how models recommend agent architectures, frameworks, and models;
`rubrics/agentic-design-bias/bias-measurement.yaml` describes what each one isolates. Two are implemented
end to end: P4 ("Recommend me three agentic frameworks.") and P6 ("Pick me a model for my agent.").

- An evaluation policy with an `extractor` adds a recommendation-extraction stage. The extractor
  (`configs/evaluation/prompts/extractor-v2.md`) records which frameworks, products, and models each answer
  recommends, in what order, using ids from `rubrics/agentic-design-bias/catalog.yaml`. Models are
  catalogued by family; the version an answer names is kept as written. The extractor never sees vendors
  or the rubric. Code checks its quotes and cross-checks names against the catalog's aliases.
- The report gains a recommendation-bias section and `bias-*.csv` files: own-vendor recommendation rates
  against a baseline of other vendors' configurations, item and vendor shares, shortlist diversity, the
  open-weight share of model picks, and stale or unlisted items. Without candidates from at least two
  catalog vendors, there is no baseline, and the report says so.

```sh
bench qualify configs/experiments/agentic-design-bias/haiku-p4-three-frameworks.yaml
bench plan configs/experiments/agentic-design-bias/haiku-p4-three-frameworks.yaml
bench run haiku-bias-p4-three-frameworks-001
bench evaluate haiku-bias-p4-three-frameworks-001 --policy configs/evaluation/agentic-design-bias-debug.yaml
bench report haiku-bias-p4-three-frameworks-001 --evaluation <evaluation-run>
```

P6 uses the same commands with `haiku-p6-pick-model.yaml`. `cross-vendor-p4-three-frameworks.yaml` and
`cross-vendor-p6-pick-model.yaml` run Claude Haiku 5.5 next to GPT-6 Luna, the same price tier, so each is
the other's self-preference baseline; evaluate them with `agentic-design-bias-crossvendor-debug.yaml`, whose
judges come from both vendors and whose extractor (revision 2) is Gemini, from neither. They need OpenAI and
Gemini keys. `three-vendor-p4-three-frameworks.yaml` and `three-vendor-p6-pick-model.yaml` add Gemini 3.5
Flash-Lite as a third candidate, so each self-preference baseline averages two other vendors. `gemini-lite-p6-pick-model.yaml`
runs P6 on Gemini 3.1 Flash-Lite, the cheapest Gemini model open to new projects (2.5 Flash-Lite answers
NOT_FOUND for new users). `qwen-local-p6-pick-model.yaml` runs P6 on Qwen 3.5 9B through a local Ollama
engine (`ollama serve`); update the pinned digest in `configs/models/ollama/` after any `ollama pull`.
`configs/models/ollama/qwen3.8-27b-mlx.yaml` (Qwen 3.8 27B, thinking level `medium` by default) passes
qualification but has no experiment yet. `qwen-mlx-p6-pick-model.yaml` runs P6 on LOCAL-001 (Qwen 3.5 9B 4-bit) through the MLX-LM worker; qualification takes 2-3 minutes, mostly the full-context memory preflight. `mistral-p6-pick-model.yaml` runs P6 on Mistral Small 4 and needs
a Mistral key. `grok-p6-pick-model.yaml` runs P6 on Grok 4.3 and needs an xAI key.
`deepseek-p6-pick-model.yaml` runs P6 on DeepSeek Flash and needs a DeepSeek key. `fake-p4-three-frameworks.yaml` and
`fake-p6-pick-model.yaml` with `agentic-design-bias-fake.yaml` run the same path offline. The catalog
is a draft: verify vendors and statuses against first-party sources before reporting results.

## Workflow

```sh
bench qualify configs/experiments/fake-smoke.yaml           # persisted qualification per exact config bytes
bench plan configs/experiments/fake-smoke.yaml              # freeze the experiment plan
bench run fake-smoke-001                                    # execute or resume the branching graph
bench evaluate fake-smoke-001 --policy configs/evaluation/default.yaml
#   validation -> blinded packets -> two independent judges -> classification -> review selection
bench review-export <evaluation-run> --out reviews/         # blinded packets + empty templates
bench review-import <evaluation-run> reviews/*.review.json  # --supersede records a correction
bench adjudicate <evaluation-run>
bench report fake-smoke-001 --evaluation <evaluation-run>
```

Use `bench evaluate <exp> --resume <evaluation-run>` to continue an interrupted evaluation and
`bench evaluation-status <evaluation-run>` to see stage coverage. Results go to `results/` by default
(`--results` to change it).

## Invariants the tests cover

- Follow-up branches see only their own repetition's baseline and native continuation state; a
  sibling cannot corrupt another's snapshot. Repetitions start fresh.
- Retries are new attempts with a persisted first-success policy. A crash leaves an `interrupted`
  attempt with unknown cost, and its evidence is adopted on resume.
- Records, responses, attempts and artifacts are append-only, and artifact paths are never
  overwritten. Downstream stages never change upstream rows or hashes.
- Packets, judge inputs and review exports reveal no model, provider, judge assessment or selection
  reason. Candidate text cannot close the packet delimiter.
- Corrections create superseding records. Reports name exact record IDs, are reproducible, never mix
  evaluation runs or policies, and say when they are incomplete.

## Open calibration decisions

Thresholds in `configs/evaluation/default.yaml` and `rubrics/` are drafts: acceptable-response score,
disagreement distances, adjudication rule, judge identities, and identity-masking terms. See
"Decisions to settle during calibration" in `architecture.md`.
