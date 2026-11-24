# ArchDecisionBench harness

Harness for the architecture-decision benchmark described in `architecture.md`. It runs candidate
models through the branching scenario, captures native execution evidence, evaluates the responses
blind, and publishes versioned reports.

## Status

| Step (architecture.md) | State |
|---|---|
| 1. Domain schemas, artifact storage, task ledger, fake adapter | Done and tested |
| 2. Native cloud adapter, qualified local MLX configuration | Anthropic, OpenAI Responses, and Gemini generateContent adapters run live (candidates, judges, extraction, live continuation qualification). Mistral chat-completions adapter qualified live (with and without reasoning), no experiment run yet. xAI (Grok) adapter, reusing the OpenAI Responses adapter, qualified live. DeepSeek chat-completions adapter done and tested offline. Kimi (Moonshot AI) chat-completions adapter done and tested offline. GLM (Z.ai / Zhipu AI) chat-completions adapter done and tested offline. MiniMax (Anthropic-compatible Messages API) adapter done and tested offline. Cohere Chat API v2 adapter done and tested offline. Amazon Bedrock Converse adapter (Amazon Nova models only) done and tested offline. Meta Model API adapter (Muse, reusing the OpenAI Responses adapter) done and tested offline. Microsoft MAI chat-completions adapter (your own Foundry deployment) qualified live with MAI-Thinking-1. OpenRouter adapter (Qwen4-Max pinned to Alibaba's own endpoint) qualified live. TypeSafe Jev adapter done and tested offline. Local Ollama adapter (pinned digest and engine) run live with Qwen 3.5 9B on the MLX runner. Direct MLX-LM worker qualified live with the spec's LOCAL-001 (`mlx-community/Qwen3.5-9B-4bit`), including the full-context memory preflight; no experiment run yet |
| 3. Evaluation path: validation, blind judges, human check of every judgement, review of flagged ones, adjudication, classification, report | Done and tested with fake judges; calibration pending |
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
request has unknown cost, with the peak-rate figure recorded beside it. The Kimi adapter uses Moonshot's
Chat Completions API on api.moonshot.ai and replays each assistant message with its `reasoning_content`, as
K3 models require. It never sends temperature or top_p, which K3 fixes and rejects if changed. It prices
cache reads, cache writes (by TTL, from the `Msh-Usage-Cache-Write-Tokens-*` headers), and misses
separately; a cache write reported without that TTL split has unknown cost.

The GLM adapter (`adapter: glm`, `provider: zhipu`) uses Z.ai's standard Chat Completions endpoint,
rejects Coding Plan endpoints during qualification, and replays complete assistant messages including
`reasoning_content`. It checks the served model id on each response and prices cache hits and misses separately.

The MiniMax adapter (`adapter: minimax`) calls MiniMax's Anthropic-compatible Messages API with plain HTTP, not
the `anthropic` SDK, so no Anthropic credential can reach MiniMax; it never reads `ANTHROPIC_API_KEY`. MiniMax
ids are undated and MiniMax documents silent rerouting of retired ids, so every response's `model` is checked.
Signed thinking blocks are replayed unchanged. A nonzero `base_resp.status_code` in a 200 body is a failure
(1026/1027 sensitive content is a refusal). MiniMax fixes temperature and top_p and ignores them if sent, so
they are not configurable. Use a pay-as-you-go key: M Plan keys have quota windows and plan-level rerouting.

The Cohere adapter (`adapter: cohere`) uses the native Chat API v2 (`/v2/chat`), not the OpenAI Compatibility API.
Responses do not name the model, so qualification checks the dated id against `GET /v1/models/{model}` (chat
support, deprecation, context length, and sampling defaults). Cohere caps `max_tokens` silently, so configurations
state the model's output limit. Assistant messages are replayed exactly, thinking blocks included. Cost is priced
on `usage.billed_units`. Cohere inserts a safety instruction into every prompt (`safety_mode` CONTEXTUAL by
default; Command A models have no OFF), which the configuration discloses. On Command A+ even production keys are
limited to 1,000 chat calls a month, and trial keys may not be used commercially.

The Bedrock adapter (`adapter: bedrock`, `provider: bedrock`) calls Amazon Bedrock's Converse API with plain HTTP
and a Bedrock API key (`Authorization: Bearer`), not boto3 and SigV4 credentials. It serves Amazon's own models
only: Bedrock also hosts other vendors' models, which this harness calls through their own APIs, and the bias
report maps `provider: bedrock` to AWS. `adapter_options.region` is required and must be a Region that accepts
API keys (not us-east-2 or us-west-1). Responses do not name the model, so qualification checks the inference
profile and foundation model on the control plane (Amazon provider, ACTIVE lifecycle) and records the Regions
a cross-Region profile may route to; which Region served a request is not reported. Nova's built-in content
filters cannot be configured; a blocked prompt or output is a refusal. Assistant messages are replayed exactly,
reasoning blocks included (Nova 2 returns reasoning text as `[REDACTED]`). Cost counts cache reads and writes
separately and is checked against `totalTokens`.

The Meta adapter (`adapter: meta`, `provider: meta`) reuses the OpenAI adapter against the Meta Model API's Responses
endpoint (`https://api.meta.ai/v1`), which Meta recommends; continuation is stateless with encrypted reasoning
replayed unchanged. It never falls back to OpenAI credentials. Only Muse models on the Standard tier qualify: the
`-contributor` ids use the same key but let Meta train on inputs and outputs. Meta adds a hidden system prompt and
developer message to every request (not billed or counted, no opt-out), which configurations disclose. Muse ids
are not dated snapshots. Reasoning is always on. A safety block (`content_policy_violation`) is a refusal, a 400
about the context length is context exceeded, and a 504 is an ambiguous timeout. Muse Ember charges a whole
request at long-context rates above 200K input tokens, so an Ember configuration needs `pricing.long_context`.

The Microsoft MAI adapter (`adapter: mai`, `provider: microsoft`) calls a model deployed in your own Microsoft
Foundry resource on its MAI chat-completions route (`{base_url}/mai/v1/chat/completions`, `api-key` header). Set
`adapter_options.base_url` to the resource endpoint; `model` is the deployment name. There is no `/models` route, so
qualification checks identity with one tiny completion, compares the reported model case-insensitively with
`served_model`, and records the deployment's `model_version` as declared, not verified. Only `max_output_tokens` is
supported: the service silently ignores unknown fields, and `reasoning_effort` is validated but undocumented. With
`replay_reasoning: true`, reasoning is requested encrypted and replayed exactly (the service rejects an altered blob).
Microsoft publishes no cached-input price and responses usually report a few cached tokens, so such a call's cost is
unknown, with the full-input-rate upper bound recorded as `api_cost_usd_if_uncached`. Microsoft's docs list a default
quota tier of 0 tokens per minute (every call is a 429) until a higher tier is granted.

The OpenRouter adapter (`adapter: openrouter`) reaches a vendor's model through OpenRouter when the vendor's own API
is unavailable, and only on the vendor's own serving infrastructure: `provider` stays the vendor (so the bias report
attributes results to it) and must be one of the routes in `openrouter_chat.ROUTES` (today `alibaba`, for Qwen).
Every request is pinned with `provider: {only: [alibaba], allow_fallbacks: false, require_parameters: true}`, never
uses model fallbacks or variants, and opts into routing metadata; a response whose metadata does not name Alibaba,
or that names another model, is a failure. Qualification reads the model's OpenRouter endpoints (exactly one Alibaba
endpoint, labelled `endpoint_model`, with the configured prices; every response's routing metadata must name it too) and streams one follow-up with
`debug.echo_upstream_body` to check what OpenRouter actually forwards to Alibaba: the vendor model id (`upstream_model`; Qwen4-Max gets the
moving `qwen4-max` alias although OpenRouter labels the endpoint with a dated id), no injected
sampling defaults, an output cap that is present and unchanged, search off, the replayed answer and reasoning forwarded with the baseline's exact text, reasoning controls that agree with the configured effort and documented
defaults, and any `upstream_expect` keys. A forwarded key the configuration does not determine leaves qualification
incomplete until it is pinned in `upstream_expect`.
Assistant messages, `reasoning` and `reasoning_details` included, are replayed exactly. Cost uses configured prices,
with OpenRouter's own charge (`usage.cost`) recorded beside it. Alibaba requests cannot be cancelled through
OpenRouter, so a timed-out request still runs and bills.

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

GPT-OSS uses native Harmony channels and its `thinking` history field; Muse uses self/user recipient messages. The MLX adapter parses these formats before scoring and preserves their native reasoning fields during continuation.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q && .venv/bin/mypy
```

The tests cover the harness code only. Every configuration, rubric, catalog, and scenario they use comes
from `tests/fixtures/project/`: frozen snapshots of a few configurations, plus synthetic rubrics, catalog,
prompts, and scenarios. Tests never read the experiment data in `configs/`, `rubrics/`, or `scenarios/`;
a guard in `tests/conftest.py` fails any test that tries. CI (`.github/workflows/ci.yml`) runs mypy and
the tests on every pull request and push to `main`.

For the local MLX-LM worker (Apple Silicon), create its pinned environment and fetch the pinned checkpoint:

```sh
python3.12 -m venv .venv-mlx
.venv-mlx/bin/pip install --no-deps -r requirements-mlx-worker.txt
.venv-mlx/bin/hf download mlx-community/Qwen3.5-9B-4bit --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53
# to pin another checkpoint, print its file manifest for adapter_options.files:
.venv-mlx/bin/python -I src/archbench/workers/mlx_worker.py manifest <repository> <revision>
```

For agentic-design-bias v2, `bench evaluate-v2 <experiment-id>` uses the dedicated
[selection policy](configs/evaluation/agentic-design-bias-v2.yaml): Luna extraction and
Jev labels for every response, with Haiku auditing a 10% sample and all flagged cases.
`bench selection-status <run-id>` shows coverage. JSON and CSV report paths are printed
by the evaluation command. The [v2 review notes](scenarios/agentic-design-bias-v2/README.md)
explain calibration, frozen inputs, resume, and the separation from v1 quality scores.

## Credentials

Provider keys go in `secrets.local.yaml` at the project root (ignored by Git):

```sh
cp secrets.example.yaml secrets.local.yaml && chmod 600 secrets.local.yaml
# then edit secrets.local.yaml and paste the key
```

For GLM, use the `zhipu` entry in `secrets.local.yaml`:

```yaml
zhipu:
  api_key: "<your Z.ai API key>"
```

The local key takes precedence over `ZAI_API_KEY`. Use a key for Z.ai's standard pay-as-you-go endpoint.

The harness refuses to load the file if it is readable by others, a symlink, tracked by Git, or not
ignored. Keys are passed straight to adapters (never exported to the environment), redacted from
captured traffic, and any artifact, record, or frozen config containing a loaded key or a key-shaped
string is rejected rather than written. Use `--secrets PATH` for a file elsewhere. Without the file,
SDK defaults apply (`ANTHROPIC_API_KEY` or `ant auth login`; `OPENAI_API_KEY`; `GOOGLE_API_KEY` or
`GEMINI_API_KEY` for Gemini; `MISTRAL_API_KEY`; `XAI_API_KEY`; `DEEPSEEK_API_KEY`; `MOONSHOT_API_KEY` for Kimi;
`ZAI_API_KEY` for GLM; `MINIMAX_API_KEY` for MiniMax; `CO_API_KEY` for Cohere; `AWS_BEARER_TOKEN_BEDROCK` for Bedrock;
`MODEL_API_KEY` for Meta; `AZURE_INFERENCE_CREDENTIAL` for Microsoft MAI; `OPENROUTER_API_KEY` for OpenRouter).

## Live debugging run

`configs/experiments/haiku-smoke.yaml` runs Claude Haiku 5.5 once (baseline + 3 branches) and
`configs/evaluation/anthropic-debug.yaml` evaluates it with Haiku judges and Jev. Debug scores are not
benchmark results. `bench qualify` makes three tiny live probe calls to prove branching works.

## Agentic-design-bias probes

The [v2 question set](scenarios/agentic-design-bias-v2/README.md) adds 24 independent,
requirement-conditioned questions for review: a personal family assistant, a swimming coach's
greenfield SaaS, and an existing on-premises Java appointment application. Each situation asks for
one choice or a ranked top three for model provider, model, agentic framework, and infrastructure.
V2 execution and evaluation configs are not yet introduced; the workflow below describes v1.

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
`deepseek-p6-pick-model.yaml` runs P6 on DeepSeek Flash and needs a DeepSeek key.
`kimi-p6-pick-model.yaml` runs P6 on Kimi K3.1 Pro and needs a Moonshot key.
`fake-p4-three-frameworks.yaml` and `fake-p6-pick-model.yaml` with `agentic-design-bias-fake.yaml` run the same path offline. The catalog
is a draft: verify vendors and statuses against first-party sources before reporting results.

## Workflow

```sh
bench qualify configs/experiments/fake-smoke.yaml           # persisted qualification per exact config bytes
bench plan configs/experiments/fake-smoke.yaml              # freeze the experiment plan
bench run fake-smoke-001                                    # execute or resume the branching graph
bench evaluate fake-smoke-001 --policy configs/evaluation/default.yaml
#   validation -> blinded packets -> two independent judges -> classification -> attention signals
bench review <evaluation-run> --reviewer <you>              # check every judgement in the review app
bench review-export <evaluation-run> --out reviews/         # flagged responses only: blinded packets + templates
bench review-import <evaluation-run> reviews/*.review.json  # --supersede records a correction
bench adjudicate <evaluation-run>
bench report fake-smoke-001 --evaluation <evaluation-run>
```

### Human checks

`bench review` opens a local page (127.0.0.1 only) that walks through every response of the run. Each
screen shows the rendered answer, with the passages the judges and the extractor cited highlighted, next to
each judge's ratings, rationales and quotes, the critical-failure candidates, and the extraction. Candidate
identity stays hidden, and responses are ordered by round and random packet id. The header shows how many
responses are checked, approved, flagged and left; the sidebar lists every response with its state and marks
the ones whose judges raised something worth a close look.

- **Looks right** (`A`): the judge mean stands. Confirm or reject each critical-failure candidate first.
- **Flag** (`F`): tick what looks wrong (a judge, the extraction, or the answer itself) and say why. A flag on
  a judge or the answer holds the score until a full blind re-score (`review-export` / `review-import`); an
  extraction-only flag is listed in the report (`human-check-flags.csv`) and does not block scoring.
- `J`/`K` move between responses, `N` jumps to the next unchecked one, and `?` lists every shortcut.

Every decision is saved as an append-only `human_check` record the moment it is made; changing your mind
supersedes the earlier record. A check names the judge and extraction records it saw. If those change (for
example a judge that was missing completes on resume), the check goes stale and the response needs another
check. Adjudication waits until every response is checked and every blocking flag is fully reviewed.

### Cross-model report

`bench cross-report` compares models across evaluation runs as one static, self-contained HTML file (inline CSS and
SVG, no scripts or external requests), ready to publish anywhere:

```sh
bench cross-report evr-69fa320525c64c75 evr-13ec817f41224e4f evr-e112dca0a93f4022 \
    evr-30111b46b2124cab evr-2850fdb90275448c \
    --scenarios scenarios/agentic-design-bias --out results/published/agentic-design-bias.html
```

Name one evaluation run per probe and model; a second run covering the same pair is rejected rather than merged.
For each probe the page shows which vendor each model recommends first, self-preference against a baseline pooled
from every other vendor's model on that probe, rubric quality and per-dimension ratings, and a coverage grid that
also lists answers no included run evaluates. Quality is the adjudicated score where one exists, otherwise the
judges' mean, marked provisional. The page lists every run, policy, judge, extractor and catalog it used, embeds
the aggregated data as JSON, and states its caveats (unchecked judgements, judges sharing a candidate's vendor,
mixed policies, draft catalog, sample size).

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
