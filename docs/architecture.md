# hfvast Architecture

Status: Milestone 1 (inspection + quote, zero spending). This document is the design contract for
the whole system; unimplemented parts are marked [M2]–[M5].

## 1. Product frame

`hfvast up <HF model>` = ephemeral, cost-optimized, OpenAI-compatible endpoint on Vast.ai.
Everything is designed around: **no spend without confirmation, no silent runaway instances,
aggressive cost optimization, transparent estimates.**

## 2. Data flow

```
HF URL/alias
   │
   ▼
┌─────────────┐   ModelInfo    ┌──────────────┐  HardwareRequirements
│  inspect/    │ ─────────────▶│  planning/    │ ─────────────────┐
│  (HF API,    │  (variants,   │  memory/      │                  ▼
│   GGUF hdrs) │   arch, ctx)  │  storage/     │            OfferQuery ──▶ providers/vast (search_offers)
└─────────────┘               │  backends/    │                          │
                               │  requirements │                          ▼
       runtimes/registry ─────▶│               │                    list[GPUOffer]
       (compat matrix)         └──────────────┘                          │
                                                                         ▼
                                            planning/ranking + quote ──▶ DeploymentQuote
                                            (cold start, session cost,   (recommendation + reasons)
                                             explainable pros/cons)
                                                                         │
                                             user confirms (--yes)       │ [M2]
                                                                         ▼
                                     deploy/: provision → download → serve → health → endpoint
                                                                         │
                                     lifecycle/: activity tracking → idle/max-runtime → destroy [M3]
```

Layer rule: **CLI handlers never touch provider HTTP or planner internals directly** — they call
`QuoteBuilder` / `DeploymentOrchestrator` [M2]. Each layer depends only on the layer below.

## 3. Core types (`hfvast.models`)

Pydantic v2 models; no giant dicts cross module boundaries.

- `HFModelRef{repo_id, revision}` — normalized input (URL or `org/model`, optional `@rev`).
- `ModelFile{path, size_bytes}`
- `ModelVariant{id, quant, size_bytes, files, tier, mmproj}` — GGUF shards of one quant = ONE variant.
- `GGUFHeaderInfo{architecture, context_length, block_count, head_count, head_count_kv, key_length,
  value_length, embedding_length, expert_count, expert_used_count, file_type}` — parsed remotely.
- `ModelInfo{ref, task, architecture, format, dtype, parameter_count, weight_bytes, context_length,
  quantization, variants, requires_trust_remote_code, gated, multimodal, mmproj_files, gguf_header,
  quantization_config, notes}` — the single product of inspection.
- `VramBreakdown{weights_gib, kv_cache_gib, runtime_overhead_gib, safety_gib, total_gib}` — shown to
  the user verbatim (§49: no buried assumptions).
- `HardwareRequirements{minimum_vram_gib, recommended_vram_gib, disk_gb, context_length, concurrency,
  breakdown, assumptions}`.
- `GPUOffer{offer_id, gpu_model, gpu_count, per_gpu_vram_gb, total_vram_gb, cpu_*, disk_gb,
  inet_down_mbps, hourly_gpu_usd, hourly_total_usd, storage_per_gb_month, inet_down_usd_per_gb,
  reliability, verified, dlperf, gpu_mem_bw_gbs, pcie_bw_gbs, nvlink_gbs, geolocation}` — normalized
  from any provider's raw record (units normalized at the provider boundary).
- `CostBreakdown{hourly_gpu, hourly_storage, hourly_total, cold_start_hours, cold_start_usd,
  bandwidth_usd, runtime_usd, total_session_usd}`.
- `RuntimeSupport{backend, supported, level: SUPPORTED|EXPERIMENTAL|UNSUPPORTED, confidence, reason}`.
- `DeploymentQuote{model, context_length, concurrency, variant_plans, recommendation, reasons,
  cost_limits}` — the output of `quote`, the input of `up` [M2].
- `Deployment` [M2/M3] — persisted state (id, model, provider, instance id, prices, endpoint, status).

## 4. Inspection (`hfvast.inspect`)

- `huggingface.HFInspector` — raw httpx against `api/models/{repo}` + `tree/{rev}?recursive=true`
  (paginated). Maps `401 x-error-code: GatedRepo` → `GatedAccessError` (friendly message), other 401 →
  `ModelNotFoundError`. Uses `HF_TOKEN` when present.
- `gguf.group_gguf_variants(files)` — shard regex `-00001-of-000NN` groups multi-part files into one
  variant; quant id parsed from the name (`Q4_K_M`, `IQ4_XS`, `F16`, …).
- `gguf.RemoteGGUFHeaderReader` — ranged GETs (first ~8 MB, streamed) parse the GGUF header per the
  ggml spec to obtain authoritative arch/context/layers/heads/kv-heads/head_dim/expert counts.
  One representative variant is parsed (arch is shared across quants); gated repos without a token
  fall back to conservative heuristics with an explicit note.
- `transformers.inspect_safetensors` — `config.json`, `model.safetensors.index.json`,
  `quantization_config`, API `safetensors` param counts; actual stored weight size preferred over
  `params × dtype`.
- `quantization.tier_for_quant` — ECONOMY (Q2/IQ2/Q3…), BALANCED (Q4*/IQ4/Q5_K_S), QUALITY (Q5_K_M/Q6/Q8/F16).

## 5. Planning (`hfvast.planning`)

- `requirements.build_requirements(model, variant, ctx, concurrency, backend)` → HardwareRequirements.
- `memory.MemoryEstimator` (pure function of inputs):
  `VRAM = weights + KV + runtime_overhead + safety`
  - weights = actual variant size (GGUF aggregate; safetensors index size) — MoE **total** weights, never
    only active params;
  - KV/token = `2 × block_count × head_count_kv × head_dim × kv_dtype_bytes` (GGUF header values when
    available; otherwise conservative fallbacks recorded in `assumptions`); × context × concurrency;
  - runtime overhead per backend (llama.cpp ~2 GiB + ~0.75 GiB/GPU CUDA context; vLLM/SGLang ~8% of
    per-GPU VRAM + ~2.5 GiB — mirrors their documented memory models);
  - safety = max(8 GiB, 5% of weights).
- `storage.estimate_disk` = weights + mmproj + 20 GiB runtime headroom + 10 GiB temp margin, ceiled.
- `backends.BackendSelector` + `runtimes.registry` — explicit compatibility matrix with support levels
  (see §7). UNSUPPORTED ⇒ hard failure before any provisioning; EXPERIMENTAL ⇒ requires explicit
  confirmation [M2].
- `hardware.build_query` → provider-agnostic `OfferQuery{min_total_vram_gb, max_gpus, min_per_gpu_vram_gb,
  disk_gb, min_download_mbps, min_reliability, gpu_filter, secure_cloud_only, max_hourly_usd}`.
- `ranking.OfferRanker` — viability filter (usable VRAM ≥ recommended, disk, reliability, bandwidth,
  gpu_count ≤ max) then explainable scoring (§9). No opaque score: every rank yields pros/cons strings.
- `quote.QuoteBuilder` — orchestrates inspect → per-variant plan (requirements, backend, offers, costs)
  → recommendation: default = BALANCED tier; upgrade to QUALITY if its cost ≤ 1.25× BALANCED; downgrade
  to ECONOMY only if BALANCED > 1.5× ECONOMY; never silently pick the smallest quant; Q2-class quants
  carry a quality warning.

## 6. Provider abstraction (`hfvast.providers`)

```python
class ComputeProvider(Protocol):
    name: str
    async def search_offers(self, query: OfferQuery) -> list[GPUOffer]: ...
    async def create_instance(self, offer_id, spec: InstanceSpec) -> InstanceHandle: ...   # [M2]
    async def get_instance(self, instance_id) -> InstanceStatus | None: ...                # [M2]
    async def destroy_instance(self, instance_id) -> None: ...                             # [M2]
    async def logs(self, instance_id, tail: int) -> str: ...                               # [M2]
```

`VastProvider` implements search in M1 (thin typed wrapper over `POST /api/v0/bundles/` with
rate-limit-aware retries via tenacity; units normalized at this boundary). Future RunPod/Lambda
providers implement the same protocol without planner changes.
Without `VAST_API_KEY`, `quote` uses a bundled **snapshot** of realistic offers that is always
displayed as `SAMPLE DATA (not live)` — never silently treated as real.

## 7. Runtime abstraction (`hfvast.runtimes`)

- `registry.SUPPORT_MATRIX`: format-level defaults + per-architecture overrides, each with
  `level/confidence/reason` and a doc pointer. Updated independently of code paths (data, not logic).
  GGUF → llama.cpp (verified arch list from llama.cpp `src/llama-arch.cpp`, checked 2026-09-02;
  unknown GGUF arch ⇒ EXPERIMENTAL/UNSUPPORTED, never a guess). Safetensors → vLLM (arch list from
  `supported_models.md`), SGLang as secondary. Diffusion/embeddings/rerankers/speech ⇒ UNSUPPORTED by task.
- `llama_cpp.py / vllm.py / sglang.py`: `build_plan(model, variant, requirements) -> RuntimePlan`
  (image, planned CLI args, health path, api prefix). Images are versioned OCI images we publish
  containing backend + bootstrap + gateway + health probes [M2].

## 8. Cost model

- `hourly_total` = provider `dph_total` (already includes storage priced at `allocated_storage`).
- `bandwidth_usd = inet_down_usd_per_gb × download_gb` (model + image), charged per byte.
- `download_hours = download_gb × 8 / (inet_down_mbps × efficiency)` (default efficiency 0.7).
- `load_minutes` heuristic from aggregate GPU memory bandwidth (capped 2–40 min), `pull_minutes`
  from image size (capped).
- `cold_start_usd = (pull + download + load) × hourly_total + bandwidth_usd`
- `total_session_usd = cold_start_usd + hourly_total × expected_session_hours`
  Expected session (default 2 h, `--expected-session`) is the ranking duration — cheap-but-slow
  offers lose for short sessions; can win for long ones.
- All numbers displayed with their inputs (§49). Estimates are always labeled "estimate"; Vast billing
  is authoritative.

## 9. Ranking strategy

Filter: usable VRAM (total − per-GPU CUDA overhead × count) ≥ recommended VRAM; disk ≥ required;
reliability ≥ 0.98 (default); inet_down ≥ 300 Mb/s; gpu_count ≤ max.
Score = `total_session_usd + penalty_usd`, penalties (each ≈2% of session cost, always surfaced as
cons): no NVLink on multi-GPU, gpu_count above the minimal viable count, pcie_bw < 8 GB/s, disk
headroom < 10%. Tiebreakers: reliability, then bandwidth. Output = ranked list + explicit
`+ reason / − reason` lines for the recommendation.

## 10. Lifecycle & auto-destruction [M2/M3]

State machine: `NEW → PROVISIONING → DOWNLOADING → LOADING → READY → IDLE(→destroy) | FAILED(→destroy)`
with resumable stages and persisted deployment state after every transition.
Idle = READY ∧ active_requests == 0 ∧ (now − last_user_activity) > idle_timeout. Health checks do NOT
count as activity. Two independent watchdogs:
1. **Local daemon** (owner of the unrestricted key) — idle timeout + hard max-runtime (`6h` default).
2. **In-container watchdog** holding only the per-instance `instance_api_key` (Vast-restricted,
   self-destroy-only) as a fail-safe if the local daemon dies (laptop sleep).
No Vast-native TTL exists (verified 2026-09-02); both watchdogs are therefore mandatory for the
runaway-bill guarantee. Limitations of the local daemon (sleep/shutdown) are documented, not hidden.

## 11. Security model

- `HF_TOKEN`/`VAST_API_KEY` come from env (preferred) or flags; never written to state/config/logs/
  exceptions; central `SecretRedactor` scrubs any output/traceback path.
- Server-side HF token: recommend fine-grained, read-only, repo-scoped tokens; documented risk on
  non-verified hosts; `--secure-cloud-only` restricts to verified/secure datacenter hosts.
- The unrestricted Vast key never leaves the local machine; containers get only
  `CONTAINER_API_KEY`; the HF token is injected at runtime, revocable, ideally read-only.
- Gateway [M2] is the only public surface: `sk-hfvast-<256-bit>` per-deployment key, backend bound to
  localhost, `/health` public, SSE proxied without buffering.
- API keys are cryptographically random (`secrets.token_urlsafe`).

## 12. Testing strategy

No real spending in CI. HF API + Vast API + runtime health are mocked (httpx `MockTransport`;
provider behind `ComputeProvider` protocol). Unit: GGUF grouping/header parsing (synthetic binaries
served over a Range-capable mock), memory/storage math, ranking, config, redaction. Integration
(marked, opt-in `HFVAST_LIVE=1`): real HF inspection. E2E [M2]: full deploy against mocks.

## 13. Milestones

1. **M1 (this)** — inspect + quote, GGUF first, live HF data, snapshot/live Vast offers, no spending.
2. **M2** — `up/down`: GGUF → llama.cpp on Vast, gateway, endpoint, failure cleanup.
3. **M3** — lifecycle: activity tracking, idle/max-runtime, watchdogs, `ps/status/logs/cost`.
4. **M4** — safetensors → vLLM.
5. **M5** — SGLang + broader compat matrix.
