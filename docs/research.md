# API & Runtime Research

Research notes for hfvast, checked against **official, live documentation and source code**.
All findings verified **2026-09-02** unless noted. Anything not confirmed is marked UNCERTAIN.

## 1. Vast.ai server API

Sources: https://docs.vast.ai/api-reference/ · https://docs.vast.ai/api-reference/search/search-offers ·
https://docs.vast.ai/api-reference/instances/create-instance · https://docs.vast.ai/api-reference/instances/destroy-instance ·
https://docs.vast.ai/api-reference/instances/show-instance · https://docs.vast.ai/api-reference/permissions ·
https://docs.vast.ai/api-reference/rate-limits-and-errors · https://docs.vast.ai/guides/reference/faq/billing ·
https://docs.vast.ai/guides/instances/connect/instance-portal (all fetched 2026-09-02)

### Auth & base
- `Authorization: Bearer $VAST_API_KEY`; base `https://console.vast.ai`, paths under `/api/v0/`
  (instance *list* is `/api/v1/instances`). Keys at https://cloud.vast.ai/manage-keys/.
- **Units trap:** `gpu_ram` is per-GPU **MB** in the REST API (GB in the CLI docs); `cpu_ram` MB;
  `disk_space` GB; `dph_total`/`dph_base`/`min_bid` are **$/hr**; `storage_cost` is **$/GB/month**;
  `inet_down_cost`/`inet_up_cost` are **$/GB**; `inet_down` is advertised in **Mb/s** (CLI docs; REST doc says
  MB/s — UNCERTAIN, we treat as Mb/s and surface raw values under `--verbose`).
- `dph_total` ≈ GPU rental + storage priced for the `allocated_storage` disk size sent with the search
  (there is an `allocated_storage` body param, default 8 GB). It **excludes bandwidth** (charged per byte on top).

### Offer search
- `POST /api/v0/bundles/`, body = JSON filter dict; operators `eq,neq,gt,lt,gte,lte,in,notin` per field:
  ```json
  {"verified":{"eq":true},"rentable":{"eq":true},"num_gpus":{"lte":4},
   "gpu_ram":{"gte":24576},"reliability":{"gte":0.98},
   "order":[["dph_total","asc"]],"type":"on-demand","limit":100,"allocated_storage":240}
  ```
- Filterable fields include: `gpu_name, gpu_arch, num_gpus, gpu_ram (MB), gpu_total_ram, gpu_mem_bw (GB/s),
  pcie_bw, bw_nvlink (GB/s), cpu_ram (MB), cpu_cores, disk_space (GB), disk_bw (MB/s), inet_down, inet_up,
  inet_down_cost, inet_up_cost, dph_total, dph_base, min_bid, dlperf, dlperf_per_dphtotal, total_flops,
  reliability, verified, rentable, duration, direct_port_count, geolocation, compute_cap, cuda_max_good,
  driver_version, host_id, storage_cost, verification`.
- Response: `{"offers": [...]}` — each offer carries all fields above plus `id` (used as the ask id at
  creation), `search{gpuCostPerHour, diskHour, totalHour}`, `geolocation`, `verification`, `external`, …

### Instance creation
- `PUT /api/v0/asks/{offer_id}/` — body: `image`, `disk` (GB, **fixed at creation**, default 8),
  `env` (JSON object; port maps are Docker-flag keys like `"-p 8000:8000": "1"`), `onstart` (script,
  ≤4048 chars), `label`, `runtype` (`ssh|jupyter|args|…` — `args` preserves the image entrypoint),
  `target_state` (`running`), `cancel_unavail` (default true → fail fast if offer vanished).
- Response: `{"success": true, "new_contract": <instance_id>, "instance_api_key": "…"}`
  - `new_contract` is the instance id (NOT the offer id).
  - `instance_api_key` is a **restricted, per-instance key** ("can only start, stop, or destroy that
    specific instance"), also injected into the container as `CONTAINER_API_KEY`.
    → This is the officially supported primitive for **remote self-destruction** (a watchdog inside the
    container can destroy its own instance without holding the user's unrestricted key).
- Errors: 400 `invalid_args`, 404 `no_such_ask`, 410 offer gone, 429 rate limit.

### Instance lifecycle
- `GET /api/v0/instances/{id}/` → `{instances: {...}}`; states: `null → loading → running`;
  terminal-failure: `exited/unknown/offline`. `ports` (Docker map) appears on running instances via
  `GET /api/v1/instances`. `ssh_host/ssh_port/public_ipaddr` available.
- `PUT /api/v0/instances/{id}/` body `{"state":"stopped"|"running"}` (stop pauses compute billing,
  **storage keeps accruing**) or `{"label": "..."}`.
- **Destroy:** `DELETE /api/v0/instances/{id}/` → irreversible.
- **Logs:** `PUT /api/v0/instances/request_logs/{id}/` body `{"tail":"1000"}` → returns `result_url`
  (S3) with the log text.
- Rate limits: per-endpoint minimum call interval (examples: create 4.5 s, show-instance 2.0 s,
  destroy 3.0 s, list 1.0 s — examples, not guarantees); HTTP 429, no `Retry-After`.

### Billing model (FAQ)
- Compute $/hr while running; storage $/GB/month billed **every second the instance exists**
  (stop does NOT stop storage billing; only destroy does); bandwidth $/GB both directions.
- Credit exhaustion auto-**stops** instances (not destroys). No documented native TTL/auto-destroy API
  → implement idle/max-runtime watchdog ourselves (local daemon + in-container restricted-key watchdog).

### HTTPS exposure (current official options)
1. **Instance Portal** (in `vastai/base-image`): per-port **Cloudflare quick tunnels**
   (`https://<words>.trycloudflare.com`) configured via `PORTAL_CONFIG`, or named tunnels with
   `CF_TUNNEL_TOKEN`. In-instance, not an API primitive.
2. Jupyter proxy/direct runtypes (Jupyter only; direct mode needs a custom root CA).
3. Raw mapped public ports = plain HTTP (no TLS).
→ hfvast strategy: gateway binds a public port; prefer Portal quick-tunnel HTTPS when available,
otherwise warn + strong API key + optional SSH tunnel guidance. No Vast-native generic HTTPS API exists.

### Restricted credentials (verified)
- Permission categories: `instance_read, instance_write, user_read, user_write, billing_read, …`;
  scoped keys like `{"api":{"instance_read":{},"instance_write":{}}}` and even per-instance constraints
  (`{"api":{"instance_read":{"api.instance.request_logs":{"constraints":{"id":{"eq":1227}}}}}}`).
- Per-instance `instance_api_key` from create response (see above).
→ Plan: keep the unrestricted key on the user's machine only; inject `CONTAINER_API_KEY` into the
container for a self-destroy watchdog; optionally create a scoped key for the daemon (M3).

## 2. Hugging Face Hub API

Sources: https://huggingface.co/docs/hub/api (→ https://huggingface.co/.well-known/openapi.json) ·
https://huggingface.co/docs/hub/gguf · https://huggingface.co/docs/hub/models-gated ·
https://huggingface.co/docs/hub/rate-limits · https://github.com/ggml-org/ggml/blob/master/docs/gguf.md ·
huggingface_hub source `hf_api.py`/`utils/_http.py` (all fetched/probed 2026-09-02)

- `GET https://huggingface.co/api/models/{repo}` (default response) includes: `gated`
  (`false|true|"auto"|"manual"`), `pipeline_tag`, `tags`, `library_name`, `config`, `safetensors`
  (per-dtype parameter counts), **`gguf`** (GGUF repos only: `architecture`, `context_length`, `total`
  params, `totalFileSize`, tokens/chat template), `siblings` (names only), `usedStorage`.
- `?expand[]=…` trims/extends fields (incl. `gguf`); `?blobs=true` adds per-file `size` + `lfs` info to
  siblings. `files_metadata=true` no longer changes the response (use `blobs=true`).
- `GET /api/models/{repo}/tree/{rev}?recursive=true` → per-file `{path, type, size, lfs{oid,size}}`;
  paginated via `Link: …rel="next"` cursor.
- **Gated repos:** unauthenticated `resolve` → `401` with `x-error-code: GatedRepo`;
  nonexistent repos deliberately return `401` (not 404) with `Invalid username or password`.
  Authenticated-but-unauthorized → `403`. Metadata APIs (model info, tree) remain readable for gated
  repos (sizes visible; blob hashes masked). Access is per-user; request via web UI.
- **Range requests work end-to-end** on `/resolve/` (302 → CDN → `206`, `accept-ranges: bytes`);
  the redirect hop already exposes `x-linked-size` (exact file size). Verified live on a GGUF shard.
- Rate limits (5-min windows): anonymous API 500 req/5 min per IP, resolvers 3000; authenticated free
  user 1000/5000. Send `HF_TOKEN` when available.
- **GGUF binary format** (ggml spec): header = magic `GGUF`, u32 version, u64 tensor_count,
  u64 metadata_kv_count, then KVs (string key = u64 len + bytes; value_type u32 ∈
  0=UINT8,1=INT8,2=UINT16,3=INT16,4=UINT32,5=INT32,6=F32,7=BOOL,8=STRING,9=ARRAY,10=UINT64,11=INT64,12=F64;
  ARRAY = u32 elem type + u64 count + elements). Useful keys: `general.architecture`, `general.file_type`,
  `{arch}.context_length`, `{arch}.block_count`, `{arch}.attention.head_count`,
  `{arch}.attention.head_count_kv`, `{arch}.attention.key_length/value_length`, `{arch}.expert_count`,
  `tokenizer.*` (large arrays, typically 2–8 MB header total). Tensor infos follow metadata.
  → Header inspection via a few ranged GETs is cheap and authoritative.

### Live probes (2026-09-02, unauthenticated)
- `orcarouter/GLM-5.3-Flash-Uncensored-GGUF`: `gated:"auto"`, `library_name:"gguf"`,
  `pipeline_tag:"image-text-to-text"`, `gguf{architecture:"glm5next", context_length:1048576,
  total:320759404382, totalFileSize:116869939744}`, 34 tree entries: variants
  Q2_K (116.9 GB, 3 shards), Q3_K_M (152.7, 4), Q4_K_M (193.0, 5), Q6_K (263.4, 6), Q8_0 (341.0, 8)
  + root `mmproj-…-F16.gguf` (1.16 GB).
- Ranged GET on that repo's GGUF unauthenticated → `401 GatedRepo` (as documented).
- llama.cpp master `src/llama-arch.cpp`: **no `glm5next` arch**; GLM-5 family is `LLM_ARCH_GLM_DSA`
  (string `glm-dsa`, cites `zai-org/GLM-5.2`). → a GGUF declaring `glm5next` does not dispatch on
  llama.cpp master today ⇒ classified EXPERIMENTAL until verified.

## 3. Inference runtimes

### llama.cpp (ggml-org/llama.cpp master, 2026-09-02)
- Server README: `tools/server/README.md` (old `tools/llama-server` path moved).
- OpenAI-compatible: `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions`,
  `POST /v1/responses`, `/v1/embeddings`; public `GET /health` (503 while loading, 200 ready).
- Auth: `--api-key` (or `--api-key-file`), env `LLAMA_API_KEY`.
- Flags: `-c/--ctx-size`, `-np/--parallel`, `-ngl/--gpu-layers`, `-sm/--split-mode {none,layer,row,tensor}`,
  `-ts/--tensor-split`, `-mg/--main-gpu`, `--fit` (default on; reserves **1024 MiB margin per device**),
  `-hf <user>/<model>[:quant]`, `-m <file>`.
- GGUF splits: loader requires the **first shard** and auto-generates the shard list from
  `<name>-00001-of-000NN.gguf` (`src/llama-model-loader.cpp`).
- Images: `ghcr.io/ggml-org/llama.cpp:server-cuda` (and per-backend tags).

### vLLM (vllm-project/vllm main, 2026-09-02)
- Supported generative architectures are matched by `config.json` `architectures`
  (`LlamaForCausalLM`, `Qwen2/3/3Moe/3Next…`, `GlmForCausalLM`, `Glm4ForCausalLM`, `Glm4MoeForCausalLM`,
  `GlmMoeDsaForCausalLM` = GLM-5/5.1/5.2, `DeepseekV2/V3/V32/V4…`, `MixtralForCausalLM`,
  `Gemma3/4ForCausalLM`, `Phi3ForCausalLM`, `GptOssForCausalLM`, …); generic Transformers fallback backend.
- Quantization: GPTQ, AWQ, FP8 (W8A8), Marlin, bitsandbytes, GGUF (subset), llm-compressor, …
- Flags: `--tensor-parallel-size/-tp`, `--gpu-memory-utilization` (default **0.92** → ~8% of each GPU is
  implicit headroom), `--max-model-len`, `--max-num-seqs` (default 128), `--quantization`, `--trust-remote-code`.
- OpenAI server: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/responses`, …;
  `GET /health` (200 / 503 on dead engine).
- **Security caveat (documented):** `--api-key` only covers `/v1`,`/v2`,`/inference` prefixes — other
  endpoints (incl. `/health`, metrics, native APIs) are unauthenticated. → hfvast must front vLLM with
  its own gateway and bind the backend to localhost.
- Image: `vllm/vllm-openai:<tag>`.

### SGLang (sgl-project/sglang main, 2026-09-02)
- Supported families: DeepSeek, Qwen, Llama, Mistral/Mixtral, Gemma, Phi, GLM-4, **GLM-5
  (`GlmMoeDsaForCausalLM`, source-confirmed `python/sglang/srt/models/glm4_moe.py`)**, GPT-OSS, …
- Quantization: fp8, mxfp4/8, awq, gptq (+marlin), blockwise_int8, w8a8, modelopt, quark, bnb, gguf load-format.
- Flags: `--tp-size` (alias `--tensor-parallel-size`), `--dp-size`, `--mem-fraction-static`
  (auto-heuristic default), `--max-running-requests`, `--context-length`, `--api-key`
  (exempts `/health` and `/metrics`), `--trust-remote-code`.
- OpenAI-compatible: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/responses`, …;
  health: `GET /health` (auth-exempt), `GET /health_generate`.
- Images: `lmsysorg/sglang:<tag>` (`:latest-cu12` for CUDA 12, immutable version tags exist).

## 4. Implications for hfvast (decisions)

1. Offer ranking must model bandwidth **on top of** `dph_total` (per-byte egress/ingress charge),
   and storage via `storage_cost` ($/GB/month → /730 for hourly) using `allocated_storage` at search time.
2. `PUT /api/v0/asks/{offer}` + `new_contract` + per-instance `instance_api_key` →
   container-side watchdog can self-destroy (restricted key), local daemon remains the coordinator.
3. No native TTL exists → idle-timeout + hard max-runtime are hfvast's responsibility (two independent
   watchdogs: local daemon primary, in-container secondary with `CONTAINER_API_KEY`).
4. GGUF headers can be read remotely with 1–3 ranged GETs (~first 8 MB) — no weight download for
   inspection; gated repos need `HF_TOKEN` for resolve (fallback to conservative estimates otherwise).
5. llama.cpp is the default GGUF backend; vLLM/SGLang for safetensors; both benefit from (and vLLM
   requires, per its own docs) an authenticating gateway in front.
6. Units normalization table (offer → hfvast): `gpu_ram` MB→GB(per GPU), `cpu_ram` MB→GB,
   `inet_down` Mb/s, `disk_space` GB, `storage_cost` $/GB/mo, `inet_down_cost` $/GB, `dph_*` $/hr.
