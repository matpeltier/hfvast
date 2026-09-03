# Changelog

All notable changes to hfvast are documented here. Format: Keep a Changelog; versioning: SemVer.

## [0.1.0] - 2026-09-02 — Milestone 1

### Added
- `hfvast inspect <model>`: remote Hugging Face repository inspection without downloading weights.
  Detects format (GGUF / safetensors), task, architecture, context length, parameter count,
  quantization variants (multi-shard GGUF grouping), multimodal projectors, gated status, and
  backend support level (SUPPORTED / EXPERIMENTAL / UNSUPPORTED) from the runtime compatibility
  registry.
- Remote GGUF header reader using HTTP range requests (authoritative arch/context/KV-shape metadata).
- `hfvast quote <model>`: hardware requirements (VRAM breakdown, disk), backend selection, live
  Vast.ai offer search (or clearly-labeled bundled sample data without `VAST_API_KEY`), explainable
  offer ranking by expected session cost, and per-quantization cost comparison.
- `hfvast doctor`: environment/credentials/connectivity checks.
- `hfvast alias add/rm/list` and optional TOML config (`~/.config/hfvast/config.toml`).
- Central secret redaction; tokens never persisted or printed.

## [0.2.0] - 2026-09-02 — Milestones 2 & 3

### Added
- `hfvast up <model>`: full deployment pipeline — quote → explicit confirmation →
  Vast instance creation (offer-fallback across the top ranked offers) →
  in-instance bootstrap (model download with resume, backend launch, health
  wait) → smoke test → `OPENAI_BASE_URL` / `OPENAI_API_KEY` output.
- Single-file stdlib gateway deployed to the instance: API-key auth
  (constant-time), `/v1/*` proxying with SSE streaming preserved chunk-by-chunk,
  `/health`, `/internal/state`, activity tracking (health checks never count).
- Backend binds to 127.0.0.1 only; the gateway is the sole public surface.
- Lifecycle (M3): deployment state persisted atomically (0600) immediately after
  instance creation; local watchdog daemon (idle timeout, hard max runtime,
  unreachable-grace) plus an in-container fail-safe watchdog using Vast's
  per-instance restricted `CONTAINER_API_KEY` (self-destroy only).
- `hfvast down/ps/status/endpoint/cost/logs`; `--dry-run`, `--yes`,
  `--keep-on-failure`, `--idle-timeout`, `--max-runtime`, `--ready-timeout`.
- Failure cleanup: bootstrap failures destroy the instance and report the
  incurred estimate (spec §30).
- Runtime image definitions + trusted release workflow (`runtime/`,
  `.github/workflows/release-images.yml`); deployment currently injects the
  bootstrap payload into pinned upstream images until images are published.
- 96 tests including a real gateway subprocess suite and a mocked-Vast
  end-to-end deployment (ready + failure-cleanup paths). No test ever spends.

### Fixed (live-deployment hardening, first real `up` session 2026-09-03)
- Vast 429 handling now honors the API's explicit `retry_after`; request pacing
  raised to 6 s (endpoint threshold is 5 s).
- Docker port maps are polled (up to 15 min) — they materialize only after the
  image pull finishes, not at `cur_state=running`.
- onstart stub rewritten in POSIX sh (dash has no `${!var}` — the original
  bashism silently killed the bootstrap).
- `/health` phase logic fixed: the backend being down during download phases is
  normal and no longer reports `error` (this destroyed healthy deployments).
- Chunked python downloader replaces curl: parallel ranges over the signed CDN
  URL (HF Xet throttles single connections after ~250 MB), per-chunk resume,
  HF token used only for URL resolution; downloader exceptions surface in
  `/health` for diagnosis.
- llama-server invoked by absolute path (`/app/llama-server`, ghcr image layout).
- `stopped` treated as a terminal provisioning failure (OCI/driver errors).
- Offers filtered on `driver_version >= 550` (CUDA 12.4 images fail on older hosts).
- Pre-rent **reachability probe**: Vast exposes host IPs before renting and its
  geolocation data is unreliable ("Washington, US" on China Unicom IP space) —
  unroutable hosts are probed (TCP 80/443) and re-ranked last; endpoints that
  never answer within 5 min fail fast to the next offer.
- Per-offer failure retry covers the whole bootstrap (create, discovery,
  download, health, smoke test), not just offer disappearance.
- Watchdog gains a 90-min bootstrap deadline; gateway exposes `/internal/log`
  and the orchestrator captures instance log tails before failure cleanup.
- `--geo` flag to restrict offer regions.

### Added (2026-09-03, after the first GLM session)
- `--budget <usd>`: hard spend cap enforced by BOTH watchdogs — the instance is
  destroyed the moment the estimated spend exceeds it (during download, load or
  serving). The runaway-bill guarantee is now a dollar figure, not just a time
  figure.
- Gated-repo UX: HF returns 403 (not 401) for authenticated users who have not
  accepted the license; both paths now raise a clear error with the model-page
  URL and the exact step, locally and in-instance (downloader → `/health`).

### Findings from the first GLM-5.3-Flash session (aborted by budget)
- The gated 194 GB download works with a fine-grained token (license accepted
  on the model page).
- Advertised host bandwidth is NOT guaranteed: the 4× A100 host advertised
  872 Mb/s and delivered ~224 Mb/s → 194 GB ≈ 2 h. Ranking is correct given
  advertised numbers; retry with `--min-download-mbps 20000` to force
  multi-Gbps hosts (B200/B300 class) where the download takes minutes.
- Vast geolocation data remains unreliable; the reachability probe and the
  never-reachable fast-fail (5 min) are the real defenses.
