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
