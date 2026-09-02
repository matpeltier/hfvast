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
