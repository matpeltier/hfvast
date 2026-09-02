# Contributing to hfvast

Thanks for helping! hfvast is a small, focused tool: turn a Hugging Face model into a temporary
OpenAI-compatible endpoint on Vast.ai — without surprise bills.

## Development setup

```bash
git clone https://github.com/hfvast/hfvast && cd hfvast
uv sync                     # creates .venv with dev tools
uv run pytest               # unit tests (no network, no cloud)
uv run ruff check .         # lint
uv run ruff format --check .
uv run mypy src
```

Integration tests that hit real services are opt-in and never spend money:

```bash
HFVAST_LIVE=1 uv run pytest -m integration
```

## Ground rules

- **No real Vast resources from tests or CI, ever.** Real-spending integration tests are run
  manually by maintainers with explicit credentials and must destroy every instance they create.
- Keep domain layers (models/inspect/planning/providers/runtimes) free of CLI and HTTP-glue code.
- Every estimate (VRAM, disk, download time, cost) must expose its inputs — no buried assumptions.
- Backend/provider compatibility is data in a registry with a support level and a reason — never an
  inline special case.
- Secrets never appear in logs, errors, state, or tests.
- New features need tests; bug fixes need a regression test.

## Style

- Ruff + mypy strict must pass.
- Prefer small pure functions for planners/estimators — they are the easiest code to test.
- Commit messages: imperative, concise (`gguf: group multi-shard variants`).
