# hfvast

**Turn a Hugging Face model into a temporary OpenAI-compatible GPU endpoint on Vast.ai.**

```bash
pipx install hfvast          # or: uv tool install hfvast

export VAST_API_KEY=...
export HF_TOKEN=...          # fine-grained, read-only token recommended

hfvast up https://huggingface.co/org/model
```

hfvast inspects the model repository, estimates VRAM/disk/runtime needs, picks a compatible
inference backend (llama.cpp / vLLM / SGLang), finds and ranks real Vast.ai offers by *expected*
cost for your session, and — only after explicit confirmation — rents a machine, downloads the
model, serves it, and hands you:

```
OPENAI_BASE_URL=https://.../v1
OPENAI_API_KEY=sk-hfvast-...
```

When you stop (or walk away and the idle timeout expires), the instance is destroyed and your bill
returns to $0. No Kubernetes, no permanent hosting — ephemeral infrastructure by default.

```python
from openai import OpenAI

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
reply = client.chat.completions.create(
    model="model",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

## Status: Milestone 1 (inspect + quote — no spending yet)

```
                       ┌─────────────────────────────┐
 HF URL/alias ──▶ inspect │ HF Hub API · GGUF headers  │
                       └──────────────┬──────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │ planning: VRAM · disk ·      │
                       │ backend · variant tiers      │
                       └──────────────┬──────────────┘
                                      ▼
                       ┌─────────────────────────────┐      ┌──────────────┐
                       │ ranking: expected session   │◀─────│ Vast.ai API  │
                       │ cost + explainable reasons  │      │ (offers)     │
                       └──────────────┬──────────────┘      └──────────────┘
                                      ▼
                     quote → you confirm → (M2: provision & serve)
```

| Works today | Coming (M2+) |
|---|---|
| `hfvast inspect <model>` — variants, sizes, arch, support level | `hfvast up <model>` — provision & serve |
| `hfvast quote <model>` — VRAM/disk plan + live Vast offers + costs | `hfvast ps / status / logs / cost` |
| `hfvast doctor`, `hfvast alias add/rm/list` | Idle timeout + hard max-runtime auto-destroy |
| GGUF (incl. multi-shard) + safetensors inspection | Gateway, streaming, OpenAI-compatible endpoint |

## Cost safety

- Nothing is ever provisioned without explicit confirmation; `--dry-run`/`quote` never can.
- Hard caps: `--max-hourly-cost`, `--max-startup-cost`, `--max-total-cost`, `--max-runtime`.
- Idle timeout (default 30 min) + independent fail-safe watchdog destroy the instance; failure
  during bootstrap destroys it too and reports the incurred estimate.
- See [SECURITY.md](SECURITY.md) for the credential/third-party-host threat model.

## Development

```bash
uv sync && uv run pytest && uv run ruff check . && uv run mypy src
```

Design docs: [docs/architecture.md](docs/architecture.md) · API research: [docs/research.md](docs/research.md) ·
[CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache-2.0
