# Security Policy

## Reporting a vulnerability

Please open a private security advisory via GitHub ("Security" → "Report a vulnerability") rather
than a public issue. We aim to acknowledge reports within 72 hours.

## Security model and assumptions

hfvast rents **third-party marketplace machines**. Assume the host can read anything inside the
container: model weights, environment variables, memory, and disk.

### Credentials

- `VAST_API_KEY` and `HF_TOKEN` are read from the environment (preferred) or CLI flags (discouraged:
  flags leak into shell history and process listings). They are **never** persisted to state/config
  files, never printed, and never included in logs, error messages, or tracebacks. A central
  redactor scrubs known secrets from all output paths.
- Prefer a **fine-grained, read-only Hugging Face token scoped to the specific repos you need**,
  especially when deploying to non-verified hosts. hfvast sends this token to the rented machine to
  download gated models; that is unavoidable for gated repos.
- The unrestricted Vast API key never leaves your machine. Instances are created with Vast's
  per-instance restricted key (`instance_api_key`, self-start/stop/destroy only) injected as
  `CONTAINER_API_KEY` for the in-container self-destroy watchdog.
- Deployments require `--secure-cloud-only` to restrict offers to verified/secure hosts if you
  cannot tolerate third-party hosts.

### Network

- Every deployment generates a cryptographically random API key (`sk-hfvast-…`); the inference
  backend never faces the internet directly — the authenticating gateway is the only public surface.
- If HTTPS cannot be guaranteed end-to-end, hfvast says so explicitly and never silently sends
  prompts over plain HTTP.

### Auto-destruction

There is no Vast-native TTL. hfvast runs a local lifecycle daemon and, as a fail-safe, an
in-container watchdog with destroy-only permissions. If your laptop sleeps, the local daemon stops;
the in-container watchdog still enforces the idle timeout and hard max runtime.

## Scope

hfvast is ephemeral infrastructure, not a multi-tenant platform: it never trains models, never runs
arbitrary user code, and never allows unauthenticated access to a public inference endpoint.
