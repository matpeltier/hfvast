# hfvast runtime images

Versioned OCI images carrying the backend + hfvast bootstrap payload
(gateway, watchdog, downloader, health probes) — no model weights.

Deployment currently uses the pinned **upstream** backend image and injects the
bootstrap payload at instance creation (see `src/hfvast/deploy/bootstrap.py`);
these Dockerfiles are the durable path and are published by the trusted
release workflow (`.github/workflows/release-images.yml`) as:

    ghcr.io/<project>/hfvast-llama-cpp:<version>
    ghcr.io/<project>/hfvast-vllm:<version>
    ghcr.io/<project>/hfvast-sglang:<version>

The gateway/watchdog sources are copied from `src/hfvast/runtime/`.
