"""Hugging Face Hub inspector.

Reads repository metadata and per-file sizes via the public Hub API — never
downloads weights. GGUF headers are parsed remotely with a few ranged GETs.

Verified against live API behavior 2026-09-02 (see docs/research.md):
  * GET /api/models/{repo} includes `gguf`, `gated`, `config`, `safetensors`,
    `pipeline_tag`, `tags`, `usedStorage` by default.
  * GET /api/models/{repo}/tree/{rev}?recursive=true yields per-file sizes,
    paginated via a `Link: <...>; rel="next"` header.
  * Gated repos: resolve → 401 with `x-error-code: GatedRepo`; nonexistent repos
    deliberately return 401 as well; tree metadata stays readable.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import httpx

from hfvast.errors import GatedAccessError, HFHubError, ModelNotFoundError, ModelNotSupportedError
from hfvast.inspect import transformers as tf
from hfvast.inspect.gguf import group_gguf_variants, header_info_from_metadata, read_gguf_header
from hfvast.models.model import (
    HFModelRef,
    ModelFile,
    ModelFormat,
    ModelInfo,
    ModelTask,
    ModelVariant,
    QuantTier,
)
from hfvast.utils.redact import redact

_BASE = "https://huggingface.co"
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')

_TASK_MAP = {
    "text-generation": ModelTask.TEXT_GENERATION,
    "image-text-to-text": ModelTask.VISION_LANGUAGE,
    "visual-question-answering": ModelTask.VISION_LANGUAGE,
    "any-to-any": ModelTask.VISION_LANGUAGE,
    "text-to-video": ModelTask.VIDEO_GENERATION,
    "image-text-to-video": ModelTask.VIDEO_GENERATION,
    "video-classification": ModelTask.VIDEO_GENERATION,
    "text-to-image": ModelTask.IMAGE_GENERATION,
    "unconditional-image-generation": ModelTask.IMAGE_GENERATION,
    "image-to-image": ModelTask.IMAGE_GENERATION,
}

_TEXTY_TAGS = {"text-generation", "conversational"}


def _map_task(pipeline_tag: str | None, tags: list[str]) -> ModelTask:
    if pipeline_tag:
        if pipeline_tag in _TASK_MAP:
            return _TASK_MAP[pipeline_tag]
        return ModelTask.OTHER
    if any(tag in _TEXTY_TAGS for tag in tags):
        return ModelTask.TEXT_GENERATION
    return ModelTask.UNKNOWN


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class HFInspector:
    """Async inspector for Hugging Face model repositories."""

    def __init__(self, client: httpx.AsyncClient | None = None, token: str | None = None) -> None:
        self._client = client or httpx.AsyncClient(follow_redirects=True, timeout=60.0)
        self._owns_client = client is None
        self._token = token

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _api_url(self, ref: HFModelRef) -> str:
        repo = quote(ref.repo_id, safe="/")
        url = f"{_BASE}/api/models/{repo}"
        if ref.revision:
            url += f"/revision/{quote(ref.revision, safe='')}"
        return url

    def _tree_url(self, ref: HFModelRef, revision: str) -> str:
        repo = quote(ref.repo_id, safe="/")
        return f"{_BASE}/api/models/{repo}/tree/{quote(revision, safe='')}?recursive=true&limit=1000"

    def _resolve_url(self, ref: HFModelRef, revision: str, path: str) -> str:
        repo = quote(ref.repo_id, safe="/")
        return f"{_BASE}/{repo}/resolve/{quote(revision, safe='')}/{quote(path)}"

    async def _get_json(self, url: str) -> Any:
        try:
            resp = await self._client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise HFHubError(f"Hugging Face is unreachable: {redact(exc)}") from exc
        return self._decode(resp, url)

    def _decode(self, resp: httpx.Response, url: str) -> Any:
        if resp.status_code in (401, 403):
            code = resp.headers.get("x-error-code", "")
            if code == "GatedRepo" or "gated" in resp.headers.get("x-error-message", "").lower():
                if self._token:
                    # repo id from the URL: .../api/models/<org>/<name>[/revision/<rev>]
                    tail = url.split("/api/models/")[-1]
                    repo_path = "/".join(tail.split("/revision/")[0].split("/")[:2])
                    raise GatedAccessError(
                        "This repository is gated and your account does not have access yet "
                        "(Hugging Face returned HTTP 403 for the authenticated user).\n"
                        f"  → Open https://huggingface.co/{repo_path} and click "
                        "'Agree and access repository' "
                        "(gated:'auto' grants instantly; gated:'manual' needs approval).\n"
                        "  → Then retry. No cloud resources were affected."
                    )
                raise GatedAccessError(
                    "This repository is gated: a valid HF_TOKEN with access is required.\n"
                    "  - Set HF_TOKEN to a fine-grained, read-only token scoped to this repo.\n"
                    "  - Then visit the model page and accept the license "
                    "(requesting access is a web-UI step)."
                )
            if resp.status_code == 401:
                raise ModelNotFoundError(
                    "Hugging Face returned 401 for this repository.\n"
                    "It may not exist (the Hub hides private/missing repos behind 401), "
                    "or your HF_TOKEN is invalid."
                )
            raise HFHubError(
                f"Hugging Face returned 403 for {url}.\n"
                "Possible causes: the token lacks this repo's scope (fine-grained tokens must "
                "explicitly include it), or you have not accepted the gated license on the "
                "model page.\n"
                f"Server message: {redact(resp.text[:200])}"
            )
        if resp.status_code == 404:
            raise ModelNotFoundError(f"Repository not found: {url}")
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HFHubError(f"Hugging Face API error {resp.status_code} for {url}: {redact(exc)}") from exc
        return resp.json()

    async def _list_tree(self, ref: HFModelRef, revision: str) -> list[ModelFile]:
        files: list[ModelFile] = []
        url: str | None = self._tree_url(ref, revision)
        while url:
            try:
                resp = await self._client.get(url, headers=self._headers())
            except httpx.HTTPError as exc:
                raise HFHubError(f"Hugging Face is unreachable: {redact(exc)}") from exc
            data = self._decode(resp, url)
            if not isinstance(data, list):
                raise HFHubError(f"Unexpected tree response for {url}")
            for entry in data:
                if entry.get("type") != "file":
                    continue
                files.append(ModelFile(path=entry.get("path", ""), size_bytes=int(entry.get("size") or 0)))
            match = _LINK_NEXT_RE.search(resp.headers.get("Link", ""))
            url = match.group(1) if match else None
        return files

    async def inspect(self, ref: HFModelRef) -> ModelInfo:
        api = await self._get_json(self._api_url(ref))
        if not isinstance(api, dict):
            raise HFHubError(f"Unexpected model-info response for {ref.repo_id}")
        revision = ref.revision or (api.get("sha") if isinstance(api.get("sha"), str) else None) or "main"
        files = await self._list_tree(ref, revision)

        tags = [t for t in api.get("tags", []) if isinstance(t, str)]
        pipeline_tag = api.get("pipeline_tag") if isinstance(api.get("pipeline_tag"), str) else None
        task = _map_task(pipeline_tag, tags)

        notes: list[str] = []
        gated = api.get("gated") in (True, "auto", "manual")
        if gated:
            notes.append("Repository is gated: downloads require an authorized HF_TOKEN.")

        gguf_files = [f for f in files if f.path.lower().endswith(".gguf")]
        safetensor_files = [f for f in files if f.path.lower().endswith(".safetensors") and "adapter" not in f.path]
        mmproj_files = [f for f in gguf_files if "mmproj" in f.path.rsplit("/", 1)[-1].lower()]
        weight_gguf = [f for f in gguf_files if f not in mmproj_files]

        raw_config = api.get("config")
        api_config: dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}

        if await self._lora_signal(api, files, revision, notes, ref):
            info = await self._inspect_lora_adapter(
                ref,
                api,
                True,
                [f for f in files if "adapter" in f.path.lower()],
                task,
                notes,
                revision,
            )
        elif gguf_files or api.get("library_name") == "gguf":
            info = await self._inspect_gguf(ref, api, api_config, weight_gguf, mmproj_files, task, notes, revision)
        elif safetensor_files:
            info = await self._inspect_safetensors(ref, api, api_config, safetensor_files, task, notes, revision)
        else:
            detected = ", ".join(sorted({f.path.rsplit("/", 1)[-1] for f in files[:6]}))
            raise ModelNotSupportedError(
                "No deployable weights found (expected GGUF or safetensors).",
                detected=detected or "no weight files",
                possible_backend="none — hfvast V1 serves GGUF and safetensors causal LMs only",
            )

        info.gated = gated
        info.notes = notes
        return info

    _KNOWN_ARCH_SUFFIXES = (
        "ForCausalLM",
        "ForConditionalGeneration",
        "ForSequenceClassification",
        "ForTokenClassification",
        "ForQuestionAnswering",
        "ForMaskedLM",
        "LMHeadModel",
        "ChatModel",
        "ModelForCausalLM",
    )

    async def _lora_signal(
        self,
        api: dict[str, Any],
        files: list[ModelFile],
        revision: str,
        notes: list[str],
        ref: HFModelRef,
    ) -> bool:
        """LoRA/PEFT adapter repos: adapter_* weights, peft tags/library, stub configs,
        or (heuristic) a README that says "lora" for a repo with a non-standard
        architecture and no task metadata (community LoRA dumps, live-verified
        2026-09-03 on AfterMidnight-MiniMax-H3)."""
        if any("adapter" in f.path.lower() for f in files):
            return True
        tags = {str(t).lower() for t in api.get("tags", [])}
        if api.get("library_name") == "peft" or tags & {"lora", "peft"}:
            return True
        raw_cfg = api.get("config")
        config: dict[str, Any] = raw_cfg if isinstance(raw_cfg, dict) else {}
        if config.get("peft_type"):
            return True

        # Heuristic: no task metadata + a non-transformers architecture name +
        # a README mentioning "lora" → community adapter dump.
        raw_tag = api.get("pipeline_tag")
        if isinstance(raw_tag, str) and raw_tag:
            return False
        archs = config.get("architectures")
        arch = str(archs[0]) if isinstance(archs, list) and archs else ""
        if arch.endswith(self._KNOWN_ARCH_SUFFIXES):
            return False
        readme = await self._fetch_text_file(ref, revision, "README.md", limit=16384)
        if readme and "lora" in readme.lower():
            notes.append("LoRA detected from repository README (no PEFT metadata present)")
            return True
        return False

    async def _inspect_lora_adapter(
        self,
        ref: HFModelRef,
        api: dict[str, Any],
        _signal: bool,
        weight_files: list[ModelFile],
        task: ModelTask,
        notes: list[str],
        revision: str,
    ) -> ModelInfo:
        size = sum(f.size_bytes for f in weight_files)
        adapter_config = await self._fetch_json_file(ref, revision, "adapter_config.json")
        cfg: dict[str, Any] = adapter_config if isinstance(adapter_config, dict) else {}
        base_model = str(cfg.get("base_model_name_or_path") or "")
        if not base_model:
            raw_card = api.get("cardData")
            card: dict[str, Any] = raw_card if isinstance(raw_card, dict) else {}
            base_model = str(card.get("base_model") or "")

        # PEFT serving layout (required by vLLM --enable-lora):
        #   adapter_config.json + adapter_model.safetensors (or .bin)
        names = {f.path.rsplit("/", 1)[-1].lower() for f in weight_files}
        has_adapter_weights = "adapter_model.safetensors" in names or "adapter_model.bin" in names
        peft_layout = bool(cfg) and has_adapter_weights

        if base_model:
            notes.append(f"base model: {base_model}")
        else:
            notes.append(
                "base model: unknown — the repo has no adapter_config.json and the card "
                "doesn't name the base; pass --base-model <org/name>"
            )
        if not peft_layout:
            notes.append(
                "adapter is NOT in the PEFT serving layout (adapter_config.json + "
                "adapter_model.safetensors) — vLLM cannot hot-load it; merge it into the "
                "base model instead"
            )
        if len(weight_files) > 1:
            notes.append(f"{len(weight_files)} adapter checkpoints found (alternative versions, not shards)")
        variant = ModelVariant(id="lora-adapter", quant=None, size_bytes=size, files=weight_files)
        return ModelInfo(
            ref=ref,
            task=task,
            format=ModelFormat.LORA_ADAPTER,
            weight_bytes=size,
            variants=[variant],
            base_model_ref=base_model or None,
            peft_layout=peft_layout,
            notes=notes,
        )

    async def _fetch_text_file(self, ref: HFModelRef, revision: str, path: str, limit: int = 65536) -> str:
        url = self._resolve_url(ref, revision, path)
        try:
            resp = await self._client.get(url, headers=self._headers())
        except httpx.HTTPError:
            return ""
        if resp.status_code != 200:
            return ""
        return resp.text[:limit]

    async def _fetch_json_file(self, ref: HFModelRef, revision: str, path: str) -> Any:
        url = self._resolve_url(ref, revision, path)
        try:
            resp = await self._client.get(url, headers=self._headers())
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    # ------------------------------------------------------------------ GGUF

    async def _inspect_gguf(
        self,
        ref: HFModelRef,
        api: dict[str, Any],
        api_config: dict[str, Any],
        weight_gguf: list[ModelFile],
        mmproj_files: list[ModelFile],
        task: ModelTask,
        notes: list[str],
        revision: str,
    ) -> ModelInfo:
        if not weight_gguf:
            raise ModelNotSupportedError(
                "Repository contains GGUF projector files (mmproj) but no model weights.",
                detected=", ".join(f.path for f in mmproj_files[:3]),
                possible_backend="none — this repo does not contain a servable model",
            )
        raw_gguf = api.get("gguf")
        api_gguf: dict[str, Any] = raw_gguf if isinstance(raw_gguf, dict) else {}
        variants = group_gguf_variants(weight_gguf)
        if mmproj_files:
            names = ", ".join(f.path for f in mmproj_files)
            notes.append(
                f"Multimodal projector (mmproj) detected: {names}. "
                "V1 serves text generation only; mmproj is included in disk sizing."
            )

        header = None
        representative = self._representative_variant(variants)
        if representative is not None and representative.files:
            shard = representative.files[0]
            url = self._resolve_url(ref, revision, shard.path)
            try:
                metadata = await read_gguf_header(self._client, url, token=self._token)
                header = header_info_from_metadata(metadata)
            except Exception as exc:
                notes.append(
                    f"GGUF header not readable ({exc.__class__.__name__}: {redact(str(exc))}); "
                    "using conservative architecture estimates for KV-cache math."
                )

        architecture = (
            (header.architecture if header else None)
            or api_gguf.get("architecture")
            or tf.architecture_from_config(api_config)
        )
        context_length = (
            (header.context_length if header else None)
            or _as_int(api_gguf.get("context_length"))
            or tf.context_length_from_config(api_config)
        )
        parameter_count = _as_int(api_gguf.get("total"))
        weight_bytes = sum(v.size_bytes for v in variants) or _as_int(api.get("usedStorage"))
        quantization = variants[0].quant if len(variants) == 1 else None

        return ModelInfo(
            ref=ref,
            task=task,
            architecture=architecture if isinstance(architecture, str) else None,
            format=ModelFormat.GGUF,
            dtype=header.file_type_name if header else None,
            parameter_count=parameter_count,
            weight_bytes=weight_bytes,
            context_length=context_length,
            quantization=quantization,
            variants=variants,
            multimodal=bool(mmproj_files) or task is ModelTask.VISION_LANGUAGE,
            mmproj_files=mmproj_files,
            gguf_header=header,
        )

    @staticmethod
    def _representative_variant(variants: list[ModelVariant]) -> ModelVariant | None:
        for tier in (QuantTier.BALANCED, QuantTier.QUALITY, QuantTier.ECONOMY):
            for variant in variants:
                if variant.tier is tier:
                    return variant
        return variants[0] if variants else None

    # ----------------------------------------------------------- safetensors

    async def _inspect_safetensors(
        self,
        ref: HFModelRef,
        api: dict[str, Any],
        api_config: dict[str, Any],
        safetensor_files: list[ModelFile],
        task: ModelTask,
        notes: list[str],
        revision: str,
    ) -> ModelInfo:
        config = await self._fetch_config_json(ref, revision)
        raw_st = api.get("safetensors")
        params: dict[str, Any] = raw_st if isinstance(raw_st, dict) else {}
        raw_params = params.get("parameters")
        param_counts: dict[str, int] | None = (
            {str(k): int(v) for k, v in raw_params.items()} if isinstance(raw_params, dict) else None
        )

        quant_label = tf.quantization_label_from_config(config)
        if quant_label:
            notes.append(f"Quantized weights detected: {quant_label}.")
        architecture = tf.architecture_from_config(config) or tf.architecture_from_config(api_config)
        requires_trc = tf.requires_trust_remote_code(config)
        if requires_trc:
            notes.append(
                "Repository declares custom modeling code (auto_map). hfvast V1 does not run "
                "arbitrary remote code (spec §9), so trust_remote_code models are not deployable."
            )
        weight_bytes = sum(f.size_bytes for f in safetensor_files) or _as_int(api.get("usedStorage"))

        variant = ModelVariant(id="safetensors", quant=None, size_bytes=weight_bytes or 0, files=safetensor_files)

        return ModelInfo(
            ref=ref,
            task=task,
            architecture=architecture,
            format=ModelFormat.SAFETENSORS,
            dtype=tf.dtype_label(config, param_counts),
            parameter_count=_as_int(params.get("total")),
            weight_bytes=weight_bytes,
            context_length=tf.context_length_from_config(config) or tf.context_length_from_config(api_config),
            quantization=quant_label,
            variants=[variant],
            requires_trust_remote_code=requires_trc,
            multimodal=task is ModelTask.VISION_LANGUAGE,
            safetensors_params=param_counts,
            quantization_config=(
                config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else None
            ),
        )

    async def _fetch_config_json(self, ref: HFModelRef, revision: str) -> dict[str, Any]:
        url = self._resolve_url(ref, revision, "config.json")
        try:
            resp = await self._client.get(url, headers=self._headers())
        except httpx.HTTPError:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}
