"""Remote GGUF inspection: variant grouping and header parsing over HTTP ranges.

GGUF binary format per https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
(value types 0–12; STRING = u64 len + bytes; ARRAY = u32 elem type + u64 count).
Hugging Face ``resolve`` endpoints support Range requests end-to-end (verified
2026-09-02), so a header can be read with a few small ranged GETs — never the weights.
"""

from __future__ import annotations

import re
import struct
from typing import Any

import httpx

from hfvast.errors import GGUFHeaderError
from hfvast.inspect.quantization import detect_quant, tier_for_quant
from hfvast.models.model import GGUFHeaderInfo, ModelFile, ModelVariant

SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)

GGUF_MAGIC = b"GGUF"

# value type -> (struct format, byte size)
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),  # UINT8
    1: ("<b", 1),  # INT8
    2: ("<H", 2),  # UINT16
    3: ("<h", 2),  # INT16
    4: ("<I", 4),  # UINT32
    5: ("<i", 4),  # INT32
    6: ("<f", 4),  # FLOAT32
    7: ("<?", 1),  # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}

MAX_STORED_ARRAY = 64  # store array values only when small; otherwise just record count


class _RangeStream:
    """Sequential byte reader over ranged HTTP GETs."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        token: str | None,
        chunk_size: int = 4 * 1024 * 1024,
        max_bytes: int = 48 * 1024 * 1024,
    ) -> None:
        self._client = client
        self._url = url
        self._token = token
        self._chunk = chunk_size
        self._max = max_bytes
        self._buffer = b""
        self._buffer_base = 0  # absolute offset of buffer[0]
        self._pos = 0  # absolute read position
        self._fetched = 0
        self._range_ok: bool | None = None

    def _tell(self) -> int:
        return self._pos

    async def _fetch(self, start: int, size: int) -> None:
        if self._fetched >= self._max:
            raise GGUFHeaderError(f"GGUF header exceeds the {self._max // (1024 * 1024)} MB inspection budget")
        end = start + size - 1
        headers = {"Range": f"bytes={start}-{end}"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            async with self._client.stream("GET", self._url, headers=headers) as resp:
                if resp.status_code == 401:
                    raise GGUFHeaderError("GGUF file is not readable without a token (gated repository)")
                if resp.status_code == 404:
                    raise GGUFHeaderError("GGUF file not found (404)")
                resp.raise_for_status()
                if self._range_ok is None:
                    self._range_ok = resp.status_code == 206
                data = b""
                async for piece in resp.aiter_bytes(1024 * 1024):
                    data += piece
                    if len(data) >= size:
                        break
        except httpx.HTTPError as exc:
            raise GGUFHeaderError(f"Failed fetching GGUF header: {exc}") from exc
        # Carry over any unconsumed tail so reads spanning chunk boundaries work.
        tail_start = self._pos - self._buffer_base
        tail = self._buffer[tail_start:] if tail_start < len(self._buffer) else b""
        self._buffer = tail + data
        self._buffer_base = self._pos
        self._fetched += len(data)

    async def _ensure(self, n: int) -> bool:
        available = len(self._buffer) - (self._pos - self._buffer_base)
        if available >= n:
            return True
        next_start = self._buffer_base + len(self._buffer)
        # If range requests are unsupported (plain 200s), we cannot skip ahead.
        if self._range_ok is False and next_start > 0:
            return False
        await self._fetch(next_start, max(self._chunk, n))
        return len(self._buffer) - (self._pos - self._buffer_base) >= n

    async def _read(self, n: int) -> bytes:
        if not await self._ensure(n):
            raise GGUFHeaderError("Unexpected end of GGUF header data")
        offset = self._pos - self._buffer_base
        data = self._buffer[offset : offset + n]
        self._pos += n
        return data

    async def _skip(self, n: int) -> None:
        # Only reachable when the bytes are within already-buffered data or the
        # next fetched chunk; large arrays advance through the byte budget too.
        remaining = n
        while remaining > 0:
            if not await self._ensure(min(remaining, self._chunk)):
                raise GGUFHeaderError("Unexpected end of GGUF header data while skipping")
            offset = self._pos - self._buffer_base
            in_buffer = len(self._buffer) - offset
            take = min(remaining, in_buffer)
            self._pos += take
            remaining -= take

    async def _unpack(self, fmt: str, size: int) -> Any:
        data = await self._read(size)
        return struct.unpack(fmt, data)[0]

    async def read_string(self) -> str:
        length = await self._unpack("<Q", 8)
        data = await self._read(length)
        return data.decode("utf-8", errors="replace")

    async def skip_value(self, vtype: int) -> None:
        if vtype == 8:
            length = await self._unpack("<Q", 8)
            await self._skip(length)
        elif vtype == 9:
            elem_type = await self._unpack("<I", 4)
            count = await self._unpack("<Q", 8)
            if elem_type == 8:
                for _ in range(count):
                    length = await self._unpack("<Q", 8)
                    await self._skip(length)
            elif elem_type in _SCALARS:
                await self._skip(_SCALARS[elem_type][1] * count)
            else:
                raise GGUFHeaderError(f"Unsupported array element type {elem_type}")
        elif vtype in _SCALARS:
            await self._skip(_SCALARS[vtype][1])
        else:
            raise GGUFHeaderError(f"Unsupported GGUF value type {vtype}")

    async def read_value(self, vtype: int) -> Any:
        if vtype == 8:
            return await self.read_string()
        if vtype == 9:
            elem_type = await self._unpack("<I", 4)
            count = await self._unpack("<Q", 8)
            if elem_type == 8:
                values: list[Any] = []
                for _ in range(min(count, MAX_STORED_ARRAY)):
                    values.append(await self.read_string())
                if count > MAX_STORED_ARRAY:
                    for _ in range(count - MAX_STORED_ARRAY):
                        length = await self._unpack("<Q", 8)
                        await self._skip(length)
                return values
            if elem_type in _SCALARS:
                fmt, size = _SCALARS[elem_type]
                if count <= MAX_STORED_ARRAY:
                    return [await self._unpack(fmt, size) for _ in range(count)]
                await self._skip(size * count)
                return None
            raise GGUFHeaderError(f"Unsupported array element type {elem_type}")
        if vtype in _SCALARS:
            fmt, size = _SCALARS[vtype]
            return await self._unpack(fmt, size)
        raise GGUFHeaderError(f"Unsupported GGUF value type {vtype}")


#: general.file_type enum (ggml spec; llama.cpp extends beyond 18 — unknowns stay None).
FILE_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
}


async def read_gguf_header(
    client: httpx.AsyncClient,
    url: str,
    token: str | None = None,
) -> dict[str, Any]:
    """Parse the metadata KVs of a remote GGUF file into a plain dict.

    Only scalar values and small arrays are retained; large tokenizer arrays are
    skipped (their bytes still flow through the read budget, which is capped).
    """
    stream = _RangeStream(client, url, token)
    magic = await stream._read(4)
    if magic != GGUF_MAGIC:
        raise GGUFHeaderError("not a GGUF file (bad magic)")
    version = await stream._unpack("<I", 4)
    if version < 2:
        raise GGUFHeaderError(f"unsupported GGUF version {version}")
    await stream._unpack("<Q", 8)  # tensor_count
    kv_count = await stream._unpack("<Q", 8)

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = await stream.read_string()
        vtype = await stream._unpack("<I", 4)
        # Stop early once the (huge) tokenizer section begins — everything we need
        # precedes it in llama.cpp-produced files.
        if key.startswith("tokenizer.") and key != "tokenizer.chat_template":
            await stream.skip_value(vtype)
            break
        try:
            metadata[key] = await stream.read_value(vtype)
        except GGUFHeaderError:
            await stream.skip_value(vtype)
            metadata[key] = None
    return metadata


def header_info_from_metadata(metadata: dict[str, Any]) -> GGUFHeaderInfo:
    """Build a :class:`GGUFHeaderInfo` from parsed metadata KVs."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str):
        arch = None
    file_type = metadata.get("general.file_type")
    head_count = _as_int(metadata.get(f"{arch}.attention.head_count") if arch else None)
    head_count_kv = _as_int(metadata.get(f"{arch}.attention.head_count_kv") if arch else None)
    return GGUFHeaderInfo(
        architecture=arch,
        file_type=_as_int(file_type),
        file_type_name=FILE_TYPE_NAMES.get(_as_int(file_type) or -1),
        name=_as_str(metadata.get("general.name")),
        context_length=_as_int(metadata.get(f"{arch}.context_length") if arch else None),
        block_count=_as_int(metadata.get(f"{arch}.block_count") if arch else None),
        head_count=head_count,
        head_count_kv=head_count_kv if head_count_kv is not None else head_count,
        key_length=_as_int(metadata.get(f"{arch}.attention.key_length") if arch else None),
        value_length=_as_int(metadata.get(f"{arch}.attention.value_length") if arch else None),
        embedding_length=_as_int(metadata.get(f"{arch}.embedding_length") if arch else None),
        expert_count=_as_int(metadata.get(f"{arch}.expert_count") if arch else None),
        expert_used_count=_as_int(metadata.get(f"{arch}.expert_used_count") if arch else None),
    )


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def group_gguf_variants(files: list[ModelFile]) -> list[ModelVariant]:
    """Group GGUF files into variants; split shards form exactly one variant each.

    Ordering follows quantization tier quality (best last: economy → balanced → quality),
    which keeps CLI tables stable.
    """
    split_groups: dict[str, list[ModelFile]] = {}
    singles: list[ModelFile] = []
    for f in sorted(files, key=lambda f: f.path):
        match = SHARD_RE.match(f.path.rsplit("/", 1)[-1])
        if match:
            split_groups.setdefault(match.group("prefix"), []).append(f)
        else:
            singles.append(f)

    variants: list[ModelVariant] = []
    for prefix, shards in split_groups.items():
        quant = detect_quant(prefix.rsplit("/", 1)[-1])
        variant_id = quant or prefix.rsplit("/", 1)[-1]
        variants.append(
            ModelVariant(
                id=variant_id,
                quant=quant,
                size_bytes=sum(s.size_bytes for s in shards),
                files=sorted(shards, key=lambda s: s.path),
                tier=tier_for_quant(quant),
                is_split=True,
            )
        )
    for single in singles:
        quant = detect_quant(single.path.rsplit("/", 1)[-1])
        variant_id = quant or single.path.rsplit("/", 1)[-1].removesuffix(".gguf")
        variants.append(
            ModelVariant(
                id=variant_id,
                quant=quant,
                size_bytes=single.size_bytes,
                files=[single],
                tier=tier_for_quant(quant),
                is_split=False,
            )
        )

    tier_rank = {None: 3, "economy": 0, "balanced": 1, "quality": 2}
    variants.sort(key=lambda v: (tier_rank.get(v.tier.value if v.tier else None, 3), v.size_bytes))
    return variants
