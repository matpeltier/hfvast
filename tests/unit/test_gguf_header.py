import re

import httpx
import pytest

from hfvast.errors import GGUFHeaderError
from hfvast.inspect.gguf import _RangeStream, read_gguf_header

# ---------------------------------------------------------------- GGUF builder


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return len(raw).to_bytes(8, "little") + raw


def build_gguf(kvs: list[tuple[str, object]], version: int = 3) -> bytes:
    """Build a minimal but spec-conformant GGUF header (metadata only)."""
    out = bytearray()
    out += b"GGUF"
    out += version.to_bytes(4, "little")
    out += (0).to_bytes(8, "little")  # tensor_count
    out += len(kvs).to_bytes(8, "little")
    for key, value in kvs:
        out += _gguf_string(key)
        if isinstance(value, str):
            out += (8).to_bytes(4, "little")
            out += _gguf_string(value)
        elif isinstance(value, bool):
            out += (7).to_bytes(4, "little")
            out += int(value).to_bytes(1, "little")
        elif isinstance(value, list):
            if value and isinstance(value[0], float):
                out += (9).to_bytes(4, "little")
                out += (6).to_bytes(4, "little")  # elem type: FLOAT32
                out += len(value).to_bytes(8, "little")
                import struct

                for item in value:
                    out += struct.pack("<f", item)
            else:
                out += (9).to_bytes(4, "little")
                out += (8).to_bytes(4, "little")  # elem type: string
                out += len(value).to_bytes(8, "little")
                for item in value:
                    out += _gguf_string(item)
        elif isinstance(value, int):
            out += (10).to_bytes(4, "little")  # UINT64
            out += value.to_bytes(8, "little")
        elif isinstance(value, float):
            out += (6).to_bytes(4, "little")  # FLOAT32
            import struct

            out += struct.pack("<f", value)
        else:
            raise TypeError(f"unsupported test value {value!r}")
    return bytes(out)


def ranged_transport(data: bytes, status_for: dict[str, int] | None = None):
    """httpx.MockTransport honoring Range headers (emulating the HF CDN)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status_for and request.url.path in status_for:
            return httpx.Response(status_for[request.url.path], text="nope")
        range_header = request.headers.get("Range", "")
        match = re.match(r"bytes=(\d+)-(\d+)?", range_header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start + 1024 * 1024
            end = min(end, len(data) - 1)
            body = data[start : end + 1]
            return httpx.Response(
                206,
                content=body,
                headers={"Content-Range": f"bytes {start}-{end}/{len(data)}", "accept-ranges": "bytes"},
            )
        return httpx.Response(200, content=data, headers={"accept-ranges": "bytes"})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------- tests


async def test_reads_scalar_metadata():
    data = build_gguf(
        [
            ("general.architecture", "qwen2"),
            ("general.name", "Qwen2.5 7B"),
            ("general.file_type", 15),
            ("qwen2.context_length", 32768),
            ("qwen2.block_count", 28),
            ("qwen2.attention.head_count", 28),
            ("qwen2.attention.head_count_kv", 4),
            ("qwen2.attention.key_length", 128),
            ("qwen2.embedding_length", 3584),
            ("qwen2.expert_count", 0),
        ]
    )
    transport = ranged_transport(data)
    async with httpx.AsyncClient(transport=transport) as client:
        metadata = await read_gguf_header(client, "https://hf.example/repo/file.gguf")
    assert metadata["general.architecture"] == "qwen2"
    assert metadata["qwen2.block_count"] == 28
    assert metadata["qwen2.attention.head_count_kv"] == 4
    assert metadata["general.file_type"] == 15


async def test_skips_large_tokenizer_arrays_and_stops():
    data = build_gguf(
        [
            ("general.architecture", "llama"),
            ("llama.context_length", 8192),
            ("llama.block_count", 32),
            ("tokenizer.ggml.tokens", [f"tok{i}" for i in range(5000)]),
            ("tokenizer.ggml.scores", [1.0] * 10),
            ("never.reached", 1),
        ]
    )
    transport = ranged_transport(data)
    async with httpx.AsyncClient(transport=transport) as client:
        metadata = await read_gguf_header(client, "https://hf.example/repo/file.gguf")
    assert metadata["llama.block_count"] == 32
    assert "tokenizer.ggml.tokens" not in metadata
    assert "never.reached" not in metadata  # parsing stopped at the tokenizer section


async def test_multi_fetch_parsing_across_chunk_boundaries():
    data = build_gguf(
        [
            ("general.architecture", "glm4moe"),
            ("general.name", "GLM-4.5"),
            ("glm4moe.context_length", 131072),
            ("glm4moe.block_count", 92),
            ("glm4moe.attention.head_count", 96),
            ("glm4moe.attention.head_count_kv", 8),
            ("glm4moe.expert_count", 160),
        ]
    )
    transport = ranged_transport(data)
    stream = _RangeStream(
        client=httpx.AsyncClient(transport=transport),
        url="https://hf.example/repo/file.gguf",
        token=None,
        chunk_size=8,  # tiny chunks → forces many ranged fetches
        max_bytes=64 * 1024,
    )
    assert await stream._read(4) == b"GGUF"  # magic
    await stream._read(4)  # version
    await stream._read(8)  # tensor_count
    await stream._read(8)  # kv_count
    key = await stream.read_string()
    assert key == "general.architecture"
    value_type = await stream._unpack("<I", 4)
    assert await stream.read_value(value_type) == "glm4moe"


async def test_gated_file_raises_header_error():
    data = build_gguf([("general.architecture", "llama")])
    transport = ranged_transport(data, status_for={"/repo/resolve/main/file.gguf": 401})
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(GGUFHeaderError):
            await read_gguf_header(client, "https://hf.example/repo/resolve/main/file.gguf")


async def test_bad_magic_raises():
    transport = ranged_transport(b"NOPE" + b"\x00" * 64)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(GGUFHeaderError):
            await read_gguf_header(client, "https://hf.example/repo/file.gguf")
