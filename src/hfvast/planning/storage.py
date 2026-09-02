"""Disk requirement estimation (spec §16).

required_disk = model download size + runtime headroom + temporary margin

The V1 download path streams GGUF shards straight into their final location, so
no duplicate HF-cache copy is planned (spec §16 "avoid downloading twice").
"""

from __future__ import annotations

from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.utils.units import ceil_to

RUNTIME_HEADROOM_GB = 20.0  # container image unpack, CUDA libs, tokenizer, bootstrap
MIN_DISK_GB = 24.0
GB = 1e9


def estimate_disk_gb(model_info: ModelInfo, variant: ModelVariant) -> float:
    weights_gb = variant.size_bytes / GB
    mmproj_gb = sum(f.size_bytes for f in model_info.mmproj_files) / GB
    temp_gb = max(10.0, 0.05 * weights_gb)
    total = weights_gb + mmproj_gb + RUNTIME_HEADROOM_GB + temp_gb
    return max(ceil_to(total, 8.0), MIN_DISK_GB)
