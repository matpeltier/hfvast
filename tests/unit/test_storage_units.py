import math

import pytest

from hfvast.models.model import ModelInfo
from hfvast.planning.storage import estimate_disk_gb
from hfvast.utils.hfref import parse_model_input
from hfvast.utils.units import ceil_to, human_bytes, human_duration, parse_duration


def _model(mmproj_gb: float = 0.0) -> ModelInfo:
    from hfvast.models.model import ModelFile

    return ModelInfo(
        ref=parse_model_input("org/model"),
        mmproj_files=[ModelFile(path="mmproj.gguf", size_bytes=int(mmproj_gb * 1e9))] if mmproj_gb else [],
    )


def test_disk_covers_model_plus_headroom():
    from hfvast.models.model import ModelVariant

    variant = ModelVariant(id="Q4_K_M", size_bytes=193_000_000_000)
    disk = estimate_disk_gb(_model(), variant)
    # 193 + 20 runtime + max(10, 5%) temp → ~222.7 → ceil to 8
    assert disk == 224.0
    assert disk >= 193 + 20 + 10


def test_disk_includes_mmproj():
    from hfvast.models.model import ModelVariant

    variant = ModelVariant(id="Q4_K_M", size_bytes=193_000_000_000)
    disk = estimate_disk_gb(_model(mmproj_gb=1.2), variant)
    assert disk >= 224.0 + 1.2


def test_minimum_disk():
    from hfvast.models.model import ModelVariant

    tiny = ModelVariant(id="Q8_0", size_bytes=2_000_000_000)
    assert estimate_disk_gb(_model(), tiny) >= 24.0


def test_parse_duration():
    assert parse_duration("30m") == 1800.0
    assert parse_duration("2h") == 7200.0
    assert parse_duration("1d") == 86400.0
    assert parse_duration("45s") == 45.0
    assert parse_duration("90") == 5400.0  # bare number = minutes
    with pytest.raises(ValueError):
        parse_duration("2x")


def test_human_helpers():
    assert human_duration(90) == "1m30s"
    assert human_duration(3600) == "1h00m"
    assert "GB" in human_bytes(193_000_000_000)
    assert ceil_to(222.6, 8) == 224.0
    assert math.isclose(ceil_to(8.0, 8), 8.0)
