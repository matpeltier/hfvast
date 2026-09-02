from hfvast.inspect.gguf import group_gguf_variants
from hfvast.inspect.quantization import detect_quant, extreme_warning, tier_for_quant
from hfvast.models.model import ModelFile, QuantTier


def _files(*specs: tuple[str, int]) -> list[ModelFile]:
    return [ModelFile(path=p, size_bytes=s) for p, s in specs]


def test_multi_shard_forms_one_variant():
    files = _files(
        ("Q4_K_M/model-Q4_K_M-00001-of-00005.gguf", 40_000_000_000),
        ("Q4_K_M/model-Q4_K_M-00002-of-00005.gguf", 40_000_000_000),
        ("Q4_K_M/model-Q4_K_M-00003-of-00005.gguf", 40_000_000_000),
        ("Q4_K_M/model-Q4_K_M-00004-of-00005.gguf", 40_000_000_000),
        ("Q4_K_M/model-Q4_K_M-00005-of-00005.gguf", 32_000_000_000),
    )
    variants = group_gguf_variants(files)
    assert len(variants) == 1
    variant = variants[0]
    assert variant.id == "Q4_K_M"
    assert variant.is_split
    assert len(variant.files) == 5
    assert variant.size_bytes == 192_000_000_000


def test_multiple_quants_grouped_and_ordered():
    files = _files(
        ("Q6_K/model-Q6_K-00001-of-00002.gguf", 100_000_000_000),
        ("Q6_K/model-Q6_K-00002-of-00002.gguf", 60_000_000_000),
        ("Q3_K_M/model-Q3_K_M.gguf", 90_000_000_000),
        ("Q2_K/model-Q2_K-00001-of-00003.gguf", 40_000_000_000),
        ("Q2_K/model-Q2_K-00002-of-00003.gguf", 40_000_000_000),
        ("Q2_K/model-Q2_K-00003-of-00003.gguf", 36_000_000_000),
    )
    variants = group_gguf_variants(files)
    assert [v.id for v in variants] == ["Q3_K_M", "Q2_K", "Q6_K"]  # economy → quality, size within tier
    assert variants[0].tier is QuantTier.ECONOMY
    assert variants[1].tier is QuantTier.ECONOMY
    assert variants[2].tier is QuantTier.QUALITY
    assert variants[2].size_bytes == 160_000_000_000


def test_single_file_and_mmproj_excluded():
    files = _files(
        ("model-Q8_0.gguf", 8_000_000_000),
        ("mmproj-model-F16.gguf", 1_200_000_000),
    )
    variants = group_gguf_variants(files)
    assert len(variants) == 2  # grouping does not filter mmproj; inspector does
    ids = {v.id for v in variants}
    assert "Q8_0" in ids
    assert "F16" in ids


def test_detect_quant_variants():
    assert detect_quant("GLM-5.3-Flash-Uncensored-Q4_K_M-00001-of-00005.gguf") == "Q4_K_M"
    assert detect_quant("qwen2.5-7b-instruct-q6_k.gguf") == "Q6_K"
    assert detect_quant("model-IQ4_XS.gguf") == "IQ4_XS"
    assert detect_quant("mmproj-model-F16.gguf") == "F16"
    assert detect_quant("some-unknown-name.gguf") is None


def test_tiers():
    assert tier_for_quant("Q3_K_M") is QuantTier.ECONOMY
    assert tier_for_quant("Q4_K_M") is QuantTier.BALANCED
    assert tier_for_quant("IQ4_XS") is QuantTier.BALANCED
    assert tier_for_quant("Q6_K") is QuantTier.QUALITY
    assert tier_for_quant("Q8_0") is QuantTier.QUALITY
    assert tier_for_quant("Q2_K") is QuantTier.ECONOMY
    assert tier_for_quant(None) is None


def test_extreme_warning():
    assert extreme_warning("Q2_K") is not None
    assert extreme_warning("Q4_K_M") is None
