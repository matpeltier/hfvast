"""End-to-end quote building against mocked HF + snapshot Vast data (no network, no spend)."""

import httpx
from conftest import hf_mock_transport

from hfvast.inspect.huggingface import HFInspector
from hfvast.models.model import QuantTier
from hfvast.planning.hardware import PlanningConstraints
from hfvast.planning.quote import QuoteBuilder, QuoteOptions
from hfvast.providers.vast.offers import SnapshotProvider
from hfvast.runtimes.base import SupportLevel
from hfvast.utils.hfref import parse_model_input


def _builder() -> QuoteBuilder:
    inspector = HFInspector(client=httpx.AsyncClient(transport=hf_mock_transport(), follow_redirects=True))
    return QuoteBuilder(inspector, SnapshotProvider())


async def test_quote_glm_gguf_end_to_end():
    builder = _builder()
    quote = await builder.build(
        parse_model_input("orcarouter/GLM-5.3-Flash-Uncensored-GGUF"),
        QuoteOptions(context=8192, concurrency=1, expected_session_hours=2.0),
    )

    # Model picked up from real fixture data
    assert quote.model.architecture == "glm5next"
    assert quote.context_length == 8192  # min(8192, model max 1M)
    assert quote.data_source == "sample"  # no VAST_API_KEY → clearly-labeled sample data

    # All five variants planned
    assert {p.variant.id for p in quote.plans} == {"Q2_K", "Q3_K_M", "Q4_K_M", "Q6_K", "Q8_0"}

    # glm5next is EXPERIMENTAL → plans still built, but support level must say so
    assert quote.plans[0].support.level is SupportLevel.EXPERIMENTAL

    # Requirements scale with variant size (weights dominate)
    sizes = {p.variant.id: p.requirements.recommended_vram_gib for p in quote.plans}
    assert sizes["Q2_K"] < sizes["Q4_K_M"] < sizes["Q6_K"]

    # Recommendation prefers the balanced tier with a viable offer
    assert quote.recommendation is not None
    assert quote.blocked_reason is None
    rec_plan = next(p for p in quote.plans if p.variant.id == quote.recommendation.variant_id)
    assert rec_plan.variant.tier in (QuantTier.BALANCED, QuantTier.QUALITY)
    assert quote.recommendation.cost.total_session_usd > 0
    assert quote.recommendation.reasons_pro  # explainable recommendation


async def test_quote_respects_hourly_cap():
    builder = _builder()
    quote = await builder.build(
        parse_model_input("orcarouter/GLM-5.3-Flash-Uncensored-GGUF"),
        QuoteOptions(
            context=8192,
            constraints=PlanningConstraints(max_hourly_usd=0.50),
        ),
    )
    # A ~320B model cannot run on any sample offer under $0.50/h
    assert quote.recommendation is None
    assert quote.blocked_reason is not None
    assert "--max-hourly-cost" in quote.blocked_reason
    assert "No Vast resources were created." in quote.blocked_reason


async def test_quote_quant_override():
    builder = _builder()
    quote = await builder.build(
        parse_model_input("orcarouter/GLM-5.3-Flash-Uncensored-GGUF"),
        QuoteOptions(quant="Q3_K_M", context=8192),
    )
    assert [p.variant.id for p in quote.plans] == ["Q3_K_M"]
    assert quote.recommendation is not None
    assert quote.recommendation.variant_id == "Q3_K_M"


async def test_quote_unknown_quant_fails():
    from hfvast.errors import PlanError

    builder = _builder()
    import pytest

    with pytest.raises(PlanError):
        await builder.build(
            parse_model_input("orcarouter/GLM-5.3-Flash-Uncensored-GGUF"),
            QuoteOptions(quant="Q9_MEGA"),
        )
