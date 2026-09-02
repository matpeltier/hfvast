"""Quote-level domain objects."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from hfvast.models.hardware import HardwareRequirements
from hfvast.models.model import ModelInfo, ModelVariant
from hfvast.models.offers import GPUOffer
from hfvast.models.pricing import CostBreakdown
from hfvast.runtimes.base import RuntimeSupport


class RankedOffer(BaseModel):
    offer: GPUOffer
    cost: CostBreakdown
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    penalty_usd: float = 0.0
    viable: bool = True

    @property
    def effective_cost_usd(self) -> float:
        return self.cost.total_session_usd + self.penalty_usd


class VariantPlan(BaseModel):
    variant: ModelVariant
    requirements: HardwareRequirements | None = None
    support: RuntimeSupport
    ranked_offers: list[RankedOffer] = Field(default_factory=list)
    cheapest_cost: CostBreakdown | None = None

    @property
    def cheapest_hourly(self) -> float | None:
        if not self.ranked_offers:
            return None
        return min(o.cost.hourly_total_usd for o in self.ranked_offers)


class QuoteRecommendation(BaseModel):
    variant_id: str
    offer: GPUOffer
    cost: CostBreakdown
    reasons_pro: list[str] = Field(default_factory=list)
    reasons_con: list[str] = Field(default_factory=list)


class DeploymentQuote(BaseModel):
    """Everything `quote` shows and `up` [M2] consumes after confirmation."""

    model: ModelInfo
    context_length: int
    concurrency: int
    plans: list[VariantPlan] = Field(default_factory=list)
    recommendation: QuoteRecommendation | None = None
    blocked_reason: str | None = Field(None, description="set when cost caps or constraints block every viable plan")
    data_source: str = Field("live", description='"live" or "sample" (bundled offers, no API key)')
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
