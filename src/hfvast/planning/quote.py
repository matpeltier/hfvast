"""Quote building: inspect → plan → search offers → rank → recommend (no spending)."""

from __future__ import annotations

from hfvast.errors import PlanError
from hfvast.inspect.base import Inspector
from hfvast.models.model import HFModelRef, ModelInfo, ModelVariant, QuantTier
from hfvast.models.quote import DeploymentQuote, QuoteRecommendation, RankedOffer, VariantPlan
from hfvast.planning.backends import select_backend
from hfvast.planning.hardware import PlanningConstraints, build_query
from hfvast.providers.base import ComputeProvider
from hfvast.runtimes.base import Backend, RuntimeSupport
from hfvast.runtimes.upstream import LiveRegistry

DEFAULT_CONTEXT = 8192

#: Recommendation heuristics (spec §11) — documented trade-offs, not magic.
QUALITY_UPGRADE_RATIO = 1.25  # QUALITY wins if it costs ≤ 1.25× BALANCED
ECONOMY_DOWNGRADE_RATIO = 1.5  # BALANCED must cost ≤ 1.5× ECONOMY to stay default


class QuoteOptions:
    """Options for building a quote (CLI flags + config merged upstream)."""

    def __init__(
        self,
        context: int | None = None,
        concurrency: int = 1,
        quant: str | None = None,
        file: str | None = None,
        backend: Backend | None = None,
        expected_session_hours: float = 2.0,
        network_efficiency: float = 0.7,
        image_size_gb: float = 8.0,
        constraints: PlanningConstraints | None = None,
    ) -> None:
        self.context = context
        self.concurrency = concurrency
        self.quant = quant
        self.file = file
        self.backend = backend
        self.expected_session_hours = expected_session_hours
        self.network_efficiency = network_efficiency
        self.image_size_gb = image_size_gb
        self.constraints = constraints or PlanningConstraints()


class QuoteBuilder:
    def __init__(self, inspector: Inspector, provider: ComputeProvider) -> None:
        self._inspector = inspector
        self._provider = provider

    async def build(self, ref: HFModelRef, options: QuoteOptions) -> DeploymentQuote:
        model = await self._inspector.inspect(ref)

        live = await self._load_live_registry()
        support = select_backend(model, override=options.backend, live=live)
        effective_context = self._effective_context(model, options.context)
        concurrency = max(1, options.concurrency)

        variants = self._select_variants(model, options)
        if not variants:
            raise PlanError("No variants matched the requested filters.")

        plans = [
            await self._plan_variant(model, variant, support, effective_context, concurrency, options)
            for variant in variants
        ]

        recommendation, blocked_reason = self._recommend(plans, options)
        return DeploymentQuote(
            model=model,
            context_length=effective_context,
            concurrency=concurrency,
            plans=plans,
            recommendation=recommendation,
            blocked_reason=blocked_reason,
            data_source=self._provider.data_source,
        )

    @staticmethod
    async def _load_live_registry() -> LiveRegistry | None:
        """Fresh upstream compatibility lists — best effort, never fatal."""
        try:
            from hfvast.runtimes.upstream import load_live_registry

            return await load_live_registry()
        except Exception:
            return None

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _effective_context(model: ModelInfo, requested: int | None) -> int:
        if requested is not None:
            if requested < 256:
                raise PlanError("--context must be at least 256 tokens")
            return requested
        max_ctx = model.context_length or DEFAULT_CONTEXT
        return min(DEFAULT_CONTEXT, max_ctx)

    @staticmethod
    def _select_variants(model: ModelInfo, options: QuoteOptions) -> list[ModelVariant]:
        if options.quant:
            variant = model.variant_by_id(options.quant)
            if variant is None:
                available = ", ".join(v.id for v in model.variants) or "none"
                raise PlanError(
                    f"Quantization {options.quant!r} not found. Available: {available}. "
                    "Use --file to pick a specific file instead."
                )
            return [variant]
        if options.file:
            for variant in model.variants:
                if any(f.path == options.file for f in variant.files):
                    return [variant]
            raise PlanError(
                f"File {options.file!r} not found among variant files. Use `hfvast inspect` to list variants."
            )
        return list(model.variants)

    async def _plan_variant(
        self,
        model: ModelInfo,
        variant: ModelVariant,
        support: RuntimeSupport,
        context: int,
        concurrency: int,
        options: QuoteOptions,
    ) -> VariantPlan:
        from hfvast.planning.ranking import OfferRanker
        from hfvast.planning.requirements import build_requirements

        requirements = build_requirements(model, variant, context, concurrency, support.backend)
        ranked: list[RankedOffer] = []
        if support.deployable:
            query = build_query(requirements, options.constraints)
            offers = await self._provider.search_offers(query)
            ranker = OfferRanker(
                session_hours=options.expected_session_hours,
                network_efficiency=options.network_efficiency,
                image_size_gb=options.image_size_gb,
            )
            ranked = ranker.rank(offers, model, variant, requirements, support.backend)
            if self._provider.data_source == "live" and ranked:
                # Pre-rent reachability probe (live offers expose host IPs;
                # sample data uses synthetic addresses — never probed).
                from hfvast.planning.reachability import probe_and_rerank

                await probe_and_rerank(ranked)
        return VariantPlan(
            variant=variant,
            requirements=requirements,
            support=support,
            ranked_offers=ranked,
            cheapest_cost=ranked[0].cost if ranked else None,
        )

    def _recommend(
        self, plans: list[VariantPlan], options: QuoteOptions
    ) -> tuple[QuoteRecommendation | None, str | None]:
        candidates = [p for p in plans if p.ranked_offers]
        if not candidates:
            if options.constraints.capped():
                return None, (
                    "No compatible offer within your limits.\n"
                    "Cost caps excluded every candidate. Raise the caps or pick another "
                    "quantization.\nNo Vast resources were created."
                )
            return None, None

        by_tier: dict[QuantTier | None, VariantPlan] = {}
        for plan in candidates:
            by_tier.setdefault(plan.variant.tier, plan)

        def best_of(plan: VariantPlan) -> RankedOffer:
            return plan.ranked_offers[0]

        chosen: VariantPlan | None = None
        if options.quant or options.file:
            chosen = candidates[0]
        else:
            balanced = by_tier.get(QuantTier.BALANCED)
            quality = by_tier.get(QuantTier.QUALITY)
            economy = by_tier.get(QuantTier.ECONOMY)
            if balanced is not None:
                chosen = balanced
                if quality is not None:
                    q_cost = best_of(quality).cost.total_session_usd
                    b_cost = best_of(balanced).cost.total_session_usd
                    if q_cost <= QUALITY_UPGRADE_RATIO * b_cost:
                        chosen = quality
                # If balanced is far more expensive than economy, note it but stay
                # balanced unless the user overrides — quality comes first (spec §11).
                if economy is not None:
                    e_cost = best_of(economy).cost.total_session_usd
                    if best_of(balanced).cost.total_session_usd > ECONOMY_DOWNGRADE_RATIO * e_cost:
                        pass  # stay balanced; reason string documents the trade-off
            elif quality is not None:
                chosen = quality
            elif economy is not None:
                chosen = economy
            else:
                # No tiers classified: pick cheapest by effective session cost.
                chosen = min(candidates, key=lambda p: best_of(p).effective_cost_usd)

        best = best_of(chosen)
        blocked = self._cap_block_reason(best, options.constraints)
        if blocked is not None:
            return None, blocked
        return (
            QuoteRecommendation(
                variant_id=chosen.variant.id,
                offer=best.offer,
                cost=best.cost,
                reasons_pro=list(best.pros),
                reasons_con=list(best.cons),
            ),
            None,
        )

    @staticmethod
    def _cap_block_reason(best: RankedOffer, constraints: PlanningConstraints) -> str | None:
        hourly = best.cost.hourly_total_usd
        if constraints.max_hourly_usd is not None and hourly > constraints.max_hourly_usd:
            return (
                f"No compatible offer under ${constraints.max_hourly_usd:.2f}/h.\n"
                f"Cheapest compatible: ${hourly:.2f}/h "
                f"({best.offer.label}).\n"
                f"Use --max-hourly-cost {hourly + 0.01:.2f} to allow this offer.\n"
                "No Vast resources were created."
            )
        if constraints.max_startup_usd is not None and best.cost.cold_start_usd > constraints.max_startup_usd:
            return (
                f"Estimated cold start ${best.cost.cold_start_usd:.2f} exceeds "
                f"--max-startup-cost ${constraints.max_startup_usd:.2f}.\n"
                "No Vast resources were created."
            )
        if constraints.max_total_usd is not None and best.cost.total_session_usd > constraints.max_total_usd:
            return (
                f"Estimated session cost ${best.cost.total_session_usd:.2f} exceeds "
                f"--max-total-cost ${constraints.max_total_usd:.2f}.\n"
                "No Vast resources were created."
            )
        return None
