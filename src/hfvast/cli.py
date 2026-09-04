"""hfvast CLI — HF model → temporary OpenAI-compatible endpoint on Vast.ai."""

from __future__ import annotations

import asyncio
import os
import platform
import sys
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import httpx
import typer
from rich.table import Table
from rich.text import Text

from hfvast import __version__
from hfvast.config import (
    ConfigError,
    alias_add,
    alias_lookup,
    alias_remove,
    load_config,
    resolve_credentials,
)
from hfvast.deploy.lifecycle import DeploymentOrchestrator, DeployOptions, DeploySecrets
from hfvast.errors import HfvastError
from hfvast.errors import HfvastError as _HfvastError
from hfvast.inspect.huggingface import HFInspector
from hfvast.logging import make_console, make_error_console
from hfvast.models.model import ModelInfo, QuantTier
from hfvast.models.quote import DeploymentQuote, VariantPlan
from hfvast.planning.backends import evaluate_support
from hfvast.planning.hardware import PlanningConstraints
from hfvast.planning.quote import QuoteBuilder, QuoteOptions
from hfvast.providers.base import ComputeProvider
from hfvast.providers.vast.offers import SnapshotProvider, VastProvider
from hfvast.runtimes.base import Backend, SupportLevel
from hfvast.state import Deployment, DeploymentStore
from hfvast.utils.hfref import parse_model_input
from hfvast.utils.paths import state_file
from hfvast.utils.redact import redact
from hfvast.utils.units import (
    human_bytes,
    human_duration,
    minutes,
    money,
    money_rate,
    parse_duration,
    percent,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Turn a Hugging Face model into a temporary OpenAI-compatible endpoint on Vast.ai.",
)
alias_app = typer.Typer(no_args_is_help=True, help="Manage model aliases.")
app.add_typer(alias_app, name="alias")

console = make_console()
err_console = make_error_console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"hfvast {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = None,
) -> None:
    """hfvast — ephemeral HF inference endpoints on Vast.ai."""


# --------------------------------------------------------------------------
# helpers


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def _model_url(name_or_url: str) -> str:
    return alias_lookup(name_or_url)


def _provider_for(vast_api_key: str | None, offline: bool) -> tuple[ComputeProvider, str | None]:
    """Pick live Vast provider when a key exists; else bundled sample provider."""
    if offline:
        return SnapshotProvider(), "Forced offline mode: using bundled SAMPLE offers (not live)."
    if vast_api_key:
        return VastProvider(api_key=vast_api_key), None
    return (
        SnapshotProvider(),
        "VAST_API_KEY is not set: showing bundled SAMPLE offers, NOT live Vast.ai data. "
        "Export VAST_API_KEY to search the real marketplace.",
    )


def _tier_label(tier: QuantTier | None) -> str:
    return {"economy": "Economy", "balanced": "Balanced", "quality": "Quality", None: "—"}.get(
        tier.value if tier else None, "—"
    )


# --------------------------------------------------------------------------
# inspect


@app.command()
def inspect(
    model: Annotated[str, typer.Argument(help="HF URL, org/model, or an alias.")],
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False,
    hf_token: Annotated[
        str | None,
        typer.Option(
            "--hf-token", envvar="HF_TOKEN", help="HF token (env preferred).", show_default=False, hidden=True
        ),
    ] = None,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity.")] = 0,
) -> None:
    """Inspect a Hugging Face repository without downloading any weights."""
    _, hf = resolve_credentials(hf_token=hf_token)
    ref = parse_model_input(_model_url(model))

    async def run() -> tuple[ModelInfo, Any]:
        inspector = HFInspector(token=hf)
        live = None
        try:
            from hfvast.runtimes.upstream import load_live_registry

            live = await load_live_registry()
        except Exception:
            live = None
        try:
            return await inspector.inspect(ref), live
        finally:
            await inspector.aclose()

    if not json_output:
        console.print("[bold]Inspecting Hugging Face model...[/bold]")
    info, live = _run_async(run())
    if json_output:
        console.print_json(info.model_dump_json())
        return

    console.print()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan", width=17)
    table.add_column()
    table.add_row("Repository", info.ref.repo_id + (f" @ {info.ref.revision}" if info.ref.revision else ""))
    table.add_row("URL", info.ref.to_url())
    table.add_row("Task", info.task.value)
    table.add_row("Architecture", info.architecture or "unknown")
    table.add_row("Format", info.format.value)
    if info.dtype:
        table.add_row("Dtype", info.dtype)
    if info.parameter_count:
        table.add_row("Parameters", f"{info.parameter_count / 1e9:.1f}B")
    if info.context_length:
        table.add_row("Context length", f"{info.context_length:,}")
    table.add_row("Gated", "yes" if info.gated else "no")
    if info.multimodal:
        table.add_row("Multimodal", "yes (mmproj detected)")
    console.print(table)

    if info.variants:
        vt = Table(title="Available variants", header_style="bold")
        vt.add_column("Tier")
        vt.add_column("Variant")
        vt.add_column("Size", justify="right")
        vt.add_column("Files", justify="right")
        vt.add_column("Split", justify="center")
        for variant in info.variants:
            vt.add_row(
                _tier_label(variant.tier),
                variant.id,
                human_bytes(variant.size_bytes),
                str(len(variant.files)),
                "yes" if variant.is_split else "no",
            )
        console.print(vt)

    st = Table(title="Backend support (runtime registry)", header_style="bold")
    st.add_column("Backend")
    st.add_column("Status")
    st.add_column("Reason", overflow="fold")
    if live and live.source == "live":
        console.print(f"  [dim]support lists: {live.label()}[/dim]")
    for support in evaluate_support(info, live):
        style = {"supported": "green", "experimental": "yellow", "unsupported": "red"}[support.level.value]
        st.add_row(support.backend.value, Text(support.level.value.upper(), style=style), support.reason or "—")
    console.print(st)

    for note in info.notes:
        console.print(f"[dim]note: {note}[/dim]")
    if verbose > 0:
        console.print(f"[dim]state: {state_file()}[/dim]")


# --------------------------------------------------------------------------
# quote


@app.command()
def quote(
    model: Annotated[str, typer.Argument(help="HF URL, org/model, or an alias.")],
    backend: Annotated[Backend | None, typer.Option(case_sensitive=False, help="Override backend selection.")] = None,
    quant: Annotated[str | None, typer.Option("--quant", help="Deploy a specific quantization (e.g. Q4_K_M).")] = None,
    file: Annotated[str | None, typer.Option("--file", help="Deploy a specific HF file.")] = None,
    base_model: Annotated[
        str | None,
        typer.Option("--base-model", help="LoRA: base model <org/name> when the adapter doesn't declare it."),
    ] = None,
    context: Annotated[
        int | None, typer.Option("--context", min=256, help="Context length (default: min(8192, model max)).")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, help="Concurrent requests to plan for.")] = 1,
    expected_session: Annotated[
        str, typer.Option("--expected-session", help="Expected session length for cost ranking (e.g. 2h).")
    ] = "2h",
    idle_timeout: Annotated[
        str, typer.Option("--idle-timeout", help="Idle timeout shown in the plan (destroyed after).")
    ] = "30m",
    min_reliability: Annotated[
        float | None, typer.Option("--min-reliability", help="Minimum host reliability (default 0.98).")
    ] = None,
    min_download_mbps: Annotated[
        float | None, typer.Option("--min-download-mbps", help="Minimum host download bandwidth.")
    ] = None,
    secure_cloud_only: Annotated[
        bool, typer.Option("--secure-cloud-only", help="Restrict to verified/secure hosts.")
    ] = False,
    geo: Annotated[
        str | None,
        typer.Option("--geo", help="Restrict to country codes, e.g. 'US,DE,NL' (some regions are unreachable)."),
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="Optional GPU model constraint.")] = None,
    max_gpus: Annotated[int, typer.Option("--max-gpus", min=1, max=8, help="Maximum GPU count.")] = 4,
    max_hourly_cost: Annotated[
        float | None, typer.Option("--max-hourly-cost", help="Reject offers above this $/h.")
    ] = None,
    max_startup_cost: Annotated[
        float | None, typer.Option("--max-startup-cost", help="Reject cold starts above this $.")
    ] = None,
    max_total_cost: Annotated[
        float | None, typer.Option("--max-total-cost", help="Reject sessions above this $.")
    ] = None,
    vast_api_key: Annotated[
        str | None,
        typer.Option(
            "--vast-api-key",
            envvar="VAST_API_KEY",
            help="Vast API key (env preferred).",
            show_default=False,
            hidden=True,
        ),
    ] = None,
    hf_token: Annotated[
        str | None,
        typer.Option(
            "--hf-token", envvar="HF_TOKEN", help="HF token (env preferred).", show_default=False, hidden=True
        ),
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Force bundled sample offers (never query Vast).")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity.")] = 0,
) -> None:
    """Plan a deployment and quote real Vast.ai offers. Never provisions anything."""
    del verbose  # used implicitly by console formatting below
    vast_key, hf = resolve_credentials(vast_api_key, hf_token)
    config = load_config()
    ref = parse_model_input(_model_url(model))

    constraints = PlanningConstraints(
        min_reliability=min_reliability if min_reliability is not None else config.defaults.min_reliability,
        min_download_mbps=min_download_mbps if min_download_mbps is not None else config.defaults.min_download_mbps,
        max_gpus=max_gpus,
        gpu_filter=gpu,
        allowed_geolocations=[c.strip().upper() for c in geo.split(",") if c.strip()] if geo else None,
        secure_cloud_only=secure_cloud_only or config.vast.secure_cloud_only,
        max_hourly_usd=max_hourly_cost if max_hourly_cost is not None else config.cost.max_hourly,
        max_startup_usd=max_startup_cost if max_startup_cost is not None else config.cost.max_startup,
        max_total_usd=max_total_cost if max_total_cost is not None else config.cost.max_total,
    )
    options = QuoteOptions(
        context=context if context is not None else config.defaults.context,
        concurrency=concurrency,
        quant=quant,
        file=file,
        backend=backend,
        base_model=base_model,
        expected_session_hours=parse_duration(expected_session) / 3600.0,
        constraints=constraints,
    )

    provider, banner = _provider_for(vast_key, offline)

    async def run() -> DeploymentQuote:
        inspector = HFInspector(token=hf)
        try:
            builder = QuoteBuilder(inspector, provider)
            return await builder.build(ref, options)
        finally:
            await inspector.aclose()

    if not json_output:
        console.print("[bold]Inspecting Hugging Face model...[/bold]")
    quote_result = _run_async(run())
    if json_output:
        console.print_json(quote_result.model_dump_json())
        return
    _print_quote(quote_result, idle_timeout, banner)


def _print_quote(q: DeploymentQuote, idle_timeout: str, banner: str | None) -> None:
    info = q.model
    console.print()
    console.print("[bold]Model[/bold]")
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan", width=17)
    table.add_column()
    table.add_row("Repository", info.ref.repo_id)
    table.add_row("Task", info.task.value)
    if info.parameter_count:
        moe_note = " (MoE: all weights resident)" if info.architecture and "moe" in info.architecture.lower() else ""
        table.add_row("Parameters", f"{info.parameter_count / 1e9:.1f}B{moe_note}")
    table.add_row("Architecture", info.architecture or "unknown")
    table.add_row("Format", info.format.value)
    if info.context_length:
        table.add_row("Max context", f"{info.context_length:,}")
    console.print(table)

    console.print()
    console.print("[bold]Available variants[/bold]")
    vt = Table(header_style="bold", box=None)
    vt.add_column("Tier")
    vt.add_column("Variant")
    vt.add_column("Model size", justify="right")
    vt.add_column("Cheapest viable GPU", overflow="fold")
    vt.add_column("Price", justify="right")
    for plan in q.plans:
        best = plan.ranked_offers[0] if plan.ranked_offers else None
        vt.add_row(
            _tier_label(plan.variant.tier),
            plan.variant.id,
            human_bytes(plan.variant.size_bytes),
            best.offer.label
            if best
            else ("—" if plan.support.deployable else f"unsupported: {plan.support.level.value}"),
            money_rate(best.cost.hourly_total_usd) if best else "—",
        )
    console.print(vt)

    console.print()
    if q.recommendation is None:
        if q.blocked_reason:
            err_console.print(f"[red]{q.blocked_reason}[/red]")
            raise typer.Exit(1)
        err_console.print("[red]No viable offer found.[/red]")
        raise typer.Exit(1)

    rec = q.recommendation
    rec_plan: VariantPlan | None = next((p for p in q.plans if p.variant.id == rec.variant_id), None)
    variant = rec_plan.variant if rec_plan else None
    console.print("[bold]Selected[/bold]")
    sel = Table(show_header=False, box=None, pad_edge=False)
    sel.add_column(style="bold cyan", width=17)
    sel.add_column()
    extreme = variant is not None and (variant.quant or "").startswith(("Q2", "IQ"))
    sel.add_row("Variant", rec.variant_id + ("  [yellow]← extreme quantization warning[/yellow]" if extreme else ""))
    sel.add_row("Context", f"{q.context_length:,}")
    sel.add_row("Concurrency", str(q.concurrency))
    console.print(sel)

    console.print()
    console.print("[bold]Estimated requirements[/bold] [dim](estimates — inputs shown)[/dim]")
    requirements = rec_plan.requirements if rec_plan else None
    breakdown = requirements.breakdown if requirements else None
    if breakdown and requirements:
        br = Table(show_header=False, box=None, pad_edge=False)
        br.add_column(width=17)
        br.add_column(justify="right")
        br.add_row("Weight storage", f"{breakdown.weights_gib:.1f} GiB")
        br.add_row("KV cache", f"{breakdown.kv_cache_gib:.1f} GiB")
        br.add_row("Runtime overhead", f"{breakdown.runtime_overhead_gib:.1f} GiB")
        br.add_row("Safety margin", f"{breakdown.safety_gib:.1f} GiB")
        br.add_row("Target VRAM", f"[bold]{breakdown.total_gib:.1f} GiB[/bold]")
        br.add_row("Disk", f"{requirements.disk_gb:.0f} GB")
        console.print(br)
        for assumption in breakdown.assumptions:
            console.print(f"  [dim]assumption: {assumption}[/dim]")
        if requirements.reference_gpu_count > 1:
            console.print(
                f"  [dim]~{requirements.reference_gpu_count}× 24 GiB-class GPUs for the target VRAM "
                "(exact topology picked from offers)[/dim]"
            )

    console.print()
    console.print("[bold]Backend[/bold]")
    support = rec_plan.support if rec_plan else None
    if support:
        color = {"supported": "green", "experimental": "yellow", "unsupported": "red"}[support.level.value]
        console.print(f"  {support.backend.value}  [bold {color}]{support.level.value.upper()}[/bold {color}]")
        console.print(f"  [dim]{support.reason}[/dim]")
        if support.level is SupportLevel.EXPERIMENTAL:
            console.print(
                "  [yellow]This architecture has not been verified with this backend. Launching may "
                "fail after GPU allocation — explicit confirmation will be required by `hfvast up`.[/yellow]"
            )

    console.print()
    console.print(f"[bold]Searching Vast.ai...[/bold] [dim]({q.data_source})[/dim]")
    ot = Table(header_style="bold", box=None)
    ot.add_column("#", justify="right")
    ot.add_column("GPU", overflow="fold")
    ot.add_column("VRAM", justify="right")
    ot.add_column("Network", justify="right")
    ot.add_column("Reliability", justify="right")
    ot.add_column("Price", justify="right")
    ot.add_column("2h session", justify="right")
    ranked = rec_plan.ranked_offers if rec_plan else []
    for idx, ranked_offer in enumerate(ranked[:5], start=1):
        offer = ranked_offer.offer
        ot.add_row(
            str(idx),
            offer.label,
            f"{offer.total_vram_gb:.0f} GB",
            f"{offer.inet_down_mbps:.0f} Mb/s",
            percent(offer.reliability),
            money_rate(ranked_offer.cost.hourly_total_usd),
            money(ranked_offer.cost.total_session_usd),
        )
    console.print(ot)

    console.print()
    console.print(f"[bold]Recommended:[/bold] {rec.offer.label}")
    console.print("Why:")
    for pro in rec.reasons_pro:
        console.print(f"  [green]+[/green] {pro}")
    for con in rec.reasons_con:
        console.print(f"  [red]-[/red] {con}")

    cost = rec.cost
    console.print()
    console.print("[bold]Estimated startup[/bold] [dim](estimate)[/dim]")
    su = Table(show_header=False, box=None, pad_edge=False)
    su.add_column(width=20)
    su.add_column(justify="right")
    su.add_row("Container pull", minutes(cost.image_pull_seconds))
    su.add_row("Model download", minutes(cost.download_seconds))
    su.add_row("Model loading", minutes(cost.model_load_seconds))
    ready_seconds = cost.image_pull_seconds + cost.download_seconds + cost.model_load_seconds
    su.add_row("Expected ready time", f"[bold]{human_duration(ready_seconds)}[/bold]")
    console.print(su)

    console.print()
    console.print("[bold]Estimated costs[/bold] [dim](estimates — Vast billing is authoritative)[/dim]")
    ct = Table(show_header=False, box=None, pad_edge=False)
    ct.add_column(width=20)
    ct.add_column(justify="right")
    ct.add_row("Compute", money_rate(cost.hourly_gpu_usd))
    ct.add_row("Storage", money_rate(cost.hourly_storage_usd))
    ct.add_row("Bandwidth (cold start)", money(cost.bandwidth_usd))
    ct.add_row("Cold start", f"[bold]{money(cost.cold_start_usd)}[/bold]")
    ct.add_row(f"{int(cost.session_hours)}-hour session", f"[bold]{money(cost.total_session_usd)}[/bold]")
    console.print(ct)

    console.print()
    console.print(f"Idle timeout      {idle_timeout}")
    console.print("Destroy instance after idle timeout: [bold]YES[/bold] (default)")

    if banner:
        console.print(f"  [yellow]! {banner}[/yellow]")
    console.print()
    console.print("[dim]This is a quote only. Nothing was created, nothing was spent.[/dim]")
    console.print("[dim]`hfvast up <model>` (Milestone 2) will provision after explicit confirmation.[/dim]")


# --------------------------------------------------------------------------
# doctor


@app.command()
def doctor() -> None:
    """Check local environment, configuration, and connectivity."""
    results: list[tuple[bool, str, str]] = []

    version_ok = sys.version_info >= (3, 12)
    results.append((version_ok, f"Python {platform.python_version()}", "requires Python 3.12+"))

    try:
        load_config()
        results.append((True, "Config file", f"{_config_path_or_none() or 'not present — defaults in use'}"))
    except ConfigError as exc:
        results.append((False, "Config file", str(exc)))

    from hfvast.utils.paths import config_dir, ensure_dirs

    try:
        ensure_dirs()
        probe = config_dir() / ".doctor"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        results.append((True, "State/config dirs writable", str(config_dir())))
    except OSError as exc:
        results.append((False, "State/config dirs writable", str(exc)))

    hf_ok = False
    hf_detail = ""
    try:
        resp = httpx.head("https://huggingface.co", timeout=10.0, follow_redirects=True)
        hf_ok = resp.status_code < 500
        hf_detail = f"HTTP {resp.status_code}"
    except httpx.HTTPError as exc:
        hf_detail = redact(str(exc))
    results.append((hf_ok, "Hugging Face reachable", hf_detail))

    hf_token = os.environ.get("HF_TOKEN")
    results.append(
        (bool(hf_token), "HF_TOKEN", "present" if hf_token else "not set — gated repos will fail (ok for public repos)")
    )
    vast_key = os.environ.get("VAST_API_KEY")
    results.append(
        (bool(vast_key), "VAST_API_KEY", "present" if vast_key else "not set — `quote` will show SAMPLE offers")
    )

    deployments = DeploymentStore().all_active()
    results.append((True, "Local deployment state", f"{len(deployments)} active deployment(s) at {state_file()}"))

    failed = False
    table = Table(header_style="bold", box=None)
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail", overflow="fold")
    for ok, name, detail in results:
        if not ok and name not in ("HF_TOKEN", "VAST_API_KEY"):
            failed = True
        style = "green" if ok else ("yellow" if name in ("HF_TOKEN", "VAST_API_KEY") else "red")
        glyph = "✓" if ok else ("!" if name in ("HF_TOKEN", "VAST_API_KEY") else "✗")
        table.add_row(name, Text(glyph, style=style), detail)
    console.print(table)
    if failed:
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# alias


@alias_app.command("add")
def alias_add_cmd(
    name: Annotated[str, typer.Argument(help="Alias name, e.g. glm-uncensored.")],
    url: Annotated[str, typer.Argument(help="HF URL or org/model.")],
) -> None:
    """Add an alias for a Hugging Face model."""
    parse_model_input(url)  # validate early
    alias_add(name, url)
    console.print(f"[green]✓[/green] alias {name} → {url}")


@alias_app.command("rm")
def alias_rm_cmd(name: Annotated[str, typer.Argument(help="Alias name.")]) -> None:
    """Remove an alias."""
    if alias_remove(name):
        console.print(f"[green]✓[/green] removed {name}")
    else:
        err_console.print(f"[red]No such alias: {name}[/red]")
        raise typer.Exit(1)


@alias_app.command("list")
def alias_list_cmd() -> None:
    """List aliases."""
    config = load_config()
    if not config.aliases:
        console.print("No aliases. Add one: hfvast alias add <name> <HF_URL>")
        return
    table = Table(header_style="bold", box=None)
    table.add_column("Name")
    table.add_column("Model")
    for name, entry in config.aliases.items():
        table.add_row(name, entry.url)
    console.print(table)


# --------------------------------------------------------------------------
# lifecycle (Milestone 2/3): up / ps / status / endpoint / cost / logs / down


@app.command()
def up(
    model: Annotated[str, typer.Argument(help="HF URL, org/model, or an alias.")],
    backend: Annotated[Backend | None, typer.Option(case_sensitive=False, help="Override backend selection.")] = None,
    quant: Annotated[str | None, typer.Option("--quant", help="Deploy a specific quantization (e.g. Q4_K_M).")] = None,
    file: Annotated[str | None, typer.Option("--file", help="Deploy a specific HF file.")] = None,
    base_model: Annotated[
        str | None,
        typer.Option("--base-model", help="LoRA: base model <org/name> when the adapter doesn't declare it."),
    ] = None,
    context: Annotated[
        int | None, typer.Option("--context", min=256, help="Context length (default: min(8192, model max)).")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, help="Concurrent requests to plan for.")] = 1,
    expected_session: Annotated[
        str, typer.Option("--expected-session", help="Expected session length for cost ranking (e.g. 2h).")
    ] = "2h",
    idle_timeout: Annotated[
        str, typer.Option("--idle-timeout", help="Destroy after this much inactivity (e.g. 30m).")
    ] = "30m",
    max_runtime: Annotated[
        str, typer.Option("--max-runtime", help="Hard runtime cap — destroyed unconditionally after (e.g. 6h).")
    ] = "6h",
    budget: Annotated[
        float | None, typer.Option("--budget", help="Hard spend cap in USD — destroyed when exceeded (e.g. 10).")
    ] = None,
    min_reliability: Annotated[
        float | None, typer.Option("--min-reliability", help="Minimum host reliability (default 0.98).")
    ] = None,
    min_download_mbps: Annotated[
        float | None, typer.Option("--min-download-mbps", help="Minimum host download bandwidth.")
    ] = None,
    secure_cloud_only: Annotated[
        bool, typer.Option("--secure-cloud-only", help="Restrict to verified/secure hosts.")
    ] = False,
    geo: Annotated[
        str | None,
        typer.Option("--geo", help="Restrict to country codes, e.g. 'US,DE,NL' (some regions are unreachable)."),
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="Optional GPU model constraint.")] = None,
    max_gpus: Annotated[int, typer.Option("--max-gpus", min=1, max=8, help="Maximum GPU count.")] = 4,
    max_hourly_cost: Annotated[
        float | None, typer.Option("--max-hourly-cost", help="Reject offers above this $/h.")
    ] = None,
    max_startup_cost: Annotated[
        float | None, typer.Option("--max-startup-cost", help="Reject cold starts above this $.")
    ] = None,
    max_total_cost: Annotated[
        float | None, typer.Option("--max-total-cost", help="Reject sessions above this $.")
    ] = None,
    ready_timeout: Annotated[
        str, typer.Option("--ready-timeout", help="Give up if not ready after this (e.g. 90m).")
    ] = "90m",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt (respects cost caps).")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and exit — never creates anything.")
    ] = False,
    keep_on_failure: Annotated[
        bool, typer.Option("--keep-on-failure", help="Keep the instance for debugging when bootstrap fails.")
    ] = False,
    vast_api_key: Annotated[
        str | None,
        typer.Option(
            "--vast-api-key",
            envvar="VAST_API_KEY",
            show_default=False,
            hidden=True,
            help="Vast API key (env preferred).",
        ),
    ] = None,
    hf_token: Annotated[
        str | None,
        typer.Option(
            "--hf-token", envvar="HF_TOKEN", help="HF token (env preferred).", show_default=False, hidden=True
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False,
) -> None:
    """Inspect, quote, confirm, provision, serve. Destroys on idle/failure."""
    vast_key, hf = resolve_credentials(vast_api_key, hf_token)
    if not vast_key:
        raise _HfvastError(
            "hfvast up requires VAST_API_KEY (real spending follows confirmation).\n"
            "Create a key at https://cloud.vast.ai/manage-keys/ and export VAST_API_KEY."
        )
    config = load_config()
    ref = parse_model_input(_model_url(model))

    constraints = PlanningConstraints(
        min_reliability=min_reliability if min_reliability is not None else config.defaults.min_reliability,
        min_download_mbps=min_download_mbps if min_download_mbps is not None else config.defaults.min_download_mbps,
        max_gpus=max_gpus,
        gpu_filter=gpu,
        allowed_geolocations=[c.strip().upper() for c in geo.split(",") if c.strip()] if geo else None,
        secure_cloud_only=secure_cloud_only or config.vast.secure_cloud_only,
        max_hourly_usd=max_hourly_cost if max_hourly_cost is not None else config.cost.max_hourly,
        max_startup_usd=max_startup_cost if max_startup_cost is not None else config.cost.max_startup,
        max_total_usd=max_total_cost if max_total_cost is not None else config.cost.max_total,
    )
    options = QuoteOptions(
        context=context if context is not None else config.defaults.context,
        concurrency=concurrency,
        quant=quant,
        file=file,
        backend=backend,
        base_model=base_model,
        expected_session_hours=parse_duration(expected_session) / 3600.0,
        constraints=constraints,
    )
    provider = VastProvider(api_key=vast_key)

    async def run_quote() -> DeploymentQuote:
        inspector = HFInspector(token=hf)
        try:
            return await QuoteBuilder(inspector, provider).build(ref, options)
        finally:
            await inspector.aclose()
            # NOTE: close the provider's httpx client too — its connections are
            # bound to THIS event loop; deploy() runs in a fresh asyncio.run()
            # loop and reuses nothing (sharing a client across loops corrupts
            # the connection pool — caught in the first live deploy run).
            await provider._client.aclose()

    console.print("[bold]Inspecting Hugging Face model...[/bold]")
    quote_result: DeploymentQuote = _run_async(run_quote())
    if json_output:
        console.print_json(quote_result.model_dump_json())
        return
    _print_quote(quote_result, idle_timeout, None)

    if quote_result.recommendation is None:
        err_console.print(f"[red]{quote_result.blocked_reason or 'No viable offer found.'}[/red]")
        raise typer.Exit(1)

    rec_plan = next((p for p in quote_result.plans if p.variant.id == quote_result.recommendation.variant_id), None)
    if rec_plan and rec_plan.support.level is SupportLevel.EXPERIMENTAL:
        console.print()
        console.print(
            "[yellow]WARNING: this architecture has not been verified with "
            f"{rec_plan.support.backend.value}. Launching may fail after GPU allocation; "
            "cleanup will destroy the instance automatically.[/yellow]"
        )
        if not yes and not typer.confirm("Proceed with the experimental backend?", default=False):
            console.print("[dim]Aborted — nothing was created.[/dim]")
            raise typer.Exit(1)

    if dry_run:
        console.print()
        console.print("[bold]--dry-run:[/bold] plan shown above; nothing was created. ✓")
        return

    console.print()
    if not yes and not typer.confirm("Launch?", default=False):
        console.print("[dim]Aborted — nothing was created.[/dim]")
        raise typer.Exit(1)

    orch_options = DeployOptions(
        keep_on_failure=keep_on_failure,
        ready_timeout_s=parse_duration(ready_timeout),
        idle_timeout_s=parse_duration(idle_timeout),
        max_runtime_s=parse_duration(max_runtime),
        budget_usd=budget,
        on_progress=_progress_printer(),
    )
    secrets = DeploySecrets(vast_api_key=vast_key, hf_token=hf)

    deploy_provider = VastProvider(api_key=vast_key)  # fresh client for the new loop

    async def run_deploy() -> object:
        orchestrator = DeploymentOrchestrator(deploy_provider, DeploymentStore(), secrets)
        try:
            return await orchestrator.deploy(quote_result, orch_options)
        finally:
            await orchestrator.aclose()

    deployment = _run_async(run_deploy())
    assert isinstance(deployment, Deployment)
    _print_up_result(deployment, idle_timeout, max_runtime)


def _progress_printer() -> Callable[[str], Awaitable[None]]:
    import time as _time

    last: dict[str, float | str] = {"t": 0.0, "msg": ""}

    async def progress(message: str) -> None:
        now = _time.time()
        if message != last["msg"] or now - float(last["t"]) > 20:
            last["msg"], last["t"] = message, now
            console.print(f"  [dim]• {message}[/dim]")

    return progress


def _print_up_result(deployment: Deployment, idle_timeout: str, max_runtime: str) -> None:
    console.print()
    console.print("[bold green]Model ready.[/bold green]")
    console.print()
    console.print(f"OPENAI_BASE_URL={deployment.endpoint}/v1")
    console.print(f"OPENAI_API_KEY={deployment.api_key}")
    console.print()
    console.print(f"Idle timeout: {idle_timeout}    Hard max runtime: {max_runtime}")
    console.print(
        f"Current cost (estimate): ${deployment.estimated_spend_usd():.2f} "
        f"({deployment.uptime_s / 60:.0f} min × ${deployment.hourly_total_usd:.2f}/h)"
    )
    console.print()
    console.print("Try:")
    console.print(
        """  curl $OPENAI_BASE_URL/chat/completions \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model": "model", "messages": [{"role": "user", "content": "Hello"}]}'"""
    )
    console.print()
    console.print(f"Destroy when done: [bold]hfvast down {deployment.id}[/bold]")


@app.command()
def ps() -> None:
    """List active deployments."""
    deployments = DeploymentStore().all_active()
    if not deployments:
        console.print("No active deployments. Start one: hfvast up <model>")
        return
    table = Table(header_style="bold", box=None)
    for col in ("NAME", "MODEL", "GPU", "STATUS", "COST", "ENDPOINT"):
        table.add_column(col, overflow="fold")
    for d in deployments:
        table.add_row(
            d.id,
            d.model_repo,
            d.gpu_label or "—",
            d.status,
            money_rate(d.hourly_total_usd),
            d.endpoint or "—",
        )
    console.print(table)
    console.print()
    detail = Table(show_header=False, box=None, pad_edge=False)
    detail.add_column(style="bold cyan", width=14)
    detail.add_column()
    for d in deployments:
        detail.add_row("Uptime", human_duration(d.uptime_s) + f"  ({d.id})")
        detail.add_row("Est. spend", money(d.estimated_spend_usd()))
        detail.add_row("Idle destroy", f"after {int(d.idle_timeout_s / 60)}m of inactivity")
    console.print(detail)


@app.command()
def status(
    deployment: Annotated[str | None, typer.Argument(help="Deployment id (defaults to the only active one).")] = None,
) -> None:
    """Show detailed status of one deployment."""
    d = DeploymentStore().resolve(deployment)
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan", width=18)
    table.add_column()
    for key, value in (
        ("Deployment", d.id),
        ("Model", f"{d.model_repo} ({d.variant_id}, {d.backend})"),
        ("Status", d.status + (f" — {d.status_message}" if d.status_message else "")),
        ("Instance", str(d.instance_id) if d.instance_id else "—"),
        ("GPU", d.gpu_label or "—"),
        ("Endpoint", d.endpoint or "—"),
        ("Uptime", human_duration(d.uptime_s)),
        ("Est. spend", money(d.estimated_spend_usd())),
        ("Idle destroy", f"{int(d.idle_timeout_s / 60)}m"),
        ("Max runtime", f"{int(d.max_runtime_s / 3600)}h"),
        ("Watchdog", f"pid {d.watchdog_pid}" if d.watchdog_pid else "not running (in-container watchdog still active)"),
        ("Last error", redact(d.last_error) if d.last_error else "—"),
    ):
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def endpoint(
    deployment: Annotated[str | None, typer.Argument(help="Deployment id.")] = None,
) -> None:
    """Print the OpenAI-compatible endpoint + key as env vars."""
    d = DeploymentStore().resolve(deployment)
    if not d.endpoint or d.status != "ready":
        err_console.print(f"[red]Deployment {d.id} is not ready (status: {d.status}).[/red]")
        raise typer.Exit(1)
    console.print(f"OPENAI_BASE_URL={d.endpoint}/v1")
    console.print(f"OPENAI_API_KEY={d.api_key}")


@app.command()
def cost(
    deployment: Annotated[str | None, typer.Argument(help="Deployment id.")] = None,
) -> None:
    """Estimate spend for a deployment (Vast billing is authoritative)."""
    d = DeploymentStore().resolve(deployment)
    hours = d.uptime_s / 3600.0
    gpu_usd = d.hourly_gpu_usd * hours
    storage_usd = max(0.0, (d.hourly_total_usd - d.hourly_gpu_usd) * hours)
    total = d.estimated_spend_usd()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(width=12)
    table.add_column(justify="right")
    table.add_row("GPU", money(gpu_usd))
    table.add_row("Storage", money(storage_usd))
    table.add_row("Bandwidth (cold start est.)", money(d.bandwidth_usd_estimate))
    table.add_row("", "--------")
    table.add_row("Total", f"[bold]{money(total)}[/bold]")
    console.print(f"Estimated deployment spend — {d.id} ({human_duration(d.uptime_s)})")
    console.print(table)
    console.print()
    console.print("[dim]This is an estimate. Check Vast.ai billing for authoritative charges.[/dim]")


@app.command()
def logs(
    deployment: Annotated[str | None, typer.Argument(help="Deployment id.")] = None,
    tail: Annotated[int, typer.Option("--tail", min=10, max=10000, help="Last N lines.")] = 400,
) -> None:
    """Fetch instance logs from Vast (requires VAST_API_KEY)."""
    d = DeploymentStore().resolve(deployment)
    vast_key, _ = resolve_credentials()
    if not vast_key or d.instance_id is None:
        raise _HfvastError("logs require VAST_API_KEY and a provisioned instance.")
    provider = VastProvider(api_key=vast_key)
    text = _run_async(provider.logs(d.instance_id, tail=tail))
    console.print(redact(text))


@app.command()
def down(
    deployment: Annotated[str | None, typer.Argument(help="Deployment id (defaults to the only active one).")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Destroy a deployment's Vast instance. Billing returns to $0."""
    store = DeploymentStore()
    d = store.resolve(deployment)
    vast_key, _ = resolve_credentials()
    if not vast_key:
        raise _HfvastError("VAST_API_KEY is required to destroy instances.")
    if d.instance_id is None:
        store.remove(d.id)
        console.print(f"[yellow]Deployment {d.id} had no instance — removed from state.[/yellow]")
        return
    if not yes:
        console.print(
            f"About to destroy [bold]{d.id}[/bold] (instance {d.instance_id}, "
            f"est. spend so far {money(d.estimated_spend_usd())})."
        )
        if not typer.confirm("Destroy?", default=False):
            console.print("[dim]Aborted.[/dim]")
            return
    provider = VastProvider(api_key=vast_key)
    orchestrator = DeploymentOrchestrator(provider, store, DeploySecrets(vast_api_key=vast_key, hf_token=None))

    async def run() -> None:
        try:
            await orchestrator.destroy(d, reason="user request")
        finally:
            await orchestrator._http.aclose()

    _run_async(run())
    console.print(f"[green]✓[/green] destroyed {d.id} — est. total spend {money(d.estimated_spend_usd())}")
    console.print("[dim]GPU and storage are no longer billed.[/dim]")


@app.command(hidden=True)
def _watch(deployment_id: Annotated[str, typer.Argument()]) -> None:
    """Internal: local lifecycle daemon (spawned by `up`)."""
    from hfvast.deploy.watchdog import main as watchdog_main

    sys.argv = ["hfvast-watch", deployment_id]
    watchdog_main()


# --------------------------------------------------------------------------


def _config_path_or_none() -> str | None:
    from hfvast.utils.paths import config_file

    path = config_file()
    return str(path) if path.exists() else None


def main() -> None:
    try:
        app()
    except HfvastError as exc:
        err_console.print(f"[red]{redact(str(exc))}[/red]")
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        err_console.print("[yellow]Interrupted — nothing else was changed.[/yellow]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
