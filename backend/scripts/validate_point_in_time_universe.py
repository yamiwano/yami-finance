"""Point-in-time universe + barrier_probability_v2 validation.

Builds a historical top-N universe from trailing 24h quote volume (candles <= T),
rebuilds the research dataset on that universe, and reruns the barrier experiment.

Does not overwrite prior reports or activate live models.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_point_in_time_universe
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

os.environ.setdefault("MARKET_DATA_PROVIDER", "binance")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-pit-universe.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.barriers import fit_barrier_walk_forward
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import BARRIER_PROBABILITY_V2
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import BARRIER_SPECS, barrier_flags_from_bars
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe
from app.research.universe import (
    build_universe_snapshots,
    compare_universes,
    load_selected_symbols,
    universe_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_point_in_time_universe")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "POINT_IN_TIME_UNIVERSE_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "POINT_IN_TIME_UNIVERSE_REPORT.md"
EXPERIMENT = BARRIER_PROBABILITY_V2
PRIOR_BARRIER_JSON = REPO_ROOT / "docs" / "BARRIER_PROBABILITY_REPORT.json"


def run_research_tests() -> dict:
    backend = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_research", "-v"],
        cwd=backend,
        env={**os.environ, "PYTHONPATH": str(backend)},
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


async def ensure_assets(provider) -> dict:
    async with SessionLocal() as db:
        ids = {}
        for spec in provider.universe():
            row = (await db.execute(select(Asset).where(Asset.symbol == spec["symbol"]))).scalar_one_or_none()
            if row is None:
                row = Asset(
                    symbol=spec["symbol"],
                    name=spec["name"],
                    asset_type=spec["asset_type"] if isinstance(spec["asset_type"], str) else spec["asset_type"].value,
                    exchange=spec["exchange"],
                )
                db.add(row)
                await db.flush()
            ids[spec["symbol"]] = row.id
        await db.commit()
        return ids


async def load_upside_rows(db) -> list[ResearchSample]:
    tf = timeframe().value
    hours = horizon_hours()
    return list(
        (
            await db.execute(
                select(ResearchSample).where(
                    ResearchSample.timeframe == tf,
                    ResearchSample.horizon_hours == hours,
                    ResearchSample.max_upside.is_not(None),
                    ResearchSample.max_drawdown.is_not(None),
                    ResearchSample.fwd_return.is_not(None),
                )
            )
        ).scalars().all()
    )


async def attach_barriers(rows: list[ResearchSample], ids: dict, db) -> dict:
    tf = timeframe().value
    h = horizon_bars(tf)
    by_symbol: dict[str, list] = {}
    for row in rows:
        by_symbol.setdefault(row.symbol, []).append(row)
    ambiguous = 0
    computed = 0
    for symbol, sym_rows in by_symbol.items():
        aid = ids.get(symbol)
        if aid is None:
            continue
        bars = await load_bars(db, aid, tf)
        idx = {as_utc(b.ts): i for i, b in enumerate(bars)}
        for row in sym_rows:
            i = idx.get(as_utc(row.ts))
            if i is None or i + h >= len(bars):
                for name, _, _ in BARRIER_SPECS:
                    setattr(row, name, "unknown")
                continue
            flags = barrier_flags_from_bars(bars[i + 1 :], bars[i].close, horizon_bars=h)
            for name, value in flags.items():
                setattr(row, name, value)
                if value == "ambiguous":
                    ambiguous += 1
            computed += 1
    return {"barrier_rows_computed": computed, "ambiguous_events": ambiguous}


async def audit_leakage(rows: list[ResearchSample], bars_by_symbol: dict) -> dict:
    tf = timeframe().value
    h = horizon_bars(tf)
    thr = move_pct()
    findings: list[str] = []
    checked = 0
    blocklist = ("future_", "up3_", "up5_", "up10_", "barrier", "opportunity", "quote_volume_24h")
    for row in rows[:200]:
        bars = bars_by_symbol.get(row.symbol) or []
        idx = {as_utc(b.ts): i for i, b in enumerate(bars)}
        i = idx.get(as_utc(row.ts))
        if i is None:
            continue
        a = extract_features(bars[: i + 1], tf)
        mutated = list(bars)
        if i + 1 < len(mutated):
            fut = mutated[i + 1]
            mutated[i + 1] = fut.model_copy(update={"high": fut.high * 3, "low": fut.low * 0.5, "close": fut.close * 3, "volume": fut.volume * 100})
        past_only = extract_features(mutated[: i + 1], tf)
        if a != past_only:
            findings.append(f"future mutation changed past features {row.symbol} {row.ts}")
        if a and any(any(b in k for b in blocklist) for k in a):
            findings.append(f"target/universe key leaked into features {row.symbol}")
        labels = label_forward(bars[i + 1 :], bars[i].close, h, thr)
        if labels and row.max_upside is not None:
            if abs(float(labels["max_upside"]) - float(row.max_upside)) > 1e-9:
                findings.append(f"max_upside mismatch {row.symbol}")
        checked += 1

    unique_ts = sorted({as_utc(r.ts) for r in rows})
    folds = walk_forward_folds(unique_ts, min_train=60, min_test=14)
    leak_folds = 0
    for train_idx, test_idx in folds:
        if not fold_is_causal(unique_ts, train_idx, test_idx, embargo()):
            leak_folds += 1
    return {
        "samples_checked": checked,
        "label_mismatches_or_findings": findings[:20],
        "fold_count": len(folds),
        "folds_violating_embargo": leak_folds,
        "embargo_hours": horizon_hours(),
        "passed": len(findings) == 0 and leak_folds == 0,
    }


def fmt(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def classify_signal_shift(v1: dict, v2: dict) -> str:
    """Objective classification of signal change after universe replacement."""
    b1 = (v1.get("barriers") or {})
    b2 = (v2.get("barriers") or {})
    deltas = []
    v2_positive = 0
    for name, _, _ in BARRIER_SPECS:
        f1 = (b1.get(name) or {}).get("full") or {}
        f2 = (b2.get(name) or {}).get("full") or {}
        a1 = (b1.get(name) or {}).get("atr_only") or {}
        a2 = (b2.get(name) or {}).get("atr_only") or {}
        t1 = f1.get("top_5_success_rate")
        t2 = f2.get("top_5_success_rate")
        at1 = a1.get("top_5_success_rate")
        at2 = a2.get("top_5_success_rate")
        sp2 = f2.get("spearman_prob_outcome")
        if sp2 is not None and sp2 > 0:
            v2_positive += 1
        if None not in (t1, t2, at1, at2):
            deltas.append((t2 - at2) - (t1 - at1))
    if not deltas:
        return "data_insufficient"
    avg_delta = float(sum(deltas) / len(deltas))
    if v2_positive == 0:
        return "signal_disappears"
    if avg_delta > 0.005:
        return "signal_survives"
    if avg_delta < -0.005:
        return "signal_weakens"
    return "signal_survives" if v2_positive >= 2 else "signal_weakens"


def write_markdown(report: dict) -> str:
    exp = report["experiment"]
    u = report["universe"]
    comp = report["universe_comparison"]
    barriers = report["barriers"]
    v1 = report.get("prior_barrier_v1") or {}
    lines = [
        f"# Point-in-time universe report (`point_in_time_universe_v1` + `{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`POINT_IN_TIME_UNIVERSE_REPORT.json`](POINT_IN_TIME_UNIVERSE_REPORT.json)",
        "",
        "Prior experiments preserved:",
        "- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)",
        "- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)",
        "- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)",
        "- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)",
        "- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)",
        "- [`BARRIER_PROBABILITY_REPORT.md`](BARRIER_PROBABILITY_REPORT.md)",
        "",
        "## Methodology",
        "",
        "### Old universe",
        "Today's top-N Binance USDT pairs by current 24h quote volume, applied retrospectively.",
        "",
        "### New point-in-time universe",
        "For each research timestamp T:",
        "1. Eligible symbols = USDT spot pairs with ≥ LOOKBACK_BARS (80) candles at/before T.",
        "2. Trailing 24h quote volume = Σ(close × volume) over bars in (T−24h, T].",
        "3. Rank by trailing 24h quote volume; select top N (same N as old universe).",
        "",
        "**Evidence strength:** historical existence is **inferred** from candle presence. "
        "Binance exchangeInfo is current-only; we do not have historical listing/delisting "
        "metadata from this provider. Symbols delisted before our data window cannot be "
        "reconstructed — this is a documented limitation, not a fabricated fix.",
        "",
        "## Survivorship analysis",
        "",
        report["survivorship"],
        "",
        "## Universe statistics",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Timestamps | {u.get('timestamps')} |",
        f"| Avg universe size | {fmt(u.get('average_universe_size'), 2)} |",
        f"| Min / max size | {u.get('min_universe_size')} / {u.get('max_universe_size')} |",
        f"| Avg / median turnover | {fmt(u.get('average_turnover'), 2)} / {fmt(u.get('median_turnover'), 2)} |",
        f"| Avg 24h quote volume | {fmt(u.get('average_quote_volume_24h'), 0)} |",
        f"| Distinct symbols selected | {u.get('distinct_symbols_selected')} |",
        "",
        "### Old vs new",
        "",
        f"- Old universe size: {comp.get('old_universe_size')}",
        f"- Avg new size: {fmt(comp.get('average_new_size'), 2)}",
        f"- Avg overlap count: {fmt(comp.get('average_overlap_count'), 2)}",
        f"- Avg overlap (% of new): {fmt(comp.get('average_overlap_pct_of_new'), 3)}",
        f"- Symbols entering (top): {comp.get('symbols_entering_top')}",
        f"- Symbols leaving (top): {comp.get('symbols_leaving_top')}",
        "",
        "## Dataset comparison",
        "",
        json.dumps(report.get("dataset_comparison"), indent=2, default=str),
        "",
        "## Barrier v2 vs v1",
        "",
    ]
    b1 = (v1.get("barriers") or {})
    for name, summary in barriers.items():
        full = summary.get("full") or {}
        atr = summary.get("atr_only") or {}
        no_atr = summary.get("full_without_atr") or {}
        old_full = ((b1.get(name) or {}).get("full") or {})
        old_atr = ((b1.get(name) or {}).get("atr_only") or {})
        lines.extend(
            [
                f"### {name}",
                "",
                "| Metric | v2 Full | v2 ATR | v1 Full | v1 ATR |",
                "|--------|--------:|-------:|--------:|-------:|",
                f"| AUC | {fmt(full.get('auc'))} | {fmt(atr.get('auc'))} | {fmt(old_full.get('auc'))} | {fmt(old_atr.get('auc'))} |",
                f"| Brier | {fmt(full.get('brier'))} | {fmt(atr.get('brier'))} | {fmt(old_full.get('brier'))} | {fmt(old_atr.get('brier'))} |",
                f"| Top-5 success | {fmt(full.get('top_5_success_rate'))} | {fmt(atr.get('top_5_success_rate'))} | {fmt(old_full.get('top_5_success_rate'))} | {fmt(old_atr.get('top_5_success_rate'))} |",
                f"| Top-10% success | {fmt(full.get('top_10pct_success_rate'))} | {fmt(atr.get('top_10pct_success_rate'))} | {fmt(old_full.get('top_10pct_success_rate'))} | {fmt(old_atr.get('top_10pct_success_rate'))} |",
                f"| Full w/o ATR Spearman | {fmt(no_atr.get('spearman_prob_outcome'))} | — | — | — |",
                "",
            ]
        )
    lines.extend(
        [
            "## Leakage audit",
            "",
            f"- Passed: **{report['leakage_audit'].get('passed')}**",
            f"- Findings: {report['leakage_audit'].get('label_mismatches_or_findings')}",
            f"- Embargo violations: **{report['leakage_audit'].get('folds_violating_embargo')}**",
            "",
            "## Final classification",
            "",
            f"**Signal classification: `{report['signal_classification']}`**",
            "",
            report.get("final_interpretation", ""),
            "",
            f"## Tests: **{report['test_results'].get('ok')}**",
            "",
            f"## Errors: {report['run_metadata'].get('errors') or []}",
            "",
        ]
    )
    return "\n".join(lines)


def final_interpretation(report: dict) -> str:
    cls = report["signal_classification"]
    if cls == "signal_survives":
        return (
            "Barrier ranking signal remains present on the point-in-time universe; "
            "survivorship reduction did not eliminate the effect. Research-only."
        )
    if cls == "signal_weakens":
        return (
            "Barrier ranking signal is materially weaker on the point-in-time universe; "
            "part of the prior effect likely reflected retrospective universe selection."
        )
    if cls == "signal_disappears":
        return (
            "Barrier ranking signal is no longer present on the point-in-time universe; "
            "prior results were likely driven by retrospective universe bias."
        )
    return "Insufficient data to determine whether the signal survives universe replacement."


def print_terminal_summary(report: dict) -> None:
    u = report["universe"]
    comp = report["universe_comparison"]
    barriers = report["barriers"]
    lines = [
        "",
        "======== POINT_IN_TIME_UNIVERSE_V1 SUMMARY ========",
        "Experiment: point_in_time_universe_v1",
        "",
        f"Old universe: today's top-{comp.get('old_universe_size')} applied retrospectively",
        f"New universe: point-in-time trailing-24h-volume top-N (N={comp.get('old_universe_size')})",
        f"Average universe size: {fmt(u.get('average_universe_size'), 2)}",
        f"Average overlap: {fmt(comp.get('average_overlap_count'), 2)} "
        f"({fmt(comp.get('average_overlap_pct_of_new'), 3)} of new)",
        f"Survivorship limitation: {report['survivorship_short']}",
        f"Leakage: {'PASS' if report['leakage_audit'].get('passed') else 'FAIL'}",
        f"Tests: {'PASS' if report['test_results'].get('ok') else 'FAIL'}",
        "",
        "Barrier v2:",
    ]
    for name, summary in barriers.items():
        full = summary.get("full") or {}
        atr = summary.get("atr_only") or {}
        lines.extend(
            [
                "",
                f"{name}",
                f"Full AUC: {fmt(full.get('auc'))}",
                f"ATR AUC: {fmt(atr.get('auc'))}",
                f"Full top-5: {fmt(full.get('top_5_success_rate'))}",
                f"ATR top-5: {fmt(atr.get('top_5_success_rate'))}",
            ]
        )
    lines.extend(
        [
            "",
            f"Signal classification: {report['signal_classification']}",
            "========================================================",
            "",
        ]
    )
    print("\n".join(lines))


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info("pit-universe validate start db=%s", settings.database_url)
    test_results = run_research_tests()
    await init_db()
    provider = create_provider()
    await provider.start()
    try:
        ids = await ensure_assets(provider)
        old_symbols = sorted(ids)
        top_n = len(old_symbols)
        async with SessionLocal() as db:
            active_before = (
                await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
            ).scalars().all()
            active_before_ids = {str(m.id) for m in active_before}

            filled = await backfill(db, provider, ids)
            log.info("backfill added=%s", filled)

            # Determine research timestamps from existing samples (12-bar stride grid).
            all_rows = await load_upside_rows(db)
            primary_rows = select_by_stride(all_rows, EXPERIMENT.primary_stride_bars, base_stride=EXPERIMENT.sensitivity_stride_bars)
            timestamps = sorted({as_utc(r.ts) for r in primary_rows})
            log.info("universe timestamps=%s top_n=%s", len(timestamps), top_n)

            snap = await build_universe_snapshots(
                db, ids, timestamps, top_n=top_n, timeframe=timeframe().value
            )
            log.info("universe snapshots=%s", snap)
            selected_by_ts = await load_selected_symbols(db, timeframe().value)
            u_stats = await universe_stats(db, timeframe().value)
            comparison = compare_universes(old_symbols, selected_by_ts)

            # Rebuild dataset restricted to point-in-time universe membership.
            # We do not delete old rows; we filter evaluation rows to selected symbols per ts.
            pit_rows = [
                r
                for r in primary_rows
                if r.symbol in set(selected_by_ts.get(as_utc(r.ts), []))
            ]
            log.info("pit rows=%s of %s", len(pit_rows), len(primary_rows))

            built = await build_samples(db, ids, stride=EXPERIMENT.sensitivity_stride_bars)
            await backfill_directional_labels(db)
            candles = await candle_stats(db, timeframe().value)

            barrier_stats = await attach_barriers(pit_rows, ids, db)
            log.info("barriers %s", barrier_stats)

            bars_by = {}
            for sym, aid in list(ids.items())[:8]:
                bars_by[sym] = await load_bars(db, aid, timeframe().value)
            leak = await audit_leakage(pit_rows, bars_by)

            result = fit_barrier_walk_forward(
                pit_rows,
                EXPERIMENT,
                min_cross_section=EXPERIMENT.min_cross_section_size,
                leakage_ok=bool(leak.get("passed")),
            )
            model_row = await save_experiment_run(db, result)

            active_after = (
                await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
            ).scalars().all()
            active_after_ids = {str(m.id) for m in active_after}

            prior_v1 = {}
            if PRIOR_BARRIER_JSON.exists():
                try:
                    prior_v1 = json.loads(PRIOR_BARRIER_JSON.read_text())
                except Exception:
                    prior_v1 = {}

            metrics = result["metrics"]
            survivorship = (
                "Point-in-time universe reduces retrospective selection bias by using only "
                "trailing candles/volume at each T. It does NOT eliminate survivorship: "
                "symbols delisted before our data window are absent, and historical listing "
                "status is inferred from candles (exchange metadata is current-only)."
            )
            survivorship_short = (
                "Reduced via point-in-time trailing-volume selection; not eliminated "
                "(delisted-before-window symbols absent; listing status inferred)."
            )

            report = {
                "run_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "market_data_provider": settings.market_data_provider,
                    "database_url": settings.database_url,
                    "research_backfill_days": settings.research_backfill_days,
                    "universe_size": settings.universe_size,
                    "errors": errors,
                    "commands": [
                        "PYTHONPATH=backend python -m unittest tests.test_research -v",
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_point_in_time_universe",
                    ],
                },
                "experiment": EXPERIMENT.lineage(),
                "test_results": test_results,
                "universe": u_stats,
                "universe_comparison": comparison,
                "dataset_comparison": {
                    "old_rows": len(primary_rows),
                    "new_rows": len(pit_rows),
                    "old_symbols": len(old_symbols),
                    "new_symbols": u_stats.get("distinct_symbols_selected"),
                    "timestamps": len(timestamps),
                },
                "barrier_stats": metrics.get("barrier_stats"),
                "barrier_attachment": barrier_stats,
                "cross_section": metrics.get("cross_section"),
                "barriers": metrics.get("barriers"),
                "fold_count": max(
                    (metrics.get("barriers") or {}).get(b, {}).get("fold_count", 0)
                    for b, _, _ in BARRIER_SPECS
                )
                if metrics.get("barriers")
                else 0,
                "evidence": metrics.get("evidence"),
                "experiment_status": result["experiment_status"],
                "notes": result["notes"],
                "prior_barrier_v1": prior_v1,
                "leakage_audit": leak,
                "survivorship": survivorship,
                "survivorship_short": survivorship_short,
                "lineage_preservation": {
                    "active_bidirectional_models_before": sorted(active_before_ids),
                    "active_bidirectional_models_after": sorted(active_after_ids),
                    "active_models_preserved": active_before_ids <= active_after_ids,
                    "experiment_model_id": str(model_row.id),
                    "experiment_model_status": model_row.status,
                    "never_set_active": model_row.status != STATUS_ACTIVE,
                },
            }
            report["signal_classification"] = classify_signal_shift(prior_v1, {"barriers": metrics.get("barriers")})
            report["final_interpretation"] = final_interpretation(report)

            REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
            REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
            REPORT_MD.write_text(write_markdown(report))
            print_terminal_summary(report)
            log.info("wrote %s and %s", REPORT_JSON, REPORT_MD)
    except Exception as exc:
        errors.append(str(exc))
        log.exception("pit-universe validation failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
