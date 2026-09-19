"""Multi-period historical robustness validation.

Expands history, segments chronologically, and evaluates frozen barrier spec
across unseen periods with walk-forward training.

Does not overwrite prior reports or activate live models.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_multi_period_robustness
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-robustness-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import MULTI_PERIOD_ROBUSTNESS_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import BARRIER_SPECS, barrier_flags_from_bars
from app.research.robustness import run_multi_period_robustness
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe
from app.research.universe import build_universe_snapshots, load_selected_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_multi_period_robustness")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "MULTI_PERIOD_ROBUSTNESS_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "MULTI_PERIOD_ROBUSTNESS_REPORT.md"
EXPERIMENT = MULTI_PERIOD_ROBUSTNESS_V1


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


def write_markdown(report: dict) -> str:
    exp = report["experiment"]
    cs = report["cross_section"]
    periods = report.get("periods") or []
    lines = [
        f"# Multi-period robustness report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`MULTI_PERIOD_ROBUSTNESS_REPORT.json`](MULTI_PERIOD_ROBUSTNESS_REPORT.json)",
        "",
        "Prior experiments preserved:",
        "- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)",
        "- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)",
        "- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)",
        "- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)",
        "- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)",
        "- [`BARRIER_PROBABILITY_REPORT.md`](BARRIER_PROBABILITY_REPORT.md)",
        "- [`POINT_IN_TIME_UNIVERSE_REPORT.md`](POINT_IN_TIME_UNIVERSE_REPORT.md)",
        "- [`OUT_OF_SAMPLE_HOLDOUT_REPORT.md`](OUT_OF_SAMPLE_HOLDOUT_REPORT.md)",
        "- [`PORTFOLIO_BACKTEST_REPORT.md`](PORTFOLIO_BACKTEST_REPORT.md)",
        "",
        "## Objective",
        "",
        "Determine whether the frozen barrier signal persists across multiple chronological "
        "market periods rather than one short holdout.",
        "",
        "## Frozen specification",
        "",
        "Same 36 features, barriers, model, universe, stride, horizon, costs, and execution "
        "as `barrier_probability_v2` + `portfolio_backtest_v1`. No tuning on results.",
        "",
        "## Dataset integrity",
        "",
        f"- Timestamps considered: {cs.get('timestamps_considered')}",
        f"- Timestamps used: {cs.get('timestamps_used')}",
        f"- Asset observations: {cs.get('asset_observations_used')}",
        f"- Avg cross-section: {fmt(cs.get('average_cross_section_size'), 2)}",
        "",
        "## Period boundaries",
        "",
    ]
    for p in periods:
        lines.append(
            f"- Period {p.get('period')}: {p.get('start')} → {p.get('end')} "
            f"({p.get('timestamps')} timestamps, train={p.get('train_rows')}, test={p.get('test_rows')})"
        )
    lines.append("")

    lines.append("## Per-period predictive metrics (Full model)")
    lines.append("")
    lines.append("| Period | Barrier | AUC | Brier | Top-5 success | Top-10% success | All-assets |")
    lines.append("|--------|---------|-----|-------|--------------:|----------------:|-----------:|")
    for p in periods:
        for barrier, _, _ in BARRIER_SPECS:
            b = (p.get("barriers") or {}).get(barrier) or {}
            full = b.get("full") or {}
            lines.append(
                f"| {p.get('period')} | {barrier} | {fmt(full.get('auc'))} | "
                f"{fmt(full.get('brier'))} | {fmt(full.get('top_5_success_rate'))} | "
                f"{fmt(full.get('top_10pct_success_rate'))} | {fmt(full.get('all_success_rate'))} |"
            )
    lines.append("")

    lines.append("## Per-period portfolio (top-5, full vs momentum vs ATR)")
    lines.append("")
    lines.append("| Period | Barrier | Full net | Momentum net | ATR net | Full Sharpe | Full maxDD |")
    lines.append("|--------|---------|---------:|-------------:|--------:|------------:|-----------:|")
    for p in periods:
        for barrier, _, _ in BARRIER_SPECS:
            b = (p.get("barriers") or {}).get(barrier) or {}
            port = b.get("portfolio") or {}
            full = (port.get("full") or {}).get("top_5") or {}
            mom = (port.get("momentum") or {}).get("top_5") or {}
            atr = (port.get("atr_only") or {}).get("top_5") or {}
            lines.append(
                f"| {p.get('period')} | {barrier} | {fmt(full.get('cumulative_net_return'))} | "
                f"{fmt(mom.get('cumulative_net_return'))} | {fmt(atr.get('cumulative_net_return'))} | "
                f"{fmt(full.get('sharpe'))} | {fmt(full.get('max_drawdown'))} |"
            )
    lines.append("")

    lines.extend(
        [
            "## Combined chronological portfolio",
            "",
            json.dumps(report.get("combined"), indent=2, default=str),
            "",
            "## Robustness counts",
            "",
            json.dumps(report.get("robust_counts"), indent=2, default=str),
            "",
            "## Leakage audit",
            "",
            f"- Passed: **{report['leakage_audit'].get('passed')}**",
            f"- Findings: {report['leakage_audit'].get('label_mismatches_or_findings')}",
            "",
            "## Multiple-testing limitation",
            "",
            report["multiple_testing_note"],
            "",
            "## Survivorship limitation",
            "",
            report["survivorship"],
            "",
            "## Final classification",
            "",
            f"**`{report['experiment_status']}`**",
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
    status = report["experiment_status"]
    if status == "robust_across_periods":
        return "Signal remains directionally consistent across multiple unseen periods."
    if status == "mixed_regime_signal":
        return "Signal appears in multiple periods but varies substantially by market environment."
    if status == "period_specific_signal":
        return "Signal is concentrated in one or a small number of periods."
    if status == "fails_robustness_test":
        return "Signal largely disappears when evaluated across multiple historical periods."
    return "Insufficient reliable history for intended evaluation."


def print_terminal_summary(report: dict) -> None:
    periods = report.get("periods") or []
    combined = report.get("combined") or {}
    lines = [
        "",
        "======== MULTI_PERIOD_ROBUSTNESS_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
        f"History: {report['dataset'].get('oldest_candle')} → {report['dataset'].get('newest_candle')}",
        f"Samples: {report['cross_section'].get('asset_observations_used')}",
        f"Periods: {len(periods)}",
        "",
    ]
    for p in periods:
        lines.append(f"Period {p.get('period')}: {p.get('start')} → {p.get('end')}")
        for barrier, _, _ in BARRIER_SPECS:
            b = (p.get("barriers") or {}).get(barrier) or {}
            full = b.get("full") or {}
            port = (b.get("portfolio") or {})
            full5 = (port.get("full") or {}).get("top_5") or {}
            mom5 = (port.get("momentum") or {}).get("top_5") or {}
            atr5 = (port.get("atr_only") or {}).get("top_5") or {}
            lines.append(
                f"  {barrier}: AUC={fmt(full.get('auc'))}, top5={fmt(full.get('top_5_success_rate'))}, "
                f"full_net={fmt(full5.get('cumulative_net_return'))}, "
                f"mom_net={fmt(mom5.get('cumulative_net_return'))}, "
                f"atr_net={fmt(atr5.get('cumulative_net_return'))}"
            )
        lines.append("")
    lines.append("Combined (full, top-5):")
    for barrier, c in combined.items():
        lines.append(
            f"  {barrier}: net={fmt(c.get('cumulative_net_return'))}, maxDD={fmt(c.get('max_drawdown'))}, "
            f"sharpe={fmt(c.get('sharpe'))}, win={fmt(c.get('win_rate'))}, trades={c.get('n_trades')}"
        )
    lines.extend(
        [
            "",
            f"Leakage: {'PASS' if report['leakage_audit'].get('passed') else 'FAIL'}",
            f"Tests: {'PASS' if report['test_results'].get('ok') else 'FAIL'}",
            f"Classification: {report['experiment_status']}",
            f"Limitations: {report['survivorship_short']}",
            "========================================================",
            "",
        ]
    )
    print("\n".join(lines))


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info("multi-period robustness start db=%s", settings.database_url)
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

            # Expand history as far as practical (existing backfill_days config).
            filled = await backfill(db, provider, ids)
            log.info("backfill added=%s", filled)

            all_rows = await load_upside_rows(db)
            primary_rows = select_by_stride(all_rows, EXPERIMENT.primary_stride_bars, base_stride=EXPERIMENT.sensitivity_stride_bars)
            timestamps = sorted({as_utc(r.ts) for r in primary_rows})
            await build_universe_snapshots(db, ids, timestamps, top_n=top_n, timeframe=timeframe().value)
            selected_by_ts = await load_selected_symbols(db, timeframe().value)
            pit_rows = [r for r in primary_rows if r.symbol in set(selected_by_ts.get(as_utc(r.ts), []))]

            built = await build_samples(db, ids, stride=EXPERIMENT.sensitivity_stride_bars)
            await backfill_directional_labels(db)
            candles = await candle_stats(db, timeframe().value)
            barrier_stats = await attach_barriers(pit_rows, ids, db)

            bars_by = {}
            for sym, aid in ids.items():
                bars_by[sym] = await load_bars(db, aid, timeframe().value)
            leak = await audit_leakage(pit_rows, {k: bars_by[k] for k in list(bars_by)[:8]})

            result = run_multi_period_robustness(
                pit_rows,
                bars_by,
                EXPERIMENT,
                min_cross_section=EXPERIMENT.min_cross_section_size,
                leakage_ok=bool(leak.get("passed")),
            )
            model_row = await save_experiment_run(db, result)

            active_after = (
                await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
            ).scalars().all()
            active_after_ids = {str(m.id) for m in active_after}

            metrics = result["metrics"]
            survivorship = (
                "Point-in-time universe reduces retrospective selection bias, but historical "
                "listing/delisting metadata is unavailable from this provider; symbols delisted "
                "before the data window are absent. Survivorship bias is reduced, not eliminated."
            )
            survivorship_short = "PIT universe; delisted-before-window absent; listing inferred."
            multiple_testing_note = (
                "This project has already tested multiple targets and model comparisons. "
                "This robustness experiment is confirmatory, not exploratory; historical "
                "results may contain selection effects from prior research."
            )

            code_commit = os.environ.get("GIT_COMMIT") or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
            ).stdout.strip()

            report = {
                "run_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "market_data_provider": settings.market_data_provider,
                    "database_url": settings.database_url,
                    "errors": errors,
                    "code_commit": code_commit,
                    "commands": [
                        "PYTHONPATH=backend python -m unittest tests.test_research -v",
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_multi_period_robustness",
                    ],
                },
                "experiment": EXPERIMENT.lineage(),
                "test_results": test_results,
                "dataset": {
                    "symbols_count": len(old_symbols),
                    "oldest_candle": candles.get("oldest"),
                    "newest_candle": candles.get("newest"),
                    "raw_candles": candles.get("count"),
                    "feature_count": len(feature_names()),
                    "lookback_bars": LOOKBACK_BARS,
                    "pit_rows": len(pit_rows),
                },
                "barrier_attachment": barrier_stats,
                "cross_section": metrics.get("cross_section"),
                "periods": metrics.get("periods"),
                "combined": metrics.get("combined"),
                "robust_counts": metrics.get("robust_counts"),
                "experiment_status": result["experiment_status"],
                "notes": result["notes"],
                "leakage_audit": leak,
                "survivorship": survivorship,
                "survivorship_short": survivorship_short,
                "multiple_testing_note": multiple_testing_note,
                "lineage_preservation": {
                    "active_bidirectional_models_before": sorted(active_before_ids),
                    "active_bidirectional_models_after": sorted(active_after_ids),
                    "active_models_preserved": active_before_ids <= active_after_ids,
                    "experiment_model_id": str(model_row.id),
                    "experiment_model_status": model_row.status,
                    "never_set_active": model_row.status != STATUS_ACTIVE,
                },
            }
            report["final_interpretation"] = final_interpretation(report)

            REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
            REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
            REPORT_MD.write_text(write_markdown(report))
            print_terminal_summary(report)
            log.info("wrote %s and %s", REPORT_JSON, REPORT_MD)
    except Exception as exc:
        errors.append(str(exc))
        log.exception("multi-period robustness failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
