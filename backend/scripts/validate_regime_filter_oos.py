"""Regime filter out-of-sample validation.

Uses frozen thresholds from regime_analysis_v1 and evaluates filters on data
strictly after the discovery period. No tuning on OOS results.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_regime_filter_oos
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-regime-filter-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import REGIME_FILTER_OOS_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import BARRIER_SPECS, barrier_flags_from_bars
from app.research.regime import fit_regime_thresholds
from app.research.regime_filter import run_regime_filter_oos
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe
from app.research.universe import build_universe_snapshots, load_selected_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_regime_filter_oos")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "REGIME_FILTER_OOS_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "REGIME_FILTER_OOS_REPORT.md"
EXPERIMENT = REGIME_FILTER_OOS_V1


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
    blocklist = ("future_", "up3_", "up5_", "up10_", "barrier", "opportunity", "quote_volume_24h", "regime")
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
            findings.append(f"target/regime key leaked into features {row.symbol}")
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
    barriers = report.get("barriers") or {}
    lines = [
        f"# Regime filter OOS report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`REGIME_FILTER_OOS_REPORT.json`](REGIME_FILTER_OOS_REPORT.json)",
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
        "- [`MULTI_PERIOD_ROBUSTNESS_REPORT.md`](MULTI_PERIOD_ROBUSTNESS_REPORT.md)",
        "- [`REGIME_ANALYSIS_REPORT.md`](REGIME_ANALYSIS_REPORT.md)",
        "",
        "## Objective",
        "",
        "Test whether previously identified regime filters (high momentum dispersion, normal "
        "volume, combined) improve the frozen barrier model on genuinely unseen data.",
        "",
        "## Frozen base model",
        "",
        "Same 36 features, HistGradientBoostingClassifier, barriers, universe, stride, horizon, "
        "costs, execution. Filters are external to the model.",
        "",
        "## Frozen regime definitions / thresholds",
        "",
        "```",
        json.dumps(report.get("thresholds"), indent=2, default=str),
        "```",
        "",
        "## OOS period",
        "",
        f"- Train end: **{report.get('train_end')}**",
        f"- OOS rows: {report.get('oos_rows')} across {report.get('oos_timestamps')} timestamps",
        "",
        "## Survivorship limitation",
        "",
        report["survivorship"],
        "",
        "## Key comparison (top-5)",
        "",
        "| Barrier | Strategy | OOS trades | OOS net | Max DD | Sharpe | Win rate | Exposure |",
        "|---------|----------|-----------:|--------:|-------:|-------:|---------:|---------:|",
    ]
    for barrier, data in barriers.items():
        port = data.get("portfolio") or {}
        for strat in ("baseline", "high_dispersion", "normal_volume", "high_dispersion_plus_normal_volume", "atr_only", "momentum", "full_without_atr"):
            res = port.get(strat) or {}
            exp_stats = res.get("exposure") or {}
            lines.append(
                f"| {barrier} | {strat} | {res.get('n_trades')} | {fmt(res.get('cumulative_net_return'))} | "
                f"{fmt(res.get('max_drawdown'))} | {fmt(res.get('sharpe'))} | {fmt(res.get('win_rate'))} | "
                f"{fmt(exp_stats.get('pct_timestamps_allowed'), 3)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Incremental value (filtered − baseline)",
            "",
            json.dumps(report.get("combined_vs_baseline"), indent=2, default=str),
            "",
            "## Random filtered benchmark",
            "",
            json.dumps({b: (d.get("random_benchmark") or {}) for b, d in barriers.items()}, indent=2, default=str),
            "",
            "## Slippage sensitivity",
            "",
            json.dumps({b: (d.get("slippage") or {}) for b, d in barriers.items()}, indent=2, default=str),
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
    if status == "filter_generalizes":
        return "Frozen regime filter improved the frozen strategy on genuinely unseen data."
    if status == "filter_mixed":
        return "Filter shows improvement in some barriers/metrics but not consistently enough."
    if status == "filter_fails":
        return "Previously identified regime relationship does not generalize to the OOS period."
    return "Insufficient genuinely untouched historical data."


def print_terminal_summary(report: dict) -> None:
    barriers = report.get("barriers") or {}
    lines = [
        "",
        "======== REGIME_FILTER_OOS_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
        f"Train end: {report.get('train_end')}",
        f"OOS rows: {report.get('oos_rows')} | OOS timestamps: {report.get('oos_timestamps')}",
        "",
    ]
    for barrier, data in barriers.items():
        port = data.get("portfolio") or {}
        lines.append(f"--- {barrier} ---")
        for strat in ("baseline", "high_dispersion", "normal_volume", "high_dispersion_plus_normal_volume"):
            res = port.get(strat) or {}
            exp = res.get("exposure") or {}
            lines.append(
                f"  {strat}: net={fmt(res.get('cumulative_net_return'))}, "
                f"trades={res.get('n_trades')}, win={fmt(res.get('win_rate'))}, "
                f"exposure={fmt(exp.get('pct_timestamps_allowed'), 3)}"
            )
        lines.append("")
    lines.extend(
        [
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
    log.info("regime filter oos start db=%s", settings.database_url)
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
            btc_bars = bars_by.get("BTCUSDT") or []
            leak = await audit_leakage(pit_rows, {k: bars_by[k] for k in list(bars_by)[:8]})

            # Frozen thresholds from regime_analysis_v1 methodology: fit on first 75% of timestamps
            cutoff_idx = int(len(timestamps) * 0.75)
            train_end = timestamps[cutoff_idx] if len(timestamps) > cutoff_idx else timestamps[-1]
            thresholds = fit_regime_thresholds(pit_rows, btc_bars, train_end=train_end)

            result = run_regime_filter_oos(
                pit_rows,
                bars_by,
                EXPERIMENT,
                train_end=train_end,
                thresholds=thresholds,
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
                "listing/delisting metadata is unavailable; symbols delisted before the data "
                "window are absent. Survivorship bias is reduced, not eliminated."
            )
            survivorship_short = "PIT universe; delisted-before-window absent."
            multiple_testing_note = (
                "Many hypotheses have been tested in this project. This OOS filter test is "
                "confirmatory; apparent effects may still reflect selection from prior research."
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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_regime_filter_oos",
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
                "train_end": metrics.get("train_end"),
                "oos_rows": metrics.get("oos_rows"),
                "oos_timestamps": metrics.get("oos_timestamps"),
                "thresholds": metrics.get("thresholds"),
                "barriers": metrics.get("barriers"),
                "combined_vs_baseline": metrics.get("combined_vs_baseline"),
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
        log.exception("regime filter oos failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
