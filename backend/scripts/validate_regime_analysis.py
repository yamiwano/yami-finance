"""Regime analysis validation.

Characterizes market regimes at signal time T using only past data, then
evaluates the frozen barrier model per regime. No model/threshold tuning.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_regime_analysis
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-regime-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import REGIME_ANALYSIS_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import BARRIER_SPECS, barrier_flags_from_bars
from app.research.portfolio import COST_PER_EVENT, run_portfolio_backtest
from app.research.regime import (
    MIN_REGIME_SAMPLE,
    assign_regimes,
    classify_regime_relationship,
    fit_regime_thresholds,
    group_rows_by_regime,
    period_composition,
)
from app.research.robustness import (
    evaluate_period_predictions,
    fit_models_on_rows,
    predict_with_models,
    segment_periods,
)
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe
from app.research.universe import build_universe_snapshots, load_selected_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_regime_analysis")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "REGIME_ANALYSIS_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "REGIME_ANALYSIS_REPORT.md"
EXPERIMENT = REGIME_ANALYSIS_V1
REGIME_DIMENSIONS = (
    "btc_trend",
    "btc_volatility",
    "market_breadth",
    "market_volatility",
    "momentum_dispersion",
    "volume",
)


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
    lines = [
        f"# Regime analysis report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`REGIME_ANALYSIS_REPORT.json`](REGIME_ANALYSIS_REPORT.json)",
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
        "",
        "## Objective",
        "",
        "Determine which observable market conditions at signal time T are associated with "
        "stronger/weaker performance of the frozen barrier signal. No model changes.",
        "",
        "## Frozen specification",
        "",
        "Same 36 features, barriers, model, universe, stride, horizon, costs, execution. "
        "Regime thresholds fit on development data only.",
        "",
        "## Regime definitions",
        "",
        "- BTC trend: 24h return > +1% bullish, < −1% bearish, else neutral",
        "- BTC volatility: trailing 24h realized vol tertiles (development-fitted)",
        "- Market breadth: % assets with positive 24h return (>60% strong, <40% weak)",
        "- Market volatility: median ATR% tertiles (development-fitted)",
        "- Momentum dispersion: cross-sectional std of 24h returns (tertiles)",
        "- Volume: median relative_volume tertiles (development-fitted)",
        "",
        "## Threshold methodology",
        "",
        json.dumps(report.get("thresholds"), indent=2, default=str),
        "",
        "## Period composition",
        "",
        json.dumps(report.get("period_composition"), indent=2, default=str),
        "",
        "## Predictive performance by regime",
        "",
    ]
    for dim, by_regime in (report.get("regime_results") or {}).items():
        lines.append(f"### {dim}")
        lines.append("")
        lines.append("| Regime | Barrier | Samples | Full AUC | Full Top5 | ATR Top5 | Mom Top5 |")
        lines.append("|--------|---------|--------:|---------:|----------:|---------:|---------:|")
        for regime, by_barrier in by_regime.items():
            for barrier, metrics in by_barrier.items():
                full = metrics.get("full") or {}
                atr = metrics.get("atr_only") or {}
                mom = metrics.get("momentum") or {}
                lines.append(
                    f"| {regime} | {barrier} | {metrics.get('n')} | {fmt(full.get('auc'))} | "
                    f"{fmt(full.get('top_5_success_rate'))} | {fmt(atr.get('top_5_success_rate'))} | "
                    f"{fmt(mom.get('top_5_success_rate'))} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Portfolio performance by regime (top-5)",
            "",
            json.dumps(report.get("regime_portfolio"), indent=2, default=str),
            "",
            "## Regime transitions",
            "",
            json.dumps(report.get("transitions"), indent=2, default=str),
            "",
            "## Regime interactions",
            "",
            json.dumps(report.get("interactions"), indent=2, default=str),
            "",
            "## Multiple-testing limitation",
            "",
            report["multiple_testing_note"],
            "",
            "## Survivorship limitation",
            "",
            report["survivorship"],
            "",
            "## Leakage audit",
            "",
            f"- Passed: **{report['leakage_audit'].get('passed')}**",
            f"- Findings: {report['leakage_audit'].get('label_mismatches_or_findings')}",
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
    if status == "strong_regime_dependence":
        return "Signal performance varies materially across measured regimes; regime context matters."
    if status == "candidate_regime_relationships":
        return "Some regime-dependent variation exists, but it is not decisive."
    if status == "no_clear_regime_relationship":
        return "No clear regime relationship detected across measured dimensions."
    return "Insufficient data for regime analysis."


def print_terminal_summary(report: dict) -> None:
    lines = [
        "",
        "======== REGIME_ANALYSIS_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
        f"History: {report['dataset'].get('oldest_candle')} → {report['dataset'].get('newest_candle')}",
        f"Samples: {report['cross_section'].get('asset_observations_used')}",
        f"Regime dimensions: {len(report.get('regime_results') or {})}",
        "",
    ]
    for dim, by_regime in (report.get("regime_results") or {}).items():
        lines.append(f"--- {dim} ---")
        for regime, by_barrier in by_regime.items():
            for barrier, metrics in by_barrier.items():
                full = metrics.get("full") or {}
                lines.append(
                    f"  {regime} {barrier}: n={metrics.get('n')}, AUC={fmt(full.get('auc'))}, "
                    f"top5={fmt(full.get('top_5_success_rate'))}"
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
    log.info("regime analysis start db=%s", settings.database_url)
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

            # Fit regime thresholds on development portion (first 75% of timestamps)
            cutoff_idx = int(len(timestamps) * 0.75)
            train_end = timestamps[cutoff_idx] if len(timestamps) > cutoff_idx else timestamps[-1]
            thresholds = fit_regime_thresholds(pit_rows, btc_bars, train_end=train_end)
            regime_rows = assign_regimes(pit_rows, btc_bars, thresholds)
            regime_by_ts = {r["ts"]: r for r in regime_rows}

            periods = segment_periods(timestamps, n_periods=4)
            period_comp = period_composition(periods, regime_rows)

            # Evaluate frozen model per regime on evaluation portion
            eval_rows = [r for r in pit_rows if as_utc(r.ts) > train_end]
            regime_results: dict[str, Any] = {}
            regime_portfolio: dict[str, Any] = {}
            transitions: dict[str, Any] = {}
            interactions: dict[str, Any] = {}

            for dim in REGIME_DIMENSIONS:
                groups = group_rows_by_regime(eval_rows, regime_by_ts, dim)
                regime_results[dim] = {}
                regime_portfolio[dim] = {}
                for regime, rows in groups.items():
                    if len(rows) < MIN_REGIME_SAMPLE:
                        regime_results[dim][regime] = {"note": "insufficient_sample", "n": len(rows)}
                        continue
                    regime_results[dim][regime] = {}
                    for barrier, up_pct, down_pct in BARRIER_SPECS:
                        models = fit_models_on_rows([r for r in pit_rows if as_utc(r.ts) <= train_end], barrier)
                        if not models:
                            continue
                        scores = predict_with_models(models, rows)
                        metrics = {}
                        for strat, sc in scores.items():
                            metrics[strat] = evaluate_period_predictions(rows, sc, barrier)
                        metrics["n"] = len(rows)
                        regime_results[dim][regime][barrier] = metrics
                        # Portfolio top-5 for full model
                        res = run_portfolio_backtest(
                            rows,
                            bars_by,
                            scores["full"],
                            barrier=barrier,
                            up_pct=up_pct,
                            down_pct=down_pct,
                            top_k=5,
                            horizon_bars=horizon_bars(timeframe().value, horizon_hours()),
                            cost=COST_PER_EVENT,
                            strategy="full",
                        )
                        regime_portfolio.setdefault(dim, {}).setdefault(regime, {})[barrier] = {
                            "n_trades": res["n_trades"],
                            "cumulative_net_return": res["cumulative_net_return"],
                            "win_rate": res["win_rate"],
                            "max_drawdown": res["max_drawdown"],
                            "sharpe": res["sharpe"],
                        }

            # Transitions: compare signal performance on regime changes vs stable
            for dim in ("btc_volatility", "market_breadth"):
                trans_groups: dict[str, list[Any]] = {}
                prev = None
                for r in regime_rows:
                    curr = r[dim]
                    if prev is not None and prev != "unknown" and curr != "unknown":
                        label = f"{prev}_to_{curr}" if prev != curr else f"stable_{curr}"
                        trans_groups.setdefault(label, []).extend(
                            [row for row in eval_rows if as_utc(row.ts) == r["ts"]]
                        )
                    prev = curr
                transitions[dim] = {
                    k: {"n": len(v), "note": "insufficient_sample" if len(v) < MIN_REGIME_SAMPLE else "ok"}
                    for k, v in trans_groups.items()
                }

            # Interactions: small fixed set
            for a, b in (
                ("btc_trend", "btc_volatility"),
                ("btc_trend", "market_breadth"),
                ("market_breadth", "market_volatility"),
                ("market_volatility", "momentum_dispersion"),
            ):
                key = f"{a}×{b}"
                groups: dict[str, list[Any]] = {}
                for row in eval_rows:
                    ts = as_utc(row.ts)
                    ra = regime_by_ts.get(ts, {}).get(a, "unknown")
                    rb = regime_by_ts.get(ts, {}).get(b, "unknown")
                    label = f"{ra}×{rb}"
                    groups.setdefault(label, []).append(row)
                interactions[key] = {
                    k: {"n": len(v), "note": "insufficient_sample" if len(v) < MIN_REGIME_SAMPLE else "ok"}
                    for k, v in groups.items()
                }

            classification = classify_regime_relationship(regime_results)
            if not leak.get("passed"):
                classification = "no_clear_regime_relationship"

            metrics = {
                "thresholds": thresholds,
                "period_composition": period_comp,
                "regime_results": regime_results,
                "regime_portfolio": regime_portfolio,
                "transitions": transitions,
                "interactions": interactions,
                "lineage": EXPERIMENT.lineage(),
                "cross_section": {
                    "asset_observations_used": len(pit_rows),
                    "eval_rows": len(eval_rows),
                    "timestamps_used": len(timestamps),
                },
            }
            result = {
                "promote": False,
                "experiment_status": classification,
                "passed_experiment_gates": classification == "strong_regime_dependence",
                "feature_names": feature_names(),
                "blob": b"",
                "metrics": metrics,
                "notes": f"Regime classification: {classification}.",
                "experiment_id": EXPERIMENT.experiment_id,
                "target_name": "barrier_first_touch",
                "sample_stride": EXPERIMENT.primary_stride_bars,
                "move_threshold": EXPERIMENT.move_pct,
            }
            model_row = await save_experiment_run(db, result)

            active_after = (
                await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
            ).scalars().all()
            active_after_ids = {str(m.id) for m in active_after}

            survivorship = (
                "Regime analysis uses the point-in-time universe. Survivorship is reduced but "
                "not eliminated: symbols delisted before the data window are absent."
            )
            survivorship_short = "PIT universe; delisted-before-window absent."
            multiple_testing_note = (
                "Many targets and comparisons have been tested in this project. Regime "
                "relationships are hypotheses for future testing, not confirmed effects."
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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_regime_analysis",
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
                "thresholds": thresholds,
                "period_composition": period_comp,
                "regime_results": regime_results,
                "regime_portfolio": regime_portfolio,
                "transitions": transitions,
                "interactions": interactions,
                "experiment_status": classification,
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
        log.exception("regime analysis failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
