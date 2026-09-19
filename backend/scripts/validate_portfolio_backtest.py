"""Realistic portfolio backtest on frozen out-of-sample barrier predictions.

Does not overwrite prior reports or activate live models.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_portfolio_backtest
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

import numpy as np
from sqlalchemy import select

os.environ.setdefault("MARKET_DATA_PROVIDER", "binance")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-portfolio-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.barriers import COST_PER_EVENT
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import PORTFOLIO_BACKTEST_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.holdout import choose_holdout_cutoff, split_development_holdout
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import BARRIER_SPECS, barrier_flags_from_bars
from app.research.portfolio import (
    HOLDOUT_CUTOFF,
    RANDOM_SIMS,
    STARTING_CAPITAL,
    classify_portfolio,
    fit_frozen_models,
    predict_scores,
    run_portfolio_backtest,
)
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe
from app.research.universe import build_universe_snapshots, load_selected_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_portfolio_backtest")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "PORTFOLIO_BACKTEST_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "PORTFOLIO_BACKTEST_REPORT.md"
EXPERIMENT = PORTFOLIO_BACKTEST_V1
TOP_KS = (1, 3, 5, "10pct")
SLIPPAGE_CASES = {"base": 0.0, "plus_0.1pct": 0.001, "plus_0.25pct": 0.0025}


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


async def audit_leakage(
    dev_rows: list[ResearchSample],
    hold_rows: list[ResearchSample],
    bars_by_symbol: dict,
    cutoff: datetime,
) -> dict:
    tf = timeframe().value
    h = horizon_bars(tf)
    thr = move_pct()
    findings: list[str] = []
    checked = 0
    blocklist = ("future_", "up3_", "up5_", "up10_", "barrier", "opportunity", "quote_volume_24h")
    for row in (dev_rows + hold_rows)[:200]:
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
    if dev_rows and hold_rows:
        dev_max = max(as_utc(r.ts) for r in dev_rows)
        hold_min = min(as_utc(r.ts) for r in hold_rows)
        if hold_min <= dev_max:
            findings.append("holdout overlaps development timestamps")
    return {
        "samples_checked": checked,
        "label_mismatches_or_findings": findings[:20],
        "embargo_hours": horizon_hours(),
        "cutoff": cutoff.isoformat(),
        "passed": len(findings) == 0,
    }


def fmt(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown(report: dict) -> str:
    exp = report["experiment"]
    split = report["split"]
    lines = [
        f"# Portfolio backtest report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`PORTFOLIO_BACKTEST_REPORT.json`](PORTFOLIO_BACKTEST_REPORT.json)",
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
        "",
        "## Objective",
        "",
        "Determine whether the frozen barrier-ranking signal translates into a realistic "
        "portfolio after next-candle execution, costs, and overlapping positions.",
        "",
        "## Frozen assumptions",
        "",
        "- Features, targets, model, universe, and holdout cutoff are frozen from "
        "`out_of_sample_holdout_v1`. No tuning on holdout.",
        f"- Cost: {report.get('cost_per_event')} per completed trade (round-trip).",
        f"- Slippage cases: {list(report.get('slippage_cases', {}).keys())}",
        "- Equal-weight positions; max 100% gross exposure; no leverage.",
        "",
        "## Periods",
        "",
        f"- Development: {split.get('development_start')} → {split.get('development_end')}",
        f"- Holdout: {split.get('holdout_start')} → {split.get('holdout_end')} "
        f"(cutoff `{split.get('cutoff')}`)",
        "",
        "## Execution methodology",
        "",
        "Signal at T → next 1h candle open entry → first barrier touch or 12h horizon exit. "
        "Same-candle dual touch ⇒ ambiguous ⇒ exit at that candle's open.",
        "",
        "## Survivorship limitation",
        "",
        report["survivorship"],
        "",
        "## Holdout results by barrier and K",
        "",
    ]
    for barrier, by_k in (report.get("holdout") or {}).items():
        lines.append(f"### {barrier}")
        lines.append("")
        lines.append("| Strategy | Top-K | Trades | Win Rate | Avg Trade | Net Return | Max DD | Sharpe | Profit Factor |")
        lines.append("|----------|------:|-------:|---------:|----------:|-----------:|-------:|-------:|--------------:|")
        for strat, ks in by_k.items():
            for k, res in ks.items():
                lines.append(
                    f"| {strat} | {k} | {res.get('n_trades')} | {fmt(res.get('win_rate'))} | "
                    f"{fmt(res.get('avg_trade_return'))} | {fmt(res.get('cumulative_net_return'))} | "
                    f"{fmt(res.get('max_drawdown'))} | {fmt(res.get('sharpe'))} | {fmt(res.get('profit_factor'))} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Development vs holdout",
            "",
            json.dumps(report.get("development_vs_holdout"), indent=2, default=str),
            "",
            "## Slippage sensitivity",
            "",
            json.dumps(report.get("slippage_sensitivity"), indent=2, default=str),
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
    if status == "economically_plausible":
        return (
            "Frozen signal produced coherent portfolio results after costs on the holdout. "
            "Research-only; not a validated trading system."
        )
    if status == "weak_economic_signal":
        return (
            "Some economic value exists, but returns/risk/execution sensitivity are not robust "
            "enough to treat as economically convincing."
        )
    if status == "not_economically_supported":
        return "The predictive signal does not translate into convincing portfolio results."
    return "Insufficient data for meaningful portfolio evaluation."


def print_terminal_summary(report: dict) -> None:
    split = report["split"]
    hold = report.get("holdout") or {}
    lines = [
        "",
        "======== PORTFOLIO_BACKTEST_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
        f"Development: {split.get('development_start')} → {split.get('development_end')}",
        f"Holdout: {split.get('holdout_start')} → {split.get('holdout_end')}",
        f"Starting capital: {STARTING_CAPITAL}",
        "",
    ]
    for barrier, by_k in hold.items():
        full5 = (by_k.get("full") or {}).get("top_5") or {}
        atr5 = (by_k.get("atr_only") or {}).get("top_5") or {}
        rand5 = (by_k.get("random") or {}).get("top_5") or {}
        lines.extend(
            [
                f"--- {barrier} (top-5) ---",
                f"Full: net={fmt(full5.get('cumulative_net_return'))}, "
                f"maxDD={fmt(full5.get('max_drawdown'))}, sharpe={fmt(full5.get('sharpe'))}, "
                f"win={fmt(full5.get('win_rate'))}, trades={full5.get('n_trades')}",
                f"ATR: net={fmt(atr5.get('cumulative_net_return'))}, "
                f"Random: net={fmt(rand5.get('cumulative_net_return'))}",
                "",
            ]
        )
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
    log.info("portfolio backtest start db=%s", settings.database_url)
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

            # Frozen holdout cutoff from out_of_sample_holdout_v1
            cutoff = datetime.fromisoformat(HOLDOUT_CUTOFF.replace("Z", "+00:00"))
            dev_rows, hold_rows, split_meta = split_development_holdout(pit_rows, cutoff)
            # Enforce embargo on development
            gap = embargo()
            hold_start = min(as_utc(r.ts) for r in hold_rows)
            dev_rows = [r for r in dev_rows if as_utc(r.ts) <= hold_start - gap]

            bars_by = {}
            for sym, aid in ids.items():
                bars_by[sym] = await load_bars(db, aid, timeframe().value)

            leak = await audit_leakage(dev_rows, hold_rows, {k: bars_by[k] for k in list(bars_by)[:8]}, cutoff)

            h_bars = horizon_bars(timeframe().value, horizon_hours())
            holdout_results: dict[str, Any] = {}
            dev_results: dict[str, Any] = {}
            slippage_results: dict[str, Any] = {}

            for barrier, up_pct, down_pct in BARRIER_SPECS:
                models = fit_frozen_models(dev_rows, barrier)
                if not models:
                    continue
                hold_scores = predict_scores(models, hold_rows)
                dev_scores = predict_scores(models, dev_rows)

                by_k_hold: dict[str, Any] = {}
                by_k_dev: dict[str, Any] = {}
                for strat, scores in hold_scores.items():
                    ks: dict[str, Any] = {}
                    for k in TOP_KS:
                        kk = k if isinstance(k, int) else max(1, int(0.10 * len(hold_rows) / max(1, len(set(r.ts for r in hold_rows)))))
                        res = run_portfolio_backtest(
                            hold_rows,
                            bars_by,
                            scores,
                            barrier=barrier,
                            up_pct=up_pct,
                            down_pct=down_pct,
                            top_k=kk,
                            horizon_bars=h_bars,
                            cost=COST_PER_EVENT,
                            strategy=strat,
                        )
                        ks[f"top_{k}" if isinstance(k, int) else "top_10pct"] = res
                    by_k_hold[strat] = ks
                # Random benchmark (multiple sims)
                rand_sims = []
                for s in range(RANDOM_SIMS):
                    res = run_portfolio_backtest(
                        hold_rows,
                        bars_by,
                        np.zeros(len(hold_rows)),
                        barrier=barrier,
                        up_pct=up_pct,
                        down_pct=down_pct,
                        top_k=5,
                        horizon_bars=h_bars,
                        cost=COST_PER_EVENT,
                        strategy="random",
                        seed=42 + s,
                    )
                    rand_sims.append(res["cumulative_net_return"])
                by_k_hold["random"] = {
                    "top_5": {
                        "cumulative_net_return_mean": float(np.mean(rand_sims)),
                        "cumulative_net_return_median": float(np.median(rand_sims)),
                        "cumulative_net_return_p5": float(np.quantile(rand_sims, 0.05)),
                        "cumulative_net_return_p95": float(np.quantile(rand_sims, 0.95)),
                        "n_sims": RANDOM_SIMS,
                    }
                }
                # Equal-weight all eligible
                ew_scores = np.zeros(len(hold_rows))
                by_k_hold["equal_weight_all"] = {
                    "top_5": run_portfolio_backtest(
                        hold_rows,
                        bars_by,
                        ew_scores,
                        barrier=barrier,
                        up_pct=up_pct,
                        down_pct=down_pct,
                        top_k=max(1, len(hold_rows) // max(1, len(set(r.ts for r in hold_rows)))),
                        horizon_bars=h_bars,
                        cost=COST_PER_EVENT,
                        strategy="equal_weight_all",
                    )
                }
                holdout_results[barrier] = by_k_hold

                # Development reference (top-5 only)
                for strat, scores in dev_scores.items():
                    res = run_portfolio_backtest(
                        dev_rows,
                        bars_by,
                        scores,
                        barrier=barrier,
                        up_pct=up_pct,
                        down_pct=down_pct,
                        top_k=5,
                        horizon_bars=h_bars,
                        cost=COST_PER_EVENT,
                        strategy=strat,
                    )
                    by_k_dev[strat] = {"top_5": res}
                dev_results[barrier] = by_k_dev

                # Slippage sensitivity on full top-5
                slip = {}
                for case, extra in SLIPPAGE_CASES.items():
                    res = run_portfolio_backtest(
                        hold_rows,
                        bars_by,
                        hold_scores["full"],
                        barrier=barrier,
                        up_pct=up_pct,
                        down_pct=down_pct,
                        top_k=5,
                        horizon_bars=h_bars,
                        cost=COST_PER_EVENT,
                        slippage=extra,
                        strategy="full",
                    )
                    slip[case] = {
                        "cumulative_net_return": res["cumulative_net_return"],
                        "n_trades": res["n_trades"],
                        "win_rate": res["win_rate"],
                    }
                slippage_results[barrier] = slip

            classification = classify_portfolio(holdout_results)
            if not leak.get("passed"):
                classification = "not_economically_supported"

            result = {
                "promote": False,
                "experiment_status": classification,
                "passed_experiment_gates": classification == "economically_plausible",
                "feature_names": feature_names(),
                "blob": b"",
                "metrics": {
                    "split": split_meta,
                    "holdout": holdout_results,
                    "development": dev_results,
                    "slippage_sensitivity": slippage_results,
                    "lineage": EXPERIMENT.lineage(),
                },
                "notes": f"Portfolio classification: {classification}.",
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
                "Portfolio backtest uses the point-in-time universe. Survivorship is reduced but "
                "not eliminated: symbols delisted before the data window are absent."
            )
            survivorship_short = "PIT universe; delisted-before-window absent."

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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_portfolio_backtest",
                    ],
                },
                "experiment": EXPERIMENT.lineage(),
                "test_results": test_results,
                "dataset": {
                    "symbols_count": len(old_symbols),
                    "raw_candles": candles.get("count"),
                    "feature_count": len(feature_names()),
                    "lookback_bars": LOOKBACK_BARS,
                },
                "split": split_meta,
                "barrier_attachment": barrier_stats,
                "holdout": holdout_results,
                "development": dev_results,
                "development_vs_holdout": {
                    b: {
                        "development_net": (dev_results.get(b, {}).get("full", {}).get("top_5") or {}).get("cumulative_net_return"),
                        "holdout_net": (holdout_results.get(b, {}).get("full", {}).get("top_5") or {}).get("cumulative_net_return"),
                        "development_max_dd": (dev_results.get(b, {}).get("full", {}).get("top_5") or {}).get("max_drawdown"),
                        "holdout_max_dd": (holdout_results.get(b, {}).get("full", {}).get("top_5") or {}).get("max_drawdown"),
                    }
                    for b, _, _ in BARRIER_SPECS
                },
                "slippage_sensitivity": slippage_results,
                "cost_per_event": COST_PER_EVENT,
                "slippage_cases": SLIPPAGE_CASES,
                "experiment_status": classification,
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
            report["final_interpretation"] = final_interpretation(report)

            REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
            REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
            REPORT_MD.write_text(write_markdown(report))
            print_terminal_summary(report)
            log.info("wrote %s and %s", REPORT_JSON, REPORT_MD)
    except Exception as exc:
        errors.append(str(exc))
        log.exception("portfolio backtest failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
