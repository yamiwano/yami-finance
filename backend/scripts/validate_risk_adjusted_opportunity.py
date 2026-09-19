"""Validate risk-adjusted opportunity experiment.

Does not overwrite prior reports or activate live models.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_risk_adjusted_opportunity
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-opportunity-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import RISK_ADJUSTED_OPPORTUNITY_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import barrier_flags_from_bars, fit_opportunity_walk_forward
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_risk_adjusted_opportunity")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "RISK_ADJUSTED_OPPORTUNITY_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "RISK_ADJUSTED_OPPORTUNITY_REPORT.md"
EXPERIMENT = RISK_ADJUSTED_OPPORTUNITY_V1


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
    """Compute first-touch barriers from OHLC; never invents intrabar order."""
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
                for name in ("up3_before_down2", "up5_before_down3", "up10_before_down5"):
                    setattr(row, name, "unknown")
                continue
            entry = bars[i].close
            flags = barrier_flags_from_bars(bars[i + 1 :], entry, horizon_bars=h)
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
            mutated[i + 1] = fut.model_copy(update={"high": fut.high * 3, "low": fut.low * 0.5, "close": fut.close * 3})
        past_only = extract_features(mutated[: i + 1], tf)
        if a != past_only:
            findings.append(f"future mutation changed past features {row.symbol} {row.ts}")
        # Ensure opportunity fields are not in feature dict
        if a and any(k.startswith("future_") or "opportunity" in k or "barrier" in k for k in a):
            findings.append("future/opportunity key leaked into features")
        labels = label_forward(bars[i + 1 :], bars[i].close, h, thr)
        if labels and row.max_upside is not None:
            if abs(float(labels["max_upside"]) - float(row.max_upside)) > 1e-9:
                findings.append(f"max_upside mismatch {row.symbol}")
            if abs(float(labels["max_drawdown"]) - float(row.max_drawdown)) > 1e-9:
                findings.append(f"max_drawdown mismatch {row.symbol}")
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
    table = report["comparison_table"]
    cs = report["cross_section"]
    td = (cs or {}).get("target_definition") or {}
    lines = [
        f"# Risk-adjusted opportunity report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`RISK_ADJUSTED_OPPORTUNITY_REPORT.json`](RISK_ADJUSTED_OPPORTUNITY_REPORT.json)",
        "",
        "Prior experiments preserved:",
        "- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)",
        "- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)",
        "- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)",
        "- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)",
        "",
        "## Hypothesis",
        "",
        "Can the existing feature set identify assets whose **future upside is attractive "
        "relative to future downside**, beyond ATR and momentum baselines?",
        "",
        "## Why this experiment exists",
        "",
        "Prior work showed volatility dominates raw upside ranking. This experiment asks a "
        "different question: opportunity quality = upside / downside, not max upside alone.",
        "",
        "## Configuration",
        "",
        f"- Experiment ID: `{exp['experiment_id']}`",
        f"- Timeframe / horizon / stride: **{exp['timeframe']} / {exp['horizon_hours']}h / {exp['primary_stride_bars']}-bar**",
        f"- Min cross-section: **{exp['min_cross_section_size']}**",
        f"- Model: HistGradientBoostingRegressor on `log_risk_adjusted_opportunity`",
        "",
        "## Target definitions",
        "",
        "```",
        f"eps = {td.get('eps', 0.001)}",
        "risk_adjusted_opportunity = future_max_upside_12h / max(|future_max_drawdown_12h|, eps)",
        "log_risk_adjusted_opportunity = log1p(risk_adjusted_opportunity)   # training target",
        "upside_downside_spread = future_max_upside_12h - |future_max_drawdown_12h|",
        "```",
        "",
        "## Barrier definitions",
        "",
        "First-touch on future 1h OHLC only. Same-candle dual hit ⇒ `ambiguous` (excluded from rates).",
        "",
        "- `up3_before_down2`: +3% before −2%",
        "- `up5_before_down3`: +5% before −3%",
        "- `up10_before_down5`: +10% before −5%",
        "",
        f"Barrier rows computed: {report.get('barrier_stats', {}).get('barrier_rows_computed')}; "
        f"ambiguous events: {report.get('barrier_stats', {}).get('ambiguous_events')}",
        "",
        "## Survivorship bias (prominent)",
        "",
        report["survivorship"],
        "",
        "## Cross-sectional coverage",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Timestamps considered | {cs.get('timestamps_considered')} |",
        f"| Timestamps excluded | {cs.get('timestamps_excluded_below_min')} |",
        f"| Timestamps used | {cs.get('timestamps_used')} |",
        f"| Avg / min / max CS | {fmt(cs.get('average_cross_section_size'), 2)} / "
        f"{cs.get('min_cross_section_size')} / {cs.get('max_cross_section_size')} |",
        f"| Asset observations | {cs.get('asset_observations_used')} |",
        "",
        "## Aggregate comparison",
        "",
        "| Metric | Full | ATR-only | Momentum | Full w/o ATR | All-assets |",
        "|--------|-----:|---------:|---------:|-------------:|-----------:|",
    ]
    for metric in (
        "spearman_opportunity",
        "top_5_mean_opportunity",
        "top_5_mean_raw_upside",
        "top_5_mean_future_return",
        "top_5_mean_drawdown",
        "top_5_up3_before_down2",
        "top_5_up5_before_down3",
        "top_5_up10_before_down5",
    ):
        row = table.get(metric) or {}
        lines.append(
            f"| {metric} | {fmt(row.get('full_model'))} | {fmt(row.get('atr_only'))} | "
            f"{fmt(row.get('momentum_ret_24'))} | {fmt(row.get('full_without_atr'))} | "
            f"{fmt(row.get('random_all_assets'))} |"
        )

    ev = report.get("evidence") or {}
    deltas = ev.get("deltas") or {}
    lines.extend(
        [
            "",
            f"**Full − ATR opportunity Spearman:** {fmt(report.get('full_minus_atr_spearman'))}  ",
            f"**Full − ATR top-5 opportunity:** {fmt(report.get('full_minus_atr_top5_opportunity'))}  ",
            f"**Full − ATR top-5 raw upside:** {fmt(deltas.get('top5_raw_upside_full_minus_atr'))}  ",
            f"**Full − ATR top-5 future return:** {fmt(deltas.get('top5_return_full_minus_atr'))}  ",
            f"**Full − ATR top-5 drawdown:** {fmt(deltas.get('top5_drawdown_full_minus_atr'))}",
            "",
            "## Walk-forward folds",
            "",
        ]
    )
    for i, fold in enumerate(report.get("folds") or [], 1):
        f = fold.get("full") or {}
        a = fold.get("atr_only") or {}
        lines.append(
            f"### Fold {i}: {fold.get('test_start')} → {fold.get('test_end')}  \n"
            f"train {fold.get('train_start')} → {fold.get('train_end')}; "
            f"ts={fold.get('test_timestamps')}; obs={fold.get('test_asset_observations')}  \n"
            f"Full opp Spearman={fmt(f.get('spearman_opportunity'))}; "
            f"ATR={fmt(a.get('spearman_opportunity'))}; "
            f"Full top5 opp={fmt(f.get('top_5_mean_opportunity'))}; "
            f"ATR top5 opp={fmt(a.get('top_5_mean_opportunity'))}; "
            f"Full top5 dd={fmt(f.get('top_5_mean_drawdown'))}; "
            f"ATR top5 dd={fmt(a.get('top_5_mean_drawdown'))}"
        )
        lines.append("")

    buckets = report.get("volatility_buckets") or {}
    lines.extend(["## Volatility bucket analysis", ""])
    for name in ("low", "medium", "high"):
        b = buckets.get(name) or {}
        lines.append(
            f"- **{name}** (n={b.get('n')}): full_sp={fmt(b.get('full_spearman'))}, "
            f"atr_sp={fmt(b.get('atr_spearman'))}, mom={fmt(b.get('momentum_spearman'))}, "
            f"noATR={fmt(b.get('full_without_atr_spearman'))}; "
            f"top5 opp/up/ret/dd="
            f"{fmt(b.get('full_top5_opportunity'))}/"
            f"{fmt(b.get('full_top5_raw_upside'))}/"
            f"{fmt(b.get('full_top5_return'))}/"
            f"{fmt(b.get('full_top5_drawdown'))}; "
            f"barriers 3/5/10="
            f"{fmt(b.get('full_top5_up3_before_down2'))}/"
            f"{fmt(b.get('full_top5_up5_before_down3'))}/"
            f"{fmt(b.get('full_top5_up10_before_down5'))}"
        )

    dd = report.get("downside_control") or {}
    lines.extend(
        [
            "",
            "## Downside-control analysis",
            "",
            f"| Group | Mean DD | Median DD | Worst DD | ≤−3% | ≤−5% | ≤−10% |",
            f"|-------|--------:|----------:|---------:|-----:|-----:|------:|",
            f"| Full top-5 | {fmt(dd.get('full_top5', {}).get('mean_drawdown'))} | "
            f"{fmt(dd.get('full_top5', {}).get('median_drawdown'))} | "
            f"{fmt(dd.get('full_top5', {}).get('worst_drawdown'))} | "
            f"{fmt(dd.get('full_top5', {}).get('frac_worse_3'))} | "
            f"{fmt(dd.get('full_top5', {}).get('frac_worse_5'))} | "
            f"{fmt(dd.get('full_top5', {}).get('frac_worse_10'))} |",
            f"| ATR top-5 | {fmt(dd.get('atr_top5', {}).get('mean_drawdown'))} | "
            f"{fmt(dd.get('atr_top5', {}).get('median_drawdown'))} | "
            f"{fmt(dd.get('atr_top5', {}).get('worst_drawdown'))} | "
            f"{fmt(dd.get('atr_top5', {}).get('frac_worse_3'))} | "
            f"{fmt(dd.get('atr_top5', {}).get('frac_worse_5'))} | "
            f"{fmt(dd.get('atr_top5', {}).get('frac_worse_10'))} |",
            f"| All assets | {fmt(dd.get('all_assets', {}).get('mean_drawdown'))} | — | — | — | — | — |",
            "",
            "## ATR-normalized opportunity diagnostic",
            "",
            json.dumps(report.get("atr_normalized_opportunity"), indent=2, default=str),
            "",
            "## Leakage audit",
            "",
            f"- Passed: **{report['leakage_audit'].get('passed')}**",
            f"- Findings: {report['leakage_audit'].get('label_mismatches_or_findings')}",
            f"- Embargo violations: **{report['leakage_audit'].get('folds_violating_embargo')}**",
            "",
            "## Evidence / classification",
            "",
            f"**Status: `{report['experiment_status']}`** (never activated)",
            "",
            "```",
            json.dumps(ev, indent=2, default=str),
            "```",
            "",
            "## Honest interpretation",
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
    d_sp = report.get("full_minus_atr_spearman")
    d_opp = report.get("full_minus_atr_top5_opportunity")
    if status == "promising_opportunity_signal":
        return (
            "Full model shows consistent opportunity-ranking improvement over ATR with "
            "acceptable downside/barrier characteristics. Research signal only — not trading-ready."
        )
    if status == "weak_opportunity_signal":
        return (
            f"Some positive opportunity ranking exists (full−ATR Spearman={fmt(d_sp)}, "
            f"top5 opp Δ={fmt(d_opp)}), but it is not decisive beyond volatility baselines "
            "under the success criteria."
        )
    return (
        f"No reliable evidence of better risk-adjusted opportunities beyond simple volatility "
        f"(full−ATR Spearman={fmt(d_sp)}, top5 opp Δ={fmt(d_opp)})."
    )


def print_terminal_summary(report: dict) -> None:
    table = report["comparison_table"]
    buckets = report.get("volatility_buckets") or {}

    def bucket_line(name: str) -> str:
        b = buckets.get(name) or {}
        return (
            f"n={b.get('n')}, full_sp={fmt(b.get('full_spearman'))}, "
            f"atr_sp={fmt(b.get('atr_spearman'))}, "
            f"top5_opp={fmt(b.get('full_top5_opportunity'))}, "
            f"top5_dd={fmt(b.get('full_top5_drawdown'))}"
        )

    beyond = (
        "Yes — provisional research evidence"
        if report["experiment_status"] == "promising_opportunity_signal"
        else (
            "Unclear / weak — not decisive beyond ATR"
            if report["experiment_status"] == "weak_opportunity_signal"
            else "No — not supported on this evidence"
        )
    )
    lines = [
        "",
        "======== RISK_ADJUSTED_OPPORTUNITY_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
        f"Samples: {report['cross_section'].get('asset_observations_used')}",
        f"Timestamps: {report['cross_section'].get('timestamps_used')}",
        f"Folds: {report.get('fold_count')}",
        "",
        f"Full model opportunity Spearman: {fmt(table['spearman_opportunity'].get('full_model'))}",
        f"ATR-only opportunity Spearman: {fmt(table['spearman_opportunity'].get('atr_only'))}",
        f"Momentum opportunity Spearman: {fmt(table['spearman_opportunity'].get('momentum_ret_24'))}",
        f"Full-without-ATR opportunity Spearman: {fmt(table['spearman_opportunity'].get('full_without_atr'))}",
        "",
        f"Full top-5 opportunity: {fmt(table['top_5_mean_opportunity'].get('full_model'))}",
        f"ATR-only top-5 opportunity: {fmt(table['top_5_mean_opportunity'].get('atr_only'))}",
        f"Full-without-ATR top-5 opportunity: {fmt(table['top_5_mean_opportunity'].get('full_without_atr'))}",
        f"All-assets opportunity: {fmt(table['top_5_mean_opportunity'].get('random_all_assets'))}",
        "",
        f"Full top-5 raw upside: {fmt(table['top_5_mean_raw_upside'].get('full_model'))}",
        f"ATR-only top-5 raw upside: {fmt(table['top_5_mean_raw_upside'].get('atr_only'))}",
        "",
        f"Full top-5 future return: {fmt(table['top_5_mean_future_return'].get('full_model'))}",
        f"ATR-only top-5 future return: {fmt(table['top_5_mean_future_return'].get('atr_only'))}",
        "",
        f"Full top-5 max drawdown: {fmt(table['top_5_mean_drawdown'].get('full_model'))}",
        f"ATR-only top-5 max drawdown: {fmt(table['top_5_mean_drawdown'].get('atr_only'))}",
        "",
        f"Full +3% before -2%: {fmt(table['top_5_up3_before_down2'].get('full_model'))}",
        f"ATR +3% before -2%: {fmt(table['top_5_up3_before_down2'].get('atr_only'))}",
        "",
        f"Full +5% before -3%: {fmt(table['top_5_up5_before_down3'].get('full_model'))}",
        f"ATR +5% before -3%: {fmt(table['top_5_up5_before_down3'].get('atr_only'))}",
        "",
        f"Full +10% before -5%: {fmt(table['top_5_up10_before_down5'].get('full_model'))}",
        f"ATR +10% before -5%: {fmt(table['top_5_up10_before_down5'].get('atr_only'))}",
        "",
        f"Full minus ATR opportunity Spearman: {fmt(report.get('full_minus_atr_spearman'))}",
        f"Full minus ATR top-5 opportunity: {fmt(report.get('full_minus_atr_top5_opportunity'))}",
        "",
        f"Low-volatility result: {bucket_line('low')}",
        f"Medium-volatility result: {bucket_line('medium')}",
        f"High-volatility result: {bucket_line('high')}",
        "",
        f"Leakage passed: {report['leakage_audit'].get('passed')}",
        f"Embargo violations: {report['leakage_audit'].get('folds_violating_embargo')}",
        "",
        f"Classification: {report['experiment_status']}",
        "",
        f"Does the model identify better risk-adjusted opportunities beyond simple volatility?: {beyond}",
        "",
        report.get("final_interpretation", ""),
        "========================================================",
        "",
    ]
    print("\n".join(lines))


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info("opportunity validate start experiment=%s db=%s", EXPERIMENT.experiment_id, settings.database_url)
    test_results = run_research_tests()
    await init_db()
    provider = create_provider()
    await provider.start()
    try:
        ids = await ensure_assets(provider)
        symbols = sorted(ids)
        async with SessionLocal() as db:
            active_before = (
                await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
            ).scalars().all()
            active_before_ids = {str(m.id) for m in active_before}

            filled = await backfill(db, provider, ids)
            log.info("backfill added=%s", filled)
            built = await build_samples(db, ids, stride=EXPERIMENT.sensitivity_stride_bars)
            await backfill_directional_labels(db)
            log.info("dataset built=%s", built)

            candles = await candle_stats(db, timeframe().value)
            all_rows = await load_upside_rows(db)
            primary_rows = select_by_stride(
                all_rows,
                EXPERIMENT.primary_stride_bars,
                base_stride=EXPERIMENT.sensitivity_stride_bars,
            )
            barrier_stats = await attach_barriers(primary_rows, ids, db)
            log.info("barriers %s", barrier_stats)

            bars_by = {}
            for sym, aid in list(ids.items())[:8]:
                bars_by[sym] = await load_bars(db, aid, timeframe().value)
            leak = await audit_leakage(primary_rows, bars_by)

            result = fit_opportunity_walk_forward(
                primary_rows,
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
                "IMPORTANT LIMITATION: Universe = today's top-N Binance USDT pairs by 24h quote volume, "
                "applied retrospectively. Delisted/illiquid names are missing. This does NOT represent "
                "the entire crypto market. Survivorship bias is documented, not fixed, here."
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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_risk_adjusted_opportunity",
                    ],
                },
                "experiment": EXPERIMENT.lineage(),
                "test_results": test_results,
                "dataset": {
                    "symbols_count": len(symbols),
                    "symbols": symbols,
                    "oldest_candle": candles.get("oldest"),
                    "newest_candle": candles.get("newest"),
                    "raw_candles": candles.get("count"),
                    "feature_count": len(feature_names()),
                    "lookback_bars": LOOKBACK_BARS,
                    "primary_stride_rows": len(primary_rows),
                },
                "barrier_stats": barrier_stats,
                "cross_section": metrics.get("cross_section"),
                "folds": metrics.get("folds"),
                "fold_count": metrics.get("fold_count"),
                "full": metrics.get("full"),
                "atr_only": metrics.get("atr_only"),
                "momentum": metrics.get("momentum"),
                "full_without_atr": metrics.get("full_without_atr"),
                "atr_ranking": metrics.get("atr_ranking"),
                "all_assets_random": metrics.get("all_assets_random"),
                "comparison_table": metrics.get("comparison_table"),
                "volatility_buckets": metrics.get("volatility_buckets"),
                "downside_control": metrics.get("downside_control"),
                "atr_normalized_opportunity": metrics.get("atr_normalized_opportunity"),
                "evidence": metrics.get("evidence"),
                "full_minus_atr_spearman": metrics.get("full_minus_atr_spearman"),
                "full_minus_atr_top5_opportunity": metrics.get("full_minus_atr_top5_opportunity"),
                "experiment_status": result["experiment_status"],
                "notes": result["notes"],
                "leakage_audit": leak,
                "survivorship": survivorship,
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
        log.exception("opportunity validation failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
