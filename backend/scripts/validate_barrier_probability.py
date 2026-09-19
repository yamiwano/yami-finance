"""Validate barrier probability experiment (first-touch up vs down).

Does not overwrite prior reports or activate live models.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_barrier_probability
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-barrier-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.barriers import fit_barrier_walk_forward
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import BARRIER_PROBABILITY_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import BARRIER_SPECS, barrier_flags_from_bars
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_barrier_probability")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "BARRIER_PROBABILITY_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "BARRIER_PROBABILITY_REPORT.md"
EXPERIMENT = BARRIER_PROBABILITY_V1


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
    feature_key_blocklist = ("future_", "up3_", "up5_", "up10_", "barrier", "opportunity")
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
        if a and any(any(b in k for b in feature_key_blocklist) for k in a):
            findings.append(f"target-derived key leaked into features {row.symbol}")
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
    cs = report["cross_section"]
    barriers = report["barriers"]
    lines = [
        f"# Barrier probability report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`BARRIER_PROBABILITY_REPORT.json`](BARRIER_PROBABILITY_REPORT.json)",
        "",
        "Prior experiments preserved:",
        "- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)",
        "- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)",
        "- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)",
        "- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)",
        "- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)",
        "",
        "## Hypothesis",
        "",
        "Can the existing features predict whether an asset reaches an **upside barrier before** "
        "a downside barrier within 12h?",
        "",
        "## Configuration",
        "",
        f"- Experiment ID: `{exp['experiment_id']}`",
        f"- Timeframe / horizon / stride: **{exp['timeframe']} / {exp['horizon_hours']}h / {exp['primary_stride_bars']}-bar**",
        f"- Min cross-section: **{exp['min_cross_section_size']}**",
        "- Model: HistGradientBoostingClassifier (success vs failure only)",
        "",
        "## Target definitions",
        "",
        "First-touch on future 1h OHLC. Same-candle dual touch ⇒ **ambiguous** (excluded).",
        "",
        "- `up3_before_down2`: +3% before −2%",
        "- `up5_before_down3`: +5% before −3%",
        "- `up10_before_down5`: +10% before −5%",
        "",
        "**Timeout/ambiguity:** TIMEOUT (neither barrier) and AMBIGUOUS are excluded from binary "
        "training/evaluation and reported separately.",
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
        "## Class balance / event mix",
        "",
    ]
    for name, stats in (report.get("barrier_stats") or {}).items():
        lines.append(
            f"### {name} (+{stats['up_pct']:.0%} / −{stats['down_pct']:.0%})  \n"
            f"total={stats['total']}; success={stats['success']}; failure={stats['failure']}; "
            f"timeout={stats['timeout']} ({fmt(stats['timeout_rate'], 3)}); "
            f"ambiguous={stats['ambiguous']} ({fmt(stats['ambiguous_rate'], 3)}); "
            f"resolved={stats['resolved_non_ambiguous']}; "
            f"success_rate_resolved={fmt(stats['success_rate_resolved'])}"
        )
        lines.append("")

    lines.append("## Per-barrier results")
    for name, summary in barriers.items():
        full = summary.get("full") or {}
        atr = summary.get("atr_only") or {}
        mom = summary.get("momentum") or {}
        no_atr = summary.get("full_without_atr") or {}
        lines.extend(
            [
                "",
                f"### {name}",
                "",
                "| Metric | Full | ATR-only | Momentum | Full w/o ATR |",
                "|--------|-----:|---------:|---------:|-------------:|",
                f"| AUC | {fmt(full.get('auc'))} | {fmt(atr.get('auc'))} | {fmt(mom.get('auc'))} | {fmt(no_atr.get('auc'))} |",
                f"| PR-AUC | {fmt(full.get('pr_auc'))} | {fmt(atr.get('pr_auc'))} | {fmt(mom.get('pr_auc'))} | {fmt(no_atr.get('pr_auc'))} |",
                f"| Brier | {fmt(full.get('brier'))} | {fmt(atr.get('brier'))} | {fmt(mom.get('brier'))} | {fmt(no_atr.get('brier'))} |",
                f"| Baseline Brier | {fmt(full.get('baseline_brier'))} | {fmt(atr.get('baseline_brier'))} | — | — |",
                f"| Log loss | {fmt(full.get('log_loss'))} | {fmt(atr.get('log_loss'))} | — | — |",
                f"| Spearman(p, outcome) | {fmt(full.get('spearman_prob_outcome'))} | {fmt(atr.get('spearman_prob_outcome'))} | {fmt(mom.get('spearman_prob_outcome'))} | {fmt(no_atr.get('spearman_prob_outcome'))} |",
                f"| Top-1 success | {fmt(full.get('top_1_success_rate'))} | {fmt(atr.get('top_1_success_rate'))} | {fmt(mom.get('top_1_success_rate'))} | {fmt(no_atr.get('top_1_success_rate'))} |",
                f"| Top-5 success | {fmt(full.get('top_5_success_rate'))} | {fmt(atr.get('top_5_success_rate'))} | {fmt(mom.get('top_5_success_rate'))} | {fmt(no_atr.get('top_5_success_rate'))} |",
                f"| Top-10% success | {fmt(full.get('top_10pct_success_rate'))} | {fmt(atr.get('top_10pct_success_rate'))} | {fmt(mom.get('top_10pct_success_rate'))} | {fmt(no_atr.get('top_10pct_success_rate'))} |",
                f"| Top-20% success | {fmt(full.get('top_20pct_success_rate'))} | {fmt(atr.get('top_20pct_success_rate'))} | {fmt(mom.get('top_20pct_success_rate'))} | {fmt(no_atr.get('top_20pct_success_rate'))} |",
                f"| All-assets success | {fmt(full.get('all_success_rate'))} | — | — | — |",
                "",
                "**Expected value (diagnostic only; not trading return):**",
                "",
                "```",
                json.dumps(summary.get("expected_value"), indent=2, default=str),
                "```",
                "",
                "**Calibration (pooled, full model):**",
                "",
                "```",
                json.dumps(
                    (summary.get("pooled_calibration") or {}).get("full", {}).get("calibration"),
                    indent=2,
                    default=str,
                ),
                "```",
                "",
                "**Volatility buckets:**",
                "",
            ]
        )
        for bname in ("low", "medium", "high"):
            b = (summary.get("volatility_buckets") or {}).get(bname) or {}
            lines.append(
                f"- **{bname}** n={b.get('n')}: base={fmt(b.get('base_success_rate'))}, "
                f"full_top5={fmt(b.get('full_top5_success_rate'))}, "
                f"atr_top5={fmt(b.get('atr_top5_success_rate'))}, "
                f"mom_top5={fmt(b.get('momentum_top5_success_rate'))}, "
                f"full_auc={fmt(b.get('full_auc'))}, full_brier={fmt(b.get('full_brier'))}"
            )
        dd = summary.get("downside") or {}
        lines.extend(
            [
                "",
                "**Downside (top-5):**",
                "",
                f"- Full: mean={fmt(dd.get('full_top5', {}).get('mean_drawdown'))}, "
                f"median={fmt(dd.get('full_top5', {}).get('median_drawdown'))}, "
                f"worst={fmt(dd.get('full_top5', {}).get('worst_drawdown'))}, "
                f"≤−3/5/10%={fmt(dd.get('full_top5', {}).get('frac_worse_3'))}/"
                f"{fmt(dd.get('full_top5', {}).get('frac_worse_5'))}/"
                f"{fmt(dd.get('full_top5', {}).get('frac_worse_10'))}",
                f"- ATR: mean={fmt(dd.get('atr_top5', {}).get('mean_drawdown'))}, "
                f"worst={fmt(dd.get('atr_top5', {}).get('worst_drawdown'))}, "
                f"≤−3/5/10%={fmt(dd.get('atr_top5', {}).get('frac_worse_3'))}/"
                f"{fmt(dd.get('atr_top5', {}).get('frac_worse_5'))}/"
                f"{fmt(dd.get('atr_top5', {}).get('frac_worse_10'))}",
                "",
                "**Folds:**",
                "",
            ]
        )
        for i, fold in enumerate(summary.get("folds") or [], 1):
            f = fold.get("full") or {}
            a = fold.get("atr_only") or {}
            lines.append(
                f"- Fold {i} ({fold.get('test_start')} → {fold.get('test_end')}): "
                f"full_auc={fmt(f.get('auc'))}, atr_auc={fmt(a.get('auc'))}, "
                f"full_brier={fmt(f.get('brier'))}, "
                f"full_top5={fmt(f.get('top_5_success_rate'))}, "
                f"atr_top5={fmt(a.get('top_5_success_rate'))}"
            )

    lines.extend(
        [
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
            json.dumps(report.get("evidence"), indent=2, default=str),
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
    if status == "promising_barrier_signal":
        return (
            "Full model shows consistent, calibrated barrier-first-touch ranking improvement over "
            "ATR/momentum across folds and regimes. Research signal only — not trading-ready."
        )
    if status == "weak_barrier_signal":
        return (
            "Some barrier ranking ability exists, but it is not consistently better than ATR/momentum "
            "across barriers, folds, and volatility regimes under the strict criteria."
        )
    return "No reliable evidence that features predict barrier-first-touch beyond simple baselines."


def print_terminal_summary(report: dict) -> None:
    barriers = report["barriers"]
    stats = report.get("barrier_stats") or {}
    lines = [
        "",
        "======== BARRIER_PROBABILITY_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
    ]
    for name, s in stats.items():
        lines.append(
            f"{name}: obs={s['total']}, success={s['success']}, failure={s['failure']}, "
            f"timeout={s['timeout']} ({fmt(s['timeout_rate'], 3)}), "
            f"ambiguous={s['ambiguous']} ({fmt(s['ambiguous_rate'], 3)}), "
            f"resolved={s['resolved_non_ambiguous']}, "
            f"success_rate_resolved={fmt(s['success_rate_resolved'])}"
        )
    lines.append(f"Folds: {report.get('fold_count')}")
    lines.append("")
    for name, summary in barriers.items():
        full = summary.get("full") or {}
        atr = summary.get("atr_only") or {}
        no_atr = summary.get("full_without_atr") or {}
        beats_atr = (
            full.get("top_5_success_rate") is not None
            and atr.get("top_5_success_rate") is not None
            and full["top_5_success_rate"] > atr["top_5_success_rate"]
        )
        lines.extend(
            [
                f"--- {name} ---",
                f"Full AUC: {fmt(full.get('auc'))}",
                f"ATR AUC: {fmt(atr.get('auc'))}",
                f"Full Brier: {fmt(full.get('brier'))}",
                f"Full top-5 success: {fmt(full.get('top_5_success_rate'))}",
                f"ATR top-5 success: {fmt(atr.get('top_5_success_rate'))}",
                f"Full top-10% success: {fmt(full.get('top_10pct_success_rate'))}",
                f"ATR top-10% success: {fmt(atr.get('top_10pct_success_rate'))}",
                f"Full EV (net): {fmt((summary.get('expected_value') or {}).get('full_top5_ev_net_of_cost'))}",
                f"ATR EV (net): {fmt((summary.get('expected_value') or {}).get('atr_top5_ev_net_of_cost'))}",
                f"Full beats ATR on top-5: {beats_atr}",
                f"Full-without-ATR Spearman: {fmt(no_atr.get('spearman_prob_outcome'))}",
                "",
            ]
        )
    lines.extend(
        [
            f"Classification: {report['experiment_status']}",
            f"Leakage passed: {report['leakage_audit'].get('passed')}",
            f"Embargo violations: {report['leakage_audit'].get('folds_violating_embargo')}",
            f"Tests passed: {report['test_results'].get('ok')}",
            "",
            report.get("final_interpretation", ""),
            "========================================================",
            "",
        ]
    )
    print("\n".join(lines))


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info("barrier validate start experiment=%s db=%s", EXPERIMENT.experiment_id, settings.database_url)
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

            result = fit_barrier_walk_forward(
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
                "applied retrospectively. Delisted/illiquid names are missing; this does NOT represent "
                "the entire crypto market. Survivorship bias remains and is not fixed by this experiment."
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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_barrier_probability",
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
        log.exception("barrier validation failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
