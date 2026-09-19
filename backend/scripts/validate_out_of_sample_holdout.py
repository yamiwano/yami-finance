"""Untouched out-of-sample holdout validation (frozen barrier_probability_v2 spec).

Does not overwrite prior reports or activate live models.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_out_of_sample_holdout
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-holdout-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import OUT_OF_SAMPLE_HOLDOUT_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.holdout import fit_holdout_experiment
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.opportunity import BARRIER_SPECS, barrier_flags_from_bars
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe
from app.research.universe import build_universe_snapshots, load_selected_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_out_of_sample_holdout")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "OUT_OF_SAMPLE_HOLDOUT_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "OUT_OF_SAMPLE_HOLDOUT_REPORT.md"
EXPERIMENT = OUT_OF_SAMPLE_HOLDOUT_V1


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

    # Holdout must be strictly after development (with embargo handled in fit).
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
    barriers = report["barriers"]
    lines = [
        f"# Out-of-sample holdout report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`OUT_OF_SAMPLE_HOLDOUT_REPORT.json`](OUT_OF_SAMPLE_HOLDOUT_REPORT.json)",
        "",
        "Prior experiments preserved:",
        "- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)",
        "- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)",
        "- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)",
        "- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)",
        "- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)",
        "- [`BARRIER_PROBABILITY_REPORT.md`](BARRIER_PROBABILITY_REPORT.md)",
        "- [`POINT_IN_TIME_UNIVERSE_REPORT.md`](POINT_IN_TIME_UNIVERSE_REPORT.md)",
        "",
        "## Objective",
        "",
        "Determine whether the existing barrier-ranking signal generalizes to a genuinely "
        "untouched future holdout. No model/target/feature changes were made.",
        "",
        "## Frozen specification",
        "",
        "- Same 36 features, same barriers, same HistGradientBoostingClassifier, same "
        "point-in-time universe, same stride/horizon as `barrier_probability_v2`.",
        "",
        "## Development / holdout cutoff",
        "",
        f"- Cutoff: **{split.get('cutoff')}**",
        f"- Development: {split.get('development_start')} → {split.get('development_end')} "
        f"({split.get('development_rows')} rows, {split.get('development_timestamps')} timestamps)",
        f"- Holdout: {split.get('holdout_start')} → {split.get('holdout_end')} "
        f"({split.get('holdout_rows')} rows, {split.get('holdout_timestamps')} timestamps)",
        f"- Embargo dropped development rows: {report.get('embargo_dropped_development_rows')}",
        "",
        "## Reproducibility",
        "",
        f"- Code commit: `{report['run_metadata'].get('code_commit')}`",
        f"- Random seed: {report.get('random_seed')}",
        f"- Bootstrap seed: {report.get('bootstrap_seed')} (n={report.get('bootstrap_n')})",
        f"- Cost per event: {report.get('cost_per_event')}",
        f"- Feature count: {report['dataset']['feature_count']}",
        "",
        "## Survivorship limitation",
        "",
        report["survivorship"],
        "",
        "## Holdout results per barrier",
        "",
    ]
    for name, summary in barriers.items():
        full = summary.get("full") or {}
        atr = summary.get("atr_only") or {}
        mom = summary.get("momentum") or {}
        no_atr = summary.get("full_without_atr") or {}
        dev = summary.get("development_full") or {}
        rand = summary.get("random_benchmark") or {}
        boot = summary.get("bootstrap") or {}
        econ = summary.get("economic_full") or {}
        lines.extend(
            [
                f"### {name}",
                "",
                "| Metric | Holdout Full | Holdout ATR | Holdout Mom | Full w/o ATR | Dev Full |",
                "|--------|-------------:|------------:|------------:|-------------:|---------:|",
                f"| AUC | {fmt(full.get('auc'))} | {fmt(atr.get('auc'))} | {fmt(mom.get('auc'))} | {fmt(no_atr.get('auc'))} | {fmt(dev.get('auc'))} |",
                f"| Brier | {fmt(full.get('brier'))} | {fmt(atr.get('brier'))} | {fmt(mom.get('brier'))} | {fmt(no_atr.get('brier'))} | {fmt(dev.get('brier'))} |",
                f"| Log loss | {fmt(full.get('log_loss'))} | — | — | — | {fmt(dev.get('log_loss'))} |",
                f"| Spearman | {fmt(full.get('spearman_prob_outcome'))} | {fmt(atr.get('spearman_prob_outcome'))} | {fmt(mom.get('spearman_prob_outcome'))} | {fmt(no_atr.get('spearman_prob_outcome'))} | {fmt(dev.get('spearman_prob_outcome'))} |",
                f"| Top-5 success | {fmt(full.get('top_5_success_rate'))} | {fmt(atr.get('top_5_success_rate'))} | {fmt(mom.get('top_5_success_rate'))} | {fmt(no_atr.get('top_5_success_rate'))} | {fmt(dev.get('top_5_success_rate'))} |",
                f"| Top-10% success | {fmt(full.get('top_10pct_success_rate'))} | {fmt(atr.get('top_10pct_success_rate'))} | {fmt(mom.get('top_10pct_success_rate'))} | {fmt(no_atr.get('top_10pct_success_rate'))} | {fmt(dev.get('top_10pct_success_rate'))} |",
                f"| All-assets success | {fmt(full.get('all_success_rate'))} | — | — | — | — |",
                f"| Random top-5 success | {fmt(rand.get('random_topk_success_rate'))} | — | — | — | — |",
                "",
                f"**Bootstrap (full):** top5={json.dumps(boot.get('top_5_success_rate'))}, "
                f"top10%={json.dumps(boot.get('top_10pct_success_rate'))}, "
                f"spearman={json.dumps(boot.get('spearman'))}",
                "",
                f"**Economic (full top-5):** gross={fmt(econ.get('gross_avg_return'))}, "
                f"net={fmt(econ.get('net_avg_return'))}, win_rate={fmt(econ.get('win_rate'))}, "
                f"avg_winner={fmt(econ.get('avg_winner'))}, avg_loser={fmt(econ.get('avg_loser'))}, "
                f"max_dd={fmt(econ.get('max_drawdown_cumulative'))}, n={econ.get('n_selections')}",
                "",
                f"**EV diagnostic:** {json.dumps(summary.get('expected_value'), default=str)}",
                "",
                "**Volatility buckets:**",
                "",
            ]
        )
        for bname in ("low", "medium", "high"):
            b = (summary.get("volatility_buckets") or {}).get(bname) or {}
            lines.append(
                f"- **{bname}** n={b.get('n')}: full_top5={fmt(b.get('full_top5_success'))}, "
                f"atr_top5={fmt(b.get('atr_top5_success'))}, mom_top5={fmt(b.get('mom_top5_success'))}, "
                f"all={fmt(b.get('all_success'))}"
            )
        lines.append("")

    lines.extend(
        [
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
    if status == "generalizes":
        return (
            "The barrier signal remains meaningfully predictive on the untouched holdout across "
            "multiple metrics. Research-only; not a trading recommendation."
        )
    if status == "weak_generalization":
        return (
            "The barrier signal is detectable on the holdout but materially weaker or inconsistent "
            "relative to development."
        )
    if status == "fails_to_generalize":
        return "The development barrier signal largely disappears on the untouched holdout."
    return "The holdout period is too short/sparse to assess generalization."


def print_terminal_summary(report: dict) -> None:
    split = report["split"]
    barriers = report["barriers"]
    lines = [
        "",
        "======== OUT_OF_SAMPLE_HOLDOUT_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
        f"Development: {split.get('development_start')} → {split.get('development_end')} "
        f"({split.get('development_rows')} rows)",
        f"Holdout: {split.get('holdout_start')} → {split.get('holdout_end')} "
        f"({split.get('holdout_rows')} rows)",
        "",
    ]
    for name, summary in barriers.items():
        full = summary.get("full") or {}
        atr = summary.get("atr_only") or {}
        rand = summary.get("random_benchmark") or {}
        econ = summary.get("economic_full") or {}
        lines.extend(
            [
                f"--- {name} ---",
                f"Full AUC: {fmt(full.get('auc'))} | ATR AUC: {fmt(atr.get('auc'))}",
                f"Full Brier: {fmt(full.get('brier'))}",
                f"Full top-5: {fmt(full.get('top_5_success_rate'))} | ATR top-5: {fmt(atr.get('top_5_success_rate'))} | Random top-5: {fmt(rand.get('random_topk_success_rate'))}",
                f"Full top-10%: {fmt(full.get('top_10pct_success_rate'))} | ATR top-10%: {fmt(atr.get('top_10pct_success_rate'))}",
                f"Economic net (full top-5): {fmt(econ.get('net_avg_return'))} (n={econ.get('n_selections')})",
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
    log.info("holdout validate start db=%s", settings.database_url)
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

            # Split for leakage audit
            from app.research.holdout import choose_holdout_cutoff, split_development_holdout

            cutoff, _ = choose_holdout_cutoff([as_utc(r.ts) for r in pit_rows])
            dev_rows, hold_rows, _split = split_development_holdout(pit_rows, cutoff)

            bars_by = {}
            for sym, aid in list(ids.items())[:8]:
                bars_by[sym] = await load_bars(db, aid, timeframe().value)
            leak = await audit_leakage(dev_rows, hold_rows, bars_by, cutoff)

            result = fit_holdout_experiment(
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

            metrics = result["metrics"]
            survivorship = (
                "Holdout uses the point-in-time universe (trailing 24h volume). Survivorship is "
                "reduced but not eliminated: symbols delisted before the data window are absent, "
                "and historical listing status is inferred from candles."
            )
            survivorship_short = (
                "PIT universe; delisted-before-window absent; listing inferred from candles."
            )

            code_commit = os.environ.get("GIT_COMMIT") or subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
            ).stdout.strip()

            report = {
                "run_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "market_data_provider": settings.market_data_provider,
                    "database_url": settings.database_url,
                    "research_backfill_days": settings.research_backfill_days,
                    "universe_size": settings.universe_size,
                    "errors": errors,
                    "code_commit": code_commit,
                    "commands": [
                        "PYTHONPATH=backend python -m unittest tests.test_research -v",
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_out_of_sample_holdout",
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
                "barrier_stats": metrics.get("barrier_stats"),
                "barrier_attachment": barrier_stats,
                "cross_section": metrics.get("cross_section"),
                "cutoff": metrics.get("cutoff"),
                "split": metrics.get("split"),
                "embargo_dropped_development_rows": metrics.get("embargo_dropped_development_rows"),
                "barriers": metrics.get("barriers"),
                "random_seed": metrics.get("random_seed"),
                "bootstrap_seed": metrics.get("bootstrap_seed"),
                "bootstrap_n": metrics.get("bootstrap_n"),
                "cost_per_event": metrics.get("cost_per_event"),
                "experiment_status": result["experiment_status"],
                "notes": result["notes"],
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
        log.exception("holdout validation failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
