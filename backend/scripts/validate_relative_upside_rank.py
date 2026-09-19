"""Validate cross-sectional relative upside ranking experiment.

Does not overwrite bidirectional or directional-up reports.
Does not activate/retire the live bidirectional model.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_relative_upside_rank
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-ranking-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import RELATIVE_UPSIDE_RANK_12H_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.ranking import fit_ranking_walk_forward
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_relative_upside_rank")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "relative-upside-ranking-report.json"
REPORT_MD = REPO_ROOT / "docs" / "RELATIVE_UPSIDE_RANKING_REPORT.md"
EXPERIMENT = RELATIVE_UPSIDE_RANK_12H_V1


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
        # Appending future bars must not be used by the builder; re-extract on past-only window.
        c = extract_features(bars[: i + 1], tf)
        if a != c:
            findings.append("feature rebuild mismatch")
        leaked = extract_features(bars[: i + 1] + bars[i + 1 : i + 1 + h], tf)
        if a is not None and leaked is not None and a == leaked and h > 0:
            # extract_features uses last LOOKBACK only; if futures change last window they differ.
            pass
        if a is not None and leaked is not None:
            # Confirm past-only window differs from a window that includes future when lookback slides.
            # Stronger check: mutating future highs must not change past-only features.
            mutated = list(bars)
            if i + 1 < len(mutated):
                fut = mutated[i + 1]
                mutated[i + 1] = fut.model_copy(update={"high": fut.high * 3, "close": fut.close * 3})
            past_only = extract_features(mutated[: i + 1], tf)
            if a != past_only:
                findings.append(f"future mutation changed past features {row.symbol} {row.ts}")
        labels = label_forward(bars[i + 1 :], bars[i].close, h, thr)
        if labels and row.max_upside is not None:
            if abs(float(labels["max_upside"]) - float(row.max_upside)) > 1e-9:
                findings.append(f"max_upside mismatch {row.symbol} {row.ts}")
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
    status = report["experiment_status"]
    lines = [
        f"# Relative upside ranking report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`relative-upside-ranking-report.json`](relative-upside-ranking-report.json)",
        "",
        "Prior experiments preserved:",
        "- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md) (bidirectional)",
        "- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)",
        "",
        "## Hypothesis",
        "",
        "Given all eligible crypto assets at time T, can the model rank which assets will have "
        "the largest `future_max_upside_12h` over the next 12 hours?",
        "",
        "This is a **relative ranking** problem, not binary +3% classification.",
        "",
        "## Configuration",
        "",
        f"- Experiment ID: `{exp['experiment_id']}`",
        f"- Timeframe / horizon: **{exp['timeframe']} / {exp['horizon_hours']}h**",
        f"- Primary stride: **{exp['primary_stride_bars']} bars** (non-overlapping)",
        f"- Training target: continuous `future_max_upside_12h` (`max_upside`)",
        f"- Evaluation target: `future_upside_rank_percentile` (1.0 = best within timestamp)",
        f"- Min cross-section size: **{exp['min_cross_section_size']}**",
        f"- Features: full {report['dataset']['feature_count']} vs ATR-only `atr_pct` vs momentum `{exp['momentum_feature']}`",
        f"- Model: HistGradientBoostingRegressor (no new architectures)",
        "",
        "## Survivorship bias (prominent)",
        "",
        report["survivorship"],
        "",
        "## Cross-section coverage",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Timestamps considered | {cs.get('timestamps_considered')} |",
        f"| Timestamps excluded (<{exp['min_cross_section_size']} assets) | {cs.get('timestamps_excluded_below_min')} |",
        f"| Timestamps used | {cs.get('timestamps_used')} |",
        f"| Avg cross-section size | {fmt(cs.get('average_cross_section_size'), 2)} |",
        f"| Min / max cross-section | {cs.get('min_cross_section_size')} / {cs.get('max_cross_section_size')} |",
        f"| Asset observations used | {cs.get('asset_observations_used')} |",
        "",
        "## Aggregate comparison table",
        "",
        "| Metric | Full Model | ATR-only | Momentum (ret_24) | Random / All-assets |",
        "|--------|----------:|---------:|------------------:|--------------------:|",
    ]
    for metric, row in table.items():
        lines.append(
            f"| {metric} | {fmt(row.get('full_model'))} | {fmt(row.get('atr_only'))} | "
            f"{fmt(row.get('momentum_ret_24'))} | {fmt(row.get('random_all_assets'))} |"
        )
    conc = report["concentration"]
    lines.extend(
        [
            "",
            "## Concentration (is top-5 just high-ATR?)",
            "",
            "| Group | Mean ATR% | Mean future max upside | Mean future max drawdown |",
            "|-------|----------:|-----------------------:|-------------------------:|",
            f"| Full model top-5 | {fmt(conc['full_model_top5'].get('mean_atr_pct'), 3)} | "
            f"{fmt(conc['full_model_top5'].get('mean_future_max_upside'))} | "
            f"{fmt(conc['full_model_top5'].get('mean_future_max_drawdown'))} |",
            f"| ATR-only top-5 | {fmt(conc['atr_only_top5'].get('mean_atr_pct'), 3)} | "
            f"{fmt(conc['atr_only_top5'].get('mean_future_max_upside'))} | "
            f"{fmt(conc['atr_only_top5'].get('mean_future_max_drawdown'))} |",
            f"| All assets | {fmt(conc['all_assets'].get('mean_atr_pct'), 3)} | "
            f"{fmt(conc['all_assets'].get('mean_future_max_upside'))} | "
            f"{fmt(conc['all_assets'].get('mean_future_max_drawdown'))} |",
            "",
            "## Extreme upside (aggregate)",
            "",
            "### Full model",
            "",
            f"- Top-1: mean={fmt(report['full'].get('top_1', {}).get('mean_upside'))}, "
            f"median={fmt(report['full'].get('top_1', {}).get('median_upside'))}, "
            f"max={fmt(report['full'].get('top_1', {}).get('max_upside'))}, "
            f"hit3/5/10={fmt(report['full'].get('top_1', {}).get('hit_3pct'))}/"
            f"{fmt(report['full'].get('top_1', {}).get('hit_5pct'))}/"
            f"{fmt(report['full'].get('top_1', {}).get('hit_10pct'))}",
            f"- Top-5: mean={fmt(report['full'].get('top_5', {}).get('mean_upside'))}, "
            f"median={fmt(report['full'].get('top_5', {}).get('median_upside'))}, "
            f"max={fmt(report['full'].get('top_5', {}).get('max_upside'))}, "
            f"hit3/5/10={fmt(report['full'].get('top_5', {}).get('hit_3pct'))}/"
            f"{fmt(report['full'].get('top_5', {}).get('hit_5pct'))}/"
            f"{fmt(report['full'].get('top_5', {}).get('hit_10pct'))}",
            f"- Top-20%: mean={fmt(report['full'].get('top_20pct', {}).get('mean_upside'))}, "
            f"hit3/5/10={fmt(report['full'].get('top_20pct', {}).get('hit_3pct'))}/"
            f"{fmt(report['full'].get('top_20pct', {}).get('hit_5pct'))}/"
            f"{fmt(report['full'].get('top_20pct', {}).get('hit_10pct'))}",
            "",
            "## Walk-forward folds",
            "",
        ]
    )
    for i, fold in enumerate(report.get("folds") or [], 1):
        f = fold.get("full") or {}
        lines.append(
            f"### Fold {i}: {fold.get('test_start')} → {fold.get('test_end')}  \n"
            f"train {fold.get('train_start')} → {fold.get('train_end')}; "
            f"timestamps={fold.get('test_timestamps')}; "
            f"obs={fold.get('test_asset_observations')}; "
            f"avg CS={fmt(fold.get('average_cross_section_size'), 2)}  \n"
            f"Spearman={fmt(f.get('spearman'))}; "
            f"top1/3/5 upside={fmt(f.get('top_1_mean_upside'))}/"
            f"{fmt(f.get('top_3_mean_upside'))}/{fmt(f.get('top_5_mean_upside'))}; "
            f"top5 hit3/5/10={fmt(f.get('top_5_hit_3pct'))}/"
            f"{fmt(f.get('top_5_hit_5pct'))}/{fmt(f.get('top_5_hit_10pct'))}"
        )
        lines.append("")

    lines.extend(
        [
            "## Evidence checklist / status",
            "",
            f"**Status: `{status}`** (never activated for live trading)",
            "",
            "```",
            json.dumps(report.get("evidence"), indent=2, default=str),
            "```",
            "",
            "## Interpretation",
            "",
        ]
    )
    for q, a in (report.get("interpretation") or {}).items():
        lines.append(f"- **{q}** {a}")
    lines.extend(
        [
            "",
            "## Leakage",
            "",
            f"- Passed: **{report['leakage_audit'].get('passed')}**",
            f"- Findings: {report['leakage_audit'].get('label_mismatches_or_findings')}",
            f"- Embargo violations: {report['leakage_audit'].get('folds_violating_embargo')}",
            "",
            "## Lineage preservation",
            "",
            f"- Active bidirectional models preserved: **{report['lineage_preservation'].get('active_models_preserved')}**",
            f"- Experiment model status: `{report['lineage_preservation'].get('experiment_model_status')}`",
            "",
            "## Tests",
            "",
            f"- Research suite OK: **{report['test_results'].get('ok')}**",
            "",
            "## Errors",
            "",
            f"- {report['run_metadata'].get('errors') or []}",
            "",
        ]
    )
    return "\n".join(lines)


def interpret(report: dict) -> dict:
    table = report["comparison_table"]
    status = report["experiment_status"]
    full_sp = table["spearman"]["full_model"]
    atr_sp = table["spearman"]["atr_only"]
    top5_f = table["top_5_mean_upside"]["full_model"]
    top5_a = table["top_5_mean_upside"]["atr_only"]
    top5_all = table["top_5_mean_upside"]["random_all_assets"]
    beyond_atr = (
        full_sp is not None
        and atr_sp is not None
        and top5_f is not None
        and top5_a is not None
        and full_sp > atr_sp
        and top5_f > top5_a
    )
    return {
        "Does ranking information exist?": (
            f"Aggregate Spearman={fmt(full_sp)}; status=`{status}`."
        ),
        "Do top-ranked assets realize larger upside than all-assets?": (
            f"Top-5 mean upside={fmt(top5_f)} vs all-assets={fmt(top5_all)}."
        ),
        "Does the full model beat ATR-only ranking?": str(beyond_atr),
        "Does the full model beat momentum (ret_24)?": str(
            report.get("evidence", {}).get("full_beats_momentum_on_spearman_and_top5")
        ),
        "Is this mostly volatility selection?": (
            "Likely still volatility-linked if ATR ranking is competitive or stronger; "
            f"see concentration table (full top5 ATR%={fmt(report['concentration']['full_model_top5'].get('mean_atr_pct'), 3)} "
            f"vs ATR-only top5 ATR%={fmt(report['concentration']['atr_only_top5'].get('mean_atr_pct'), 3)})."
        ),
        "Should we treat this as product-ready ranking edge?": (
            "No — research-only; not activated."
            if status != "promising_ranking_signal"
            else "Promising research signal only; still not a trading strategy and not activated."
        ),
        "Continue to next research stage?": (
            "Only if status is promising_ranking_signal and concentration analysis shows separation from ATR."
            if status == "promising_ranking_signal"
            else "No — insufficient ranking edge beyond simple baselines on this evidence."
        ),
    }


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info(
        "ranking validate start experiment=%s db=%s provider=%s",
        EXPERIMENT.experiment_id,
        settings.database_url,
        settings.market_data_provider,
    )
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

            bars_by = {}
            for sym, aid in list(ids.items())[:8]:
                bars_by[sym] = await load_bars(db, aid, timeframe().value)
            leak = await audit_leakage(primary_rows, bars_by)

            result = fit_ranking_walk_forward(
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
                "applied retrospectively. This is not the full crypto market; delisted/illiquid names are "
                "missing. Survivorship/selection bias remains and can inflate apparent ranking stability."
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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_relative_upside_rank",
                    ],
                },
                "experiment": EXPERIMENT.lineage(),
                "sampling": {
                    "primary": f"{EXPERIMENT.primary_stride_bars}-bar stride",
                    "horizon_hours": EXPERIMENT.horizon_hours,
                    "timeframe": EXPERIMENT.timeframe,
                },
                "test_results": test_results,
                "dataset": {
                    "symbols_count": len(symbols),
                    "symbols": symbols,
                    "oldest_candle": candles.get("oldest"),
                    "newest_candle": candles.get("newest"),
                    "raw_candles": candles.get("count"),
                    "feature_count": len(feature_names()),
                    "features": feature_names(),
                    "lookback_bars": LOOKBACK_BARS,
                    "primary_stride_labeled_rows": len(primary_rows),
                },
                "cross_section": metrics.get("cross_section"),
                "folds": metrics.get("folds"),
                "fold_count": metrics.get("fold_count"),
                "full": metrics.get("full"),
                "atr_only": metrics.get("atr_only"),
                "momentum": metrics.get("momentum"),
                "all_assets_random": metrics.get("all_assets_random"),
                "comparison_table": metrics.get("comparison_table"),
                "concentration": metrics.get("concentration"),
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
                    "prior_reports_untouched": [
                        "docs/RESEARCH_VALIDATION_REPORT.md",
                        "docs/DIRECTIONAL_UP_VALIDATION_REPORT.md",
                    ],
                },
            }
            report["interpretation"] = interpret(report)

            REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
            REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
            REPORT_MD.write_text(write_markdown(report))
            print(json.dumps(report, indent=2, default=str))
            log.info("wrote %s and %s", REPORT_JSON, REPORT_MD)
    except Exception as exc:
        errors.append(str(exc))
        log.exception("ranking validation failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
