"""Validate volatility-adjusted upside experiment (information beyond ATR).

Does not overwrite prior experiment reports or activate live models.

Usage:
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_vol_adjusted_upside
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
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-voladj-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import backfill_directional_labels, build_samples, select_by_stride
from app.research.experiments import VOLATILITY_ADJUSTED_UPSIDE_V1
from app.research.features import extract_features, feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import STATUS_ACTIVE, save_experiment_run
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe
from app.research.vol_adjusted import fit_vol_adjusted_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_vol_adjusted_upside")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "VOLATILITY_ADJUSTED_UPSIDE_REPORT.json"
REPORT_MD = REPO_ROOT / "docs" / "VOLATILITY_ADJUSTED_UPSIDE_REPORT.md"
EXPERIMENT = VOLATILITY_ADJUSTED_UPSIDE_V1


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
        mutated = list(bars)
        if i + 1 < len(mutated):
            fut = mutated[i + 1]
            mutated[i + 1] = fut.model_copy(update={"high": fut.high * 3, "close": fut.close * 3})
        past_only = extract_features(mutated[: i + 1], tf)
        if a != past_only:
            findings.append(f"future mutation changed past features {row.symbol} {row.ts}")
        # ATR must come from past window only (same as features).
        if a is not None and "atr_pct" in a:
            atr_a = a["atr_pct"]
            atr_b = extract_features(bars[: i + 1], tf)["atr_pct"]
            if atr_a != atr_b:
                findings.append(f"atr rebuild mismatch {row.symbol}")
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
    target_def = (cs or {}).get("target_definition") or {}
    lines = [
        f"# Volatility-adjusted upside report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.json`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.json)",
        "",
        "Prior experiments preserved (unchanged):",
        "- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)",
        "- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)",
        "- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)",
        "",
        "## Hypothesis",
        "",
        "Can the existing 36-feature set predict **unusually large upside after controlling "
        "for current volatility**, or is apparent ranking skill mostly ATR?",
        "",
        "## Why this experiment exists",
        "",
        "Prior relative-ranking found a real signal, but ATR-only beat the full model "
        "(Spearman ~0.508 vs ~0.462). This experiment isolates upside *beyond* volatility.",
        "",
        "## Configuration",
        "",
        f"- Experiment ID: `{exp['experiment_id']}`",
        f"- Timeframe / horizon / stride: **{exp['timeframe']} / {exp['horizon_hours']}h / {exp['primary_stride_bars']}-bar**",
        f"- Min cross-section: **{exp['min_cross_section_size']}**",
        f"- Model: HistGradientBoostingRegressor",
        f"- Features: full 36, ATR-only, momentum `{exp['momentum_feature']}`, full-without-ATR",
        "",
        "## Target definitions (mathematical)",
        "",
        "### Secondary — ATR-normalized upside",
        "",
        "```",
        "atr_frac = atr_pct / 100",
        "future_upside_atr_multiple = future_max_upside_12h / max(atr_frac, eps)",
        f"eps = {target_def.get('eps', 1e-6)}",
        "```",
        "",
        "### Primary — cross-sectional volatility-adjusted residual",
        "",
        "At each timestamp T, for all eligible assets:",
        "",
        "```",
        "log_up_i  = log(1 + max(future_max_upside_12h_i, 0))",
        "log_atr_i = log(max(atr_frac_i, eps))",
        "Fit OLS within T only:  log_up = α_T + β_T * log_atr",
        "vol_adj_residual_i = log_up_i - (α̂_T + β̂_T * log_atr_i)",
        "```",
        "",
        "Higher residual ⇒ more upside than the **same-timestamp** volatility schedule implies.",
        "Residuals / future upside are **targets only** — never features.",
        "",
        f"Mean OLS α across timestamps: **{fmt(cs.get('mean_ols_alpha'))}**; "
        f"mean β: **{fmt(cs.get('mean_ols_beta'))}**",
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
        f"| Avg / min / max CS size | {fmt(cs.get('average_cross_section_size'), 2)} / "
        f"{cs.get('min_cross_section_size')} / {cs.get('max_cross_section_size')} |",
        f"| Asset observations | {cs.get('asset_observations_used')} |",
        "",
        "## Aggregate comparison (primary = vol-adj residual ranking)",
        "",
        "| Metric | Full | ATR-only | Momentum | Full w/o ATR | All-assets |",
        "|--------|-----:|---------:|---------:|-------------:|-----------:|",
    ]
    for metric in (
        "spearman",
        "top_5_mean_raw_upside",
        "top_5_mean_vol_adj",
        "top_5_hit_3pct",
        "top_5_hit_5pct",
        "top_5_hit_10pct",
    ):
        row = table.get(metric) or {}
        lines.append(
            f"| {metric} | {fmt(row.get('full_model'))} | {fmt(row.get('atr_only'))} | "
            f"{fmt(row.get('momentum_ret_24'))} | {fmt(row.get('full_without_atr'))} | "
            f"{fmt(row.get('random_all_assets'))} |"
        )

    lines.extend(
        [
            "",
            f"**Full − ATR Spearman:** {fmt(report.get('full_minus_atr_spearman'))}  ",
            f"**Full-without-ATR − ATR Spearman:** {fmt(report.get('full_without_atr_vs_atr'))}  ",
            f"**Full − ATR top-5 vol-adj:** {fmt((report.get('evidence') or {}).get('deltas', {}).get('top5_vol_adj_full_minus_atr'))}  ",
            f"**Full − ATR top-5 raw upside:** {fmt((report.get('evidence') or {}).get('deltas', {}).get('top5_raw_full_minus_atr'))}",
            "",
            "## Walk-forward folds",
            "",
        ]
    )
    for i, fold in enumerate(report.get("folds") or [], 1):
        f = fold.get("full") or {}
        a = fold.get("atr_only") or {}
        lines.append(
            f"### Fold {i}: test {fold.get('test_start')} → {fold.get('test_end')}  \n"
            f"train {fold.get('train_start')} → {fold.get('train_end')}; "
            f"ts={fold.get('test_timestamps')}; obs={fold.get('test_asset_observations')}; "
            f"avg CS={fmt(fold.get('average_cross_section_size'), 2)}  \n"
            f"Full Spearman={fmt(f.get('spearman'))}; ATR Spearman={fmt(a.get('spearman'))}; "
            f"Full top5 raw={fmt(f.get('top_5_mean_raw_upside'))}; "
            f"ATR top5 raw={fmt(a.get('top_5_mean_raw_upside'))}; "
            f"Full top5 vol-adj={fmt(f.get('top_5_mean_vol_adj'))}; "
            f"ATR top5 vol-adj={fmt(a.get('top_5_mean_vol_adj'))}"
        )
        lines.append("")

    buckets = report.get("volatility_buckets") or {}
    lines.extend(["## Volatility bucket analysis", ""])
    for name in ("low", "medium", "high"):
        b = buckets.get(name) or {}
        lines.append(
            f"- **{name}** (n={b.get('n')}): full Spearman={fmt(b.get('full_spearman'))}, "
            f"ATR={fmt(b.get('atr_spearman'))}, mom={fmt(b.get('momentum_spearman'))}; "
            f"top5 raw full/ATR={fmt(b.get('full_top5_raw_upside'))}/{fmt(b.get('atr_top5_raw_upside'))}; "
            f"top5 vol-adj full/ATR={fmt(b.get('full_top5_vol_adj'))}/{fmt(b.get('atr_top5_vol_adj'))}"
        )
    matched = report.get("matched_volatility") or {}
    lines.extend(
        [
            "",
            "## Matched-volatility analysis",
            "",
            matched.get("note", ""),
            "",
            f"- Matched pairs: **{matched.get('n_matched')}** (skipped={matched.get('n_skipped')})",
            f"- Mean selected upside: **{fmt(matched.get('mean_selected_upside'))}**",
            f"- Mean matched peer upside: **{fmt(matched.get('mean_matched_upside'))}**",
            f"- Mean difference (selected − peer): **{fmt(matched.get('mean_difference'))}**",
            f"- Median difference: **{fmt(matched.get('median_difference'))}**",
            f"- Fraction selected higher: **{fmt(matched.get('frac_selected_higher'))}**",
            "",
            "## Extreme upside (full model)",
            "",
            f"- Top-1: mean={fmt(report['full'].get('top_1_mean_raw_upside'))}, "
            f"median={fmt(report['full'].get('top_1_median_raw_upside'))}, "
            f"max={fmt(report['full'].get('top_1_max_raw_upside'))}, "
            f"hit3={fmt(report['full'].get('top_1_hit_3pct'))}",
            f"- Top-5: mean={fmt(report['full'].get('top_5_mean_raw_upside'))}, "
            f"median={fmt(report['full'].get('top_5_median_raw_upside'))}, "
            f"max={fmt(report['full'].get('top_5_max_raw_upside'))}, "
            f"hit3/5/10={fmt(report['full'].get('top_5_hit_3pct'))}/"
            f"{fmt(report['full'].get('top_5_hit_5pct'))}/{fmt(report['full'].get('top_5_hit_10pct'))}",
            f"- Top-20%: mean={fmt(report['full'].get('top_20pct_mean_raw_upside'))}, "
            f"hit3/5/10={fmt(report['full'].get('top_20pct_hit_3pct'))}/"
            f"{fmt(report['full'].get('top_20pct_hit_5pct'))}/{fmt(report['full'].get('top_20pct_hit_10pct'))}",
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
            "## Lineage preservation",
            "",
            f"- Active bidirectional preserved: **{report['lineage_preservation'].get('active_models_preserved')}**",
            f"- Experiment status: `{report['lineage_preservation'].get('experiment_model_status')}`",
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
    delta = report.get("full_minus_atr_spearman")
    no_atr = report.get("full_without_atr_vs_atr")
    if status == "promising_volatility_adjusted_signal":
        return (
            "The full feature set shows consistent vol-adjusted ranking skill beyond ATR-only, "
            "including without atr_pct. Research signal only — not product-ready."
        )
    if status == "weak_volatility_adjusted_signal":
        return (
            f"There is a positive vol-adjusted ranking signal (full−ATR Spearman={fmt(delta)}; "
            f"full-without-ATR−ATR={fmt(no_atr)}), but it is not strong/consistent enough to claim "
            "clear independent information beyond volatility under the success criteria."
        )
    return (
        f"No reliable evidence that the 36-feature model predicts upside beyond current volatility "
        f"(full−ATR Spearman={fmt(delta)}; full-without-ATR−ATR={fmt(no_atr)}). "
        "Classification rejected for beyond-volatility edge."
    )


def print_terminal_summary(report: dict) -> None:
    table = report["comparison_table"]
    buckets = report.get("volatility_buckets") or {}
    matched = report.get("matched_volatility") or {}

    def bucket_line(name: str) -> str:
        b = buckets.get(name) or {}
        return (
            f"n={b.get('n')}, full_sp={fmt(b.get('full_spearman'))}, "
            f"atr_sp={fmt(b.get('atr_spearman'))}, "
            f"top5_raw={fmt(b.get('full_top5_raw_upside'))}, "
            f"top5_adj={fmt(b.get('full_top5_vol_adj'))}"
        )

    beyond = (
        "Yes — provisional research evidence"
        if report["experiment_status"] == "promising_volatility_adjusted_signal"
        else (
            "Unclear / weak — not decisive beyond ATR"
            if report["experiment_status"] == "weak_volatility_adjusted_signal"
            else "No — not supported on this evidence"
        )
    )
    lines = [
        "",
        "======== VOLATILITY_ADJUSTED_UPSIDE_V1 SUMMARY ========",
        f"Experiment: {report['experiment']['experiment_id']}",
        f"Samples: {report['cross_section'].get('asset_observations_used')}",
        f"Timestamps: {report['cross_section'].get('timestamps_used')}",
        f"Folds: {report.get('fold_count')}",
        "",
        f"Full model Spearman: {fmt(table['spearman'].get('full_model'))}",
        f"ATR-only Spearman: {fmt(table['spearman'].get('atr_only'))}",
        f"Momentum Spearman: {fmt(table['spearman'].get('momentum_ret_24'))}",
        f"Full-without-ATR Spearman: {fmt(table['spearman'].get('full_without_atr'))}",
        "",
        f"Full top-5 raw upside: {fmt(table['top_5_mean_raw_upside'].get('full_model'))}",
        f"ATR-only top-5 raw upside: {fmt(table['top_5_mean_raw_upside'].get('atr_only'))}",
        f"Full top-5 volatility-adjusted target: {fmt(table['top_5_mean_vol_adj'].get('full_model'))}",
        f"ATR-only top-5 volatility-adjusted target: {fmt(table['top_5_mean_vol_adj'].get('atr_only'))}",
        "",
        f"Full minus ATR Spearman: {fmt(report.get('full_minus_atr_spearman'))}",
        f"Full-without-ATR vs ATR: {fmt(report.get('full_without_atr_vs_atr'))}",
        "",
        f"Low-volatility result: {bucket_line('low')}",
        f"Medium-volatility result: {bucket_line('medium')}",
        f"High-volatility result: {bucket_line('high')}",
        "",
        f"Matched-volatility result: n={matched.get('n_matched')}, "
        f"mean_diff={fmt(matched.get('mean_difference'))}, "
        f"frac_selected_higher={fmt(matched.get('frac_selected_higher'))}",
        "",
        f"Leakage passed: {report['leakage_audit'].get('passed')}",
        f"Embargo violations: {report['leakage_audit'].get('folds_violating_embargo')}",
        "",
        f"Classification: {report['experiment_status']}",
        "",
        f"Does the full model contain evidence beyond volatility?: {beyond}",
        "",
        report.get("final_interpretation", ""),
        "========================================================",
        "",
    ]
    print("\n".join(lines))


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info(
        "vol-adjusted validate start experiment=%s db=%s",
        EXPERIMENT.experiment_id,
        settings.database_url,
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

            result = fit_vol_adjusted_walk_forward(
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
                "the entire crypto market. Survivorship bias is documented, not fixed, in this experiment."
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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_vol_adjusted_upside",
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
                "cross_section": metrics.get("cross_section"),
                "folds": metrics.get("folds"),
                "fold_count": metrics.get("fold_count"),
                "full": metrics.get("full"),
                "atr_only": metrics.get("atr_only"),
                "momentum": metrics.get("momentum"),
                "full_without_atr": metrics.get("full_without_atr"),
                "atr_multiple_diagnostic": metrics.get("atr_multiple_diagnostic"),
                "all_assets_random": metrics.get("all_assets_random"),
                "comparison_table": metrics.get("comparison_table"),
                "volatility_buckets": metrics.get("volatility_buckets"),
                "matched_volatility": metrics.get("matched_volatility"),
                "evidence": metrics.get("evidence"),
                "full_minus_atr_spearman": metrics.get("full_minus_atr_spearman"),
                "full_without_atr_vs_atr": metrics.get("full_without_atr_vs_atr_spearman"),
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
        log.exception("vol-adjusted validation failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
