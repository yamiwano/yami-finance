"""Validate directional-up experiment (large_up_move) vs ATR-only baseline.

Does not overwrite the bidirectional validation report.

Usage (from repo root):
  MARKET_DATA_PROVIDER=binance \\
  DATABASE_URL=sqlite+aiosqlite:////ABS/PATH/backend/data/research.db \\
  PYTHONPATH=backend python -m scripts.validate_directional_up
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

from sqlalchemy import func, select

os.environ.setdefault("MARKET_DATA_PROVIDER", "binance")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-directional-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, engine, init_db
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import (
    backfill_directional_labels,
    build_samples,
    select_by_stride,
)
from app.research.experiments import DIRECTIONAL_UP_3PCT_12H_V1
from app.research.features import feature_names
from app.research.history import as_utc, backfill, candle_stats, load_bars
from app.research.labels import label_forward
from app.research.model import (
    STATUS_ACTIVE,
    fit_experiment_walk_forward,
    labeled_rows,
    save_experiment_run,
)
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, timeframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_directional_up")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "directional-up-validation-report.json"
REPORT_MD = REPO_ROOT / "docs" / "DIRECTIONAL_UP_VALIDATION_REPORT.md"
EXPERIMENT = DIRECTIONAL_UP_3PCT_12H_V1


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


def target_breakdown(rows: list[ResearchSample]) -> dict:
    thr = move_pct()
    labeled = [r for r in rows if r.large_up_move is not None]
    pos = [r for r in labeled if r.large_up_move]
    clean = [r for r in labeled if r.clean_up_move]
    sig = [r for r in labeled if r.significant_move]
    return {
        "primary_target": "large_up_move",
        "secondary_target": "clean_up_move",
        "preserved_target": "significant_move",
        "labeled": len(labeled),
        "large_up_move_positives": len(pos),
        "large_up_move_rate": (len(pos) / len(labeled)) if labeled else None,
        "clean_up_move_positives": len(clean),
        "clean_up_move_rate": (len(clean) / len(labeled)) if labeled else None,
        "significant_move_positives": len(sig),
        "significant_move_rate": (len(sig) / len(labeled)) if labeled else None,
        "definition_primary": (
            f"large_up_move=True if max_upside>={thr} during the next {horizon_hours()}h "
            "(final close need not remain above +3%)."
        ),
        "definition_secondary": (
            f"clean_up_move=True if max_upside>={thr} AND max_drawdown>{EXPERIMENT.clean_drawdown_floor} "
            f"over the next {horizon_hours()}h (analysis only; not the training target)."
        ),
    }


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
        from app.research.features import extract_features

        a = extract_features(bars[: i + 1], tf)
        c = extract_features(bars[: i + 1], tf)
        if a != c:
            findings.append("feature rebuild mismatch")
        labels = label_forward(bars[i + 1 :], bars[i].close, h, thr)
        if labels and row.large_up_move is not None:
            if bool(labels["large_up_move"]) != bool(row.large_up_move):
                findings.append(f"large_up_move mismatch {row.symbol} {row.ts}")
            if bool(labels["clean_up_move"]) != bool(row.clean_up_move):
                findings.append(f"clean_up_move mismatch {row.symbol} {row.ts}")
            if row.significant_move is not None and bool(labels["significant_move"]) != bool(row.significant_move):
                findings.append(f"significant_move mismatch {row.symbol} {row.ts}")
        checked += 1

    timestamps = [as_utc(r.ts) for r in sorted(rows, key=lambda r: as_utc(r.ts))]
    folds = walk_forward_folds(timestamps)
    leak_folds = 0
    for train_idx, test_idx in folds:
        if not fold_is_causal(timestamps, train_idx, test_idx, embargo()):
            leak_folds += 1
    return {
        "samples_checked": checked,
        "label_mismatches_or_findings": findings[:20],
        "fold_count": len(folds),
        "folds_violating_embargo": leak_folds,
        "embargo_hours": horizon_hours(),
    }


def summarize_model(result: dict) -> dict:
    metrics = result["metrics"]
    comparison = result["comparison"]
    return {
        "experiment_status": result["experiment_status"],
        "passed_experiment_gates": result["passed_experiment_gates"],
        "promote_to_live": False,
        "notes": result["notes"],
        "fold_count": metrics.get("fold_count"),
        "folds": metrics.get("folds"),
        "mean_auc": metrics.get("mean_auc"),
        "std_auc": metrics.get("std_auc"),
        "mean_brier": metrics.get("mean_brier"),
        "std_brier": metrics.get("std_brier"),
        "mean_baseline_brier": metrics.get("mean_baseline_brier"),
        "mean_top_quintile_lift": metrics.get("mean_top_quintile_lift"),
        "mean_top_quintile_precision": metrics.get("mean_top_quintile_precision"),
        "atr_only_mean_auc": comparison["atr_only"].get("mean_auc"),
        "atr_only_std_auc": comparison["atr_only"].get("std_auc"),
        "atr_only_mean_brier": comparison["atr_only"].get("mean_brier"),
        "atr_only_mean_baseline_brier": comparison["atr_only"].get("mean_baseline_brier"),
        "atr_only_mean_top_quintile_lift": comparison["atr_only"].get("mean_top_quintile_lift"),
        "atr_only_mean_top_quintile_precision": comparison["atr_only"].get("mean_top_quintile_precision"),
        "atr_only_folds": comparison["atr_only"].get("folds"),
        "auc_delta_full_minus_atr": comparison.get("auc_delta_full_minus_atr"),
        "brier_delta_full_minus_atr": comparison.get("brier_delta_full_minus_atr"),
        "full_beats_constant_baseline": comparison.get("full_beats_constant_baseline"),
        "full_adds_value_beyond_atr": comparison.get("full_adds_value_beyond_atr"),
        "atr_auc_margin_required": comparison.get("atr_auc_margin_required"),
    }


def write_markdown(report: dict) -> str:
    exp = report["experiment"]
    primary = report["primary_stride_12"]
    sens = report["sensitivity_stride_6"]
    wf = primary["walk_forward"]
    lines = [
        f"# Directional-up validation report (`{exp['experiment_id']}`)",
        "",
        f"Generated: **{report['run_metadata']['generated_at']}**  ",
        f"Provider: **{report['run_metadata']['market_data_provider']}**  ",
        f"Database: `{report['run_metadata']['database_url']}`",
        "",
        "Machine-readable: [`directional-up-validation-report.json`](directional-up-validation-report.json)",
        "",
        "Prior bidirectional report is unchanged: [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md).",
        "",
        "## Experiment configuration",
        "",
        f"- **Experiment ID:** `{exp['experiment_id']}`",
        f"- **Primary target:** `{exp['target_name']}` = max_upside >= {exp['move_pct']} over {exp['horizon_hours']}h",
        f"- **Secondary (analysis only):** `{exp['secondary_target']}`",
        f"- **Timeframe / horizon:** {exp['timeframe']} / {exp['horizon_hours']}h",
        f"- **Primary sampling:** **{exp['primary_stride_bars']}-bar stride** (non-overlapping 12h label windows)",
        f"- **Sensitivity sampling:** existing **{exp['sensitivity_stride_bars']}-bar stride**",
        f"- **Features:** full {report['dataset']['feature_count']} vs ATR-only `{list(exp['atr_feature_names'])}`",
        "",
        "## Dataset",
        "",
        f"| Metric | Primary (stride {exp['primary_stride_bars']}) | Sensitivity (stride {exp['sensitivity_stride_bars']}) |",
        "|--------|---:|---:|",
        f"| Labeled samples | {primary['labeled_samples']} | {sens['labeled_samples']} |",
        f"| Positive rate (large_up_move) | {primary['positive_rate']} | {sens['positive_rate']} |",
        f"| Raw candles | {report['dataset']['raw_candles']} | (same) |",
        f"| Symbols | {report['dataset']['symbols_count']} | (same) |",
        f"| Candle range | {report['dataset']['oldest_candle']} → {report['dataset']['newest_candle']} | (same) |",
        "",
        "## Primary walk-forward (stride 12): full vs ATR-only",
        "",
        f"| Aggregate | Full model | ATR-only |",
        "|-----------|----------:|---------:|",
        f"| Mean AUC ± std | {wf.get('mean_auc')} ± {wf.get('std_auc')} | {wf.get('atr_only_mean_auc')} ± {wf.get('atr_only_std_auc')} |",
        f"| Mean Brier | {wf.get('mean_brier')} | {wf.get('atr_only_mean_brier')} |",
        f"| Mean baseline Brier | {wf.get('mean_baseline_brier')} | {wf.get('atr_only_mean_baseline_brier')} |",
        f"| Mean top-quintile lift | {wf.get('mean_top_quintile_lift')} | {wf.get('atr_only_mean_top_quintile_lift')} |",
        f"| Mean top-quintile precision | {wf.get('mean_top_quintile_precision')} | {wf.get('atr_only_mean_top_quintile_precision')} |",
        f"| AUC delta (full − ATR) | {wf.get('auc_delta_full_minus_atr')} | — |",
        f"| Brier delta (full − ATR) | {wf.get('brier_delta_full_minus_atr')} | — |",
        "",
        f"- Beats constant base-rate baseline: **{wf.get('full_beats_constant_baseline')}**",
        f"- Full adds measurable value beyond ATR: **{wf.get('full_adds_value_beyond_atr')}** "
        f"(required AUC margin ≥ {wf.get('atr_auc_margin_required')})",
        f"- Experiment status: **`{wf.get('experiment_status')}`**",
        f"- Promoted over live bidirectional model: **No** (by design)",
        "",
        "## Interpretation answers",
        "",
    ]
    answers = report.get("interpretation", {})
    for i, (q, a) in enumerate(answers.items(), 1):
        lines.append(f"{i}. **{q}** {a}")
        lines.append("")
    lines.extend(
        [
            "## Leakage / bias",
            "",
            f"- Leakage findings: {report['leakage_audit'].get('label_mismatches_or_findings')}",
            f"- Embargo violations: {report['leakage_audit'].get('folds_violating_embargo')}",
            f"- Survivorship: {report['survivorship']}",
            "",
            "## Tests",
            "",
            f"- Research suite OK: **{report['test_results'].get('ok')}**",
            "",
            "## Errors",
            "",
            f"- {report['run_metadata'].get('errors') or '[]'}",
            "",
        ]
    )
    return "\n".join(lines)


def interpret(report_core: dict) -> dict:
    wf = report_core["primary_stride_12"]["walk_forward"]
    primary_n = report_core["primary_stride_12"]["labeled_samples"]
    pos_rate = report_core["primary_stride_12"]["positive_rate"]
    auc_f = wf.get("mean_auc")
    auc_a = wf.get("atr_only_mean_auc")
    delta = wf.get("auc_delta_full_minus_atr")
    beyond = wf.get("full_adds_value_beyond_atr")
    beats = wf.get("full_beats_constant_baseline")
    status = wf.get("experiment_status")
    leak = report_core["leakage_audit"]
    still_vol = (
        delta is not None
        and abs(delta) < EXPERIMENT.atr_auc_margin
        and auc_f is not None
        and auc_a is not None
    )
    continue_next = bool(beyond and beats and status == "experiment_pass")
    return {
        "How many samples were produced?": (
            f"Primary stride-12 labeled n={primary_n}; "
            f"sensitivity stride-6 labeled n={report_core['sensitivity_stride_6']['labeled_samples']}."
        ),
        "What was the positive class rate?": f"large_up_move rate (primary)={pos_rate}.",
        "What were the walk-forward AUC/Brier results?": (
            f"Full mean AUC={auc_f}, mean Brier={wf.get('mean_brier')}; "
            f"ATR-only mean AUC={auc_a}, mean Brier={wf.get('atr_only_mean_brier')}."
        ),
        "How did the full model compare with ATR-only?": (
            f"AUC delta (full−ATR)={delta}; Brier delta (full−ATR)={wf.get('brier_delta_full_minus_atr')}."
        ),
        "Did the model identify upward moves specifically, or mostly volatility?": (
            "Still appears largely volatility-driven: full-model AUC is within the ATR-only margin."
            if still_vol
            else (
                "Full model exceeded the ATR-only AUC margin; directional features may contribute, "
                "but treat cautiously given fold count and survivorship."
                if beyond
                else "Insufficient separation from ATR-only to claim directional (non-volatility) edge."
            )
        ),
        "Did the model beat the constant base-rate baseline?": str(beats),
        "Did the full feature model materially outperform ATR-only?": str(beyond),
        "What were the top-quintile precision and lift?": (
            f"precision={wf.get('mean_top_quintile_precision')}, lift={wf.get('mean_top_quintile_lift')}."
        ),
        "Were there any leakage issues?": (
            f"findings={leak.get('label_mismatches_or_findings')}, "
            f"embargo_violations={leak.get('folds_violating_embargo')}."
        ),
        "What survivorship/selection bias remains?": report_core["survivorship"],
        "Should this experiment be considered evidence of directional predictive value?": (
            "Yes — provisional evidence only; not a trading signal."
            if continue_next
            else (
                f"No. Status=`{status}`. Do not treat as directional predictive edge; "
                "do not proceed to the next research stage on this evidence alone."
            )
        ),
    }


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info(
        "directional validate start experiment=%s db=%s provider=%s",
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
            # Preserve any existing active bidirectional model ids before we write experiment rows.
            active_before = (
                await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
            ).scalars().all()
            active_before_ids = {str(m.id) for m in active_before}

            filled = await backfill(db, provider, ids)
            log.info("backfill added=%s", filled)
            # Keep legacy 6-bar grid intact; stride-12 evaluation thins it deterministically.
            built = await build_samples(db, ids, stride=EXPERIMENT.sensitivity_stride_bars)
            log.info("dataset built=%s", built)
            n_dir = await backfill_directional_labels(db)
            log.info("directional labels backfilled=%s", n_dir)

            candles = await candle_stats(db, timeframe().value)
            all_labeled = await labeled_rows(db, target_name="large_up_move")
            primary_rows = select_by_stride(
                all_labeled,
                EXPERIMENT.primary_stride_bars,
                base_stride=EXPERIMENT.sensitivity_stride_bars,
            )
            sensitivity_rows = select_by_stride(
                all_labeled,
                EXPERIMENT.sensitivity_stride_bars,
                base_stride=EXPERIMENT.sensitivity_stride_bars,
            )

            dup = (
                await db.execute(
                    select(ResearchSample.symbol, ResearchSample.ts, func.count())
                    .where(
                        ResearchSample.timeframe == timeframe().value,
                        ResearchSample.horizon_hours == horizon_hours(),
                    )
                    .group_by(ResearchSample.symbol, ResearchSample.ts)
                    .having(func.count() > 1)
                )
            ).all()

            primary_result = fit_experiment_walk_forward(primary_rows, EXPERIMENT)
            sensitivity_result = fit_experiment_walk_forward(sensitivity_rows, EXPERIMENT)
            model_row = await save_experiment_run(db, primary_result)

            active_after = (
                await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
            ).scalars().all()
            active_after_ids = {str(m.id) for m in active_after}
            preserved_active = active_before_ids <= active_after_ids

            bars_by = {}
            for sym, aid in list(ids.items())[:8]:
                bars_by[sym] = await load_bars(db, aid, timeframe().value)
            leak = await audit_leakage(primary_rows, bars_by)
            breakdown = target_breakdown(primary_rows)

            pos_primary = sum(1 for r in primary_rows if r.large_up_move)
            pos_sens = sum(1 for r in sensitivity_rows if r.large_up_move)

            survivorship = (
                "Universe remains today's top-N USDT pairs by 24h quote volume applied to history. "
                "Delisted/illiquid names absent from today's set are missing — survivorship/selection bias remains."
            )

            primary_summary = summarize_model(primary_result)
            sens_summary = summarize_model(sensitivity_result)

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
                        "MARKET_DATA_PROVIDER=binance DATABASE_URL=... PYTHONPATH=backend python -m scripts.validate_directional_up",
                    ],
                },
                "experiment": EXPERIMENT.lineage(),
                "sampling": {
                    "primary_experiment": f"{EXPERIMENT.primary_stride_bars}-bar stride",
                    "secondary_sensitivity": f"existing {EXPERIMENT.sensitivity_stride_bars}-bar stride",
                    "horizon_hours": EXPERIMENT.horizon_hours,
                    "timeframe": EXPERIMENT.timeframe,
                    "note": (
                        "Primary evaluation thins the stored 6-bar sample grid to every other sample "
                        "per symbol (equivalent to 12-bar stride from the same lookback origin), "
                        "so 12h outcome windows do not overlap."
                    ),
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
                    "duplicate_symbol_ts_groups": len(dup),
                    "build_samples": built,
                    "directional_labels_updated": n_dir,
                },
                "label": breakdown,
                "primary_stride_12": {
                    "labeled_samples": len(primary_rows),
                    "positives": pos_primary,
                    "positive_rate": (pos_primary / len(primary_rows)) if primary_rows else None,
                    "walk_forward": primary_summary,
                    "saved_model_id": str(model_row.id),
                    "saved_status": model_row.status,
                },
                "sensitivity_stride_6": {
                    "labeled_samples": len(sensitivity_rows),
                    "positives": pos_sens,
                    "positive_rate": (pos_sens / len(sensitivity_rows)) if sensitivity_rows else None,
                    "walk_forward": sens_summary,
                },
                "leakage_audit": leak,
                "survivorship": survivorship,
                "lineage_preservation": {
                    "active_bidirectional_models_before": sorted(active_before_ids),
                    "active_bidirectional_models_after": sorted(active_after_ids),
                    "active_models_preserved": preserved_active,
                    "experiment_model_status": model_row.status,
                    "experiment_model_never_set_active": model_row.status != STATUS_ACTIVE,
                },
                "promotion_gates": {
                    "min_folds": 2,
                    "min_mean_auc": EXPERIMENT.promote_min_auc,
                    "require_brier_better_than_constant_baseline": True,
                    "require_auc_margin_vs_atr": EXPERIMENT.atr_auc_margin,
                    "require_brier_better_than_atr": EXPERIMENT.require_brier_better_than_atr,
                    "live_promotion": False,
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
        log.exception("directional validation failed")
        raise
    finally:
        await provider.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
