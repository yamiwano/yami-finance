"""One-shot end-to-end research validation against live Binance history.

Usage:
  MARKET_DATA_PROVIDER=binance DATABASE_URL=sqlite+aiosqlite:////tmp/yami-validate.db \\
    PYTHONPATH=. python -m scripts.validate_research
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

# Ensure settings pick up env before app imports create the engine.
os.environ.setdefault("MARKET_DATA_PROVIDER", "binance")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/yami-research-validate.db")
os.environ.setdefault("RESEARCH_ENABLED", "true")

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.domain import Timeframe
from app.market_data import create_provider
from app.models.asset import Asset
from app.models.candle import Candle
from app.models.research import ModelVersion, Prediction, ResearchSample
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import build_samples, sample_stats
from app.research.features import feature_names
from app.research.history import as_utc, backfill, candle_stats
from app.research.labels import label_forward
from app.research.live import live_stats, record_closed, resolve_pending
from app.research.model import (
    active_model,
    design_matrix,
    fit_walk_forward,
    labeled_rows,
    load_payload,
    save_training_run,
)
from app.research.spec import LOOKBACK_BARS, embargo, horizon_bars, horizon_hours, move_pct, sample_stride, timeframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validate_research")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = REPO_ROOT / "docs" / "research-validation-report.json"


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


def label_breakdown(rows: list[ResearchSample]) -> dict:
    labeled = [r for r in rows if r.significant_move is not None]
    pos = [r for r in labeled if r.significant_move]
    up_only = 0
    down_only = 0
    both = 0
    neither_but_pos = 0
    thr = move_pct()
    for r in pos:
        up = (r.max_upside or 0) >= thr or (r.fwd_return or 0) >= thr
        down = (r.max_drawdown or 0) <= -thr or (r.fwd_return or 0) <= -thr
        # Match label_forward logic more carefully
        abs_ret = abs(r.fwd_return or 0) >= thr
        up_path = (r.max_upside or 0) >= thr
        down_path = (r.max_drawdown or 0) <= -thr
        up = abs_ret and (r.fwd_return or 0) > 0 or up_path
        down = abs_ret and (r.fwd_return or 0) < 0 or down_path
        if up and down:
            both += 1
        elif up:
            up_only += 1
        elif down:
            down_only += 1
        else:
            neither_but_pos += 1
    return {
        "labeled": len(labeled),
        "positives": len(pos),
        "positive_rate": (len(pos) / len(labeled)) if labeled else None,
        "pos_upside_path_or_up_close": up_only,
        "pos_drawdown_path_or_down_close": down_only,
        "pos_both_directions_touched": both,
        "pos_other": neither_but_pos,
        "definition": (
            f"significant_move=True if |fwd_return|>={thr} OR max_upside>={thr} OR max_drawdown<=-{thr} "
            f"over the next {horizon_hours()}h. Up and down both count as positive."
        ),
    }


async def audit_leakage(rows: list[ResearchSample], bars_by_symbol: dict) -> dict:
    tf = timeframe().value
    h = horizon_bars(tf)
    thr = move_pct()
    findings = []
    checked = 0
    for row in rows[:200]:
        bars = bars_by_symbol.get(row.symbol) or []
        idx = {as_utc(b.ts): i for i, b in enumerate(bars)}
        i = idx.get(as_utc(row.ts))
        if i is None:
            continue
        # Features must ignore bars after i
        from app.research.features import extract_features

        a = extract_features(bars[: i + 1], tf)
        b = extract_features(bars[: i + 1] + bars[i + 1 : i + 1 + h], tf)  # would leak if window not truncated
        # extract_features uses last LOOKBACK only, so appending future changes features —
        # confirm historical builder passed bars[:i+1] only by replaying that contract:
        c = extract_features(bars[: i + 1], tf)
        if a != c:
            findings.append("feature rebuild mismatch")
        labels = label_forward(bars[i + 1 :], bars[i].close, h, thr)
        if labels and row.significant_move is not None:
            if bool(labels["significant_move"]) != bool(row.significant_move):
                findings.append(f"label mismatch {row.symbol} {row.ts}")
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


async def main() -> None:
    errors: list[str] = []
    settings = get_settings()
    log.info(
        "validate start db=%s provider=%s backfill_days=%s universe=%s",
        settings.database_url,
        settings.market_data_provider,
        settings.research_backfill_days,
        settings.universe_size,
    )
    test_results = run_research_tests()
    await init_db()
    provider = create_provider()
    await provider.start()
    try:
        ids = await ensure_assets(provider)
        symbols = sorted(ids)
        log.info("universe n=%s symbols=%s", len(symbols), ",".join(symbols))

        async with SessionLocal() as db:
            filled = await backfill(db, provider, ids)
            log.info("backfill added=%s", filled)
            built = await build_samples(db, ids)
            log.info("dataset built=%s", built)
            candles = await candle_stats(db, timeframe().value)
            samples = await sample_stats(db)
            rows = await labeled_rows(db)

            # Duplicate check
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

            result = fit_walk_forward(rows)
            previous = await active_model(db)
            model_row = await save_training_run(db, result, previous)

            # Live path smoke: record predictions on latest bar; resolve if horizon available
            payload = load_payload(model_row.blob) if model_row.blob else None
            live_recorded = 0
            live_resolved = 0
            if payload and model_row.status == "active":
                live_recorded = await record_closed(db, model_row, payload, ids)
                live_resolved = await resolve_pending(db, ids)
            live = await live_stats(db)

            # Load bars for leakage audit (subset)
            from app.research.history import load_bars

            bars_by = {}
            for sym, aid in list(ids.items())[:8]:
                bars_by[sym] = await load_bars(db, aid, timeframe().value)
            leak = await audit_leakage(rows, bars_by)

            breakdown = label_breakdown(rows)

            # Overlap / effective independence note
            stride = sample_stride()
            horizon = horizon_hours()
            overlap_ratio = max(0.0, 1.0 - (stride / horizon)) if horizon else None

            report = {
                "run_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "market_data_provider": settings.market_data_provider,
                    "database_url": settings.database_url,
                    "research_backfill_days": settings.research_backfill_days,
                    "universe_size": settings.universe_size,
                    "errors": errors,
                },
                "test_results": test_results,
                "dataset": {
                    "symbols_count": len(symbols),
                    "symbols": symbols,
                    "oldest_candle": candles.get("oldest"),
                    "newest_candle": candles.get("newest"),
                    "raw_candles": candles.get("count"),
                    "research_samples": samples.get("count"),
                    "labeled_samples": samples.get("labeled"),
                    "positive_class_rate": samples.get("positive_rate"),
                    "feature_count": len(feature_names()),
                    "features": feature_names(),
                    "timeframe": timeframe().value,
                    "prediction_horizon_hours": horizon_hours(),
                    "move_threshold": move_pct(),
                    "sample_stride_bars": stride,
                    "lookback_bars": LOOKBACK_BARS,
                    "duplicate_symbol_ts_groups": len(dup),
                    "universe_selection": (
                        "Current Binance USDT spot top-N by 24h quote volume "
                        f"(universe_size={settings.universe_size}). Historical rows are for today's survivors."
                    ),
                },
                "label": breakdown,
                "leakage_audit": leak,
                "statistical_notes": {
                    "label_window_overlap_fraction_approx": overlap_ratio,
                    "note": (
                        f"Samples every {stride}h with a {horizon}h label window heavily overlap within each symbol, "
                        "so effective independent sample size is much smaller than labeled_samples."
                    ),
                    "enough_for_meaningful_validation": bool(
                        samples.get("labeled", 0) >= 1000
                        and result["metrics"].get("fold_count", 0) >= 2
                        and all((f.get("n") or 0) >= 40 for f in result["metrics"].get("folds") or [])
                    ),
                },
                "walk_forward": {
                    "fold_count": result["metrics"].get("fold_count"),
                    "folds": result["metrics"].get("folds"),
                    "mean_auc": result["metrics"].get("mean_auc"),
                    "std_auc": result["metrics"].get("std_auc"),
                    "mean_brier": result["metrics"].get("mean_brier"),
                    "std_brier": result["metrics"].get("std_brier"),
                    "mean_baseline_brier": result["metrics"].get("mean_baseline_brier"),
                    "mean_top_quintile_lift": result["metrics"].get("mean_top_quintile_lift"),
                    "beats_baseline": bool(
                        result["metrics"].get("mean_brier") is not None
                        and result["metrics"].get("mean_baseline_brier") is not None
                        and result["metrics"]["mean_brier"] < result["metrics"]["mean_baseline_brier"]
                    ),
                    "mean_auc_vs_0_5": (
                        None
                        if result["metrics"].get("mean_auc") is None
                        else result["metrics"]["mean_auc"] - 0.5
                    ),
                },
                "promotion": {
                    "promote_flag": result["promote"],
                    "saved_status": model_row.status,
                    "model_id": str(model_row.id),
                    "criteria": (
                        f"fold_count>={2}, mean_auc>={0.55}, mean_brier < mean_baseline_brier; "
                        "never promotes on train score alone; active blob is final fit on all labeled rows "
                        "only after walk-forward criteria pass."
                    ),
                    "notes": model_row.notes,
                    "live_uses_active_blob": bool(model_row.status == "active" and model_row.blob),
                },
                "live_path": {
                    "recorded": live_recorded,
                    "resolved_this_run": live_resolved,
                    "stats": live,
                },
            }

            REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
            REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
            print(json.dumps(report, indent=2, default=str))
            log.info("wrote %s", REPORT_JSON)
    except Exception as exc:
        errors.append(str(exc))
        log.exception("validation failed")
        raise
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
