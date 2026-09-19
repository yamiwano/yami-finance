"""Background research loop. Does not replace the live scanner."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal
from app.models.research import ModelVersion
from app.research.dataset import build_samples, sample_stats
from app.research.history import asset_map, backfill, candle_stats, persist_provider_window
from app.research.live import live_stats, record_closed, resolve_pending
from app.research.model import DISCLAIMER, active_model, fit_walk_forward, labeled_rows, load_payload, save_training_run
from app.research.spec import MIN_TRAIN_SAMPLES, timeframe

log = logging.getLogger("radar.research")

RETRAIN_SECONDS = 12 * 3600


def _model_out(row: ModelVersion | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": str(row.id),
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "timeframe": row.timeframe,
        "horizon_hours": row.horizon_hours,
        "metrics": row.metrics,
        "notes": row.notes,
    }


class ResearchService:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._lock = asyncio.Lock()
        self._payload: dict[str, Any] | None = None
        self._last_train = 0.0
        self.status: dict[str, Any] = {
            "phase": "idle",
            "message": "Research loop has not started.",
            "error": None,
            "busy": False,
            "last_train_at": None,
        }

    def _set(self, phase: str, message: str, error: str | None = None) -> None:
        self.status["phase"] = phase
        self.status["message"] = message
        self.status["error"] = error

    async def start(self) -> None:
        if not get_settings().research_enabled:
            self._set("disabled", "Research loop is off (RESEARCH_ENABLED=false). Scanner is unchanged.")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="research-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        await asyncio.sleep(2)
        try:
            await self._bootstrap()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("research bootstrap failed")
            self.status["busy"] = False
            self._set("error", "Research bootstrap failed. Scanner continues.", str(exc))
        while self._running:
            retrain = False
            try:
                async with SessionLocal() as db:
                    await self._tick(db)
                    retrain = self._last_train == 0 or (time.monotonic() - self._last_train >= RETRAIN_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("research tick failed")
                self.status["error"] = str(exc)
            if retrain:
                try:
                    await self.train_now()
                except Exception:
                    log.exception("scheduled retrain failed")
            await asyncio.sleep(60)

    async def _ids(self, db) -> dict:
        from app.scanner import scanner

        if scanner.asset_ids:
            return dict(scanner.asset_ids)
        return await asset_map(db)

    async def _bootstrap(self) -> None:
        from app.scanner import scanner

        async with SessionLocal() as db:
            ids = await self._ids(db)
            if not ids:
                self._set("waiting", "Waiting for scanner universe before backfill.")
                return
            self.status["busy"] = True
            self._set("backfill", "Writing closed 1h candles into the research store.")
            await persist_provider_window(db, scanner.provider, ids)
            await backfill(db, scanner.provider, ids)
            self._set("dataset", "Building point-in-time samples and forward labels.")
            built = await build_samples(db, ids)
            stats = await sample_stats(db)
        await self._load_active()
        await self.train_now()
        self.status["busy"] = False
        self._set(
            "live",
            f"Research store ready: {stats.get('labeled', 0)} labeled {stats.get('timeframe', '1h')} samples "
            f"(+{built.get('created', 0)} this pass).",
        )

    async def _tick(self, db) -> None:
        from app.scanner import scanner

        ids = await self._ids(db)
        if not ids:
            return
        await persist_provider_window(db, scanner.provider, ids, limit=4)
        await build_samples(db, ids)
        await resolve_pending(db, ids)
        model = await active_model(db)
        if model and model.blob:
            if self._payload is None:
                self._payload = load_payload(model.blob)
            if self._payload:
                await record_closed(db, model, self._payload, ids)

    async def _load_active(self) -> None:
        async with SessionLocal() as db:
            model = await active_model(db)
            if model and model.blob:
                self._payload = load_payload(model.blob)
            else:
                self._payload = None

    async def train_now(self) -> dict[str, Any]:
        async with self._lock:
            async with SessionLocal() as db:
                rows = await labeled_rows(db)
                if len(rows) < MIN_TRAIN_SAMPLES:
                    msg = f"Need {MIN_TRAIN_SAMPLES} labeled samples to train; have {len(rows)}."
                    self._set("waiting", msg)
                    return {"ok": False, "reason": msg, "labeled": len(rows)}
                self.status["busy"] = True
                self._set("train", f"Walk-forward training on {len(rows)} labeled samples.")
                try:
                    result = await asyncio.to_thread(fit_walk_forward, rows)
                    previous = await active_model(db)
                    row = await save_training_run(db, result, previous)
                    self._last_train = time.monotonic()
                    self.status["last_train_at"] = row.created_at.isoformat() if row.created_at else None
                    if row.status == "active" and row.blob:
                        self._payload = load_payload(row.blob)
                    self._set(
                        "live" if row.status == "active" else "idle",
                        f"Training finished: {row.status}. {row.notes}",
                    )
                    return {
                        "ok": True,
                        "status": row.status,
                        "metrics": row.metrics,
                        "notes": row.notes,
                        "id": str(row.id),
                    }
                except Exception as exc:
                    log.exception("research train failed")
                    self._set("error", "Training failed.", str(exc))
                    return {"ok": False, "reason": str(exc)}
                finally:
                    self.status["busy"] = False

    async def snapshot(self) -> dict[str, Any]:
        async with SessionLocal() as db:
            latest = (
                await db.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(1))
            ).scalar_one_or_none()
            return {
                "status": dict(self.status),
                "candles": await candle_stats(db, timeframe().value),
                "samples": await sample_stats(db),
                "live": await live_stats(db),
                "active_model": _model_out(await active_model(db)),
                "last_run": _model_out(latest),
                "disclaimer": DISCLAIMER,
            }


research_service = ResearchService()
