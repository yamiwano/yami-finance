"""SQLite-friendly additive migrations for research experiment columns."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger("radar.research.schema")

_SAMPLE_COLUMNS = {
    "large_up_move": "BOOLEAN",
    "clean_up_move": "BOOLEAN",
    "sample_stride": "INTEGER",
}

_MODEL_COLUMNS = {
    "experiment_id": "VARCHAR(64)",
    "target_name": "VARCHAR(64)",
    "sample_stride": "INTEGER",
    "move_threshold": "FLOAT",
}

_PRED_COLUMNS = {
    "large_up_move": "BOOLEAN",
    "clean_up_move": "BOOLEAN",
}


def _existing_columns(sync_conn, table: str) -> set[str]:
    rows = sync_conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    # PRAGMA: cid, name, type, notnull, dflt_value, pk
    return {r[1] for r in rows}


def _add_missing(sync_conn, table: str, columns: dict[str, str]) -> list[str]:
    existing = _existing_columns(sync_conn, table)
    added: list[str] = []
    for name, coltype in columns.items():
        if name in existing:
            continue
        sync_conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}"))
        added.append(name)
    return added


def migrate_sqlite_research(sync_conn) -> dict[str, list[str]]:
    """Add experiment columns if missing. Safe to call repeatedly on SQLite."""
    dialect = sync_conn.dialect.name
    if dialect != "sqlite":
        # create_all handles new installs; Postgres ALTER is out of scope for this experiment.
        return {"dialect": [dialect], "skipped": ["non-sqlite"]}
    added = {
        "research_samples": _add_missing(sync_conn, "research_samples", _SAMPLE_COLUMNS),
        "model_versions": _add_missing(sync_conn, "model_versions", _MODEL_COLUMNS),
        "predictions": _add_missing(sync_conn, "predictions", _PRED_COLUMNS),
    }
    return added


async def ensure_research_schema(engine: AsyncEngine) -> dict:
    async with engine.begin() as conn:
        result = await conn.run_sync(migrate_sqlite_research)
    log.info("Research schema migrate: %s", result)
    return result
