"""Point-in-time historical universe construction.

Selection at timestamp T uses ONLY candles with ts <= T (trailing window).
No future volume, listing status, delisting status, or candles are used.

Limitations (documented, not fabricated):
- Binance exchangeInfo is current-only; historical listing/delisting status is
  INFERRED from candle existence, not known from exchange metadata.
- We only have candles for symbols we can fetch today; symbols delisted before
  our data window cannot be reconstructed from this provider.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import Bar
from app.models.research import ResearchUniverseMembership
from app.research.history import as_utc, load_bars
from app.research.spec import LOOKBACK_BARS

SOURCE_KNOWN = "known"
SOURCE_INFERRED = "inferred"
SOURCE_UNKNOWN = "unknown"


def trailing_quote_volume_24h(bars: Sequence[Bar], ts: datetime) -> float | None:
    """Sum(close * volume) over bars with (ts - 24h, ts]. Uses only past data."""
    end = as_utc(ts)
    start = end - timedelta(hours=24)
    total = 0.0
    used = 0
    for bar in bars:
        bt = as_utc(bar.ts)
        if start < bt <= end:
            total += float(bar.close) * float(bar.volume)
            used += 1
    if used == 0:
        return None
    return float(total)


def warmup_bars_available(bars: Sequence[Bar], ts: datetime) -> int:
    """Count of closed bars at or before ts."""
    end = as_utc(ts)
    return sum(1 for b in bars if as_utc(b.ts) <= end)


def select_universe_at(
    bars_by_symbol: dict[str, list[Bar]],
    ts: datetime,
    *,
    top_n: int,
    warmup_bars: int = LOOKBACK_BARS,
) -> list[dict[str, Any]]:
    """Rank symbols by trailing 24h quote volume at ts; select top_n.

    A symbol is eligible only if it has >= warmup_bars candles at/before ts
    and a computable trailing 24h quote volume. Historical existence is
    INFERRED from candle presence (no exchange metadata history available).
    """
    rows: list[dict[str, Any]] = []
    for symbol, bars in bars_by_symbol.items():
        warm = warmup_bars_available(bars, ts)
        qv = trailing_quote_volume_24h(bars, ts)
        eligible = warm >= warmup_bars and qv is not None
        reason = "ok" if eligible else ("insufficient_warmup" if warm < warmup_bars else "no_trailing_volume")
        rows.append(
            {
                "ts": as_utc(ts),
                "symbol": symbol,
                "eligible": bool(eligible),
                "quote_volume_24h": qv,
                "warmup_bars": warm,
                "source": SOURCE_INFERRED,
                "reason": reason,
            }
        )
    eligible_rows = [r for r in rows if r["eligible"]]
    eligible_rows.sort(key=lambda r: (-(r["quote_volume_24h"] or 0.0), r["symbol"]))
    for rank, row in enumerate(eligible_rows, start=1):
        row["selection_rank"] = rank
        row["selected"] = rank <= top_n
    for row in rows:
        if not row["eligible"]:
            row["selection_rank"] = None
            row["selected"] = False
    return rows


async def build_universe_snapshots(
    db: AsyncSession,
    asset_ids: dict[str, Any],
    timestamps: Sequence[datetime],
    *,
    top_n: int,
    timeframe: str,
    warmup_bars: int = LOOKBACK_BARS,
) -> dict[str, Any]:
    """Persist point-in-time membership for each timestamp (idempotent)."""
    bars_by_symbol: dict[str, list[Bar]] = {}
    for symbol, aid in asset_ids.items():
        bars_by_symbol[symbol] = await load_bars(db, aid, timeframe)

    inserted = 0
    for ts in sorted({as_utc(t) for t in timestamps}):
        rows = select_universe_at(bars_by_symbol, ts, top_n=top_n, warmup_bars=warmup_bars)
        for row in rows:
            stmt = sqlite_insert(ResearchUniverseMembership).values(
                ts=row["ts"],
                symbol=row["symbol"],
                timeframe=timeframe,
                eligible=row["eligible"],
                selected=row["selected"],
                selection_rank=row["selection_rank"],
                quote_volume_24h=row["quote_volume_24h"],
                warmup_bars=row["warmup_bars"],
                source=row["source"],
                reason=row["reason"],
            )
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["ts", "symbol", "timeframe"]
            )
            result = await db.execute(stmt)
            if result.rowcount and result.rowcount > 0:
                inserted += 1
    await db.commit()
    return {"snapshots": len(set(timestamps)), "rows_inserted": inserted}


async def load_selected_symbols(db: AsyncSession, timeframe: str) -> dict[datetime, list[str]]:
    rows = (
        await db.execute(
            select(ResearchUniverseMembership.ts, ResearchUniverseMembership.symbol)
            .where(
                ResearchUniverseMembership.timeframe == timeframe,
                ResearchUniverseMembership.selected.is_(True),
            )
            .order_by(ResearchUniverseMembership.ts, ResearchUniverseMembership.selection_rank)
        )
    ).all()
    out: dict[datetime, list[str]] = {}
    for ts, symbol in rows:
        out.setdefault(as_utc(ts), []).append(symbol)
    return out


async def universe_stats(db: AsyncSession, timeframe: str) -> dict[str, Any]:
    rows = (
        await db.execute(
            select(ResearchUniverseMembership).where(
                ResearchUniverseMembership.timeframe == timeframe,
                ResearchUniverseMembership.selected.is_(True),
            )
        )
    ).scalars().all()
    by_ts: dict[datetime, list[Any]] = {}
    for row in rows:
        by_ts.setdefault(as_utc(row.ts), []).append(row)
    sizes = [len(v) for v in by_ts.values()]
    turnovers: list[int] = []
    prev: set[str] | None = None
    for ts in sorted(by_ts):
        current = {r.symbol for r in by_ts[ts]}
        if prev is not None:
            turnovers.append(len(current.symmetric_difference(prev)))
        prev = current
    vols = [r.quote_volume_24h for r in rows if r.quote_volume_24h is not None]
    freq = Counter(r.symbol for r in rows)
    return {
        "timestamps": len(by_ts),
        "average_universe_size": float(np.mean(sizes)) if sizes else None,
        "min_universe_size": int(min(sizes)) if sizes else None,
        "max_universe_size": int(max(sizes)) if sizes else None,
        "average_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "median_turnover": float(np.median(turnovers)) if turnovers else 0.0,
        "average_quote_volume_24h": float(np.mean(vols)) if vols else None,
        "median_quote_volume_24h": float(np.median(vols)) if vols else None,
        "most_frequent_symbols": freq.most_common(15),
        "distinct_symbols_selected": len(freq),
    }


def compare_universes(
    old_symbols: Sequence[str],
    new_by_ts: dict[datetime, list[str]],
) -> dict[str, Any]:
    old = set(old_symbols)
    overlaps = []
    entering: Counter[str] = Counter()
    leaving: Counter[str] = Counter()
    for _ts, symbols in sorted(new_by_ts.items()):
        current = set(symbols)
        overlaps.append(len(current & old))
        for s in current - old:
            entering[s] += 1
        for s in old - current:
            leaving[s] += 1
    avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    avg_size = float(np.mean([len(v) for v in new_by_ts.values()])) if new_by_ts else 0.0
    return {
        "old_universe_size": len(old),
        "average_new_size": avg_size,
        "average_overlap_count": avg_overlap,
        "average_overlap_pct_of_new": (avg_overlap / avg_size) if avg_size else None,
        "average_overlap_pct_of_old": (avg_overlap / len(old)) if old else None,
        "symbols_entering_top": entering.most_common(15),
        "symbols_leaving_top": leaving.most_common(15),
    }
