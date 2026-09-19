"""Forward labels from bars strictly after T."""

from __future__ import annotations

from app.domain import Bar


def label_forward(
    future_bars: list[Bar],
    entry_close: float,
    horizon_bars: int,
    move_pct: float,
) -> dict[str, float | bool] | None:
    """Labels use only bars after the decision bar. Incomplete windows return None."""
    if entry_close <= 0 or horizon_bars <= 0:
        return None
    window = future_bars[:horizon_bars]
    if len(window) < horizon_bars:
        return None
    last_close = window[-1].close
    max_high = max(b.high for b in window)
    min_low = min(b.low for b in window)
    fwd_return = last_close / entry_close - 1.0
    max_upside = max_high / entry_close - 1.0
    max_drawdown = min_low / entry_close - 1.0
    significant = bool(
        abs(fwd_return) >= move_pct or max_upside >= move_pct or max_drawdown <= -move_pct
    )
    return {
        "fwd_return": fwd_return,
        "max_upside": max_upside,
        "max_drawdown": max_drawdown,
        "significant_move": significant,
    }
