"""Forward labels from bars strictly after T."""

from __future__ import annotations

from app.domain import Bar

# Secondary "clean upward" adverse-move floor (not the primary target).
CLEAN_DRAWDOWN_FLOOR = -0.015


def label_forward(
    future_bars: list[Bar],
    entry_close: float,
    horizon_bars: int,
    move_pct: float,
    *,
    clean_drawdown_floor: float = CLEAN_DRAWDOWN_FLOOR,
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
    large_up_move = bool(max_upside >= move_pct)
    clean_up_move = bool(large_up_move and max_drawdown > clean_drawdown_floor)
    return {
        "fwd_return": fwd_return,
        "max_upside": max_upside,
        "max_drawdown": max_drawdown,
        "significant_move": significant,
        "large_up_move": large_up_move,
        "clean_up_move": clean_up_move,
    }


def labels_from_outcomes(
    fwd_return: float | None,
    max_upside: float | None,
    max_drawdown: float | None,
    move_pct: float,
    *,
    clean_drawdown_floor: float = CLEAN_DRAWDOWN_FLOOR,
) -> dict[str, bool] | None:
    """Recompute classification targets from stored continuous outcomes."""
    if fwd_return is None or max_upside is None or max_drawdown is None:
        return None
    significant = bool(
        abs(fwd_return) >= move_pct or max_upside >= move_pct or max_drawdown <= -move_pct
    )
    large_up_move = bool(max_upside >= move_pct)
    clean_up_move = bool(large_up_move and max_drawdown > clean_drawdown_floor)
    return {
        "significant_move": significant,
        "large_up_move": large_up_move,
        "clean_up_move": clean_up_move,
    }
