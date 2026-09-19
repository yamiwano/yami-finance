"""Realistic portfolio backtest on frozen barrier predictions.

Execution model:
- Signal at timestamp T uses features/universe available at T.
- Entry at NEXT 1h candle open (no same-candle execution).
- Exit on first barrier touch after entry, or horizon expiry.
- Same-candle dual touch => ambiguous => exit at that candle's open (conservative).
- Equal-weight positions, max 100% gross exposure, no leverage.
- Cost: 0.2% per completed trade (round-trip), per existing research assumption.

Frozen: features, targets, model, universe, holdout cutoff. No tuning on holdout.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from app.domain import Bar
from app.research.barriers import COST_PER_EVENT
from app.research.experiments import ExperimentSpec
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.opportunity import BARRIER_SPECS
from app.research.ranking import MIN_CROSS_SECTION, group_by_timestamp

STARTING_CAPITAL = 10_000.0
RANDOM_SEED = 42
RANDOM_SIMS = 100
HOLDOUT_CUTOFF = "2026-08-28T14:00:00+00:00"

STATUS_PLAUSIBLE = "economically_plausible"
STATUS_WEAK = "weak_economic_signal"
STATUS_NOT_SUPPORTED = "not_economically_supported"
STATUS_INSUFFICIENT = "insufficient_data"


def _feat(row: Any, name: str) -> float:
    feats = getattr(row, "features", None) or {}
    value = feats.get(name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _matrix(rows: Sequence[Any], names: Sequence[str]) -> np.ndarray:
    return np.array([[_feat(r, n) for n in names] for r in rows], dtype=float)


def _new_classifier() -> HistGradientBoostingClassifier:
    kwargs: dict[str, Any] = dict(
        max_depth=4,
        learning_rate=0.06,
        max_iter=140,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=42,
    )
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", **kwargs)
    except TypeError:
        return HistGradientBoostingClassifier(**kwargs)


def _positive_proba(clf: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(x)
    classes = list(clf.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    if 0 in classes and len(classes) == 1:
        return np.zeros(len(x), dtype=float)
    return proba[:, -1]


def _barrier_binary(row: Any, name: str) -> int | None:
    state = str(getattr(row, name, "unknown"))
    if state == "up_first":
        return 1
    if state == "down_first":
        return 0
    return None


@dataclass
class Position:
    symbol: str
    signal_ts: datetime
    entry_ts: datetime
    entry_price: float
    weight: float
    prob: float
    rank: int
    barrier: str
    up_pct: float
    down_pct: float
    horizon_bars: int
    entry_bar_index: int


@dataclass
class Trade:
    strategy: str
    barrier: str
    symbol: str
    signal_ts: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    prob: float
    rank: int
    weight: float
    gross_return: float
    cost: float
    net_return: float
    exit_reason: str
    holding_hours: float


@dataclass
class PortfolioState:
    cash: float = STARTING_CAPITAL
    equity: float = STARTING_CAPITAL
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)


def simulate_trade_exit(
    bars: Sequence[Bar],
    entry_index: int,
    entry_price: float,
    up_pct: float,
    down_pct: float,
    horizon_bars: int,
) -> tuple[int, float, str] | None:
    """Scan bars after entry_index for first barrier touch or horizon expiry.

    Returns (exit_index, exit_price, exit_reason). Same-candle dual touch => ambiguous,
    exit at that candle's open (conservative, no favorable choice).
    """
    up_level = entry_price * (1.0 + up_pct)
    down_level = entry_price * (1.0 - down_pct)
    last_index = min(entry_index + horizon_bars, len(bars) - 1)
    for j in range(entry_index + 1, last_index + 1):
        bar = bars[j]
        hit_up = bar.high >= up_level
        hit_down = bar.low <= down_level
        if hit_up and hit_down:
            return j, float(bar.open), "ambiguous"
        if hit_up:
            return j, float(up_level), "up_barrier"
        if hit_down:
            return j, float(down_level), "down_barrier"
    if last_index > entry_index:
        return last_index, float(bars[last_index].close), "timeout"
    return None


def run_portfolio_backtest(
    rows: Sequence[Any],
    bars_by_symbol: dict[str, list[Bar]],
    scores: np.ndarray,
    *,
    barrier: str,
    up_pct: float,
    down_pct: float,
    top_k: int,
    horizon_bars: int,
    cost: float = COST_PER_EVENT,
    slippage: float = 0.0,
    strategy: str = "full",
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Event-driven portfolio backtest over signal timestamps.

    At each timestamp: rank by score, select top_k, enter at next candle open,
    manage open positions, enforce <=100% exposure.
    """
    by_ts = group_by_timestamp(rows)
    score_by_id = {id(r): float(s) for r, s in zip(rows, scores)}
    ts_list = sorted(by_ts)
    # Map symbol -> {ts: index} for bar lookup
    bar_index: dict[str, dict[datetime, int]] = {}
    for sym, bars in bars_by_symbol.items():
        bar_index[sym] = {as_utc(b.ts): i for i, b in enumerate(bars)}

    state = PortfolioState()
    rng = np.random.default_rng(seed)
    total_cost_rate = cost + slippage

    for ts in ts_list:
        bucket = by_ts[ts]
        # 1) Exit positions whose exit condition is met at/before this ts
        still_open: list[Position] = []
        for pos in state.positions:
            bars = bars_by_symbol.get(pos.symbol) or []
            idx = bar_index.get(pos.symbol, {}).get(ts)
            if idx is None:
                still_open.append(pos)
                continue
            exit_info = simulate_trade_exit(
                bars, pos.entry_bar_index, pos.entry_price, pos.up_pct, pos.down_pct, pos.horizon_bars
            )
            if exit_info is None:
                still_open.append(pos)
                continue
            exit_idx, exit_price, reason = exit_info
            exit_ts = as_utc(bars[exit_idx].ts)
            if exit_ts > ts:
                still_open.append(pos)
                continue
            gross = exit_price / pos.entry_price - 1.0
            net = gross - total_cost_rate
            notional = pos.weight * STARTING_CAPITAL
            state.cash += notional * (1.0 + net)
            state.trades.append(
                Trade(
                    strategy=strategy,
                    barrier=barrier,
                    symbol=pos.symbol,
                    signal_ts=pos.signal_ts.isoformat(),
                    entry_ts=pos.entry_ts.isoformat(),
                    exit_ts=exit_ts.isoformat(),
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    prob=pos.prob,
                    rank=pos.rank,
                    weight=pos.weight,
                    gross_return=gross,
                    cost=total_cost_rate,
                    net_return=net,
                    exit_reason=reason,
                    holding_hours=(exit_ts - pos.entry_ts).total_seconds() / 3600.0,
                )
            )
        state.positions = still_open

        # 2) New signals: rank and select top_k
        eligible = [r for r in bucket if id(r) in score_by_id]
        if len(eligible) < MIN_CROSS_SECTION:
            state.equity = state.cash + sum(
                p.weight * STARTING_CAPITAL for p in state.positions
            )
            state.equity_curve.append(
                {
                    "ts": ts.isoformat(),
                    "equity": state.equity,
                    "cash": state.cash,
                    "gross_exposure": sum(p.weight for p in state.positions),
                    "n_positions": len(state.positions),
                }
            )
            continue
        if strategy == "random":
            order = rng.permutation(len(eligible))
        else:
            order = np.argsort([score_by_id[id(r)] for r in eligible])[::-1]
        selected = [eligible[i] for i in order[: max(1, min(top_k, len(eligible)))]]

        # 3) Enter at next candle open with available capital
        available = state.cash
        n_slots = max(0, len(selected))
        if n_slots and available > 0:
            weight_per = min(1.0 / n_slots, available / STARTING_CAPITAL / n_slots)
            weight_per = min(weight_per, 1.0 - sum(p.weight for p in state.positions))
            if weight_per <= 0:
                weight_per = 0.0
            for rank, row in enumerate(selected, start=1):
                if weight_per <= 0:
                    break
                bars = bars_by_symbol.get(row.symbol) or []
                idx = bar_index.get(row.symbol, {}).get(as_utc(row.ts))
                if idx is None or idx + 1 >= len(bars):
                    continue
                entry_bar = bars[idx + 1]
                entry_price = float(entry_bar.open)
                if entry_price <= 0:
                    continue
                notional = weight_per * STARTING_CAPITAL
                state.cash -= notional
                state.positions.append(
                    Position(
                        symbol=row.symbol,
                        signal_ts=as_utc(row.ts),
                        entry_ts=as_utc(entry_bar.ts),
                        entry_price=entry_price,
                        weight=weight_per,
                        prob=score_by_id[id(row)],
                        rank=rank,
                        barrier=barrier,
                        up_pct=up_pct,
                        down_pct=down_pct,
                        horizon_bars=horizon_bars,
                        entry_bar_index=idx + 1,
                    )
                )

        state.equity = state.cash + sum(p.weight * STARTING_CAPITAL for p in state.positions)
        state.equity_curve.append(
            {
                "ts": ts.isoformat(),
                "equity": state.equity,
                "cash": state.cash,
                "gross_exposure": sum(p.weight for p in state.positions),
                "n_positions": len(state.positions),
            }
        )

    # Force-close remaining positions at last available bar close
    for pos in state.positions:
        bars = bars_by_symbol.get(pos.symbol) or []
        if not bars:
            continue
        exit_price = float(bars[-1].close)
        exit_ts = as_utc(bars[-1].ts)
        gross = exit_price / pos.entry_price - 1.0
        net = gross - total_cost_rate
        notional = pos.weight * STARTING_CAPITAL
        state.cash += notional * (1.0 + net)
        state.trades.append(
            Trade(
                strategy=strategy,
                barrier=barrier,
                symbol=pos.symbol,
                signal_ts=pos.signal_ts.isoformat(),
                entry_ts=pos.entry_ts.isoformat(),
                exit_ts=exit_ts.isoformat(),
                entry_price=pos.entry_price,
                exit_price=exit_price,
                prob=pos.prob,
                rank=pos.rank,
                weight=pos.weight,
                gross_return=gross,
                cost=total_cost_rate,
                net_return=net,
                exit_reason="end_of_data",
                holding_hours=(exit_ts - pos.entry_ts).total_seconds() / 3600.0,
            )
        )
    state.positions = []
    state.equity = state.cash

    return summarize_portfolio(state, strategy=strategy, barrier=barrier, top_k=top_k, cost=total_cost_rate)


def summarize_portfolio(
    state: PortfolioState, *, strategy: str, barrier: str, top_k: int, cost: float
) -> dict[str, Any]:
    trades = state.trades
    equity = np.array([p["equity"] for p in state.equity_curve], dtype=float)
    if not len(equity):
        equity = np.array([STARTING_CAPITAL])
    returns = np.array([t.net_return for t in trades], dtype=float)
    gross_returns = np.array([t.gross_return for t in trades], dtype=float)

    ending = float(equity[-1])
    cumulative_net = ending / STARTING_CAPITAL - 1.0
    cumulative_gross = float(np.prod(1.0 + gross_returns) - 1.0) if len(gross_returns) else 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(np.min(dd)) if len(dd) else 0.0
    vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else None
    downside = returns[returns < 0]
    downside_dev = float(np.std(downside, ddof=1)) if len(downside) > 1 else None
    sharpe = float(np.mean(returns) / vol) if vol and vol > 0 else None
    sortino = (
        float(np.mean(returns) / downside_dev) if downside_dev and downside_dev > 0 else None
    )
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    profit_factor = (
        float(np.sum(wins) / abs(np.sum(losses))) if len(losses) and np.sum(losses) != 0 else None
    )
    exposures = [p["gross_exposure"] for p in state.equity_curve]
    positions = [p["n_positions"] for p in state.equity_curve]

    return {
        "strategy": strategy,
        "barrier": barrier,
        "top_k": top_k,
        "starting_capital": STARTING_CAPITAL,
        "ending_capital": ending,
        "cumulative_net_return": cumulative_net,
        "cumulative_gross_return": cumulative_gross,
        "cost_rate_per_trade": cost,
        "n_trades": len(trades),
        "n_completed": len(trades),
        "win_rate": float(np.mean(returns > 0)) if len(returns) else None,
        "avg_trade_return": float(np.mean(returns)) if len(returns) else None,
        "median_trade_return": float(np.median(returns)) if len(returns) else None,
        "avg_winner": float(np.mean(wins)) if len(wins) else None,
        "avg_loser": float(np.mean(losses)) if len(losses) else None,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "volatility": vol,
        "downside_deviation": downside_dev,
        "sharpe": sharpe,
        "sortino": sortino,
        "avg_holding_hours": float(np.mean([t.holding_hours for t in trades])) if trades else None,
        "turnover": float(len(trades) / max(1, len(state.equity_curve))),
        "avg_simultaneous_positions": float(np.mean(positions)) if positions else None,
        "max_simultaneous_positions": int(max(positions)) if positions else 0,
        "peak_equity": float(np.max(equity)),
        "equity_curve": state.equity_curve,
        "trade_ledger": [t.__dict__ for t in trades],
    }


def fit_frozen_models(
    dev_rows: Sequence[Any],
    barrier: str,
) -> dict[str, Any]:
    """Train frozen classifiers on development period only."""
    full_names = feature_names()
    no_atr_names = [n for n in full_names if n != "atr_pct"]
    y = np.array(
        [_barrier_binary(r, barrier) if _barrier_binary(r, barrier) is not None else -1 for r in dev_rows],
        dtype=int,
    )
    mask = y >= 0
    if mask.sum() < 50 or len(set(y[mask].tolist())) < 2:
        return {}
    idx = np.where(mask)[0]
    y_bin = y[mask].astype(int)

    def _fit(names: list[str]):
        x = _matrix(dev_rows, names)
        clf = _new_classifier()
        clf.fit(x[idx], y_bin)
        return clf, names

    return {
        "full": _fit(full_names),
        "atr_only": _fit(["atr_pct"]),
        "momentum": _fit(["ret_24"]),
        "full_without_atr": _fit(no_atr_names),
    }


def predict_scores(models: dict[str, Any], rows: Sequence[Any]) -> dict[str, np.ndarray]:
    out = {}
    for name, (clf, names) in models.items():
        x = _matrix(rows, names)
        out[name] = _positive_proba(clf, x)
    return out


def classify_portfolio(holdout_results: dict[str, Any]) -> str:
    """Objective classification based on holdout net returns vs baselines."""
    if not holdout_results:
        return STATUS_INSUFFICIENT
    full_net = []
    random_net = []
    for _barrier, by_k in holdout_results.items():
        full = (by_k.get("full") or {}).get("top_5") or {}
        rand = (by_k.get("random") or {}).get("top_5") or {}
        if full.get("cumulative_net_return") is not None:
            full_net.append(full["cumulative_net_return"])
        if rand.get("cumulative_net_return") is not None:
            random_net.append(rand["cumulative_net_return"])
    if not full_net:
        return STATUS_INSUFFICIENT
    positive = sum(1 for v in full_net if v > 0)
    beats_random = sum(
        1 for f, r in zip(full_net, random_net) if r is not None and f > r
    )
    if positive == len(full_net) and beats_random >= max(1, len(full_net) - 1):
        return STATUS_PLAUSIBLE
    if positive >= 1:
        return STATUS_WEAK
    return STATUS_NOT_SUPPORTED
