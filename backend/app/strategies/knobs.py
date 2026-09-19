"""Default, clamped strategy knobs. The learner may nudge these; it does not invent new rules."""

from __future__ import annotations

from typing import Any

DEFAULT_KNOBS: dict[str, dict[str, float | bool]] = {
    "volume_confirmed_breakout": {
        "min_rel_vol": 1.65,
        "rsi_lo": 48.0,
        "rsi_hi": 76.0,
        "min_close_in_range": 0.62,
        "min_body": 0.38,
        "max_range_atr": 4.8,
        "stop_atr": 0.35,
    },
    "trend_pullback": {
        "rsi_lo": 38.0,
        "rsi_hi": 62.0,
        "max_rel_vol_dry": 1.25,
        "require_dry": False,
        "stop_atr": 0.25,
        "t1_r": 1.7,
    },
    "momentum_continuation": {
        "min_rel_vol": 1.15,
        "rsi_lo": 52.0,
        "rsi_hi": 78.0,
        "stop_atr": 0.15,
        "t1_r": 1.5,
    },
    "failed_breakout": {
        "min_rel_vol": 1.2,
        "min_wick": 0.28,
        "rsi_hi": 68.0,
        "stop_atr": 0.25,
        "t1_r": 1.6,
    },
    "parabolic_exhaustion": {
        "min_extension_atr": 2.6,
        "rsi_ob": 72.0,
        "climax_mult": 2.1,
        "fade_frac": 0.75,
        "require_roll_and_fade": False,
        "stop_atr": 0.25,
    },
    "breakdown_continuation": {
        "min_rel_vol": 1.25,
        "rsi_hi": 50.0,
        "stop_atr": 0.3,
        "t1_r": 1.6,
    },
}

KNOB_RANGE: dict[str, tuple[float, float]] = {
    "min_rel_vol": (1.05, 2.45),
    "max_rel_vol_dry": (0.85, 1.55),
    "rsi_lo": (32.0, 62.0),
    "rsi_hi": (48.0, 84.0),
    "rsi_ob": (66.0, 84.0),
    "min_close_in_range": (0.50, 0.80),
    "min_body": (0.25, 0.55),
    "max_range_atr": (3.2, 6.0),
    "stop_atr": (0.12, 0.55),
    "t1_r": (1.2, 2.2),
    "min_wick": (0.18, 0.48),
    "min_extension_atr": (1.8, 3.8),
    "climax_mult": (1.6, 2.8),
    "fade_frac": (0.55, 0.90),
}

KNOB_LABELS: dict[str, str] = {
    "min_rel_vol": "Min relative volume",
    "max_rel_vol_dry": "Pullback volume cap (dry)",
    "rsi_lo": "RSI floor",
    "rsi_hi": "RSI ceiling",
    "rsi_ob": "RSI overbought",
    "min_close_in_range": "Min close-in-bar",
    "min_body": "Min body %",
    "max_range_atr": "Max bar range (ATR)",
    "stop_atr": "Stop distance (ATR)",
    "t1_r": "Target 1 (R)",
    "min_wick": "Min rejection wick",
    "min_extension_atr": "Min extension (ATR)",
    "climax_mult": "Volume climax multiple",
    "fade_frac": "Volume fade fraction",
    "require_dry": "Require dry pullback",
    "require_roll_and_fade": "Require rollover and volume fade",
}

STANCE_LABELS = {
    "observe": "Watching",
    "working": "Holding",
    "salvage": "Needs work",
    "strict": "Firing less",
}


def resolve(strategy: str, overlay: dict[str, Any] | None = None) -> dict[str, Any]:
    base = dict(DEFAULT_KNOBS.get(strategy) or {})
    if overlay:
        for key, value in overlay.items():
            if key in base:
                base[key] = value
    return clamp_knobs(strategy, base)


def clamp_knobs(strategy: str, knobs: dict[str, Any]) -> dict[str, Any]:
    priors = DEFAULT_KNOBS.get(strategy) or {}
    out: dict[str, Any] = {}
    for key, prior in priors.items():
        value = knobs.get(key, prior)
        if isinstance(prior, bool):
            out[key] = bool(value)
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = float(prior)
        lo, hi = KNOB_RANGE.get(key, (num, num))
        out[key] = round(max(lo, min(hi, num)), 4)
    if "rsi_lo" in out and "rsi_hi" in out and float(out["rsi_lo"]) >= float(out["rsi_hi"]) - 6:
        out["rsi_hi"] = round(min(KNOB_RANGE["rsi_hi"][1], float(out["rsi_lo"]) + 8), 4)
    return out


def format_knob(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if key in {"min_rel_vol", "max_rel_vol_dry", "climax_mult"}:
        return f"{float(value):.2f}x"
    if "rsi" in key:
        return f"{float(value):.0f}"
    if key in {"min_close_in_range", "min_body", "min_wick", "fade_frac"}:
        return f"{float(value) * 100:.0f}%"
    return f"{float(value):.2f}"
