"""Named research experiments. Keep contracts explicit so runs are comparable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    """Frozen lineage for one research run. Changing fields requires a new experiment_id."""

    experiment_id: str
    target_name: str
    secondary_target: str
    timeframe: str
    horizon_hours: int
    move_pct: float
    primary_stride_bars: int
    sensitivity_stride_bars: int
    feature_set: str  # "full" contract name; ATR set is always atr_feature_names
    atr_feature_names: tuple[str, ...] = ("atr_pct",)
    momentum_feature: str = "ret_24"
    promote_min_auc: float = 0.55
    # Full model must beat ATR-only by at least this AUC margin to count as evidence.
    atr_auc_margin: float = 0.02
    # Full model Brier must be strictly lower than ATR-only Brier.
    require_brier_better_than_atr: bool = True
    clean_drawdown_floor: float = -0.015
    min_cross_section_size: int = 10
    notes: str = ""
    version: str = "v1"

    def lineage(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "target_name": self.target_name,
            "secondary_target": self.secondary_target,
            "timeframe": self.timeframe,
            "horizon_hours": self.horizon_hours,
            "move_pct": self.move_pct,
            "primary_stride_bars": self.primary_stride_bars,
            "sensitivity_stride_bars": self.sensitivity_stride_bars,
            "feature_set": self.feature_set,
            "atr_feature_names": list(self.atr_feature_names),
            "momentum_feature": self.momentum_feature,
            "promote_min_auc": self.promote_min_auc,
            "atr_auc_margin": self.atr_auc_margin,
            "require_brier_better_than_atr": self.require_brier_better_than_atr,
            "clean_drawdown_floor": self.clean_drawdown_floor,
            "min_cross_section_size": self.min_cross_section_size,
            "notes": self.notes,
        }


# Prior bidirectional volatility/move experiment (reference; do not overwrite its artifacts).
BIDIRECTIONAL_V1 = ExperimentSpec(
    experiment_id="bidirectional_significant_move_v1",
    target_name="significant_move",
    secondary_target="",
    timeframe="1h",
    horizon_hours=12,
    move_pct=0.03,
    primary_stride_bars=6,
    sensitivity_stride_bars=6,
    feature_set="full_36",
    notes="Original bidirectional significant_move target with 6-bar stride.",
)

# Primary directional-up experiment for this controlled run.
DIRECTIONAL_UP_3PCT_12H_V1 = ExperimentSpec(
    experiment_id="directional_up_3pct_12h_v1",
    target_name="large_up_move",
    secondary_target="clean_up_move",
    timeframe="1h",
    horizon_hours=12,
    move_pct=0.03,
    primary_stride_bars=12,
    sensitivity_stride_bars=6,
    feature_set="full_36",
    notes=(
        "Primary target large_up_move=max_upside>=3% over 12h; primary stride 12 bars "
        "(non-overlapping label windows); sensitivity uses existing 6-bar stride."
    ),
)

# Cross-sectional relative upside ranking (product-aligned hypothesis).
RELATIVE_UPSIDE_RANK_12H_V1 = ExperimentSpec(
    experiment_id="relative_upside_rank_12h_v1",
    target_name="future_upside_rank_percentile",
    secondary_target="future_max_upside_12h",
    timeframe="1h",
    horizon_hours=12,
    move_pct=0.03,
    primary_stride_bars=12,
    sensitivity_stride_bars=6,
    feature_set="full_36",
    momentum_feature="ret_24",
    min_cross_section_size=10,
    notes=(
        "Cross-sectional ranking of future_max_upside_12h within each timestamp; "
        "evaluate Spearman / top-K vs ATR-only and ret_24 momentum; 12-bar stride."
    ),
)

# Volatility-adjusted upside: information beyond current ATR?
VOLATILITY_ADJUSTED_UPSIDE_V1 = ExperimentSpec(
    experiment_id="volatility_adjusted_upside_v1",
    target_name="vol_adj_residual",
    secondary_target="future_upside_atr_multiple",
    timeframe="1h",
    horizon_hours=12,
    move_pct=0.03,
    primary_stride_bars=12,
    sensitivity_stride_bars=6,
    feature_set="full_36",
    momentum_feature="ret_24",
    min_cross_section_size=10,
    notes=(
        "Primary target = cross-sectional OLS residual of log1p(future_max_upside) on "
        "log(atr_frac); secondary = upside/ATR multiple. Ablation without atr_pct."
    ),
)

# Risk-adjusted opportunity: upside quality relative to downside.
RISK_ADJUSTED_OPPORTUNITY_V1 = ExperimentSpec(
    experiment_id="risk_adjusted_opportunity_v1",
    target_name="log_risk_adjusted_opportunity",
    secondary_target="risk_adjusted_opportunity",
    timeframe="1h",
    horizon_hours=12,
    move_pct=0.03,
    primary_stride_bars=12,
    sensitivity_stride_bars=6,
    feature_set="full_36",
    momentum_feature="ret_24",
    min_cross_section_size=10,
    notes=(
        "Train on log1p(upside/max(|drawdown|,eps)); evaluate opportunity ranking, "
        "barriers, downside control, and ATR ablation."
    ),
)

EXPERIMENTS: dict[str, ExperimentSpec] = {
    BIDIRECTIONAL_V1.experiment_id: BIDIRECTIONAL_V1,
    DIRECTIONAL_UP_3PCT_12H_V1.experiment_id: DIRECTIONAL_UP_3PCT_12H_V1,
    RELATIVE_UPSIDE_RANK_12H_V1.experiment_id: RELATIVE_UPSIDE_RANK_12H_V1,
    VOLATILITY_ADJUSTED_UPSIDE_V1.experiment_id: VOLATILITY_ADJUSTED_UPSIDE_V1,
    RISK_ADJUSTED_OPPORTUNITY_V1.experiment_id: RISK_ADJUSTED_OPPORTUNITY_V1,
}


def get_experiment(experiment_id: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise KeyError(f"Unknown experiment_id={experiment_id!r}") from exc
