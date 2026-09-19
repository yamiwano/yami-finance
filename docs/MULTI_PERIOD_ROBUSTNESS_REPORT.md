# Multi-period robustness report (`multi_period_robustness_v1`)

Generated: **2026-09-19T16:32:39.602659+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`MULTI_PERIOD_ROBUSTNESS_REPORT.json`](MULTI_PERIOD_ROBUSTNESS_REPORT.json)

Prior experiments preserved:
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)
- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)
- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)
- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)
- [`BARRIER_PROBABILITY_REPORT.md`](BARRIER_PROBABILITY_REPORT.md)
- [`POINT_IN_TIME_UNIVERSE_REPORT.md`](POINT_IN_TIME_UNIVERSE_REPORT.md)
- [`OUT_OF_SAMPLE_HOLDOUT_REPORT.md`](OUT_OF_SAMPLE_HOLDOUT_REPORT.md)
- [`PORTFOLIO_BACKTEST_REPORT.md`](PORTFOLIO_BACKTEST_REPORT.md)

## Objective

Determine whether the frozen barrier signal persists across multiple chronological market periods rather than one short holdout.

## Frozen specification

Same 36 features, barriers, model, universe, stride, horizon, costs, and execution as `barrier_probability_v2` + `portfolio_backtest_v1`. No tuning on results.

## Dataset integrity

- Timestamps considered: 488
- Timestamps used: 174
- Asset observations: 8173
- Avg cross-section: 46.97

## Period boundaries

- Period 1: 2026-06-24T14:00:00+00:00 → 2026-07-15T14:00:00+00:00 (43 timestamps, train=None, test=None)
- Period 2: 2026-07-16T02:00:00+00:00 → 2026-08-06T14:00:00+00:00 (44 timestamps, train=2021, test=2068)
- Period 3: 2026-08-07T02:00:00+00:00 → 2026-08-28T02:00:00+00:00 (43 timestamps, train=4089, test=2021)
- Period 4: 2026-08-28T14:00:00+00:00 → 2026-09-19T02:00:00+00:00 (44 timestamps, train=6110, test=2063)

## Per-period predictive metrics (Full model)

| Period | Barrier | AUC | Brier | Top-5 success | Top-10% success | All-assets |
|--------|---------|-----|-------|--------------:|----------------:|-----------:|
| 1 | up3_before_down2 | — | — | — | — | — |
| 1 | up5_before_down3 | — | — | — | — | — |
| 1 | up10_before_down5 | — | — | — | — | — |
| 2 | up3_before_down2 | 0.4971 | 0.2891 | 0.3909 | 0.4110 | 0.3616 |
| 2 | up5_before_down3 | 0.5271 | 0.2643 | 0.3421 | 0.2412 | 0.3401 |
| 2 | up10_before_down5 | 0.4660 | 0.3193 | 0.1600 | 0.1000 | 0.1705 |
| 3 | up3_before_down2 | 0.4698 | 0.2854 | 0.3674 | 0.3372 | 0.4096 |
| 3 | up5_before_down3 | 0.5564 | 0.2758 | 0.3333 | 0.3495 | 0.3409 |
| 3 | up10_before_down5 | 0.6830 | 0.2371 | 0.3231 | 0.3846 | 0.2893 |
| 4 | up3_before_down2 | 0.6116 | 0.2440 | 0.4744 | 0.4787 | 0.4326 |
| 4 | up5_before_down3 | 0.6304 | 0.2348 | 0.4537 | 0.4898 | 0.4441 |
| 4 | up10_before_down5 | 0.6542 | 0.2315 | 0.3059 | 0.2647 | 0.3030 |

## Per-period portfolio (top-5, full vs momentum vs ATR)

| Period | Barrier | Full net | Momentum net | ATR net | Full Sharpe | Full maxDD |
|--------|---------|---------:|-------------:|--------:|------------:|-----------:|
| 1 | up3_before_down2 | — | — | — | — | — |
| 1 | up5_before_down3 | — | — | — | — | — |
| 1 | up10_before_down5 | — | — | — | — | — |
| 2 | up3_before_down2 | -0.0831 | -0.0473 | 0.0206 | 0.0861 | -0.0866 |
| 2 | up5_before_down3 | -0.0670 | -0.0287 | 0.0940 | 0.0984 | -0.0675 |
| 2 | up10_before_down5 | -0.0700 | -0.0824 | 0.0968 | 0.1642 | -0.0933 |
| 3 | up3_before_down2 | -0.0858 | 0.0391 | -0.0559 | 0.0858 | -0.1242 |
| 3 | up5_before_down3 | -0.0620 | 0.0718 | -0.0393 | 0.0921 | -0.1245 |
| 3 | up10_before_down5 | -0.0321 | 0.0649 | 0.0645 | 0.0954 | -0.1052 |
| 4 | up3_before_down2 | 0.1203 | 0.0403 | 0.0219 | 0.2146 | -0.0697 |
| 4 | up5_before_down3 | 0.1052 | -0.0153 | -0.0531 | 0.1769 | -0.0584 |
| 4 | up10_before_down5 | 0.0369 | -0.0683 | -0.0054 | 0.0733 | -0.0744 |

## Combined chronological portfolio

{
  "up3_before_down2": {
    "n_trades": 583,
    "cumulative_net_return": -0.06089698937938359,
    "max_drawdown": -0.08582323591794172,
    "sharpe": 0.08955585339703434,
    "sortino": 0.8328423016213701,
    "profit_factor": 1.7304775459061228,
    "win_rate": 0.4648370497427101,
    "avg_trade_return": 0.007538383882937122,
    "period_net_returns": [
      -0.0830754042062265,
      -0.08582323591794172,
      0.12033894175903992
    ],
    "note": "Compounds period-level net returns; per-period portfolios are equal-capital normalized."
  },
  "up5_before_down3": {
    "n_trades": 580,
    "cumulative_net_return": -0.03272876468931463,
    "max_drawdown": -0.061955503158093395,
    "sharpe": 0.0999892477433425,
    "sortino": 0.7255071658529844,
    "profit_factor": 1.7691967002039433,
    "win_rate": 0.44310344827586207,
    "avg_trade_return": 0.009577359553571446,
    "period_net_returns": [
      -0.06700108636297752,
      -0.061955503158093395,
      0.10520717301419613
    ],
    "note": "Compounds period-level net returns; per-period portfolios are equal-capital normalized."
  },
  "up10_before_down5": {
    "n_trades": 571,
    "cumulative_net_return": -0.06657450678762,
    "max_drawdown": -0.032117488428768826,
    "sharpe": 0.11494936758586198,
    "sortino": 0.5772600944332482,
    "profit_factor": 2.021818657159304,
    "win_rate": 0.36777583187390545,
    "avg_trade_return": 0.013286077672538643,
    "period_net_returns": [
      -0.06995624522376764,
      -0.032117488428768826,
      0.03694001577419992
    ],
    "note": "Compounds period-level net returns; per-period portfolios are equal-capital normalized."
  }
}

## Robustness counts

{
  "auc_gt_050": 6,
  "auc_gt_055": 5,
  "full_gt_atr": 5,
  "full_gt_mom": 4,
  "full_gt_all": 6,
  "periods_evaluated": 3
}

## Leakage audit

- Passed: **True**
- Findings: []

## Multiple-testing limitation

This project has already tested multiple targets and model comparisons. This robustness experiment is confirmatory, not exploratory; historical results may contain selection effects from prior research.

## Survivorship limitation

Point-in-time universe reduces retrospective selection bias, but historical listing/delisting metadata is unavailable from this provider; symbols delisted before the data window are absent. Survivorship bias is reduced, not eliminated.

## Final classification

**`mixed_regime_signal`**

Signal appears in multiple periods but varies substantially by market environment.

## Tests: **True**

## Errors: []
