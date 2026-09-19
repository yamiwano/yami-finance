# Barrier probability report (`barrier_probability_v1`)

Generated: **2026-09-19T07:22:48.533055+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`BARRIER_PROBABILITY_REPORT.json`](BARRIER_PROBABILITY_REPORT.json)

Prior experiments preserved:
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)
- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)
- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)
- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)

## Hypothesis

Can the existing features predict whether an asset reaches an **upside barrier before** a downside barrier within 12h?

## Configuration

- Experiment ID: `barrier_probability_v1`
- Timeframe / horizon / stride: **1h / 12h / 12-bar**
- Min cross-section: **10**
- Model: HistGradientBoostingClassifier (success vs failure only)

## Target definitions

First-touch on future 1h OHLC. Same-candle dual touch ⇒ **ambiguous** (excluded).

- `up3_before_down2`: +3% before −2%
- `up5_before_down3`: +5% before −3%
- `up10_before_down5`: +10% before −5%

**Timeout/ambiguity:** TIMEOUT (neither barrier) and AMBIGUOUS are excluded from binary training/evaluation and reported separately.

## Survivorship bias (prominent)

IMPORTANT LIMITATION: Universe = today's top-N Binance USDT pairs by 24h quote volume, applied retrospectively. Delisted/illiquid names are missing; this does NOT represent the entire crypto market. Survivorship bias remains and is not fixed by this experiment.

## Cross-sectional coverage

| Metric | Value |
|--------|------:|
| Timestamps considered | 313 |
| Timestamps excluded | 140 |
| Timestamps used | 173 |
| Avg / min / max CS | 47.00 / 47 / 47 |
| Asset observations | 8131 |

## Class balance / event mix

### up3_before_down2 (+3% / −2%)  
total=8131; success=1728; failure=2652; timeout=3719 (0.457); ambiguous=32 (0.004); resolved=4380; success_rate_resolved=0.3945

### up5_before_down3 (+5% / −3%)  
total=8131; success=854; failure=1613; timeout=5652 (0.695); ambiguous=12 (0.001); resolved=2467; success_rate_resolved=0.3462

### up10_before_down5 (+10% / −5%)  
total=8131; success=216; failure=593; timeout=7319 (0.900); ambiguous=3 (0.000); resolved=809; success_rate_resolved=0.2670

## Per-barrier results

### up3_before_down2

| Metric | Full | ATR-only | Momentum | Full w/o ATR |
|--------|-----:|---------:|---------:|-------------:|
| AUC | 0.5194 | 0.5102 | 0.5345 | 0.5189 |
| PR-AUC | 0.4687 | 0.4520 | 0.4655 | 0.4655 |
| Brier | 0.2670 | 0.2640 | 0.2535 | 0.2665 |
| Baseline Brier | 0.2424 | 0.2424 | — | — |
| Log loss | 0.7322 | 0.7226 | — | — |
| Spearman(p, outcome) | 0.0321 | 0.0176 | 0.0595 | 0.0313 |
| Top-1 success | 0.3246 | 0.2829 | 0.3913 | 0.4141 |
| Top-5 success | 0.4224 | 0.3955 | 0.4551 | 0.4484 |
| Top-10% success | 0.4224 | 0.3955 | 0.4551 | 0.4484 |
| Top-20% success | 0.4183 | 0.4261 | 0.4445 | 0.4307 |
| All-assets success | 0.4424 | — | — | — |

**Expected value (diagnostic only; not trading return):**

```
{
  "assumption": "success=+3%, failure=-2%; diagnostic only; no leverage/compounding/sizing.",
  "cost_per_event": 0.002,
  "full_top5_ev_gross": 0.001118276014109347,
  "atr_top5_ev_gross": -0.00022321428571428492,
  "full_top5_ev_net_of_cost": -0.000881723985890653,
  "atr_top5_ev_net_of_cost": -0.002223214285714285
}
```

**Calibration (pooled, full model):**

```
[
  {
    "bucket": "0%-20%",
    "n": 27,
    "mean_pred": 0.16782145776149443,
    "observed": 0.4444444444444444
  },
  {
    "bucket": "20%-40%",
    "n": 342,
    "mean_pred": 0.32439976664106684,
    "observed": 0.42105263157894735
  },
  {
    "bucket": "40%-60%",
    "n": 761,
    "mean_pred": 0.502826732436328,
    "observed": 0.4178712220762155
  },
  {
    "bucket": "60%-80%",
    "n": 343,
    "mean_pred": 0.6711699960130761,
    "observed": 0.4606413994169096
  },
  {
    "bucket": "80%-100%",
    "n": 23,
    "mean_pred": 0.8287064832344491,
    "observed": 0.6086956521739131
  }
]
```

**Volatility buckets:**

- **low** n=815: base=0.4413, full_top5=0.4969, atr_top5=0.4543, mom_top5=0.5192, full_auc=0.4628, full_brier=0.2769
- **medium** n=661: base=0.4057, full_top5=0.3788, atr_top5=0.3859, mom_top5=0.4035, full_auc=0.4763, full_brier=0.2705
- **high** n=1156: base=0.4395, full_top5=0.4369, atr_top5=0.4054, mom_top5=0.4131, full_auc=0.5465, full_brier=0.2620

**Downside (top-5):**

- Full: mean=-0.0244, median=-0.0218, worst=-0.0454, ≤−3/5/10%=0.2929/0.1000/0.0071
- ATR: mean=-0.0318, worst=-0.0563, ≤−3/5/10%=0.4071/0.1893/0.0286

**Folds:**

- Fold 1 (2026-08-09T02:00:00+00:00 → 2026-08-22T14:00:00+00:00): full_auc=0.4857, atr_auc=0.5195, full_brier=0.2774, full_top5=0.4025, atr_top5=0.3804
- Fold 2 (2026-08-23T02:00:00+00:00 → 2026-09-05T14:00:00+00:00): full_auc=0.5530, atr_auc=0.5009, full_brier=0.2567, full_top5=0.4423, atr_top5=0.4107

### up5_before_down3

| Metric | Full | ATR-only | Momentum | Full w/o ATR |
|--------|-----:|---------:|---------:|-------------:|
| AUC | 0.5379 | 0.4949 | 0.5683 | 0.5424 |
| PR-AUC | 0.4600 | 0.3976 | 0.4614 | 0.4741 |
| Brier | 0.2635 | 0.2725 | 0.2487 | 0.2610 |
| Baseline Brier | 0.2282 | 0.2282 | — | — |
| Log loss | 0.7305 | 0.7443 | — | — |
| Spearman(p, outcome) | 0.0647 | -0.0091 | 0.1126 | 0.0728 |
| Top-1 success | 0.2784 | 0.3853 | 0.3681 | 0.3977 |
| Top-5 success | 0.4500 | 0.3440 | 0.3778 | 0.4441 |
| Top-10% success | 0.4500 | 0.3440 | 0.3778 | 0.4441 |
| Top-20% success | 0.4408 | 0.3780 | 0.4271 | 0.4307 |
| All-assets success | 0.4352 | — | — | — |

**Expected value (diagnostic only; not trading return):**

```
{
  "assumption": "success=+5%, failure=-3%; diagnostic only; no leverage/compounding/sizing.",
  "cost_per_event": 0.002,
  "full_top5_ev_gross": 0.005999999999999998,
  "atr_top5_ev_gross": -0.0024761904761904756,
  "full_top5_ev_net_of_cost": 0.003999999999999998,
  "atr_top5_ev_net_of_cost": -0.004476190476190476
}
```

**Calibration (pooled, full model):**

```
[
  {
    "bucket": "0%-20%",
    "n": 61,
    "mean_pred": 0.1493820919089923,
    "observed": 0.3770491803278688
  },
  {
    "bucket": "20%-40%",
    "n": 230,
    "mean_pred": 0.3129531029139764,
    "observed": 0.34782608695652173
  },
  {
    "bucket": "40%-60%",
    "n": 329,
    "mean_pred": 0.5048288913576754,
    "observed": 0.3556231003039514
  },
  {
    "bucket": "60%-80%",
    "n": 235,
    "mean_pred": 0.6832629218298969,
    "observed": 0.4553191489361702
  },
  {
    "bucket": "80%-100%",
    "n": 21,
    "mean_pred": 0.8284427243691537,
    "observed": 0.5714285714285714
  }
]
```

**Volatility buckets:**

- **low** n=815: base=0.4706, full_top5=0.6667, atr_top5=0.5714, mom_top5=0.5769, full_auc=0.4661, full_brier=0.2999
- **medium** n=661: base=0.3519, full_top5=0.3922, atr_top5=0.3468, mom_top5=0.3438, full_auc=0.6112, full_brier=0.2471
- **high** n=1156: base=0.3870, full_top5=0.4321, atr_top5=0.3767, mom_top5=0.4024, full_auc=0.5575, full_brier=0.2633

**Downside (top-5):**

- Full: mean=-0.0240, median=-0.0221, worst=-0.0443, ≤−3/5/10%=0.2714/0.1143/0.0143
- ATR: mean=-0.0312, worst=-0.0563, ≤−3/5/10%=0.3929/0.1679/0.0286

**Folds:**

- Fold 1 (2026-08-09T02:00:00+00:00 → 2026-08-22T14:00:00+00:00): full_auc=0.5633, atr_auc=0.4857, full_brier=0.2657, full_top5=0.4042, atr_top5=0.3381
- Fold 2 (2026-08-23T02:00:00+00:00 → 2026-09-05T14:00:00+00:00): full_auc=0.5126, atr_auc=0.5041, full_brier=0.2613, full_top5=0.4958, atr_top5=0.3500

### up10_before_down5

| Metric | Full | ATR-only | Momentum | Full w/o ATR |
|--------|-----:|---------:|---------:|-------------:|
| AUC | 0.5648 | 0.5302 | 0.5200 | 0.5743 |
| PR-AUC | 0.3426 | 0.3174 | 0.2995 | 0.3380 |
| Brier | 0.2545 | 0.2561 | 0.2630 | 0.2501 |
| Baseline Brier | 0.1896 | 0.1896 | — | — |
| Log loss | 0.7233 | 0.7163 | — | — |
| Spearman(p, outcome) | 0.0783 | 0.0483 | 0.0233 | 0.0915 |
| Top-1 success | 0.3095 | 0.3910 | 0.3205 | 0.1000 |
| Top-5 success | 0.3846 | 0.3685 | 0.3409 | 0.4028 |
| Top-10% success | 0.3846 | 0.3685 | 0.3409 | 0.4028 |
| Top-20% success | 0.4498 | 0.3733 | 0.3391 | 0.4161 |
| All-assets success | 0.3624 | — | — | — |

**Expected value (diagnostic only; not trading return):**

```
{
  "assumption": "success=+10%, failure=-5%; diagnostic only; no leverage/compounding/sizing.",
  "cost_per_event": 0.002,
  "full_top5_ev_gross": 0.007692307692307693,
  "atr_top5_ev_gross": 0.005273268398268388,
  "full_top5_ev_net_of_cost": 0.005692307692307693,
  "atr_top5_ev_net_of_cost": 0.003273268398268388
}
```

**Calibration (pooled, full model):**

```
[
  {
    "bucket": "0%-20%",
    "n": 92,
    "mean_pred": 0.1109731578194567,
    "observed": 0.14130434782608695
  },
  {
    "bucket": "20%-40%",
    "n": 69,
    "mean_pred": 0.30193774223182357,
    "observed": 0.2753623188405797
  },
  {
    "bucket": "40%-60%",
    "n": 65,
    "mean_pred": 0.5016397824433618,
    "observed": 0.3384615384615385
  },
  {
    "bucket": "60%-80%",
    "n": 63,
    "mean_pred": 0.7044945098242903,
    "observed": 0.3492063492063492
  },
  {
    "bucket": "80%-100%",
    "n": 23,
    "mean_pred": 0.8382015744519186,
    "observed": 0.30434782608695654
  }
]
```

**Volatility buckets:**

- **low** n=815: base=0.2222, full_top5=0.5000, atr_top5=0.5000, mom_top5=0.2000, full_auc=0.8036, full_brier=0.1463
- **medium** n=661: base=0.2564, full_top5=0.2500, atr_top5=0.1250, mom_top5=0.1667, full_auc=0.5483, full_brier=0.2378
- **high** n=1156: base=0.2706, full_top5=0.3377, atr_top5=0.3417, mom_top5=0.3333, full_auc=0.6073, full_brier=0.2519

**Downside (top-5):**

- Full: mean=-0.0220, median=-0.0204, worst=-0.0418, ≤−3/5/10%=0.2179/0.1107/0.0214
- ATR: mean=-0.0317, worst=-0.0610, ≤−3/5/10%=0.3786/0.2000/0.0286

**Folds:**

- Fold 1 (2026-08-09T02:00:00+00:00 → 2026-08-22T14:00:00+00:00): full_auc=0.4722, atr_auc=0.5455, full_brier=0.3252, full_top5=0.6154, atr_top5=0.5135
- Fold 2 (2026-08-23T02:00:00+00:00 → 2026-09-05T14:00:00+00:00): full_auc=0.6575, atr_auc=0.5149, full_brier=0.1838, full_top5=0.1538, atr_top5=0.2235

## Leakage audit

- Passed: **True**
- Findings: []
- Embargo violations: **0**

## Evidence / classification

**Status: `weak_barrier_signal`** (never activated)

```
{
  "leakage_audit_passed": true,
  "barriers_full_beats_atr_top5": 3,
  "barriers_full_beats_all_assets_top5": 2,
  "barriers_consistent_positive_folds": 1,
  "barriers_calibration_better_than_baseline": 0,
  "barriers_no_atr_retains_signal": 3,
  "barriers_positive_across_vol_buckets": 0,
  "barriers_topk_material_vs_all": 0,
  "n_barriers": 3
}
```

## Honest interpretation

Some barrier ranking ability exists, but it is not consistently better than ATR/momentum across barriers, folds, and volatility regimes under the strict criteria.

## Tests: **True**

## Errors: []
