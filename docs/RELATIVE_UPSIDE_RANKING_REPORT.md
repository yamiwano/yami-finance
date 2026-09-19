# Relative upside ranking report (`relative_upside_rank_12h_v1`)

Generated: **2026-09-19T06:51:55.138747+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`relative-upside-ranking-report.json`](relative-upside-ranking-report.json)

Prior experiments preserved:
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md) (bidirectional)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)

## Hypothesis

Given all eligible crypto assets at time T, can the model rank which assets will have the largest `future_max_upside_12h` over the next 12 hours?

This is a **relative ranking** problem, not binary +3% classification.

## Configuration

- Experiment ID: `relative_upside_rank_12h_v1`
- Timeframe / horizon: **1h / 12h**
- Primary stride: **12 bars** (non-overlapping)
- Training target: continuous `future_max_upside_12h` (`max_upside`)
- Evaluation target: `future_upside_rank_percentile` (1.0 = best within timestamp)
- Min cross-section size: **10**
- Features: full 36 vs ATR-only `atr_pct` vs momentum `ret_24`
- Model: HistGradientBoostingRegressor (no new architectures)

## Survivorship bias (prominent)

IMPORTANT LIMITATION: Universe = today's top-N Binance USDT pairs by 24h quote volume, applied retrospectively. This is not the full crypto market; delisted/illiquid names are missing. Survivorship/selection bias remains and can inflate apparent ranking stability.

## Cross-section coverage

| Metric | Value |
|--------|------:|
| Timestamps considered | 313 |
| Timestamps excluded (<10 assets) | 140 |
| Timestamps used | 173 |
| Avg cross-section size | 47.00 |
| Min / max cross-section | 47 / 47 |
| Asset observations used | 8131 |

## Aggregate comparison table

| Metric | Full Model | ATR-only | Momentum (ret_24) | Random / All-assets |
|--------|----------:|---------:|------------------:|--------------------:|
| spearman | 0.4624 | 0.5083 | 0.0652 | 0.0000 |
| top_1_mean_upside | 0.0627 | 0.0708 | 0.0554 | 0.0264 |
| top_3_mean_upside | 0.0574 | 0.0596 | 0.0443 | 0.0264 |
| top_5_mean_upside | 0.0494 | 0.0512 | 0.0403 | 0.0264 |
| top_20pct_mean_upside | 0.0424 | 0.0444 | 0.0341 | 0.0264 |
| top_1_hit_3pct | 0.5714 | 0.5714 | 0.5893 | 0.2869 |
| top_5_hit_3pct | 0.4643 | 0.5143 | 0.4071 | 0.2869 |
| top_5_hit_5pct | 0.3000 | 0.3107 | 0.2500 | 0.1432 |
| top_5_hit_10pct | 0.1250 | 0.1214 | 0.0679 | 0.0342 |

## Concentration (is top-5 just high-ATR?)

| Group | Mean ATR% | Mean future max upside | Mean future max drawdown |
|-------|----------:|-----------------------:|-------------------------:|
| Full model top-5 | 2.606 | 0.0494 | -0.0361 |
| ATR-only top-5 | 2.821 | 0.0512 | -0.0367 |
| All assets | 1.255 | 0.0264 | -0.0202 |

## Extreme upside (aggregate)

### Full model

- Top-1: mean=0.0627, median=0.0627, max=0.0627, hit3/5/10=0.5714/0.4286/0.1964
- Top-5: mean=0.0494, median=0.0316, max=0.1211, hit3/5/10=0.4643/0.3000/0.1250
- Top-20%: mean=0.0424, hit3/5/10=0.4429/0.2643/0.0875

## Walk-forward folds

### Fold 1: 2026-08-09T02:00:00+00:00 → 2026-08-22T14:00:00+00:00  
train 2026-06-24T14:00:00+00:00 → 2026-08-08T02:00:00+00:00; timestamps=28; obs=1316; avg CS=47.00  
Spearman=0.4424; top1/3/5 upside=0.0590/0.0581/0.0495; top5 hit3/5/10=0.4357/0.3071/0.1429

### Fold 2: 2026-08-23T02:00:00+00:00 → 2026-09-05T14:00:00+00:00  
train 2026-06-24T14:00:00+00:00 → 2026-08-22T02:00:00+00:00; timestamps=28; obs=1316; avg CS=47.00  
Spearman=0.4825; top1/3/5 upside=0.0663/0.0567/0.0494; top5 hit3/5/10=0.4929/0.2929/0.1071

## Evidence checklist / status

**Status: `weak_ranking_signal`** (never activated for live trading)

```
{
  "positive_aggregate_spearman": true,
  "consistent_positive_spearman_across_folds": true,
  "top5_beats_all_assets": true,
  "full_beats_atr_on_spearman_and_top5": false,
  "full_beats_momentum_on_spearman_and_top5": true,
  "effect_not_single_fold": true,
  "leakage_audit_passed": true,
  "fold_spearmans": [
    0.44236452937737847,
    0.4825337741365217
  ],
  "deltas": {
    "spearman_full_minus_atr": -0.04589847940055125,
    "spearman_full_minus_momentum": 0.39722910191361405,
    "top5_upside_full_minus_all": 0.023040057375939615,
    "top5_upside_full_minus_atr": -0.0017810296192431047,
    "top5_upside_full_minus_momentum": 0.009177300750621321
  }
}
```

## Interpretation

- **Does ranking information exist?** Aggregate Spearman=0.4624; status=`weak_ranking_signal`.
- **Do top-ranked assets realize larger upside than all-assets?** Top-5 mean upside=0.0494 vs all-assets=0.0264.
- **Does the full model beat ATR-only ranking?** False
- **Does the full model beat momentum (ret_24)?** True
- **Is this mostly volatility selection?** Likely still volatility-linked if ATR ranking is competitive or stronger; see concentration table (full top5 ATR%=2.606 vs ATR-only top5 ATR%=2.821).
- **Should we treat this as product-ready ranking edge?** No — research-only; not activated.
- **Continue to next research stage?** No — insufficient ranking edge beyond simple baselines on this evidence.

## Leakage

- Passed: **True**
- Findings: []
- Embargo violations: 0

## Lineage preservation

- Active bidirectional models preserved: **True**
- Experiment model status: `weak_ranking_signal`

## Tests

- Research suite OK: **True**

## Errors

- []
