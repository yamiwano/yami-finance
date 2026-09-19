# Volatility-adjusted upside report (`volatility_adjusted_upside_v1`)

Generated: **2026-09-19T07:01:15.839609+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.json`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.json)

Prior experiments preserved (unchanged):
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)
- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)

## Hypothesis

Can the existing 36-feature set predict **unusually large upside after controlling for current volatility**, or is apparent ranking skill mostly ATR?

## Why this experiment exists

Prior relative-ranking found a real signal, but ATR-only beat the full model (Spearman ~0.508 vs ~0.462). This experiment isolates upside *beyond* volatility.

## Configuration

- Experiment ID: `volatility_adjusted_upside_v1`
- Timeframe / horizon / stride: **1h / 12h / 12-bar**
- Min cross-section: **10**
- Model: HistGradientBoostingRegressor
- Features: full 36, ATR-only, momentum `ret_24`, full-without-ATR

## Target definitions (mathematical)

### Secondary — ATR-normalized upside

```
atr_frac = atr_pct / 100
future_upside_atr_multiple = future_max_upside_12h / max(atr_frac, eps)
eps = 1e-06
```

### Primary — cross-sectional volatility-adjusted residual

At each timestamp T, for all eligible assets:

```
log_up_i  = log(1 + max(future_max_upside_12h_i, 0))
log_atr_i = log(max(atr_frac_i, eps))
Fit OLS within T only:  log_up = α_T + β_T * log_atr
vol_adj_residual_i = log_up_i - (α̂_T + β̂_T * log_atr_i)
```

Higher residual ⇒ more upside than the **same-timestamp** volatility schedule implies.
Residuals / future upside are **targets only** — never features.

Mean OLS α across timestamps: **0.0783**; mean β: **0.0114**

## Survivorship bias (prominent)

IMPORTANT LIMITATION: Universe = today's top-N Binance USDT pairs by 24h quote volume, applied retrospectively. Delisted/illiquid names are missing. This does NOT represent the entire crypto market. Survivorship bias is documented, not fixed, in this experiment.

## Cross-sectional coverage

| Metric | Value |
|--------|------:|
| Timestamps considered | 313 |
| Timestamps excluded | 140 |
| Timestamps used | 173 |
| Avg / min / max CS size | 47.00 / 47 / 47 |
| Asset observations | 8131 |

## Aggregate comparison (primary = vol-adj residual ranking)

| Metric | Full | ATR-only | Momentum | Full w/o ATR | All-assets |
|--------|-----:|---------:|---------:|-------------:|-----------:|
| spearman | 0.1491 | 0.1480 | 0.1264 | 0.1390 | 0.0000 |
| top_5_mean_raw_upside | 0.0383 | 0.0311 | 0.0299 | 0.0361 | 0.0264 |
| top_5_mean_vol_adj | 0.0160 | 0.0115 | 0.0105 | 0.0138 | -0.0000 |
| top_5_hit_3pct | 0.3464 | 0.2679 | 0.3071 | 0.3821 | 0.2869 |
| top_5_hit_5pct | 0.2500 | 0.1786 | 0.1857 | 0.2643 | 0.1432 |
| top_5_hit_10pct | 0.1036 | 0.0929 | 0.0643 | 0.1036 | 0.0342 |

**Full − ATR Spearman:** 0.0011  
**Full-without-ATR − ATR Spearman:** -0.0090  
**Full − ATR top-5 vol-adj:** 0.0045  
**Full − ATR top-5 raw upside:** 0.0072

## Walk-forward folds

### Fold 1: test 2026-08-09T02:00:00+00:00 → 2026-08-22T14:00:00+00:00  
train 2026-06-24T14:00:00+00:00 → 2026-08-08T02:00:00+00:00; ts=28; obs=1316; avg CS=47.00  
Full Spearman=0.1463; ATR Spearman=0.1551; Full top5 raw=0.0433; ATR top5 raw=0.0343; Full top5 vol-adj=0.0175; ATR top5 vol-adj=0.0127

### Fold 2: test 2026-08-23T02:00:00+00:00 → 2026-09-05T14:00:00+00:00  
train 2026-06-24T14:00:00+00:00 → 2026-08-22T02:00:00+00:00; ts=28; obs=1316; avg CS=47.00  
Full Spearman=0.1519; ATR Spearman=0.1408; Full top5 raw=0.0332; ATR top5 raw=0.0279; Full top5 vol-adj=0.0145; ATR top5 vol-adj=0.0104

## Volatility bucket analysis

- **low** (n=815): full Spearman=0.4556, ATR=0.4663, mom=0.4373; top5 raw full/ATR=0.0081/0.0044; top5 vol-adj full/ATR=0.0046/0.0024
- **medium** (n=661): full Spearman=-0.0273, ATR=-0.0336, mom=0.0163; top5 raw full/ATR=0.0221/0.0223; top5 vol-adj full/ATR=-0.0041/-0.0043
- **high** (n=1156): full Spearman=-0.0016, ATR=0.0667, mom=0.0269; top5 raw full/ATR=0.0497/0.0482; top5 vol-adj full/ATR=0.0118/0.0102

## Matched-volatility analysis

For each model top-5 asset, matched nearest non-top peer with |Δatr_frac|/atr_frac <= 0.15.

- Matched pairs: **112** (skipped=168)
- Mean selected upside: **0.0540**
- Mean matched peer upside: **0.0427**
- Mean difference (selected − peer): **0.0113**
- Median difference: **0.0010**
- Fraction selected higher: **0.5357**

## Extreme upside (full model)

- Top-1: mean=0.0492, median=0.0492, max=0.0492, hit3=0.4643
- Top-5: mean=0.0383, median=0.0213, max=0.1095, hit3/5/10=0.3464/0.2500/0.1036
- Top-20%: mean=0.0354, hit3/5/10=0.3429/0.2179/0.0821

## Leakage audit

- Passed: **True**
- Findings: []
- Embargo violations: **0**

## Evidence / classification

**Status: `weak_volatility_adjusted_signal`** (never activated)

```
{
  "positive_aggregate_spearman": true,
  "consistent_positive_across_folds": true,
  "full_beats_atr_on_spearman_and_top5_vol_adj": true,
  "material_margin_vs_atr": false,
  "full_without_atr_retains_ranking": true,
  "positive_across_vol_buckets": false,
  "matched_vol_selected_higher": true,
  "leakage_audit_passed": true,
  "fold_spearmans": [
    0.14633929468056378,
    0.15187148762009167
  ],
  "deltas": {
    "spearman_full_minus_atr": 0.0011355688879518544,
    "spearman_full_without_atr_minus_atr": -0.008958022380841618,
    "top5_vol_adj_full_minus_atr": 0.004498990332424457,
    "top5_raw_full_minus_atr": 0.007157631139941983
  }
}
```

## Honest interpretation

There is a positive vol-adjusted ranking signal (full−ATR Spearman=0.0011; full-without-ATR−ATR=-0.0090), but it is not strong/consistent enough to claim clear independent information beyond volatility under the success criteria.

## Lineage preservation

- Active bidirectional preserved: **True**
- Experiment status: `weak_volatility_adjusted_signal`

## Tests: **True**

## Errors: []
