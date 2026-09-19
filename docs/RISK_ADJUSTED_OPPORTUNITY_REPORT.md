# Risk-adjusted opportunity report (`risk_adjusted_opportunity_v1`)

Generated: **2026-09-19T07:13:06.590272+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`RISK_ADJUSTED_OPPORTUNITY_REPORT.json`](RISK_ADJUSTED_OPPORTUNITY_REPORT.json)

Prior experiments preserved:
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)
- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)
- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)

## Hypothesis

Can the existing feature set identify assets whose **future upside is attractive relative to future downside**, beyond ATR and momentum baselines?

## Why this experiment exists

Prior work showed volatility dominates raw upside ranking. This experiment asks a different question: opportunity quality = upside / downside, not max upside alone.

## Configuration

- Experiment ID: `risk_adjusted_opportunity_v1`
- Timeframe / horizon / stride: **1h / 12h / 12-bar**
- Min cross-section: **10**
- Model: HistGradientBoostingRegressor on `log_risk_adjusted_opportunity`

## Target definitions

```
eps = 0.001
risk_adjusted_opportunity = future_max_upside_12h / max(|future_max_drawdown_12h|, eps)
log_risk_adjusted_opportunity = log1p(risk_adjusted_opportunity)   # training target
upside_downside_spread = future_max_upside_12h - |future_max_drawdown_12h|
```

## Barrier definitions

First-touch on future 1h OHLC only. Same-candle dual hit ⇒ `ambiguous` (excluded from rates).

- `up3_before_down2`: +3% before −2%
- `up5_before_down3`: +5% before −3%
- `up10_before_down5`: +10% before −5%

Barrier rows computed: 8271; ambiguous events: 47

## Survivorship bias (prominent)

IMPORTANT LIMITATION: Universe = today's top-N Binance USDT pairs by 24h quote volume, applied retrospectively. Delisted/illiquid names are missing. This does NOT represent the entire crypto market. Survivorship bias is documented, not fixed, here.

## Cross-sectional coverage

| Metric | Value |
|--------|------:|
| Timestamps considered | 313 |
| Timestamps excluded | 140 |
| Timestamps used | 173 |
| Avg / min / max CS | 47.00 / 47 / 47 |
| Asset observations | 8131 |

## Aggregate comparison

| Metric | Full | ATR-only | Momentum | Full w/o ATR | All-assets |
|--------|-----:|---------:|---------:|-------------:|-----------:|
| spearman_opportunity | 0.0852 | 0.0495 | 0.0667 | 0.0908 | 0.0000 |
| top_5_mean_opportunity | 6.9893 | 5.0318 | 5.6922 | 6.5446 | 5.5246 |
| top_5_mean_raw_upside | 0.0336 | 0.0388 | 0.0294 | 0.0334 | 0.0264 |
| top_5_mean_future_return | 0.0054 | 0.0076 | 0.0053 | 0.0056 | 0.0050 |
| top_5_mean_drawdown | -0.0243 | -0.0298 | -0.0229 | -0.0244 | -0.0202 |
| top_5_up3_before_down2 | 0.2973 | 0.3152 | 0.2723 | 0.2750 | 0.2464 |
| top_5_up5_before_down3 | 0.1857 | 0.2000 | 0.1643 | 0.1607 | 0.1288 |
| top_5_up10_before_down5 | 0.0464 | 0.0607 | 0.0357 | 0.0464 | 0.0315 |

**Full − ATR opportunity Spearman:** 0.0357  
**Full − ATR top-5 opportunity:** 1.9575  
**Full − ATR top-5 raw upside:** -0.0052  
**Full − ATR top-5 future return:** -0.0022  
**Full − ATR top-5 drawdown:** 0.0055

## Walk-forward folds

### Fold 1: 2026-08-09T02:00:00+00:00 → 2026-08-22T14:00:00+00:00  
train 2026-06-24T14:00:00+00:00 → 2026-08-08T02:00:00+00:00; ts=28; obs=1316  
Full opp Spearman=0.0717; ATR=0.0490; Full top5 opp=8.0450; ATR top5 opp=5.9256; Full top5 dd=-0.0223; ATR top5 dd=-0.0264

### Fold 2: 2026-08-23T02:00:00+00:00 → 2026-09-05T14:00:00+00:00  
train 2026-06-24T14:00:00+00:00 → 2026-08-22T02:00:00+00:00; ts=28; obs=1316  
Full opp Spearman=0.0987; ATR=0.0500; Full top5 opp=5.9337; ATR top5 opp=4.1380; Full top5 dd=-0.0262; ATR top5 dd=-0.0332

## Volatility bucket analysis

- **low** (n=815): full_sp=0.3280, atr_sp=0.3007, mom=0.2974, noATR=0.3364; top5 opp/up/ret/dd=5.3539/0.0136/0.0029/-0.0091; barriers 3/5/10=0.0893/0.0357/0.0071
- **medium** (n=661): full_sp=-0.0338, atr_sp=-0.0638, mom=-0.0477, noATR=-0.0366; top5 opp/up/ret/dd=5.7484/0.0214/0.0048/-0.0168; barriers 3/5/10=0.2104/0.0839/0.0107
- **high** (n=1156): full_sp=-0.0311, atr_sp=-0.0585, mom=-0.0547, noATR=-0.0144; top5 opp/up/ret/dd=6.3256/0.0400/0.0059/-0.0298; barriers 3/5/10=0.3393/0.2250/0.0679

## Downside-control analysis

| Group | Mean DD | Median DD | Worst DD | ≤−3% | ≤−5% | ≤−10% |
|-------|--------:|----------:|---------:|-----:|-----:|------:|
| Full top-5 | -0.0243 | -0.0206 | -0.0481 | 0.2714 | 0.1250 | 0.0143 |
| ATR top-5 | -0.0298 | -0.0262 | -0.0559 | 0.3857 | 0.1821 | 0.0250 |
| All assets | -0.0202 | — | — | — | — | — |

## ATR-normalized opportunity diagnostic

{
  "full_top5_norm_upside": 2.1953590761702984,
  "atr_top5_norm_upside": 1.957777198913373,
  "full_top5_norm_drawdown": 1.5633569333087032,
  "atr_top5_norm_drawdown": 1.6658604991511372,
  "full_top5_norm_opp": 554.8954213711877,
  "atr_top5_norm_opp": 326.9594863234742
}

## Leakage audit

- Passed: **True**
- Findings: []
- Embargo violations: **0**

## Evidence / classification

**Status: `weak_opportunity_signal`** (never activated)

```
{
  "positive_aggregate_spearman": true,
  "consistent_positive_across_folds": true,
  "full_beats_atr_spearman": true,
  "full_beats_atr_top5_opportunity": true,
  "material_spearman_margin_vs_atr": true,
  "full_without_atr_retains_signal": true,
  "downside_not_materially_worse_vs_atr": true,
  "barrier_improves_vs_atr": false,
  "positive_across_vol_buckets": false,
  "leakage_audit_passed": true,
  "fold_spearmans": [
    0.07173645789133011,
    0.09870388188270515
  ],
  "deltas": {
    "spearman_full_minus_atr": 0.03568562631269999,
    "top5_opp_full_minus_atr": 1.9575333471533014,
    "top5_raw_upside_full_minus_atr": -0.005205322090896955,
    "top5_return_full_minus_atr": -0.002167017306579892,
    "top5_drawdown_full_minus_atr": 0.005518328539877056
  }
}
```

## Honest interpretation

Some positive opportunity ranking exists (full−ATR Spearman=0.0357, top5 opp Δ=1.9575), but it is not decisive beyond volatility baselines under the success criteria.

## Tests: **True**

## Errors: []
