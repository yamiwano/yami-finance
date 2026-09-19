# Point-in-time universe report (`point_in_time_universe_v1` + `barrier_probability_v2`)

Generated: **2026-09-19T07:42:01.790499+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`POINT_IN_TIME_UNIVERSE_REPORT.json`](POINT_IN_TIME_UNIVERSE_REPORT.json)

Prior experiments preserved:
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)
- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)
- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)
- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)
- [`BARRIER_PROBABILITY_REPORT.md`](BARRIER_PROBABILITY_REPORT.md)

## Methodology

### Old universe
Today's top-N Binance USDT pairs by current 24h quote volume, applied retrospectively.

### New point-in-time universe
For each research timestamp T:
1. Eligible symbols = USDT spot pairs with ≥ LOOKBACK_BARS (80) candles at/before T.
2. Trailing 24h quote volume = Σ(close × volume) over bars in (T−24h, T].
3. Rank by trailing 24h quote volume; select top N (same N as old universe).

**Evidence strength:** historical existence is **inferred** from candle presence. Binance exchangeInfo is current-only; we do not have historical listing/delisting metadata from this provider. Symbols delisted before our data window cannot be reconstructed — this is a documented limitation, not a fabricated fix.

## Survivorship analysis

Point-in-time universe reduces retrospective selection bias by using only trailing candles/volume at each T. It does NOT eliminate survivorship: symbols delisted before our data window are absent, and historical listing status is inferred from candles (exchange metadata is current-only).

## Universe statistics

| Metric | Value |
|--------|------:|
| Timestamps | 313 |
| Avg universe size | 47.89 |
| Min / max size | 47 / 48 |
| Avg / median turnover | 0.00 / 0.00 |
| Avg 24h quote volume | 58947499 |
| Distinct symbols selected | 48 |

### Old vs new

- Old universe size: 48
- Avg new size: 47.89
- Avg overlap count: 47.89
- Avg overlap (% of new): 1.000
- Symbols entering (top): []
- Symbols leaving (top): [('SPYBUSDT', 33)]

## Dataset comparison

{
  "old_rows": 8271,
  "new_rows": 8271,
  "old_symbols": 48,
  "new_symbols": 48,
  "timestamps": 313
}

## Barrier v2 vs v1

### up3_before_down2

| Metric | v2 Full | v2 ATR | v1 Full | v1 ATR |
|--------|--------:|-------:|--------:|-------:|
| AUC | 0.5194 | 0.5102 | 0.5194 | 0.5102 |
| Brier | 0.2670 | 0.2640 | 0.2670 | 0.2640 |
| Top-5 success | 0.4224 | 0.3955 | 0.4224 | 0.3955 |
| Top-10% success | 0.4224 | 0.3955 | 0.4224 | 0.3955 |
| Full w/o ATR Spearman | 0.0313 | — | — | — |

### up5_before_down3

| Metric | v2 Full | v2 ATR | v1 Full | v1 ATR |
|--------|--------:|-------:|--------:|-------:|
| AUC | 0.5379 | 0.4949 | 0.5379 | 0.4949 |
| Brier | 0.2635 | 0.2725 | 0.2635 | 0.2725 |
| Top-5 success | 0.4500 | 0.3440 | 0.4500 | 0.3440 |
| Top-10% success | 0.4500 | 0.3440 | 0.4500 | 0.3440 |
| Full w/o ATR Spearman | 0.0728 | — | — | — |

### up10_before_down5

| Metric | v2 Full | v2 ATR | v1 Full | v1 ATR |
|--------|--------:|-------:|--------:|-------:|
| AUC | 0.5648 | 0.5302 | 0.5648 | 0.5302 |
| Brier | 0.2545 | 0.2561 | 0.2545 | 0.2561 |
| Top-5 success | 0.3846 | 0.3685 | 0.3846 | 0.3685 |
| Top-10% success | 0.3846 | 0.3685 | 0.3846 | 0.3685 |
| Full w/o ATR Spearman | 0.0915 | — | — | — |

## Leakage audit

- Passed: **True**
- Findings: []
- Embargo violations: **0**

## Final classification

**Signal classification: `signal_survives`**

Barrier ranking signal remains present on the point-in-time universe; survivorship reduction did not eliminate the effect. Research-only.

## Tests: **True**

## Errors: []
