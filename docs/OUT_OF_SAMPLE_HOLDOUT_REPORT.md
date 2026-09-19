# Out-of-sample holdout report (`out_of_sample_holdout_v1`)

Generated: **2026-09-19T07:54:50.386630+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`OUT_OF_SAMPLE_HOLDOUT_REPORT.json`](OUT_OF_SAMPLE_HOLDOUT_REPORT.json)

Prior experiments preserved:
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)
- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)
- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)
- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)
- [`BARRIER_PROBABILITY_REPORT.md`](BARRIER_PROBABILITY_REPORT.md)
- [`POINT_IN_TIME_UNIVERSE_REPORT.md`](POINT_IN_TIME_UNIVERSE_REPORT.md)

## Objective

Determine whether the existing barrier-ranking signal generalizes to a genuinely untouched future holdout. No model/target/feature changes were made.

## Frozen specification

- Same 36 features, same barriers, same HistGradientBoostingClassifier, same point-in-time universe, same stride/horizon as `barrier_probability_v2`.

## Development / holdout cutoff

- Cutoff: **2026-08-28T14:00:00+00:00**
- Development: 2026-06-24T14:00:00+00:00 → 2026-08-28T02:00:00+00:00 (6110 rows, 130 timestamps)
- Holdout: 2026-08-28T14:00:00+00:00 → 2026-09-18T14:00:00+00:00 (2021 rows, 43 timestamps)
- Embargo dropped development rows: 0

## Reproducibility

- Code commit: `1d21d83c3fecde94837c7f6e4e0276afe2a89b88`
- Random seed: 42
- Bootstrap seed: 42 (n=1000)
- Cost per event: 0.002
- Feature count: 36

## Survivorship limitation

Holdout uses the point-in-time universe (trailing 24h volume). Survivorship is reduced but not eliminated: symbols delisted before the data window are absent, and historical listing status is inferred from candles.

## Holdout results per barrier

### up3_before_down2

| Metric | Holdout Full | Holdout ATR | Holdout Mom | Full w/o ATR | Dev Full |
|--------|-------------:|------------:|------------:|-------------:|---------:|
| AUC | 0.6051 | 0.5086 | 0.5154 | 0.6096 | 0.8717 |
| Brier | 0.2426 | 0.2579 | 0.2572 | 0.2412 | 0.1678 |
| Log loss | 0.6792 | — | — | — | 0.5160 |
| Spearman | 0.1784 | 0.0146 | 0.0261 | 0.1859 | 0.6287 |
| Top-5 success | 0.4540 | 0.4047 | 0.3857 | 0.4635 | 0.6846 |
| Top-10% success | 0.4540 | 0.4047 | 0.3857 | 0.4635 | 0.7514 |
| All-assets success | 0.4398 | — | — | — | — |
| Random top-5 success | 0.3647 | — | — | — | — |

**Bootstrap (full):** top5={"mean": 0.4682518751436395, "ci_low": 0.3602272727272727, "ci_high": 0.5856439393939393}, top10%={"mean": 0.45470800896767355, "ci_low": 0.3741011494252873, "ci_high": 0.5413657407407406}, spearman={"mean": 0.1751270755704752, "ci_low": 0.03944837915277358, "ci_high": 0.29000733991219735}

**Economic (full top-5):** gross=0.0178, net=0.0158, win_rate=0.5442, avg_winner=0.0519, avg_loser=-0.0229, max_dd=-0.5218, n=215

**EV diagnostic:** {"assumption": "success=+3%, failure=-2%; diagnostic only.", "cost_per_event": 0.002, "full_top5_ev_net": 0.0006984126984126964, "atr_top5_ev_net": -0.0017674418604651174}

**Volatility buckets:**

- **low** n=455: full_top5=0.5227, atr_top5=0.4750, mom_top5=0.4917, all=0.4710
- **medium** n=571: full_top5=0.4789, atr_top5=0.4231, mom_top5=0.4368, all=0.4719
- **high** n=995: full_top5=0.4264, atr_top5=0.4101, mom_top5=0.3795, all=0.4135

### up5_before_down3

| Metric | Holdout Full | Holdout ATR | Holdout Mom | Full w/o ATR | Dev Full |
|--------|-------------:|------------:|------------:|-------------:|---------:|
| AUC | 0.6192 | 0.5026 | 0.5603 | 0.6010 | 0.9241 |
| Brier | 0.2400 | 0.2583 | 0.2529 | 0.2477 | 0.1356 |
| Log loss | 0.6723 | — | — | — | 0.4391 |
| Spearman | 0.1990 | 0.0043 | 0.1007 | 0.1687 | 0.6939 |
| Top-5 success | 0.5131 | 0.4512 | 0.3898 | 0.4671 | 0.5481 |
| Top-10% success | 0.5131 | 0.4512 | 0.3898 | 0.4671 | 0.7121 |
| All-assets success | 0.4622 | — | — | — | — |
| Random top-5 success | 0.3833 | — | — | — | — |

**Bootstrap (full):** top5={"mean": 0.48554231043810364, "ci_low": 0.3175219298245614, "ci_high": 0.6500181159420289}, top10%={"mean": 0.5151320043101486, "ci_low": 0.3849206349206349, "ci_high": 0.6491875}, spearman={"mean": 0.19350887757317106, "ci_low": 0.032363119190661246, "ci_high": 0.34774878117660424}

**Economic (full top-5):** gross=0.0194, net=0.0174, win_rate=0.5581, avg_winner=0.0544, avg_loser=-0.0247, max_dd=-0.3904, n=215

**EV diagnostic:** {"assumption": "success=+5%, failure=-3%; diagnostic only.", "cost_per_event": 0.002, "full_top5_ev_net": 0.009045045045045047, "atr_top5_ev_net": 0.004097560975609758}

**Volatility buckets:**

- **low** n=455: full_top5=0.7500, atr_top5=0.6250, mom_top5=0.6500, all=0.6778
- **medium** n=571: full_top5=0.4361, atr_top5=0.3611, mom_top5=0.4744, all=0.4934
- **high** n=995: full_top5=0.5258, atr_top5=0.4301, mom_top5=0.3885, all=0.4410

### up10_before_down5

| Metric | Holdout Full | Holdout ATR | Holdout Mom | Full w/o ATR | Dev Full |
|--------|-------------:|------------:|------------:|-------------:|---------:|
| AUC | 0.6212 | 0.5524 | 0.5408 | 0.6199 | 0.9854 |
| Brier | 0.2412 | 0.2511 | 0.2548 | 0.2387 | 0.0782 |
| Log loss | 0.6930 | — | — | — | 0.2886 |
| Spearman | 0.1944 | 0.0842 | 0.0655 | 0.1923 | 0.7225 |
| Top-5 success | 0.4808 | 0.4171 | 0.2865 | 0.4867 | 0.3325 |
| Top-10% success | 0.4808 | 0.4171 | 0.2865 | 0.4867 | 0.5709 |
| All-assets success | 0.4286 | — | — | — | — |
| Random top-5 success | 0.4123 | — | — | — | — |

**Bootstrap (full):** top5={"mean": 0.4489914045468249, "ci_low": 0.2635416666666667, "ci_high": 0.6208333333333333}, top10%={"mean": 0.476334059407796, "ci_low": 0.3333333333333333, "ci_high": 0.611111111111111}, spearman={"mean": 0.17492810709194997, "ci_low": -0.13789227632853307, "ci_high": 0.4239381158689202}

**Economic (full top-5):** gross=0.0106, net=0.0086, win_rate=0.5163, avg_winner=0.0413, avg_loser=-0.0222, max_dd=-0.3452, n=215

**EV diagnostic:** {"assumption": "success=+10%, failure=-5%; diagnostic only.", "cost_per_event": 0.002, "full_top5_ev_net": 0.02011538461538462, "atr_top5_ev_net": 0.010569444444444446}

**Volatility buckets:**

- **low** n=455: full_top5=0.5000, atr_top5=0.0000, mom_top5=0.5000, all=0.6667
- **medium** n=571: full_top5=0.4286, atr_top5=0.2500, mom_top5=0.3333, all=0.5385
- **high** n=995: full_top5=0.4213, atr_top5=0.4079, mom_top5=0.3137, all=0.3918

## Leakage audit

- Passed: **True**
- Findings: []

## Final classification

**`weak_generalization`**

The barrier signal is detectable on the holdout but materially weaker or inconsistent relative to development.

## Tests: **True**

## Errors: []
