# Portfolio backtest report (`portfolio_backtest_v1`)

Generated: **2026-09-19T08:10:07.823939+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`PORTFOLIO_BACKTEST_REPORT.json`](PORTFOLIO_BACKTEST_REPORT.json)

Prior experiments preserved:
- [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md)
- [`DIRECTIONAL_UP_VALIDATION_REPORT.md`](DIRECTIONAL_UP_VALIDATION_REPORT.md)
- [`RELATIVE_UPSIDE_RANKING_REPORT.md`](RELATIVE_UPSIDE_RANKING_REPORT.md)
- [`VOLATILITY_ADJUSTED_UPSIDE_REPORT.md`](VOLATILITY_ADJUSTED_UPSIDE_REPORT.md)
- [`RISK_ADJUSTED_OPPORTUNITY_REPORT.md`](RISK_ADJUSTED_OPPORTUNITY_REPORT.md)
- [`BARRIER_PROBABILITY_REPORT.md`](BARRIER_PROBABILITY_REPORT.md)
- [`POINT_IN_TIME_UNIVERSE_REPORT.md`](POINT_IN_TIME_UNIVERSE_REPORT.md)
- [`OUT_OF_SAMPLE_HOLDOUT_REPORT.md`](OUT_OF_SAMPLE_HOLDOUT_REPORT.md)

## Objective

Determine whether the frozen barrier-ranking signal translates into a realistic portfolio after next-candle execution, costs, and overlapping positions.

## Frozen assumptions

- Features, targets, model, universe, and holdout cutoff are frozen from `out_of_sample_holdout_v1`. No tuning on holdout.
- Cost: 0.002 per completed trade (round-trip).
- Slippage cases: ['base', 'plus_0.1pct', 'plus_0.25pct']
- Equal-weight positions; max 100% gross exposure; no leverage.

## Periods

- Development: 2026-06-24T14:00:00+00:00 → 2026-08-28T08:00:00+00:00
- Holdout: 2026-08-28T14:00:00+00:00 → 2026-09-18T14:00:00+00:00 (cutoff `2026-08-28T14:00:00+00:00`)

## Execution methodology

Signal at T → next 1h candle open entry → first barrier touch or 12h horizon exit. Same-candle dual touch ⇒ ambiguous ⇒ exit at that candle's open.

## Survivorship limitation

Portfolio backtest uses the point-in-time universe. Survivorship is reduced but not eliminated: symbols delisted before the data window are absent.

## Holdout results by barrier and K

### up3_before_down2

| Strategy | Top-K | Trades | Win Rate | Avg Trade | Net Return | Max DD | Sharpe | Profit Factor |
|----------|------:|-------:|---------:|----------:|-----------:|-------:|-------:|--------------:|
| full | top_1 | 33 | 0.3333 | -0.0025 | -0.0956 | -0.2182 | -0.1168 | 0.7721 |
| full | top_3 | 129 | 0.5271 | 0.0048 | 0.1303 | -0.0550 | 0.2195 | 1.6317 |
| full | top_5 | 215 | 0.4930 | 0.0031 | 0.0787 | -0.0740 | 0.1373 | 1.3536 |
| full | top_10pct | 80 | 0.4750 | 0.0034 | 0.0891 | -0.0879 | 0.1496 | 1.3968 |
| atr_only | top_1 | 39 | 0.4359 | -0.0005 | -0.0420 | -0.2092 | -0.0205 | 0.9586 |
| atr_only | top_3 | 126 | 0.4365 | -0.0002 | -0.0695 | -0.1381 | -0.0088 | 0.9801 |
| atr_only | top_5 | 215 | 0.4465 | 0.0004 | -0.0344 | -0.1444 | 0.0131 | 1.0309 |
| atr_only | top_10pct | 84 | 0.4643 | 0.0015 | -0.0147 | -0.1400 | 0.0543 | 1.1374 |
| momentum | top_1 | 39 | 0.4872 | 0.0031 | 0.1221 | -0.1200 | 0.1327 | 1.3342 |
| momentum | top_3 | 129 | 0.4341 | -0.0001 | 0.0215 | -0.0963 | -0.0030 | 0.9936 |
| momentum | top_5 | 215 | 0.4047 | -0.0006 | 0.0097 | -0.0814 | -0.0254 | 0.9467 |
| momentum | top_10pct | 86 | 0.4302 | 0.0006 | 0.0556 | -0.1068 | 0.0248 | 1.0546 |
| full_without_atr | top_1 | 35 | 0.4000 | -0.0020 | -0.0764 | -0.1747 | -0.0906 | 0.8240 |
| full_without_atr | top_3 | 126 | 0.4603 | 0.0025 | 0.0542 | -0.0896 | 0.1141 | 1.2852 |
| full_without_atr | top_5 | 215 | 0.4930 | 0.0029 | 0.0589 | -0.0699 | 0.1319 | 1.3344 |
| full_without_atr | top_10pct | 82 | 0.4512 | 0.0023 | 0.0507 | -0.1177 | 0.1016 | 1.2483 |
| random | top_5 | None | — | — | — | — | — | — |
| equal_weight_all | top_5 | 1032 | 0.4089 | -0.0001 | -0.0316 | -0.0798 | -0.0045 | 0.9884 |

### up5_before_down3

| Strategy | Top-K | Trades | Win Rate | Avg Trade | Net Return | Max DD | Sharpe | Profit Factor |
|----------|------:|-------:|---------:|----------:|-----------:|-------:|-------:|--------------:|
| full | top_1 | 29 | 0.4483 | 0.0031 | 0.0865 | -0.1515 | 0.0992 | 1.2635 |
| full | top_3 | 123 | 0.4553 | 0.0070 | 0.0761 | -0.1006 | 0.1627 | 1.6475 |
| full | top_5 | 210 | 0.4714 | 0.0062 | 0.1043 | -0.0530 | 0.1628 | 1.5936 |
| full | top_10pct | 80 | 0.4125 | 0.0053 | -0.0072 | -0.1671 | 0.1084 | 1.4306 |
| atr_only | top_1 | 28 | 0.3571 | -0.0064 | -0.1482 | -0.2028 | -0.2319 | 0.5884 |
| atr_only | top_3 | 123 | 0.4715 | 0.0022 | 0.0552 | -0.0928 | 0.0650 | 1.1602 |
| atr_only | top_5 | 210 | 0.5048 | 0.0034 | 0.0544 | -0.0765 | 0.0954 | 1.2515 |
| atr_only | top_10pct | 72 | 0.4583 | 0.0007 | -0.0250 | -0.1289 | 0.0194 | 1.0465 |
| momentum | top_1 | 30 | 0.5667 | 0.0103 | 0.3035 | -0.1052 | 0.3129 | 2.1250 |
| momentum | top_3 | 129 | 0.4651 | 0.0067 | 0.1678 | -0.0466 | 0.1582 | 1.5227 |
| momentum | top_5 | 210 | 0.4667 | 0.0040 | 0.1189 | -0.0631 | 0.1023 | 1.2966 |
| momentum | top_10pct | 76 | 0.5000 | 0.0070 | 0.2169 | -0.1011 | 0.2033 | 1.5821 |
| full_without_atr | top_1 | 30 | 0.5000 | 0.0053 | 0.1373 | -0.1546 | 0.1619 | 1.4450 |
| full_without_atr | top_3 | 126 | 0.4762 | 0.0057 | 0.0530 | -0.0893 | 0.1300 | 1.4557 |
| full_without_atr | top_5 | 210 | 0.4714 | 0.0062 | 0.0633 | -0.0788 | 0.1415 | 1.5211 |
| full_without_atr | top_10pct | 80 | 0.5000 | 0.0060 | 0.1403 | -0.1285 | 0.1856 | 1.5268 |
| random | top_5 | None | — | — | — | — | — | — |
| equal_weight_all | top_5 | 1032 | 0.4196 | 0.0012 | -0.0132 | -0.0971 | 0.0413 | 1.1163 |

### up10_before_down5

| Strategy | Top-K | Trades | Win Rate | Avg Trade | Net Return | Max DD | Sharpe | Profit Factor |
|----------|------:|-------:|---------:|----------:|-----------:|-------:|-------:|--------------:|
| full | top_1 | 28 | 0.3571 | 0.0091 | 0.2297 | -0.1716 | 0.1582 | 1.5103 |
| full | top_3 | 129 | 0.3876 | 0.0052 | 0.1025 | -0.1031 | 0.1042 | 1.3715 |
| full | top_5 | 215 | 0.4093 | 0.0049 | 0.0608 | -0.0567 | 0.1083 | 1.3898 |
| full | top_10pct | 82 | 0.3415 | 0.0007 | 0.0466 | -0.1077 | 0.0147 | 1.0406 |
| atr_only | top_1 | 30 | 0.4000 | 0.0009 | -0.1526 | -0.3798 | 0.0147 | 1.0370 |
| atr_only | top_3 | 129 | 0.4729 | 0.0085 | 0.1862 | -0.1124 | 0.1528 | 1.4563 |
| atr_only | top_5 | 215 | 0.4744 | 0.0068 | 0.1266 | -0.1190 | 0.1277 | 1.3775 |
| atr_only | top_10pct | 78 | 0.4744 | 0.0073 | 0.0927 | -0.1803 | 0.1293 | 1.3797 |
| momentum | top_1 | 28 | 0.3929 | 0.0064 | -0.0909 | -0.2483 | 0.1142 | 1.3526 |
| momentum | top_3 | 129 | 0.4651 | 0.0027 | -0.0207 | -0.1436 | 0.0564 | 1.1597 |
| momentum | top_5 | 215 | 0.4651 | 0.0000 | -0.0749 | -0.1394 | 0.0002 | 1.0005 |
| momentum | top_10pct | 86 | 0.4767 | 0.0061 | 0.0148 | -0.1247 | 0.1205 | 1.3681 |
| full_without_atr | top_1 | 28 | 0.3929 | 0.0095 | 0.2003 | -0.1893 | 0.1669 | 1.5486 |
| full_without_atr | top_3 | 129 | 0.4109 | 0.0045 | 0.0584 | -0.0902 | 0.1026 | 1.3284 |
| full_without_atr | top_5 | 215 | 0.3953 | 0.0036 | 0.0370 | -0.0673 | 0.0886 | 1.2872 |
| full_without_atr | top_10pct | 78 | 0.4231 | 0.0065 | 0.1580 | -0.1004 | 0.1389 | 1.4592 |
| random | top_5 | None | — | — | — | — | — | — |
| equal_weight_all | top_5 | 1032 | 0.4273 | 0.0022 | -0.0123 | -0.0839 | 0.0593 | 1.1926 |

## Development vs holdout

{
  "up3_before_down2": {
    "development_net": 1.2476195061640367,
    "holdout_net": 0.07872431406562219,
    "development_max_dd": -0.01865451660452644,
    "holdout_max_dd": -0.07398460032893894
  },
  "up5_before_down3": {
    "development_net": 1.2862842304085227,
    "holdout_net": 0.10433253166471501,
    "development_max_dd": -0.024635494638093114,
    "holdout_max_dd": -0.052962074106447043
  },
  "up10_before_down5": {
    "development_net": 1.1160050511955721,
    "holdout_net": 0.06083303014117858,
    "development_max_dd": -0.0380400651117343,
    "holdout_max_dd": -0.0567034505976004
  }
}

## Slippage sensitivity

{
  "up3_before_down2": {
    "base": {
      "cumulative_net_return": 0.07872431406562219,
      "n_trades": 215,
      "win_rate": 0.4930232558139535
    },
    "plus_0.1pct": {
      "cumulative_net_return": 0.044580886497509686,
      "n_trades": 215,
      "win_rate": 0.48372093023255813
    },
    "plus_0.25pct": {
      "cumulative_net_return": -0.004669071576428374,
      "n_trades": 215,
      "win_rate": 0.4790697674418605
    }
  },
  "up5_before_down3": {
    "base": {
      "cumulative_net_return": 0.10433253166471501,
      "n_trades": 210,
      "win_rate": 0.4714285714285714
    },
    "plus_0.1pct": {
      "cumulative_net_return": 0.07468631086025534,
      "n_trades": 210,
      "win_rate": 0.4666666666666667
    },
    "plus_0.25pct": {
      "cumulative_net_return": 0.03163480658622797,
      "n_trades": 210,
      "win_rate": 0.44761904761904764
    }
  },
  "up10_before_down5": {
    "base": {
      "cumulative_net_return": 0.06083303014117858,
      "n_trades": 215,
      "win_rate": 0.40930232558139534
    },
    "plus_0.1pct": {
      "cumulative_net_return": 0.035962973964410505,
      "n_trades": 215,
      "win_rate": 0.4046511627906977
    },
    "plus_0.25pct": {
      "cumulative_net_return": -0.0002956667472466812,
      "n_trades": 215,
      "win_rate": 0.37209302325581395
    }
  }
}

## Leakage audit

- Passed: **True**
- Findings: []

## Final classification

**`weak_economic_signal`**

Some economic value exists, but returns/risk/execution sensitivity are not robust enough to treat as economically convincing.

## Tests: **True**

## Errors: []
