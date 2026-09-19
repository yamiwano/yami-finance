# Regime filter OOS report (`regime_filter_oos_v1`)

Generated: **2026-09-19T17:18:57.403807+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`REGIME_FILTER_OOS_REPORT.json`](REGIME_FILTER_OOS_REPORT.json)

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
- [`MULTI_PERIOD_ROBUSTNESS_REPORT.md`](MULTI_PERIOD_ROBUSTNESS_REPORT.md)
- [`REGIME_ANALYSIS_REPORT.md`](REGIME_ANALYSIS_REPORT.md)

## Objective

Test whether previously identified regime filters (high momentum dispersion, normal volume, combined) improve the frozen barrier model on genuinely unseen data.

## Frozen base model

Same 36 features, HistGradientBoostingClassifier, barriers, universe, stride, horizon, costs, execution. Filters are external to the model.

## Frozen regime definitions / thresholds

```
{
  "btc_volatility": [
    0.0024927660335913623,
    0.0037175525844377233
  ],
  "market_volatility": [
    0.7799124976222205,
    1.2708427219468243
  ],
  "momentum_dispersion": [
    0.0272705267408798,
    0.04756759248191777
  ],
  "volume": [
    0.6332251316814317,
    1.1700550807850245
  ],
  "train_end": "2026-08-30T00:00:00+00:00",
  "n_dev_rows": 6751
}
```

## OOS period

- Train end: **2026-08-30T00:00:00+00:00**
- OOS rows: 1922 across 41 timestamps

## Survivorship limitation

Point-in-time universe reduces retrospective selection bias, but historical listing/delisting metadata is unavailable; symbols delisted before the data window are absent. Survivorship bias is reduced, not eliminated.

## Key comparison (top-5)

| Barrier | Strategy | OOS trades | OOS net | Max DD | Sharpe | Win rate | Exposure |
|---------|----------|-----------:|--------:|-------:|-------:|---------:|---------:|
| up3_before_down2 | baseline | 184 | 0.0456 | -0.1016 | 0.1191 | 0.4837 | 1.000 |
| up3_before_down2 | high_dispersion | 95 | 0.0609 | -0.0480 | 0.2662 | 0.5368 | 0.537 |
| up3_before_down2 | normal_volume | 71 | 0.0757 | -0.0272 | 0.3088 | 0.5634 | 0.390 |
| up3_before_down2 | high_dispersion_plus_normal_volume | 39 | 0.0337 | -0.0240 | 0.2609 | 0.5128 | 0.220 |
| up3_before_down2 | atr_only | 194 | 0.0089 | -0.1140 | 0.0542 | 0.4588 | — |
| up3_before_down2 | momentum | 182 | 0.0387 | -0.1050 | 0.0832 | 0.4725 | — |
| up3_before_down2 | full_without_atr | 184 | 0.0588 | -0.0892 | 0.1266 | 0.4728 | — |

| up5_before_down3 | baseline | 177 | 0.1127 | -0.0467 | 0.1772 | 0.5085 | 1.000 |
| up5_before_down3 | high_dispersion | 94 | 0.1066 | -0.0189 | 0.2674 | 0.5745 | 0.537 |
| up5_before_down3 | normal_volume | 66 | 0.0625 | -0.0146 | 0.2590 | 0.6364 | 0.390 |
| up5_before_down3 | high_dispersion_plus_normal_volume | 37 | 0.0223 | -0.0092 | 0.2094 | 0.5946 | 0.220 |
| up5_before_down3 | atr_only | 178 | 0.0236 | -0.1349 | 0.0770 | 0.4719 | — |
| up5_before_down3 | momentum | 179 | 0.0293 | -0.1239 | 0.0603 | 0.4693 | — |
| up5_before_down3 | full_without_atr | 178 | 0.0667 | -0.0646 | 0.1537 | 0.5000 | — |

| up10_before_down5 | baseline | 179 | 0.0770 | -0.0876 | 0.1191 | 0.4190 | 1.000 |
| up10_before_down5 | high_dispersion | 93 | 0.0745 | -0.0308 | 0.2293 | 0.4731 | 0.537 |
| up10_before_down5 | normal_volume | 63 | 0.0628 | -0.0232 | 0.2498 | 0.5079 | 0.390 |
| up10_before_down5 | high_dispersion_plus_normal_volume | 34 | 0.0188 | -0.0217 | 0.2004 | 0.4706 | 0.220 |
| up10_before_down5 | atr_only | 168 | 0.0570 | -0.1656 | 0.0836 | 0.4762 | — |
| up10_before_down5 | momentum | 170 | 0.0375 | -0.1039 | 0.1148 | 0.5000 | — |
| up10_before_down5 | full_without_atr | 179 | 0.0799 | -0.0930 | 0.1103 | 0.4637 | — |

## Incremental value (filtered − baseline)

{
  "up3_before_down2": {
    "baseline_net": 0.04556789453749066,
    "combined_net": 0.03368468354429832,
    "delta": -0.011883210993192339
  },
  "up5_before_down3": {
    "baseline_net": 0.11266237269042123,
    "combined_net": 0.02234530121291578,
    "delta": -0.09031707147750545
  },
  "up10_before_down5": {
    "baseline_net": 0.07695770909523736,
    "combined_net": 0.018831055636530758,
    "delta": -0.0581266534587066
  }
}

## Random filtered benchmark

{
  "up3_before_down2": {
    "baseline": {
      "filter": "baseline",
      "n_sims": 100,
      "cumulative_net_return_mean": -0.002759616151792179,
      "cumulative_net_return_median": 0.00025898808213559166,
      "cumulative_net_return_p5": -0.055322152957125656,
      "cumulative_net_return_p95": 0.05286651122817723
    },
    "high_dispersion": {
      "filter": "high_dispersion",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.030184948652451866,
      "cumulative_net_return_median": 0.02959557671751667,
      "cumulative_net_return_p5": -0.010436234701019464,
      "cumulative_net_return_p95": 0.07086258489312465
    },
    "normal_volume": {
      "filter": "normal_volume",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.03384868711287932,
      "cumulative_net_return_median": 0.03300107857154211,
      "cumulative_net_return_p5": -0.00904305004854684,
      "cumulative_net_return_p95": 0.0735343317288736
    },
    "high_dispersion_plus_normal_volume": {
      "filter": "high_dispersion_plus_normal_volume",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.011538109685831757,
      "cumulative_net_return_median": 0.01200835686071322,
      "cumulative_net_return_p5": -0.01955215136803402,
      "cumulative_net_return_p95": 0.04279343625656674
    }
  },
  "up5_before_down3": {
    "baseline": {
      "filter": "baseline",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.030524447244128713,
      "cumulative_net_return_median": 0.03273075714194695,
      "cumulative_net_return_p5": -0.03040395853557513,
      "cumulative_net_return_p95": 0.0946086715269124
    },
    "high_dispersion": {
      "filter": "high_dispersion",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.05441144470022539,
      "cumulative_net_return_median": 0.05436056341479656,
      "cumulative_net_return_p5": 0.000430222803709641,
      "cumulative_net_return_p95": 0.10097041981837203
    },
    "normal_volume": {
      "filter": "normal_volume",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.06772841058076913,
      "cumulative_net_return_median": 0.0696731881500482,
      "cumulative_net_return_p5": 0.02161302161225074,
      "cumulative_net_return_p95": 0.11489675355180187
    },
    "high_dispersion_plus_normal_volume": {
      "filter": "high_dispersion_plus_normal_volume",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.035242947797516556,
      "cumulative_net_return_median": 0.032631062256651466,
      "cumulative_net_return_p5": 0.0020029133995575934,
      "cumulative_net_return_p95": 0.07066566434445158
    }
  },
  "up10_before_down5": {
    "baseline": {
      "filter": "baseline",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.0713502911694885,
      "cumulative_net_return_median": 0.06845299697121121,
      "cumulative_net_return_p5": 0.003308259604290844,
      "cumulative_net_return_p95": 0.15365876132435932
    },
    "high_dispersion": {
      "filter": "high_dispersion",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.0754056276133293,
      "cumulative_net_return_median": 0.07025787530244654,
      "cumulative_net_return_p5": 0.008938900277732044,
      "cumulative_net_return_p95": 0.13658940917740647
    },
    "normal_volume": {
      "filter": "normal_volume",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.0801632734474822,
      "cumulative_net_return_median": 0.07745934684292777,
      "cumulative_net_return_p5": 0.03596942693034805,
      "cumulative_net_return_p95": 0.13660803324993964
    },
    "high_dispersion_plus_normal_volume": {
      "filter": "high_dispersion_plus_normal_volume",
      "n_sims": 100,
      "cumulative_net_return_mean": 0.044120916220709544,
      "cumulative_net_return_median": 0.04383143219019503,
      "cumulative_net_return_p5": 0.001506785254963739,
      "cumulative_net_return_p95": 0.08656721711026068
    }
  }
}

## Slippage sensitivity

{
  "up3_before_down2": {
    "baseline": {
      "base": {
        "cumulative_net_return": 0.04556789453749066,
        "n_trades": 184,
        "win_rate": 0.483695652173913
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.015314368055473704,
        "n_trades": 184,
        "win_rate": 0.483695652173913
      },
      "plus_0.25pct": {
        "cumulative_net_return": -0.028364290975222306,
        "n_trades": 184,
        "win_rate": 0.4782608695652174
      }
    },
    "high_dispersion": {
      "base": {
        "cumulative_net_return": 0.06093259485643676,
        "n_trades": 95,
        "win_rate": 0.5368421052631579
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.04521828339693634,
        "n_trades": 95,
        "win_rate": 0.5368421052631579
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.021965881248611208,
        "n_trades": 95,
        "win_rate": 0.5368421052631579
      }
    },
    "normal_volume": {
      "base": {
        "cumulative_net_return": 0.07572873896381394,
        "n_trades": 71,
        "win_rate": 0.5633802816901409
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.06367475911874276,
        "n_trades": 71,
        "win_rate": 0.5633802816901409
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.04565729038157329,
        "n_trades": 71,
        "win_rate": 0.5633802816901409
      }
    },
    "high_dispersion_plus_normal_volume": {
      "base": {
        "cumulative_net_return": 0.03368468354429832,
        "n_trades": 39,
        "win_rate": 0.5128205128205128
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.027455477159107344,
        "n_trades": 39,
        "win_rate": 0.5128205128205128
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.018098552554460134,
        "n_trades": 39,
        "win_rate": 0.5128205128205128
      }
    }
  },
  "up5_before_down3": {
    "baseline": {
      "base": {
        "cumulative_net_return": 0.11266237269042123,
        "n_trades": 177,
        "win_rate": 0.5084745762711864
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.08701940967834121,
        "n_trades": 177,
        "win_rate": 0.5084745762711864
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.04957633824619734,
        "n_trades": 177,
        "win_rate": 0.4858757062146893
      }
    },
    "high_dispersion": {
      "base": {
        "cumulative_net_return": 0.10662619481250313,
        "n_trades": 94,
        "win_rate": 0.574468085106383
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.09264462003202856,
        "n_trades": 94,
        "win_rate": 0.574468085106383
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.07196010873278191,
        "n_trades": 94,
        "win_rate": 0.5638297872340425
      }
    },
    "normal_volume": {
      "base": {
        "cumulative_net_return": 0.06253615895434494,
        "n_trades": 66,
        "win_rate": 0.6363636363636364
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.05267815775992957,
        "n_trades": 66,
        "win_rate": 0.6363636363636364
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.037928572280819806,
        "n_trades": 66,
        "win_rate": 0.5909090909090909
      }
    },
    "high_dispersion_plus_normal_volume": {
      "base": {
        "cumulative_net_return": 0.02234530121291578,
        "n_trades": 37,
        "win_rate": 0.5945945945945946
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.01679509565329096,
        "n_trades": 37,
        "win_rate": 0.5945945945945946
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.008436565086390235,
        "n_trades": 37,
        "win_rate": 0.5675675675675675
      }
    }
  },
  "up10_before_down5": {
    "baseline": {
      "base": {
        "cumulative_net_return": 0.07695770909523736,
        "n_trades": 179,
        "win_rate": 0.41899441340782123
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.055351513987298206,
        "n_trades": 179,
        "win_rate": 0.4134078212290503
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.02371476260269656,
        "n_trades": 179,
        "win_rate": 0.39664804469273746
      }
    },
    "high_dispersion": {
      "base": {
        "cumulative_net_return": 0.07453665187295799,
        "n_trades": 93,
        "win_rate": 0.4731182795698925
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.06293923404395763,
        "n_trades": 103,
        "win_rate": 0.46601941747572817
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.04575744412403293,
        "n_trades": 93,
        "win_rate": 0.46236559139784944
      }
    },
    "normal_volume": {
      "base": {
        "cumulative_net_return": 0.06275158620394206,
        "n_trades": 63,
        "win_rate": 0.5079365079365079
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.05254374704730447,
        "n_trades": 63,
        "win_rate": 0.5079365079365079
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.037252045860808325,
        "n_trades": 63,
        "win_rate": 0.4603174603174603
      }
    },
    "high_dispersion_plus_normal_volume": {
      "base": {
        "cumulative_net_return": 0.018831055636530758,
        "n_trades": 34,
        "win_rate": 0.47058823529411764
      },
      "plus_0.1pct": {
        "cumulative_net_return": 0.012955303584206934,
        "n_trades": 38,
        "win_rate": 0.4473684210526316
      },
      "plus_0.25pct": {
        "cumulative_net_return": 0.004194579677053278,
        "n_trades": 38,
        "win_rate": 0.42105263157894735
      }
    }
  }
}

## Leakage audit

- Passed: **True**
- Findings: []

## Multiple-testing limitation

Many hypotheses have been tested in this project. This OOS filter test is confirmatory; apparent effects may still reflect selection from prior research.

## Final classification

**`filter_fails`**

Previously identified regime relationship does not generalize to the OOS period.

## Tests: **True**

## Errors: []
