# Regime analysis report (`regime_analysis_v1`)

Generated: **2026-09-19T16:57:41.835135+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`REGIME_ANALYSIS_REPORT.json`](REGIME_ANALYSIS_REPORT.json)

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

## Objective

Determine which observable market conditions at signal time T are associated with stronger/weaker performance of the frozen barrier signal. No model changes.

## Frozen specification

Same 36 features, barriers, model, universe, stride, horizon, costs, execution. Regime thresholds fit on development data only.

## Regime definitions

- BTC trend: 24h return > +1% bullish, < −1% bearish, else neutral
- BTC volatility: trailing 24h realized vol tertiles (development-fitted)
- Market breadth: % assets with positive 24h return (>60% strong, <40% weak)
- Market volatility: median ATR% tertiles (development-fitted)
- Momentum dispersion: cross-sectional std of 24h returns (tertiles)
- Volume: median relative_volume tertiles (development-fitted)

## Threshold methodology

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

## Period composition

[
  {
    "period": 1,
    "start": "2026-06-24T14:00:00+00:00",
    "end": "2026-07-20T02:00:00+00:00",
    "timestamps": 122,
    "btc_trend": {
      "bearish": 0.27049180327868855,
      "neutral": 0.4180327868852459,
      "bullish": 0.3114754098360656
    },
    "btc_volatility": {
      "high": 0.38524590163934425,
      "medium": 0.39344262295081966,
      "low": 0.22131147540983606
    },
    "market_breadth": {
      "weak": 0.5327868852459017,
      "strong": 0.39344262295081966,
      "neutral": 0.07377049180327869
    },
    "market_volatility": {
      "medium": 0.5,
      "high": 0.30327868852459017,
      "low": 0.19672131147540983
    },
    "momentum_dispersion": {
      "low": 0.2786885245901639,
      "high": 0.28688524590163933,
      "medium": 0.2786885245901639,
      "unknown": 0.1557377049180328
    },
    "volume": {
      "high": 0.3770491803278688,
      "low": 0.29508196721311475,
      "normal": 0.32786885245901637
    }
  },
  {
    "period": 2,
    "start": "2026-07-20T08:00:00+00:00",
    "end": "2026-08-09T12:00:00+00:00",
    "timestamps": 122,
    "btc_trend": {
      "bearish": 0.22131147540983606,
      "neutral": 0.5819672131147541,
      "bullish": 0.19672131147540983
    },
    "btc_volatility": {
      "medium": 0.3770491803278688,
      "high": 0.19672131147540983,
      "low": 0.4262295081967213
    },
    "market_breadth": {
      "strong": 0.4426229508196721,
      "weak": 0.4918032786885246,
      "neutral": 0.06557377049180328
    },
    "market_volatility": {
      "low": 0.4426229508196721,
      "high": 0.20491803278688525,
      "medium": 0.3524590163934426
    },
    "momentum_dispersion": {
      "unknown": 0.3360655737704918,
      "high": 0.32786885245901637,
      "low": 0.16393442622950818,
      "medium": 0.1721311475409836
    },
    "volume": {
      "low": 0.36885245901639346,
      "normal": 0.30327868852459017,
      "high": 0.32786885245901637
    }
  },
  {
    "period": 3,
    "start": "2026-08-09T14:00:00+00:00",
    "end": "2026-08-29T20:00:00+00:00",
    "timestamps": 122,
    "btc_trend": {
      "neutral": 0.5409836065573771,
      "bearish": 0.16393442622950818,
      "bullish": 0.29508196721311475
    },
    "btc_volatility": {
      "low": 0.4098360655737705,
      "medium": 0.2540983606557377,
      "high": 0.3360655737704918
    },
    "market_breadth": {
      "neutral": 0.07377049180327869,
      "strong": 0.4344262295081967,
      "weak": 0.4918032786885246
    },
    "market_volatility": {
      "low": 0.4426229508196721,
      "high": 0.3524590163934426,
      "medium": 0.20491803278688525
    },
    "momentum_dispersion": {
      "medium": 0.27049180327868855,
      "unknown": 0.3360655737704918,
      "low": 0.2786885245901639,
      "high": 0.11475409836065574
    },
    "volume": {
      "high": 0.29508196721311475,
      "low": 0.3442622950819672,
      "normal": 0.36065573770491804
    }
  },
  {
    "period": 4,
    "start": "2026-08-30T00:00:00+00:00",
    "end": "2026-09-19T02:00:00+00:00",
    "timestamps": 122,
    "btc_trend": {
      "neutral": 0.5245901639344263,
      "bullish": 0.23770491803278687,
      "bearish": 0.23770491803278687
    },
    "btc_volatility": {
      "low": 0.28688524590163933,
      "medium": 0.4180327868852459,
      "high": 0.29508196721311475
    },
    "market_breadth": {
      "weak": 0.4918032786885246,
      "neutral": 0.06557377049180328,
      "strong": 0.4426229508196721
    },
    "market_volatility": {
      "medium": 0.319672131147541,
      "low": 0.3524590163934426,
      "high": 0.32786885245901637
    },
    "momentum_dispersion": {
      "low": 0.18032786885245902,
      "unknown": 0.32786885245901637,
      "medium": 0.19672131147540983,
      "high": 0.29508196721311475
    },
    "volume": {
      "normal": 0.36885245901639346,
      "low": 0.319672131147541,
      "high": 0.3114754098360656
    }
  }
]

## Predictive performance by regime

### btc_trend

| Regime | Barrier | Samples | Full AUC | Full Top5 | ATR Top5 | Mom Top5 |
|--------|---------|--------:|---------:|----------:|---------:|---------:|
| neutral | up3_before_down2 | 1113 | 0.6372 | 0.4190 | 0.3333 | 0.3905 |
| neutral | up5_before_down3 | 1113 | 0.6698 | 0.3600 | 0.4000 | 0.3600 |
| neutral | up10_before_down5 | 1113 | 0.5317 | 0.2222 | 0.2000 | 0.1778 |
| bullish | up3_before_down2 | 460 | 0.4336 | 0.5556 | 0.3778 | 0.4889 |
| bullish | up5_before_down3 | 460 | 0.4413 | 0.5333 | 0.4444 | 0.5111 |
| bullish | up10_before_down5 | 460 | 0.6289 | 0.5667 | 0.4333 | 0.4667 |
| bearish | up3_before_down2 | 509 | 0.6138 | 0.4000 | 0.5600 | 0.4600 |
| bearish | up5_before_down3 | 509 | 0.6138 | 0.4800 | 0.4000 | 0.4400 |
| bearish | up10_before_down5 | 509 | 0.6176 | 0.4000 | 0.2000 | 0.4000 |

### btc_volatility

| Regime | Barrier | Samples | Full AUC | Full Top5 | ATR Top5 | Mom Top5 |
|--------|---------|--------:|---------:|----------:|---------:|---------:|
| low | up3_before_down2 | 564 | 0.6046 | 0.3818 | 0.3273 | 0.4364 |
| low | up5_before_down3 | 564 | 0.5729 | 0.3091 | 0.4000 | 0.3455 |
| low | up10_before_down5 | 564 | 0.4530 | 0.4333 | 0.3667 | 0.4000 |
| medium | up3_before_down2 | 913 | 0.6472 | 0.4778 | 0.4444 | 0.4444 |
| medium | up5_before_down3 | 913 | 0.6761 | 0.4824 | 0.3882 | 0.4353 |
| medium | up10_before_down5 | 913 | 0.6457 | 0.3200 | 0.2800 | 0.2800 |
| high | up3_before_down2 | 605 | 0.5195 | 0.4545 | 0.4000 | 0.4000 |
| high | up5_before_down3 | 605 | 0.5186 | 0.4727 | 0.4545 | 0.4545 |
| high | up10_before_down5 | 605 | 0.6630 | 0.3200 | 0.2000 | 0.2000 |

### market_breadth

| Regime | Barrier | Samples | Full AUC | Full Top5 | ATR Top5 | Mom Top5 |
|--------|---------|--------:|---------:|----------:|---------:|---------:|
| neutral | up3_before_down2 | 376 | 0.5757 | 0.4000 | 0.1750 | 0.3750 |
| neutral | up5_before_down3 | 376 | 0.4355 | 0.2750 | 0.4000 | 0.3000 |
| neutral | up10_before_down5 | 376 | 0.4375 | 0.0667 | 0.0667 | 0.0667 |
| weak | up3_before_down2 | 879 | 0.6054 | 0.4588 | 0.5647 | 0.4824 |
| weak | up5_before_down3 | 879 | 0.6262 | 0.4500 | 0.4125 | 0.4375 |
| weak | up10_before_down5 | 879 | 0.6675 | 0.2000 | 0.1333 | 0.2000 |
| strong | up3_before_down2 | 827 | 0.5935 | 0.4533 | 0.3333 | 0.4000 |
| strong | up5_before_down3 | 827 | 0.6173 | 0.4933 | 0.4133 | 0.4533 |
| strong | up10_before_down5 | 827 | 0.5897 | 0.5000 | 0.4000 | 0.4000 |

### market_volatility

| Regime | Barrier | Samples | Full AUC | Full Top5 | ATR Top5 | Mom Top5 |
|--------|---------|--------:|---------:|----------:|---------:|---------:|
| low | up3_before_down2 | 181 | 0.4818 | 0.2667 | 0.1333 | 0.4000 |
| low | up5_before_down3 | 181 | 0.3846 | 0.2000 | 0.3333 | 0.2667 |
| low | up10_before_down5 | 181 | 0.6667 | — | — | — |
| medium | up3_before_down2 | 906 | 0.6180 | 0.5333 | 0.4000 | 0.4778 |
| medium | up5_before_down3 | 906 | 0.5418 | 0.4889 | 0.5222 | 0.5111 |
| medium | up10_before_down5 | 906 | 0.4404 | 0.3750 | 0.3500 | 0.3500 |
| high | up3_before_down2 | 995 | 0.6008 | 0.3895 | 0.4421 | 0.3895 |
| high | up5_before_down3 | 995 | 0.6619 | 0.4111 | 0.3111 | 0.3444 |
| high | up10_before_down5 | 995 | 0.6912 | 0.3500 | 0.2250 | 0.2500 |

### momentum_dispersion

| Regime | Barrier | Samples | Full AUC | Full Top5 | ATR Top5 | Mom Top5 |
|--------|---------|--------:|---------:|----------:|---------:|---------:|
| low | up3_before_down2 | 195 | 0.5858 | 0.6667 | 0.4667 | 0.5333 |
| low | up5_before_down3 | 195 | 0.6495 | 0.5333 | 0.6000 | 0.5333 |
| low | up10_before_down5 | 195 | 0.8333 | — | — | — |
| unknown | up3_before_down2 | 40 | — | — | — | — |
| unknown | up5_before_down3 | 40 | — | — | — | — |
| unknown | up10_before_down5 | 40 | — | — | — | — |
| medium | up3_before_down2 | 776 | 0.5875 | 0.4000 | 0.4000 | 0.3750 |
| medium | up5_before_down3 | 776 | 0.5678 | 0.3600 | 0.2800 | 0.2933 |
| medium | up10_before_down5 | 776 | 0.5670 | 0.0857 | 0.0571 | 0.0857 |
| high | up3_before_down2 | 1071 | 0.6089 | 0.4476 | 0.3905 | 0.4571 |
| high | up5_before_down3 | 1071 | 0.6098 | 0.4667 | 0.4762 | 0.4857 |
| high | up10_before_down5 | 1071 | 0.5845 | 0.5778 | 0.4667 | 0.4667 |

### volume

| Regime | Barrier | Samples | Full AUC | Full Top5 | ATR Top5 | Mom Top5 |
|--------|---------|--------:|---------:|----------:|---------:|---------:|
| normal | up3_before_down2 | 809 | 0.5545 | 0.5125 | 0.4000 | 0.5125 |
| normal | up5_before_down3 | 809 | 0.5196 | 0.4133 | 0.4267 | 0.4533 |
| normal | up10_before_down5 | 809 | 0.5034 | 0.6667 | 0.5333 | 0.5333 |
| low | up3_before_down2 | 379 | 0.5284 | 0.3000 | 0.3333 | 0.2000 |
| low | up5_before_down3 | 379 | 0.5647 | 0.4333 | 0.3000 | 0.3333 |
| low | up10_before_down5 | 379 | 0.5354 | 0.0000 | 0.0000 | 0.0000 |
| high | up3_before_down2 | 894 | 0.6236 | 0.4333 | 0.4222 | 0.4333 |
| high | up5_before_down3 | 894 | 0.6424 | 0.4444 | 0.4333 | 0.4111 |
| high | up10_before_down5 | 894 | 0.6499 | 0.3800 | 0.3000 | 0.3200 |

## Portfolio performance by regime (top-5)

{
  "btc_trend": {
    "neutral": {
      "up3_before_down2": {
        "n_trades": 95,
        "cumulative_net_return": -0.0006252110974322989,
        "win_rate": 0.4842105263157895,
        "max_drawdown": -0.05307235471543659,
        "sharpe": 0.15224614548056295
      },
      "up5_before_down3": {
        "n_trades": 94,
        "cumulative_net_return": 0.015498170607708195,
        "win_rate": 0.46808510638297873,
        "max_drawdown": -0.08872840212903021,
        "sharpe": 0.1435685487171205
      },
      "up10_before_down5": {
        "n_trades": 91,
        "cumulative_net_return": -0.03789068908353532,
        "win_rate": 0.3626373626373626,
        "max_drawdown": -0.13324183914665388,
        "sharpe": 0.08569501990923466
      }
    },
    "bullish": {
      "up3_before_down2": {
        "n_trades": 42,
        "cumulative_net_return": 0.07157271639617102,
        "win_rate": 0.6666666666666666,
        "max_drawdown": -0.01034575678332983,
        "sharpe": 0.43435775403117044
      },
      "up5_before_down3": {
        "n_trades": 41,
        "cumulative_net_return": 0.0158180883429051,
        "win_rate": 0.4878048780487805,
        "max_drawdown": -0.040819826897668054,
        "sharpe": 0.1864316737122915
      },
      "up10_before_down5": {
        "n_trades": 39,
        "cumulative_net_return": 0.050157236396339844,
        "win_rate": 0.5128205128205128,
        "max_drawdown": -0.02952926824524038,
        "sharpe": 0.18394285653131123
      }
    },
    "bearish": {
      "up3_before_down2": {
        "n_trades": 41,
        "cumulative_net_return": 0.04102057825448946,
        "win_rate": 0.5365853658536586,
        "max_drawdown": -0.023915497728516755,
        "sharpe": 0.227331792841592
      },
      "up5_before_down3": {
        "n_trades": 41,
        "cumulative_net_return": 0.030878822517329096,
        "win_rate": 0.5121951219512195,
        "max_drawdown": -0.009865259545627447,
        "sharpe": 0.20318059295782875
      },
      "up10_before_down5": {
        "n_trades": 43,
        "cumulative_net_return": 0.06072096195478527,
        "win_rate": 0.4883720930232558,
        "max_drawdown": -0.019647623897430355,
        "sharpe": 0.22984616152911766
      }
    }
  },
  "btc_volatility": {
    "low": {
      "up3_before_down2": {
        "n_trades": 50,
        "cumulative_net_return": 0.02748880945503629,
        "win_rate": 0.46,
        "max_drawdown": -0.034128859025337466,
        "sharpe": 0.05590276002182854
      },
      "up5_before_down3": {
        "n_trades": 50,
        "cumulative_net_return": -0.03546729308605978,
        "win_rate": 0.38,
        "max_drawdown": -0.05638325212890816,
        "sharpe": -0.05465233353120536
      },
      "up10_before_down5": {
        "n_trades": 39,
        "cumulative_net_return": 0.028850627024382902,
        "win_rate": 0.46153846153846156,
        "max_drawdown": -0.018792050062671373,
        "sharpe": 0.17387017975507518
      }
    },
    "medium": {
      "up3_before_down2": {
        "n_trades": 76,
        "cumulative_net_return": 0.0305025643711061,
        "win_rate": 0.5526315789473685,
        "max_drawdown": -0.03953144909780526,
        "sharpe": 0.21579404724826154
      },
      "up5_before_down3": {
        "n_trades": 75,
        "cumulative_net_return": 0.06514314163102553,
        "win_rate": 0.5333333333333333,
        "max_drawdown": -0.01677399316589201,
        "sharpe": 0.2603632610061553
      },
      "up10_before_down5": {
        "n_trades": 79,
        "cumulative_net_return": 0.014820857093457818,
        "win_rate": 0.4430379746835443,
        "max_drawdown": -0.06551809393672858,
        "sharpe": 0.3049376931883413
      }
    },
    "high": {
      "up3_before_down2": {
        "n_trades": 52,
        "cumulative_net_return": 0.05339320503572553,
        "win_rate": 0.5961538461538461,
        "max_drawdown": -0.019884545040603108,
        "sharpe": 0.2943519630449543
      },
      "up5_before_down3": {
        "n_trades": 51,
        "cumulative_net_return": -0.00047841095629264085,
        "win_rate": 0.47058823529411764,
        "max_drawdown": -0.04255941159609267,
        "sharpe": 0.1760683207989808
      },
      "up10_before_down5": {
        "n_trades": 47,
        "cumulative_net_return": 0.04550412811822002,
        "win_rate": 0.44680851063829785,
        "max_drawdown": -0.041286129444700004,
        "sharpe": 0.1371766223960562
      }
    }
  },
  "market_breadth": {
    "neutral": {
      "up3_before_down2": {
        "n_trades": 32,
        "cumulative_net_return": 0.015186276130415122,
        "win_rate": 0.5625,
        "max_drawdown": -0.025837931150871296,
        "sharpe": 0.22588441862529066
      },
      "up5_before_down3": {
        "n_trades": 30,
        "cumulative_net_return": -0.024556284747056956,
        "win_rate": 0.4666666666666667,
        "max_drawdown": -0.042581734500622526,
        "sharpe": 0.1824286200659839
      },
      "up10_before_down5": {
        "n_trades": 37,
        "cumulative_net_return": -0.06787822245436359,
        "win_rate": 0.3783783783783784,
        "max_drawdown": -0.07132658546304038,
        "sharpe": 0.19387688347815787
      }
    },
    "weak": {
      "up3_before_down2": {
        "n_trades": 70,
        "cumulative_net_return": 0.053493330973842346,
        "win_rate": 0.5285714285714286,
        "max_drawdown": -0.0392906798462086,
        "sharpe": 0.1870422339327281
      },
      "up5_before_down3": {
        "n_trades": 72,
        "cumulative_net_return": 0.07108070255754351,
        "win_rate": 0.4861111111111111,
        "max_drawdown": -0.022806296009662885,
        "sharpe": 0.2702453699307859
      },
      "up10_before_down5": {
        "n_trades": 70,
        "cumulative_net_return": 0.06453790837829754,
        "win_rate": 0.38571428571428573,
        "max_drawdown": -0.06700678392744308,
        "sharpe": 0.12808203695727707
      }
    },
    "strong": {
      "up3_before_down2": {
        "n_trades": 76,
        "cumulative_net_return": 0.04184795228351468,
        "win_rate": 0.5657894736842105,
        "max_drawdown": -0.03795111416307595,
        "sharpe": 0.2135318065557682
      },
      "up5_before_down3": {
        "n_trades": 74,
        "cumulative_net_return": 0.02127382817639334,
        "win_rate": 0.5,
        "max_drawdown": -0.06245213484406387,
        "sharpe": 0.1679381048944285
      },
      "up10_before_down5": {
        "n_trades": 72,
        "cumulative_net_return": 0.06250252929129774,
        "win_rate": 0.5138888888888888,
        "max_drawdown": -0.05550350588973729,
        "sharpe": 0.18018432329769932
      }
    }
  },
  "market_volatility": {
    "low": {
      "up3_before_down2": {
        "n_trades": 12,
        "cumulative_net_return": 0.0024670856988218137,
        "win_rate": 0.4166666666666667,
        "max_drawdown": -0.0043999999999999595,
        "sharpe": -0.011976816924680108
      },
      "up5_before_down3": {
        "n_trades": 8,
        "cumulative_net_return": 0.008197832300950614,
        "win_rate": 0.5,
        "max_drawdown": 0.0,
        "sharpe": 0.320683864719857
      },
      "up10_before_down5": {
        "n_trades": 10,
        "cumulative_net_return": 0.0075359949314719454,
        "win_rate": 0.4,
        "max_drawdown": 0.0,
        "sharpe": 0.18507738661917333
      }
    },
    "medium": {
      "up3_before_down2": {
        "n_trades": 76,
        "cumulative_net_return": 0.07802627237603899,
        "win_rate": 0.6052631578947368,
        "max_drawdown": -0.020473433807335684,
        "sharpe": 0.20131948825710091
      },
      "up5_before_down3": {
        "n_trades": 78,
        "cumulative_net_return": -0.006258004432345254,
        "win_rate": 0.48717948717948717,
        "max_drawdown": -0.053501657481164644,
        "sharpe": 0.16606401148787445
      },
      "up10_before_down5": {
        "n_trades": 75,
        "cumulative_net_return": -0.051535120932540646,
        "win_rate": 0.44,
        "max_drawdown": -0.07672385937079651,
        "sharpe": 0.1966182969788986
      }
    },
    "high": {
      "up3_before_down2": {
        "n_trades": 90,
        "cumulative_net_return": 0.027551414603891766,
        "win_rate": 0.5,
        "max_drawdown": -0.07103943114160849,
        "sharpe": 0.11726554352425686
      },
      "up5_before_down3": {
        "n_trades": 86,
        "cumulative_net_return": 0.04514325719061829,
        "win_rate": 0.45348837209302323,
        "max_drawdown": -0.05925108718633176,
        "sharpe": 0.17765755712035824
      },
      "up10_before_down5": {
        "n_trades": 89,
        "cumulative_net_return": 0.07396974768226339,
        "win_rate": 0.4157303370786517,
        "max_drawdown": -0.07261635507680231,
        "sharpe": 0.1117894118107342
      }
    }
  },
  "momentum_dispersion": {
    "low": {
      "up3_before_down2": {
        "n_trades": 13,
        "cumulative_net_return": 0.012207849861955333,
        "win_rate": 0.5384615384615384,
        "max_drawdown": -0.0074523842908531535,
        "sharpe": 0.2166082133627071
      },
      "up5_before_down3": {
        "n_trades": 13,
        "cumulative_net_return": 0.03902303198029844,
        "win_rate": 0.6923076923076923,
        "max_drawdown": 0.0,
        "sharpe": 0.586633302716459
      },
      "up10_before_down5": {
        "n_trades": 13,
        "cumulative_net_return": 0.02306723128048982,
        "win_rate": 0.46153846153846156,
        "max_drawdown": -0.005721303606012307,
        "sharpe": 0.29982934509675074
      }
    },
    "unknown": {
      "up3_before_down2": {
        "n_trades": 0,
        "cumulative_net_return": 0.0,
        "win_rate": null,
        "max_drawdown": 0.0,
        "sharpe": null
      },
      "up5_before_down3": {
        "n_trades": 0,
        "cumulative_net_return": 0.0,
        "win_rate": null,
        "max_drawdown": 0.0,
        "sharpe": null
      },
      "up10_before_down5": {
        "n_trades": 0,
        "cumulative_net_return": 0.0,
        "win_rate": null,
        "max_drawdown": 0.0,
        "sharpe": null
      }
    },
    "medium": {
      "up3_before_down2": {
        "n_trades": 73,
        "cumulative_net_return": 0.011800181664475273,
        "win_rate": 0.4657534246575342,
        "max_drawdown": -0.03613330965567885,
        "sharpe": 0.027954661176125155
      },
      "up5_before_down3": {
        "n_trades": 68,
        "cumulative_net_return": -0.057371219183302125,
        "win_rate": 0.3382352941176471,
        "max_drawdown": -0.07051564574135694,
        "sharpe": -0.09481325848481414
      },
      "up10_before_down5": {
        "n_trades": 72,
        "cumulative_net_return": -0.08463987397288708,
        "win_rate": 0.3055555555555556,
        "max_drawdown": -0.10582154071292205,
        "sharpe": -0.1578992411307073
      }
    },
    "high": {
      "up3_before_down2": {
        "n_trades": 92,
        "cumulative_net_return": 0.08416588172708694,
        "win_rate": 0.5978260869565217,
        "max_drawdown": -0.04217059136338763,
        "sharpe": 0.32785542053305944
      },
      "up5_before_down3": {
        "n_trades": 95,
        "cumulative_net_return": 0.06365760121618491,
        "win_rate": 0.5368421052631579,
        "max_drawdown": -0.0387526421701635,
        "sharpe": 0.22288145811306376
      },
      "up10_before_down5": {
        "n_trades": 97,
        "cumulative_net_return": 0.07301914862144221,
        "win_rate": 0.4536082474226804,
        "max_drawdown": -0.04270233399155754,
        "sharpe": 0.15932314440934148
      }
    }
  },
  "volume": {
    "normal": {
      "up3_before_down2": {
        "n_trades": 70,
        "cumulative_net_return": 0.09966136339081011,
        "win_rate": 0.6285714285714286,
        "max_drawdown": -0.027172126715839462,
        "sharpe": 0.3778064759352106
      },
      "up5_before_down3": {
        "n_trades": 66,
        "cumulative_net_return": 0.05241468729326648,
        "win_rate": 0.5757575757575758,
        "max_drawdown": -0.024753094415980947,
        "sharpe": 0.23058135669372565
      },
      "up10_before_down5": {
        "n_trades": 59,
        "cumulative_net_return": 0.05500323384364836,
        "win_rate": 0.4915254237288136,
        "max_drawdown": -0.02450360687504005,
        "sharpe": 0.20263549028686717
      }
    },
    "low": {
      "up3_before_down2": {
        "n_trades": 28,
        "cumulative_net_return": -0.014343540306588687,
        "win_rate": 0.39285714285714285,
        "max_drawdown": -0.026960533119999908,
        "sharpe": -0.110991683428867
      },
      "up5_before_down3": {
        "n_trades": 28,
        "cumulative_net_return": -0.0010245910431933014,
        "win_rate": 0.32142857142857145,
        "max_drawdown": -0.020228860831203566,
        "sharpe": -0.0009385575545655588
      },
      "up10_before_down5": {
        "n_trades": 32,
        "cumulative_net_return": -0.05979570831475389,
        "win_rate": 0.3125,
        "max_drawdown": -0.07794375091297845,
        "sharpe": -0.2447525991121115
      }
    },
    "high": {
      "up3_before_down2": {
        "n_trades": 80,
        "cumulative_net_return": 0.03667228736923667,
        "win_rate": 0.5125,
        "max_drawdown": -0.04736433616907498,
        "sharpe": 0.15135201658026287
      },
      "up5_before_down3": {
        "n_trades": 78,
        "cumulative_net_return": 0.02465582215487827,
        "win_rate": 0.4358974358974359,
        "max_drawdown": -0.09278760817873555,
        "sharpe": 0.09396896275595573
      },
      "up10_before_down5": {
        "n_trades": 79,
        "cumulative_net_return": 0.028130233114954883,
        "win_rate": 0.34177215189873417,
        "max_drawdown": -0.10062303770705006,
        "sharpe": 0.053073613762043136
      }
    }
  }
}

## Regime transitions

{
  "btc_volatility": {
    "stable_high": {
      "n": 457,
      "note": "ok"
    },
    "high_to_medium": {
      "n": 102,
      "note": "ok"
    },
    "stable_medium": {
      "n": 716,
      "note": "ok"
    },
    "medium_to_low": {
      "n": 48,
      "note": "ok"
    },
    "low_to_medium": {
      "n": 95,
      "note": "ok"
    },
    "medium_to_high": {
      "n": 101,
      "note": "ok"
    },
    "stable_low": {
      "n": 513,
      "note": "ok"
    },
    "high_to_low": {
      "n": 3,
      "note": "insufficient_sample"
    },
    "low_to_high": {
      "n": 47,
      "note": "ok"
    }
  },
  "market_breadth": {
    "stable_weak": {
      "n": 520,
      "note": "ok"
    },
    "weak_to_strong": {
      "n": 318,
      "note": "ok"
    },
    "stable_strong": {
      "n": 507,
      "note": "ok"
    },
    "strong_to_weak": {
      "n": 353,
      "note": "ok"
    },
    "strong_to_neutral": {
      "n": 188,
      "note": "ok"
    },
    "neutral_to_weak": {
      "n": 6,
      "note": "insufficient_sample"
    },
    "weak_to_neutral": {
      "n": 188,
      "note": "ok"
    },
    "neutral_to_strong": {
      "n": 2,
      "note": "insufficient_sample"
    }
  }
}

## Regime interactions

{
  "btc_trend\u00d7btc_volatility": {
    "neutral\u00d7low": {
      "n": 460,
      "note": "ok"
    },
    "bullish\u00d7low": {
      "n": 54,
      "note": "ok"
    },
    "neutral\u00d7medium": {
      "n": 450,
      "note": "ok"
    },
    "neutral\u00d7high": {
      "n": 203,
      "note": "ok"
    },
    "bearish\u00d7high": {
      "n": 157,
      "note": "ok"
    },
    "bullish\u00d7medium": {
      "n": 161,
      "note": "ok"
    },
    "bearish\u00d7medium": {
      "n": 302,
      "note": "ok"
    },
    "bullish\u00d7high": {
      "n": 245,
      "note": "ok"
    },
    "bearish\u00d7low": {
      "n": 50,
      "note": "ok"
    }
  },
  "btc_trend\u00d7market_breadth": {
    "neutral\u00d7neutral": {
      "n": 329,
      "note": "ok"
    },
    "neutral\u00d7weak": {
      "n": 369,
      "note": "ok"
    },
    "bullish\u00d7strong": {
      "n": 402,
      "note": "ok"
    },
    "neutral\u00d7strong": {
      "n": 415,
      "note": "ok"
    },
    "bearish\u00d7weak": {
      "n": 499,
      "note": "ok"
    },
    "bullish\u00d7weak": {
      "n": 11,
      "note": "insufficient_sample"
    },
    "bearish\u00d7strong": {
      "n": 10,
      "note": "insufficient_sample"
    },
    "bullish\u00d7neutral": {
      "n": 47,
      "note": "ok"
    }
  },
  "market_breadth\u00d7market_volatility": {
    "neutral\u00d7low": {
      "n": 94,
      "note": "ok"
    },
    "weak\u00d7low": {
      "n": 70,
      "note": "ok"
    },
    "strong\u00d7medium": {
      "n": 409,
      "note": "ok"
    },
    "strong\u00d7low": {
      "n": 17,
      "note": "insufficient_sample"
    },
    "weak\u00d7high": {
      "n": 547,
      "note": "ok"
    },
    "weak\u00d7medium": {
      "n": 262,
      "note": "ok"
    },
    "strong\u00d7high": {
      "n": 401,
      "note": "ok"
    },
    "neutral\u00d7medium": {
      "n": 235,
      "note": "ok"
    },
    "neutral\u00d7high": {
      "n": 47,
      "note": "ok"
    }
  },
  "market_volatility\u00d7momentum_dispersion": {
    "low\u00d7low": {
      "n": 47,
      "note": "ok"
    },
    "low\u00d7unknown": {
      "n": 40,
      "note": "ok"
    },
    "medium\u00d7medium": {
      "n": 250,
      "note": "ok"
    },
    "high\u00d7low": {
      "n": 59,
      "note": "ok"
    },
    "high\u00d7medium": {
      "n": 526,
      "note": "ok"
    },
    "medium\u00d7low": {
      "n": 89,
      "note": "ok"
    },
    "medium\u00d7high": {
      "n": 567,
      "note": "ok"
    },
    "high\u00d7high": {
      "n": 410,
      "note": "ok"
    },
    "low\u00d7high": {
      "n": 94,
      "note": "ok"
    }
  }
}

## Multiple-testing limitation

Many targets and comparisons have been tested in this project. Regime relationships are hypotheses for future testing, not confirmed effects.

## Survivorship limitation

Regime analysis uses the point-in-time universe. Survivorship is reduced but not eliminated: symbols delisted before the data window are absent.

## Leakage audit

- Passed: **True**
- Findings: []

## Final classification

**`strong_regime_dependence`**

Signal performance varies materially across measured regimes; regime context matters.

## Tests: **True**

## Errors: []
