# Research validation report (Binance live run)

Generated: **2026-09-19T06:27:21Z**  
Provider: **binance**  
Database: SQLite research store (`backend/data/research.db`) — Postgres was not reachable on this host; schema and pipeline match production.

Machine-readable output: [`research-validation-report.json`](research-validation-report.json)

## Dataset summary

| Metric | Value |
|--------|-------|
| Symbols | 48 |
| Raw 1h candles | 103,242 |
| Research samples | 16,591 |
| Labeled samples | 16,495 |
| Date range (candles) | 2026-06-21 07:00 UTC → 2026-09-19 05:00 UTC |
| Positive-class rate | 40.30% (6,647 / 16,495) |
| Features | 36 |
| Timeframe / horizon / move | 1h / 12h / 3% |
| Sample stride | 6 bars |
| Duplicate (symbol, ts) groups | 0 |

### Symbols

AAVEUSDT, ADAUSDT, APTUSDT, ARBUSDT, ARUSDT, AVAXUSDT, BANKUSDT, BCHUSDT, BNBUSDT, BTCUSDT, CRCLBUSDT, DASHUSDT, DOGEUSDT, ENAUSDT, ETHUSDT, FETUSDT, FILUSDT, FUSDT, GUSDT, INJUSDT, LINKUSDT, LSKUSDT, LTCUSDT, NEARUSDT, ONDOUSDT, ONEUSDT, OPUSDT, PAXGUSDT, PEPEUSDT, PROVEUSDT, PUMPUSDT, REZUSDT, RLUSDUSDT, SNDKBUSDT, SOLUSDT, SPYBUSDT, STRKUSDT, SUIUSDT, TAOUSDT, TRUMPUSDT, TRXUSDT, UNIUSDT, UUSDT, WLDUSDT, XLMUSDT, XRPUSDT, ZAMAUSDT, ZECUSDT

## Label breakdown

Definition: `significant_move=True` if |fwd_return|≥3% OR max_upside≥3% OR max_drawdown≤−3% over the next 12h. **Up and down both count as positive.**

| Bucket | Count |
|--------|------:|
| Upside-path / up close (positive) | 3,306 |
| Drawdown-path / down close (positive) | 2,747 |
| Both directions touched (positive) | 594 |
| Other positives | 0 |

## Walk-forward results

| Fold | N | Pos rate | AUC | Brier | Baseline Brier | Top-quintile lift | Test window |
|------|---:|---:|---:|---:|---:|---:|---|
| 1 | 2,688 | 37.6% | 0.827 | 0.164 | 0.235 | 2.22 | 2026-08-09 → 2026-08-22 |
| 2 | 2,688 | 49.9% | 0.793 | 0.188 | 0.250 | 1.71 | 2026-08-23 → 2026-09-05 |

| Aggregate | Value |
|-----------|------:|
| Fold count | 2 |
| Mean AUC ± std | **0.810 ± 0.024** |
| Mean Brier ± std | **0.176 ± 0.017** |
| Mean baseline Brier | 0.242 |
| Mean top-quintile lift | 1.97× |
| Beats baseline (lower Brier) | **Yes** |
| Mean AUC vs 0.5 | +0.310 |

## Model promotion

| Field | Value |
|-------|-------|
| Promoted (walk-forward criteria) | Yes |
| Saved status | active |
| Model ID | 385924ad-aa97-4039-a490-87cf9c88e5e4 |
| Live uses active blob | Yes |

Criteria: ≥2 folds, mean AUC ≥ 0.55, mean Brier < mean constant base-rate Brier. Training score alone does not promote.

## Leakage audit

- Samples checked: 200  
- Label/feature mismatches: **none**  
- Folds violating 12h embargo: **0**

## Survivorship bias

Universe = **today’s top 48 USDT pairs by 24h quote volume**, applied to historical rows. Delisted or illiquid names that were not in today’s set are excluded. **This is potential survivorship bias** and can inflate apparent stability of patterns.

## Statistical caveats

- Label windows overlap (~50% with stride 6h and horizon 12h); effective independent N is smaller than 16,495.  
- Only **2** walk-forward folds.  
- Positive rate shifts between folds (37.6% → 49.9%).

## Live prediction path

- Recorded this run: 48  
- Resolved this run: 0 (12h horizon not elapsed)  
- Live AUC/Brier: not yet available

## Tests

`tests.test_research`: **7/7 passed** (leakage / causality unit tests).

## Errors

None.

## Reproduce

```bash
MARKET_DATA_PROVIDER=binance \
DATABASE_URL='sqlite+aiosqlite:///./backend/data/research.db' \
PYTHONPATH=backend python -m scripts.validate_research
```

Use your Postgres `DATABASE_URL` from `.env` when Docker Postgres is running (replace `postgres` host with `localhost` from the host machine).
