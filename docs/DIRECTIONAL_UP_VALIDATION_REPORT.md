# Directional-up validation report (`directional_up_3pct_12h_v1`)

Generated: **2026-09-19T06:37:59.557001+00:00**  
Provider: **binance**  
Database: `sqlite+aiosqlite:////home/yamiwano/Projects/Yami-finance/backend/data/research.db`

Machine-readable: [`directional-up-validation-report.json`](directional-up-validation-report.json)

Prior bidirectional report is unchanged: [`RESEARCH_VALIDATION_REPORT.md`](RESEARCH_VALIDATION_REPORT.md).

## Experiment configuration

- **Experiment ID:** `directional_up_3pct_12h_v1`
- **Primary target:** `large_up_move` = max_upside >= 0.03 over 12h
- **Secondary (analysis only):** `clean_up_move`
- **Timeframe / horizon:** 1h / 12h
- **Primary sampling:** **12-bar stride** (non-overlapping 12h label windows)
- **Sensitivity sampling:** existing **6-bar stride**
- **Features:** full 36 vs ATR-only `['atr_pct']`

## Dataset

| Metric | Primary (stride 12) | Sensitivity (stride 6) |
|--------|---:|---:|
| Labeled samples | 8271 | 16495 |
| Positive rate (large_up_move) | 0.24313867730625074 | 0.23643528341921793 |
| Raw candles | 103242 | (same) |
| Symbols | 48 | (same) |
| Candle range | 2026-06-21T07:00:00+00:00 → 2026-09-19T05:00:00+00:00 | (same) |

## Primary walk-forward (stride 12): full vs ATR-only

| Aggregate | Full model | ATR-only |
|-----------|----------:|---------:|
| Mean AUC ± std | 0.7463397879749243 ± 0.0374866830314543 | 0.7479460488863716 ± 0.04723373328877474 |
| Mean Brier | 0.20962431327891662 | 0.21575099931234498 |
| Mean baseline Brier | 0.20197432814271543 | 0.20197432814271543 |
| Mean top-quintile lift | 1.9614156384256447 | 2.0888248565295027 |
| Mean top-quintile precision | 0.5503731343283582 | 0.585820895522388 |
| AUC delta (full − ATR) | -0.0016062609114473325 | — |
| Brier delta (full − ATR) | -0.0061266860334283635 | — |

- Beats constant base-rate baseline: **False**
- Full adds measurable value beyond ATR: **False** (required AUC margin ≥ 0.02)
- Experiment status: **`rejected`**
- Promoted over live bidirectional model: **No** (by design)

## Interpretation answers

1. **How many samples were produced?** Primary stride-12 labeled n=8271; sensitivity stride-6 labeled n=16495.

2. **What was the positive class rate?** large_up_move rate (primary)=0.24313867730625074.

3. **What were the walk-forward AUC/Brier results?** Full mean AUC=0.7463397879749243, mean Brier=0.20962431327891662; ATR-only mean AUC=0.7479460488863716, mean Brier=0.21575099931234498.

4. **How did the full model compare with ATR-only?** AUC delta (full−ATR)=-0.0016062609114473325; Brier delta (full−ATR)=-0.0061266860334283635.

5. **Did the model identify upward moves specifically, or mostly volatility?** Still appears largely volatility-driven: full-model AUC is within the ATR-only margin.

6. **Did the model beat the constant base-rate baseline?** False

7. **Did the full feature model materially outperform ATR-only?** False

8. **What were the top-quintile precision and lift?** precision=0.5503731343283582, lift=1.9614156384256447.

9. **Were there any leakage issues?** findings=[], embargo_violations=0.

10. **What survivorship/selection bias remains?** Universe remains today's top-N USDT pairs by 24h quote volume applied to history. Delisted/illiquid names absent from today's set are missing — survivorship/selection bias remains.

11. **Should this experiment be considered evidence of directional predictive value?** No. Status=`rejected`. Do not treat as directional predictive edge; do not proceed to the next research stage on this evidence alone.

## Leakage / bias

- Leakage findings: []
- Embargo violations: 0
- Survivorship: Universe remains today's top-N USDT pairs by 24h quote volume applied to history. Delisted/illiquid names absent from today's set are missing — survivorship/selection bias remains.

## Tests

- Research suite OK: **True**

## Errors

- []
