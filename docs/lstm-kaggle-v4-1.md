# LSTM V4.1 development experiment

## Why V4.1 exists

The first V4 run produced a small five-session MAE advantage but its 95% block-bootstrap interval crossed zero. It also selected the smallest model at its first epoch and lost 171,950 possible rows to short fragments. The result indicated three concrete problems:

- activity and liquidity rules removed context rows instead of only controlling forecast eligibility;
- the volatility-scaled excess-return plus market-return decomposition was more complex than the available signal justified; and
- the classification objective competed too strongly with return regression.

V4 is preserved unchanged. V4.1 is a separate iteration, not a replacement for its historical result.

## Changes

V4.1 makes the following predeclared changes:

- Finite, continuous rows remain available as causal sequence context even when they are not eligible forecast endpoints.
- Activity and liquidity thresholds are applied only to each sequence endpoint.
- The regression head predicts train-winsorized and train-standardized next-session and five-session returns directly.
- The future market decomposition and volatility target inversion are removed.
- Regression uses MAE, matching the actual selection objective.
- Auxiliary `DOWN/NEUTRAL/UP` loss weight is reduced from `0.20` to `0.05`.
- Artifact names contain `v4_1`, so an existing V4 result cannot be overwritten accidentally.

All V4 protections remain: chronological split, five-session embargo, validation-only sweep and calibration, train-only preprocessing, validation-selected baselines, block bootstrap, segmented reporting and separate horizon eligibility.

## Important evaluation status

The original V4 output exposed the final period of the current one-million-row dataset. Consequently, that period is no longer an independent holdout for changes made in V4.1.

V4.1 therefore uses these statuses:

- `EXPERIMENTAL`: it does not clear the numerical gates.
- `DEVELOPMENT_CANDIDATE`: it clears the numerical gates on the reused period, but is not independently confirmed.
- `RESEARCH_CANDIDATE`: reserved for a future run on newly collected, previously unseen dates.

With the current dataset, `integration_eligible` remains false even if V4.1 clears every numerical gate. This prevents iterative test-set tuning from being presented as independent evidence.

## Run on Kaggle

1. Keep the private Kaggle Dataset containing `vci_ohlcv_lstm_1m.csv` attached.
2. Upload `notebooks/train_lstm_kaggle_v4_1.ipynb` as a new notebook.
3. Enable a GPU and select **Run all**.
4. Download all six `stock_lstm_v4_1.*` artifacts from `/kaggle/working`.

Expected outputs:

- `stock_lstm_v4_1.keras`
- `stock_lstm_v4_1.spec.json`
- `stock_lstm_v4_1.metrics.json`
- `stock_lstm_v4_1.history.csv`
- `stock_lstm_v4_1.validation_sweep.csv`
- `stock_lstm_v4_1.test_predictions.csv`

Do not overwrite the V4 output files. For an independent confirmation, append genuinely new market dates to a newly versioned dataset and change `INDEPENDENT_HOLDOUT` only when none of those dates or aggregate results has been inspected during model development.
