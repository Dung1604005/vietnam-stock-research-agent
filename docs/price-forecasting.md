# Price Forecasting Boundary

## What is available now

Every single-ticker analysis displays the latest observed close, its market-data `as_of` date, and a next-session estimate. Until a trained model is supplied, that estimate uses `rolling-mean-return-baseline-v1`: the mean daily return of the latest 20 sessions plus a volatility diagnostic range.

Its status is always `BASELINE_UNTRAINED` and confidence is `0`. It is a visible integration baseline, not a validated prediction, investment recommendation, or trade instruction.

## Deep-learning integration later

`PriceForecastService` accepts a `PriceForecastModel` implementation. A trained adapter must implement `predict_next_session(market_data)` and return a `PriceForecast` with:

- `status=TRAINED_MODEL`;
- the same ticker and `as_of` date as the input data;
- model version, horizon, point forecast, uncertainty bounds and calibrated confidence;
- a reproducible model artifact and feature/data version.

Do not train or report a deep-learning forecast until Phase F has a point-in-time dataset with adjusted-price policy, train/validation/test time splits, a naive baseline comparison, and out-of-sample error/calibration metrics. The forecasting service is deterministic infrastructure, not another LLM agent. A valid one-session `TRAINED_MODEL` forecast is converted to a bounded 0-10 signal and contributes 20% to the Technical Agent score; indicator evidence contributes 80%. Baseline/unavailable forecasts contribute nothing.

The runtime blend is `final_score = 0.80 * indicator_score + 0.20 * lstm_score`. The next-session predicted return is mapped linearly onto 0-10: -3% maps to 0, 0% maps to 5, and +3% maps to 10; values outside that range are clipped. The response records the pre-blend score, LSTM score, predicted return, weights and score delta in Technical diagnostics.

## Collecting an offline LSTM experiment dataset

The project includes a resumable real-data collector. It writes one row per observed ticker/session with raw `date/open/high/low/close/volume`, provenance, and labels from the *next observed session* (`target_date`, `next_session_close`, `next_session_return`). It does not fabricate missing sessions or OHLCV values.

```powershell
.\.venv\Scripts\python.exe scripts\collect_lstm_ohlcv_dataset.py --target-rows 100000
```

The output is kept outside Git at `.runtime/ml/vci_ohlcv_lstm_100k.csv` alongside metadata containing the source, ticker failures, requested lookback, schema, row count and SHA-256. The command can be safely rerun to continue after a rate limit or interrupted terminal session: already completed tickers are skipped.

By default it requests four calendar years per current VCI-listed ticker, so it usually needs materially fewer API calls than the chat-facing 550-day history. Its current-listing catalogue has survivorship bias and VCI prices have `UNKNOWN` adjustment status. It is suitable for a classroom LSTM experiment only, not a historical performance claim or trading recommendation.
