# Price forecasting: implemented models and policy

Reviewed 23/10/2026. Forecasting is a Python model service, not an additional LLM agent.

## Construction and loading

`PriceForecastService(trained_model=...)` accepts an adapter. A bare
`PriceForecastService()` and the default constructor of `AnalysisService()` use
the explicitly untrained rolling-return baseline. Environment-aware construction
uses `PriceForecastService.from_environment(...)`; UI can defer model loading and
select V2 or V4 per session without modifying .env. The current CLI constructs
bare `AnalysisService()` and therefore also uses the baseline. A configured model
name does not prove its artifact was loaded.

| Path | Runtime behaviour / limit |
|---|---|
| Baseline | Mean of last 20 observed session returns; BASELINE_UNTRAINED, confidence 0, zero Technical weight |
| V2 | Existing frozen adapter/default selection when LSTM is enabled; legacy demo policy, not revalidated by newer gates |
| V3 | Explicit artifact directory; eligibility and blend weight checked; optional shadow remains zero weight |
| V4 | Explicit selected archive; UI opt-in experimental blend; gate/eligibility metadata remains unchanged |
| Invalid/missing artifact or inference error | Labelled baseline fallback, no baseline scoring; no silent V4-to-V2 substitution |

Deferred UI forecast uses baseline in QUICK/OFFLINE and avoids loading TensorFlow
solely for startup/readiness. This is a property of that construction path, not a
claim that arbitrary injected model services have the same loading behaviour.

## Blend contract

TechnicalAgent._apply_lstm_blend checks TRAINED_MODEL, one-session horizon,
valid current/forecast prices, usable Technical history, eligibility and weight.
The explicit V4 experimental path additionally checks model version,
VERIFIED_FILES, opt-in flag and 0.2 weight; it does not change policy_eligibility
to true. Merely returning TRAINED_MODEL is insufficient for newer artifacts.

For the 20% demo blend:

```text
return = forecast_price / current_price - 1
lstm_score = 5 + 5 * clip(return, -0.03, 0.03) / 0.03
final_technical_score = 0.8 * indicator_score + 0.2 * lstm_score
```

The code caps declared model weight at 0.2 and can accept a smaller eligible
weight. Baseline/unavailable/shadow forecasts contribute zero. Diagnostics retain
version, pre-blend score, weights, return and score delta. 20% is a policy weight,
not accuracy or expected return. One/five-session estimates are not three-month targets.
Intervals/confidence are diagnostics unless independently calibrated.

## Code and model evidence

- `src/stock_multi_agent/services/price_forecast.py`: loading, fallback and deferred service.
- `src/stock_multi_agent/services/lstm_forecast.py`, `lstm_forecast_v3.py`, `lstm_forecast_v4.py`: adapters.
- `src/stock_multi_agent/agents/technical_agent.py`: _apply_lstm_blend.
- `app/streamlit_app.py`: session model selection and service construction.
- [V2 reproduction](lstm-kaggle-training.md), [V3 model card](lstm-v3-model-card.md), [V4 development review](lstm-v4-research-and-training.md), [UI selection](lstm-ui-model-selection.md).

V3/V4 historical GPU artifact reviews are separate from software regression.
This documentation audit does not rerun GPU training, large-artifact checksum
audits or forward-market evaluation; their prior reports retain dated scope.

## Dataset collection

```powershell
.\.venv\Scripts\python.exe scripts\collect_lstm_ohlcv_dataset.py --target-rows 100000
```

The resumable collector writes observed OHLCV and next-observed-session labels
with provenance/metadata in .runtime/ml on D. Current-listing selection has
survivorship bias and unknown adjustment status; it is not sufficient for a
validated historical investment study. New experiments need time-aware splits,
train-only preprocessing, baseline comparisons and separate eligibility gates.
