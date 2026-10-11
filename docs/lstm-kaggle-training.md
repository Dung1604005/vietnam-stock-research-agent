# LSTM training on Kaggle

## Dataset

The real, untracked dataset is `.runtime/ml/vci_ohlcv_lstm_1m.csv`. Collection is complete: 1,000,000 labelled rows from 991 current-listed tickers. Its paired metadata file records the source, schema, failures, row count and SHA-256.

Prices have `UNKNOWN` adjustment status and the current listing catalogue has survivorship bias. The dataset is suitable for a course demonstration, not a historical performance claim.

## Recommended version 2 run

1. Create a private Kaggle Dataset and upload `vci_ohlcv_lstm_1m.csv` plus `vci_ohlcv_lstm_1m.metadata.json`.
2. Create a new Kaggle Notebook and attach that private dataset.
3. Select **Accelerator → GPU**.
4. Copy all Python from `notebooks/train_lstm_kaggle_v2.py` into the first code cell. Do not paste raw `.ipynb` JSON into a code cell.
5. Run the cell and download the four files produced under `/kaggle/working`:
   - `stock_lstm_multihorizon_demo.keras`
   - `stock_lstm_multihorizon_demo.spec.json`
   - `stock_lstm_multihorizon_demo.metrics.json`
   - `stock_lstm_multihorizon_demo.history.csv`

The improved version uses 22 price, volume, liquidity and cross-sectional market-context features; a 30-session sequence; ticker embeddings; and both next-session and five-session targets. It optimizes MAE directly, filters only on information available at prediction time, and calibrates forecast shrinkage on validation only. Feature clipping, feature scaling and target scaling are fitted on train only. Train, validation and test are separated chronologically according to the date when the five-session label becomes available.

## Artifact policy

The model is always exported so the application can demonstrate a complete deep-learning lifecycle:

- `RESEARCH_CANDIDATE`: at least one horizon beats the zero-return baseline by holdout MAE. Only passed horizons may be considered for research display.
- `EXPERIMENTAL_REJECTED`: neither horizon passes. The model may appear in an explicitly labelled classroom-demo panel, but it must not influence BUY/SELL/HOLD or confidence.

Deep learning remains a deterministic forecasting component behind `PriceForecastService`; it is not an additional LLM agent and never routes orders.

## Final classroom selection: V2

The frozen artifacts supplied from Kaggle are stored in `notebooks/trainResult`. `manifest.json` pins every file by SHA-256. Runtime loading rejects a missing, modified, ineligible or metadata-inconsistent artifact.

V2 is the final classroom-demo selection because its original declared holdout gate passed both horizons. This is not a claim that V2 is a strong forecasting model: its next-session MAE advantage over zero return was only 0.0842%, its next-session directional accuracy was 45.41%, and it did not use the later V4 embargo/bootstrap protocol.

Install the optional inference dependency into the workspace-local virtual environment on drive D:

```powershell
$env:PIP_CACHE_DIR = "D:\Download\thầy quế\Agent\.cache\pip"
$env:TEMP = "D:\Download\thầy quế\Agent\.tmp\pip"
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = "1"
$env:PIP_PROGRESS_BAR = "off"
.\.venv\Scripts\python.exe -m pip install --progress-bar off -e ".[ml]"
```

The Streamlit app then loads the checksummed V2 artifact automatically. Set `STOCK_AGENT_LSTM_ENABLED=false` to disable it, or set `STOCK_AGENT_LSTM_ARTIFACT_DIR` to an alternative directory on drive D.

Single-ticker application requests cannot reproduce V2's cross-sectional median-return and liquidity-percentile features. The adapter substitutes their frozen train means, which map to zero after standardization, and exposes this approximation in every forecast warning. The displayed interval is RMSE-based diagnostic dispersion rather than a calibrated prediction interval. V2 output is shown as a low-confidence research estimate and never changes BUY/SELL/HOLD.

## Leakage-aware experiment v4

The newer V4 pipeline is documented in [lstm-kaggle-v4.md](lstm-kaggle-v4.md). It preserves V2/V3, performs a validation-only four-configuration sweep, adds five-session embargoes, generates windows by batch, evaluates stronger baselines and applies a stricter statistical eligibility gate. Run `notebooks/train_lstm_kaggle_v4.ipynb` directly on Kaggle; do not use the test output to tune another V4 run.

After the first V4 result was inspected, the follow-up [V4.1 development experiment](lstm-kaggle-v4-1.md) was frozen as separate code and artifact names. It retains valid context rows, filters only prediction endpoints, returns to direct-return targets and reduces the auxiliary classification weight. Its current holdout is explicitly marked as reused; it cannot become integration-eligible until confirmed on newly collected dates.

## Clean-data experiment v3

The original 1M dataset and v2 training code are preserved. A separate reproducible cleaning command creates a smaller quality-focused dataset:

```powershell
.\.venv\Scripts\python.exe scripts\build_clean_lstm_dataset.py
```

New files:

- `.runtime/ml/vci_ohlcv_lstm_clean_v1.csv`
- `.runtime/ml/vci_ohlcv_lstm_clean_v1.metadata.json`
- `notebooks/train_lstm_kaggle_clean_v3.py`

The cleaner selects its ticker universe using only observations available by the train cutoff, requires sufficient history and trading activity, keeps the upper half by historical liquidity, rejects abnormal calendar gaps and removes returns above 20% that cannot be trusted while adjustment status is unknown. It also records every rule, count and checksum in metadata.

For the new run, upload the clean CSV and metadata as a new private Kaggle Dataset, then paste `train_lstm_kaggle_clean_v3.py` into a new GPU notebook. Version 3 prevents rolling features and 30-session windows from crossing rows removed by cleaning, uses a smaller regularized LSTM to reduce the early overfitting seen in v2, and writes:

- `stock_lstm_clean_v3.keras`
- `stock_lstm_clean_v3.spec.json`
- `stock_lstm_clean_v3.metrics.json`
- `stock_lstm_clean_v3.history.csv`
