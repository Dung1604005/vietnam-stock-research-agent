# Phase F — Evaluation and Point-in-Time Backtesting

## Status

The reproducible evaluation and backtesting **harness** is implemented. It does not claim that the existing current-data provider can produce a valid historical result. A full Vietnam-market performance study remains blocked until a point-in-time historical source is ingested.

## Implemented components

`BacktestDataset` is a versioned offline manifest. It requires:

- a historical universe snapshot for every rebalance date, with `INSTRUMENT_UNIVERSE` provenance and `point_in_time_supported=true`;
- immutable signal snapshots with `as_of`, timezone-aware `available_at`, source IDs and one policy version;
- adjusted OHLCV observations with source references for both securities and the benchmark;
- an explicit dataset version, benchmark and strategy configuration.

`PointInTimeBacktestEngine` is deterministic and has no LLM, live fetch, broker, or order-routing capability. It ranks eligible signals in each historical universe, enters only after a configured session lag, applies disclosed transaction costs and reports:

- gross/net/benchmark/active return;
- turnover and total transaction cost;
- annualized return/volatility, zero-rate Sharpe, maximum drawdown and hit rate;
- a deterministic bootstrap confidence interval for mean period excess return;
- skipped periods/failure slices and explicit walk-forward date windows.

The engine refuses to run when it detects a current/non-point-in-time universe, post-decision signal availability, unadjusted prices under an adjusted-price policy, future-dated provenance, duplicate records, policy mixing, missing executable prices, or missing benchmark sessions. This is intentional fail-closed behaviour.

`EvaluationRegistry` writes an append-only report envelope to `.runtime/evaluations` with input version, configuration, code revision (when supplied as `GIT_COMMIT`) and SHA-256 of the report. Nothing is written to the repository or used as a live trading signal.

## Run an offline backtest

Create a validated historical JSON manifest outside Git-tracked files, then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_point_in_time_backtest.py path\to\dataset.json `
  --walk-forward 24 6 6 --write-report
```

The script will reject a current VCI catalogue or normal current chat/screen output. It must receive a separate historical source with point-in-time membership and adjusted historical prices.

## Free demo path: collect forward data, do not reconstruct history

For a course demo, use the free VCI/vnstock provider only to record observations **from now onward**. Run once after the market close on each trading day for a small bounded set:

```powershell
.\.venv\Scripts\python.exe scripts\capture_forward_evaluation.py FPT VNM HPG MWG VCB SSI
```

The command saves an immutable capture under `.runtime/forward_evaluation/captures/`. It captures the active listing catalogue, each available technical score, the latest price and VNINDEX with source provenance. It does not backfill, scrape historical membership, or invent missing observations.

After at least three distinct capture dates, compile the observed data. Three captures are needed for at least one observable holding period: a signal date, the next captured session for entry, and a later captured session for exit.

```powershell
.\.venv\Scripts\python.exe scripts\build_forward_backtest_dataset.py --version demo-forward-2026q4-v1
```

VCI price-adjustment status can be `UNKNOWN`. The strict backtest default refuses those records. For a classroom-only exploratory report, explicitly opt in to the checked-in non-adjusted config and label the output accordingly:

```powershell
.\.venv\Scripts\python.exe scripts\run_point_in_time_backtest.py .runtime\forward_evaluation\datasets\demo-forward-2026q4-v1.json `
  --config evaluation\configs\demo-forward-unadjusted-v1.json --write-report
```

This demo configuration is not evidence of investment performance. It is only an auditable demonstration that Data Service, Technical Engine, Evaluation Registry and Risk discipline can work together on real observations without a broker or paid API.

## Existing evaluation coverage

- Semantic evaluation already has a deterministic evaluator, holdout overlap guard, calibration summary and optional reports under `.runtime/evaluations`.
- The CI suite includes unit, routing, provider-contract, research and new point-in-time backtest tests.
- Current research tests verify provenance/citation behaviour, but there is not yet a human-labelled retrieval or faithfulness benchmark.

## Remaining external-data gates

Phase F cannot be labelled performance-validated until all of the following are available:

1. Licensed or official historical instrument membership, listing/delisting and corporate-action data.
2. Adjusted price history with a documented revision policy.
3. Financial-statement/document publication timestamps for point-in-time fundamental and RAG features.
4. A pre-registered strategy/metric plan, held-out walk-forward periods, realistic liquidity/cost assumptions and failure-slice review.

Until then, `BUY`/`HOLD`/`WATCH` labels remain uncalibrated analytical outputs, never predictions or backtested investment recommendations.
