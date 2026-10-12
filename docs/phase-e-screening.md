# Phase E — Screening and Comparison

## Status

E2 is implemented for **current-data screening and comparison**. It is deliberately not presented as a historical screener, a backtest, a complete fundamental ranking system, or investment advice.

## E1: bounded and deterministic screen

`ScreenSpec` supports the versioned progressive policy (`screen-v3-progressive`) and declares the universe, candidate ceiling, technical eligibility, score threshold, missing-data policy and a price-date alignment requirement. `ScreeningService` sends each bounded candidate through the existing cache-aware real-data provider and technical-analysis code.

Each candidate is visible as one of:

- `PASS`: verified current data met the disclosed technical rule and can be ranked;
- `FAIL`: verified current data did not meet that rule;
- `UNAVAILABLE`: upstream data failed, did not pass quality validation, or cannot be compared safely.

`UNAVAILABLE` is never ranked. A `PASS` is a screen result only, not a buy recommendation or a return forecast.

## E2: instrument universe and provenance

`VnstockInstrumentUniverseProvider` fetches the real current VCI listing catalogue through vnstock's `Listing.symbols_by_exchange` endpoint. It accepts only records labelled `STOCK` on an active board (`HSX`/`HOSE`, `HNX`, `UPCOM`), explicitly excludes VCI's `DELISTED` board, normalizes ticker/name/exchange/currency, and stores:

- an immutable raw capture in `D:\Download\thầy quế\Agent\.runtime\snapshots`;
- provider source URL, request parameters, SHA-256, snapshot ID and retrieval timestamp in `SourceReference`;
- a short-lived convenience cache at `.runtime/cache/current_instrument_universe_v2.json` (default 24 hours).

The service exposes the explicit universes `market_current`, `hose_current`, `hnx_current` and `upcom_current`. A normal chat request can say, for example, “Sàng lọc sàn HOSE” or “Tìm mã trên UPCOM”. The request remains bounded by `ScreenSpec.maximum_candidates` (default and maximum 100), so it does not create an unbounded upstream market scan.

The configured default universe now contains 100 unique Vietnamese tickers. `run_progressive()` ranks all successfully evaluated candidates once and uses successive halving until top-K remains. It is a bounded operational screen, **not** a claim that the configured 100 symbols represent the whole market. A liquidity/market-cap prefilter still requires a source that supplies those fields in bulk.

The VCI endpoint is a **retrieval-time catalogue**, not a source of historical index membership or verified listing/delisting effective dates. Therefore its universe has `captured_at`, source provenance and a symbol count, but deliberately has:

- `membership_as_of = null`;
- `point_in_time_supported = false`.

Any `ScreenSpec.as_of` request is rejected before results are generated. This prevents the current catalogue from being silently reused as a past universe and producing survivorship bias.

## Ranking and comparison guards

Technical scores are only ranked when all usable symbols have exactly the same price `as_of` date. If dates differ or are missing, the screen shows the affected result as `UNAVAILABLE`, attaches an explicit warning, and withholds the ranking. The response exposes universe size, evaluated count, ranks, `as_of`, source IDs, policy version and the universe capture time.

`ComparisonSpec` version `comparison-v1` independently checks currency, price `as_of` and financial reporting period. The comparison response exposes price date and financial period per ticker. It withholds the technical or fundamental score field when its relevant alignment fails, instead of comparing mismatched figures.

Current Phase D financial observations can be used for per-company analysis, but they are not yet a cross-company fundamental/valuation factor ranking: the free provider does not reliably supply a publication timestamp for every reporting period. That constraint is intentional.

## Verification

`tests/test_phase_e_screening.py` covers:

- bounded screens, reasons, provenance and provider failures;
- filtering the real-provider schema to listed equities;
- current exchange universe provenance and its explicit non-historical status;
- withholding a screen ranking when price dates differ;
- withholding comparison fields when price or financial-period alignment fails.

## Remaining external gate

Full historical screening/backtesting needs a licensed or official archive that supplies point-in-time constituent membership plus listing/delisting and corporate-action history. It cannot be completed honestly from a current VCI catalogue. That work belongs with Phase F once such a source is selected and ingested.
