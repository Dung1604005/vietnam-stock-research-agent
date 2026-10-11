# Phase D — Analysis Engine Baseline

## Status

The deterministic Phase D baseline is implemented. It improves analysis quality only where a verified structured input exists. It does not claim that missing benchmark, event, portfolio, publication-time, or adjustment data exists.

## Completion review

| Roadmap outcome | Status | Evidence / boundary |
|---|---|---|
| One technical implementation for analysis and charts | Done | `TechnicalIndicatorToolkit` is the only indicator implementation. |
| Horizon changes calculation inputs | Done | Short/medium/long profiles alter windows and minimum history. |
| Liquidity, volatility, regime and adjustment diagnostics | Done | Deterministic result fields with source/as-of lineage. |
| Benchmark-relative return | Done for aligned sessions | Data acquisition attempts to retrieve VNINDEX through the same verified provider; unavailable or misaligned series stays `UNKNOWN`. |
| Multi-period fundamental inputs | Done with availability guards | Annual and provider-limited quarterly recognised rows are retained; only annual, period-labelled rows feed current growth/cash-conversion calculations. |
| TTM, dilution, dividend and valuation history | Blocked safely | Quarterly rows and corporate-action records exist, but the free source has no authoritative statement publication timestamp or validated adjustment policy. |
| Sector-specific treatment | Done for supported templates | Golden calculation fixtures cover bank, securities, insurance, real estate and generic templates; VCI profile sector is current metadata, not a point-in-time sector master. |
| Data, market, business, event and portfolio risk sections | Done with explicit availability | Verified VCI issuer events and user-supplied portfolio weights are assessed when present; otherwise they are `UNKNOWN`. |
| Versioned score/rating policy and evidence lineage | Done, uncalibrated | `technical-v2`, `fundamental-v2`, `risk-v2`, `synthesis-v2`; no predictive calibration claim. |

## Technical analysis

`TechnicalIndicatorToolkit` is now the single indicator implementation for both `TechnicalAgent` and `ChartService`. The requested horizon selects a deterministic profile:

| Profile | Typical horizon | SMA | RSI | Volatility / return window |
|---|---:|---|---:|---:|
| short | up to 45 days | 10 / 20 | 7 | 20 sessions |
| medium | 46–269 days | 20 / 50 | 14 | 60 sessions |
| long | 270+ days or graph re-analysis | 50 / 150 | 14 | 120 sessions |

The technical result records `technical-v2`, profile, feature contributions, liquidity diagnostic, regime, price-adjustment status, source lineage, and as-of date. It uses average traded value (`close × volume`) as a configurable liquidity diagnostic. The default is `STOCK_AGENT_MIN_AVG_TRADED_VALUE_VND=1000000000`; it is a data-quality filter, not a personalised recommendation threshold.

When `STOCK_AGENT_BENCHMARK_ENABLED=true` (the default), `DataAgent` also attempts to acquire VNINDEX OHLCV from the same provider, with its own source reference, timestamp and snapshot. Technical analysis calculates an issuer-minus-VNINDEX return only over common trading dates for the selected horizon. The technical and full-analysis responses disclose those VNINDEX provenance records alongside issuer sources. A benchmark failure or insufficient overlap leaves relative strength explicitly `UNKNOWN` and never invalidates the issuer-data response. Beta and correlation are not calculated. Price-adjustment status `UNKNOWN` reduces confidence and produces a warning rather than pretending that corporate actions were handled.

## Fundamental analysis

Fundamental output now uses typed evidence with input fields, source IDs and as-of date. It records `fundamental-v2`, feature contributions, the selected template, and unavailable inputs.

The current templates are `BANK`, `SECURITIES`, `INSURANCE`, `REAL_ESTATE`, and `NON_FINANCIAL_GENERIC`. The live VCI provider now makes a bounded `Company.overview` call and preserves its returned company name, sector and exchange as a `COMPANY_PROFILE` source reference/snapshot. A missing profile keeps the template as `NON_FINANCIAL_GENERIC_UNVERIFIED_SECTOR`. Banking/financial templates do not treat debt-to-equity as generic corporate leverage.

The provider retrieves annual income statement, balance sheet and cash-flow tables when `STOCK_AGENT_FINANCIAL_STATEMENTS_ENABLED=true`, plus provider-limited quarterly rows when `STOCK_AGENT_QUARTERLY_STATEMENTS_ENABLED=true`. Only recognised rows with explicit period labels are retained, each with a statement-specific source ID, immutable snapshot, checksum and reporting period. The free source does not supply an authoritative statement publication timestamp, so quarterly rows are evidence only and do not feed a point-in-time TTM calculation. Where ratio growth is missing, or ratios are stale while annual statements remain current, the engine calculates two-period reported revenue/profit growth; it calculates operating-cash conversion only when operating cash flow and net profit share the same reported period.

`DataAgent` also retrieves explicit issuer events when `STOCK_AGENT_ISSUER_EVENTS_ENABLED=true`. VCI event records are saved with their source reference and raw snapshot. Recognised dividend, share-issue, additional-listing and trading-suspension records populate the corporate-action contract; unrecognised event codes remain events, not inferred actions.

## Risk analysis

`risk-v2` decomposes output into these sections:

- `market`: realised volatility and drawdown;
- `liquidity`: average traded value;
- `data`: data-quality report;
- `business_and_accounting`: available leverage and fundamental weaknesses;
- `event_news`: assessed from verified issuer-event records when available, otherwise `UNKNOWN`;
- `portfolio`: deterministic concentration (normalized maximum weight and HHI) when a caller supplies `PortfolioPosition` weights, otherwise `UNKNOWN`. Cross-asset correlation remains unavailable until return series are explicitly supplied for every holding.

Risk confidence is reduced by data-quality issues and fundamental abstention. The existing veto remains a deterministic safety constraint, not a prediction of loss.

## Evidence contract

`EvidenceItem` replaces free-form analytical evidence in Technical and Fundamental output. Every item carries a label, value, input-field names, source IDs, as-of date, confidence, and optional note. `RiskSection` makes unavailable risk inputs visible with `UNKNOWN` instead of silently omitting them.

## Verification

`tests/test_phase_d_analysis.py` covers horizon-dependent parameter selection, shared chart/analysis RSI, typed evidence lineage, aligned VNINDEX relative-return logic, safe benchmark failure, hand-calculated golden scores for every supported sector template, risk sections, verified-event/portfolio handling, financial-sector leverage treatment, statement-row normalisation, stale-ratio/current-statement handling, reported growth and cash conversion. The full pytest suite must remain green before changes are accepted.

## Remaining Phase D gates

- Validate provider adjustment semantics across issuer and index series before adding beta or correlation.
- Add an instrument master with historical sector classifications; `Company.overview` is current-profile metadata, not point-in-time classification history.
- Ingest an official filing feed with publication timestamps before point-in-time TTM, dilution, dividends, or valuation history can be calculated.
- Add reviewed news/filing evidence from Phase C before interpreting issuer events as a full event-risk assessment.
- Supply return series for every portfolio holding before correlation or scenario risk can be calculated.
- Run Phase F point-in-time evaluation before interpreting score thresholds as economically validated.
