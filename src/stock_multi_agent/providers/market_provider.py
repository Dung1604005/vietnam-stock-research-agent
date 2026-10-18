from __future__ import annotations

import json
import math
import re
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from stock_multi_agent.models.analysis import FinancialMetrics, MarketData, PriceBar
from stock_multi_agent.models.data import (
    DataKind,
    CorporateAction,
    FinancialMetricObservation,
    FinancialStatementObservation,
    Instrument,
    InstrumentType,
    IssuerEvent,
    PriceAdjustment,
    QualitySeverity,
    QualityStatus,
    SourceReference,
)
from stock_multi_agent.providers.data_store import MarketDataCache, RawSnapshotStore, SQLiteMarketDataCache
from stock_multi_agent.providers.resilience import ProviderHealth, ResiliencePolicy, ResilientExecutor
from stock_multi_agent.services.data_quality import DataQualityChecker


def _env_number(name: str, default: int | float, cast):
    try:
        return cast(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class MarketDataError(RuntimeError):
    """Raised when real market data cannot be acquired or normalized."""


class MarketDataProvider(Protocol):
    def get_market_data(self, ticker: str, include_financials: bool = True) -> MarketData: ...


class VnstockMarketDataProvider:
    """Real Vietnamese market data via vnstock 4.x's unified stock API."""

    SCHEMA_VERSION = "vnstock-4-normalized-v7-financial-history"

    def __init__(
        self,
        source: str = "VCI",
        lookback_days: int = 550,
        cache: MarketDataCache | None = None,
        snapshot_store: RawSnapshotStore | None = None,
        quality_checker: DataQualityChecker | None = None,
        resilience_policy: ResiliencePolicy | None = None,
        enable_cache: bool | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.source = source.upper()
        self.lookback_days = lookback_days
        runtime_root = self._configure_runtime_storage()
        cache_enabled = enable_cache if enable_cache is not None else os.getenv("STOCK_AGENT_CACHE_ENABLED", "true").lower() not in {"0", "false", "off"}
        self.cache = cache or (SQLiteMarketDataCache(runtime_root / "cache" / "market_data.sqlite3") if cache_enabled else None)
        self.snapshot_store = snapshot_store or RawSnapshotStore(runtime_root / "snapshots")
        self.quality_checker = quality_checker or DataQualityChecker()
        self.api_key_configured = bool(os.getenv("VNSTOCK_API_KEY", "").strip())
        default_request_interval = 1.05 if self.api_key_configured else 3.2
        policy = resilience_policy or ResiliencePolicy(
            max_attempts=_env_number("STOCK_AGENT_PROVIDER_MAX_ATTEMPTS", 1, int),
            base_delay_seconds=_env_number("STOCK_AGENT_PROVIDER_RETRY_DELAY_SECONDS", 0.4, float),
            circuit_failure_threshold=_env_number("STOCK_AGENT_PROVIDER_CIRCUIT_THRESHOLD", 5, int),
            circuit_cooldown_seconds=_env_number("STOCK_AGENT_PROVIDER_CIRCUIT_COOLDOWN_SECONDS", 60, int),
            min_request_interval_seconds=_env_number(
                "STOCK_AGENT_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS",
                default_request_interval,
                float,
            ),
        )
        self.executor = ResilientExecutor(policy)
        self.cache_ttl_seconds = cache_ttl_seconds if cache_ttl_seconds is not None else _env_number(
            "STOCK_AGENT_CACHE_TTL_SECONDS", 259200, int,
        )
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None

    def get_market_data(self, ticker: str, include_financials: bool = True) -> MarketData:
        symbol = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", symbol):
            raise MarketDataError(f"Mã cổ phiếu không hợp lệ: {ticker!r}")

        self._configure_runtime_storage()
        cache_key = self._cache_key(symbol, include_financials)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                from stock_multi_agent.services.request_telemetry import current_metrics
                metrics = current_metrics()
                if metrics:
                    metrics.cache_hits += 1
                result = self._apply_quality(cached, include_financials)
                result.access_mode = 'CACHE'
                self.last_success_at = datetime.now(timezone.utc)
                return result
        self.cache_misses += 1
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('market_provider.fetch')
        try:
            from vnstock.api.financial import Finance
            from vnstock.api.quote import Quote
        except ImportError as exc:
            raise MarketDataError("Chưa cài vnstock. Chạy: pip install -e .") from exc

        end = date.today()
        start = end - timedelta(days=self.lookback_days)
        fetched_at = datetime.now(timezone.utc)
        warnings: list[str] = []
        source_refs: list[SourceReference] = []
        quote_source = self.source
        quote_parameters: dict[str, str] = {}
        quote_errors: list[str] = []
        frame: Any | None = None
        for candidate_source in self._price_sources():
            try:
                quote = Quote(source=candidate_source, symbol=symbol, show_log=False)
                quote_parameters = {
                    "source": candidate_source,
                    "symbol": symbol,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "interval": "1D",
                }
                frame = self.executor.run(
                    lambda quote=quote: quote.history(start=start.isoformat(), end=end.isoformat(), interval="1D")
                )
                quote_source = candidate_source
                break
            except Exception as exc:
                quote_errors.append(f"{candidate_source}: {str(exc)[:240]}")

        if frame is None:
            self.last_error = "; ".join(quote_errors)
            tried = ", ".join(self._price_sources())
            raise MarketDataError(
                f"Không thể tải lịch sử giá thật cho {symbol} từ các nguồn {tried}: {self.last_error}"
            )
        if quote_source != self.source:
            warnings.append(
                f"Nguồn giá chính {self.source} không khả dụng; đã dùng OHLCV từ {quote_source}."
            )

        bars = self._normalize_bars(frame)
        if len(bars) < 50:
            raise MarketDataError(
                f"{symbol} chỉ có {len(bars)} phiên hợp lệ; cần ít nhất 50 phiên để phân tích."
            )

        quote_digest, quote_snapshot = self.snapshot_store.save(
            provider=f"vnstock-{quote_source}",
            data_kind=DataKind.OHLCV.value,
            ticker=symbol,
            payload=self._frame_payload(frame),
            fetched_at=fetched_at,
            request_parameters=quote_parameters,
        )
        source_refs.append(SourceReference(
            provider="vnstock",
            source_id=f"{quote_source}:Quote.history:{symbol}",
            data_kind=DataKind.OHLCV,
            fetched_at=fetched_at,
            as_of=bars[-1].date,
            request_parameters=quote_parameters,
            schema_version=self.SCHEMA_VERSION,
            content_sha256=quote_digest,
            snapshot_id=quote_snapshot,
        ))

        instrument = Instrument(
            ticker=symbol,
            instrument_type=InstrumentType.INDEX if symbol == "VNINDEX" else InstrumentType.EQUITY,
        )
        company_name: str | None = None
        if symbol != "VNINDEX" and self._feature_enabled("STOCK_AGENT_COMPANY_PROFILE_ENABLED", "false"):
            try:
                instrument, company_name, profile_ref = self._fetch_company_profile(symbol, fetched_at)
                if profile_ref is not None:
                    source_refs.append(profile_ref)
            except Exception as exc:
                warnings.append(f"Company profile unavailable: {exc}")

        issuer_events: list[IssuerEvent] = []
        corporate_actions: list[CorporateAction] = []
        if symbol != "VNINDEX" and self._feature_enabled("STOCK_AGENT_ISSUER_EVENTS_ENABLED", "false"):
            try:
                issuer_events, corporate_actions, event_ref = self._fetch_issuer_events(symbol, fetched_at)
                if event_ref is not None:
                    source_refs.append(event_ref)
            except Exception as exc:
                warnings.append(f"Issuer events unavailable: {exc}")

        financials = FinancialMetrics()
        observations: list[FinancialMetricObservation] = []
        statement_observations: list[FinancialStatementObservation] = []
        if include_financials:
            finance: Any | None = None
            try:
                ratio_parameters = {"symbol": symbol, "period": "year", "lang": "en", "dropna": True}
                finance = Finance(
                    source=self.source,
                    symbol=symbol,
                    period="year",
                    get_all=False,
                    show_log=False,
                )
                ratio_frame = self.executor.run(
                    lambda: finance.ratio(period="year", lang="en", dropna=True, show_log=False)
                )
                financial_source_id = f"{self.source}:Finance.ratio:{symbol}"
                financials, observations = self._normalize_ratios_with_observations(ratio_frame, financial_source_id)
                ratio_digest, ratio_snapshot = self.snapshot_store.save(
                    provider=f"vnstock-{self.source}",
                    data_kind=DataKind.FINANCIAL_RATIOS.value,
                    ticker=symbol,
                    payload=self._frame_payload(ratio_frame),
                    fetched_at=fetched_at,
                    request_parameters=ratio_parameters,
                )
                period_dates = [item.period_end for item in observations if item.period_end]
                source_refs.append(SourceReference(
                    provider="vnstock",
                    source_id=financial_source_id,
                    data_kind=DataKind.FINANCIAL_RATIOS,
                    fetched_at=fetched_at,
                    as_of=max(period_dates) if period_dates else None,
                    request_parameters=ratio_parameters,
                    schema_version=self.SCHEMA_VERSION,
                    content_sha256=ratio_digest,
                    snapshot_id=ratio_snapshot,
                ))
            except Exception as exc:
                warnings.append(f"Không tải được đầy đủ chỉ số tài chính: {exc}")

            if finance is not None and self._feature_enabled("STOCK_AGENT_FINANCIAL_STATEMENTS_ENABLED", "false"):
                try:
                    statement_observations, statement_refs = self._fetch_statement_series(finance, symbol, fetched_at)
                    source_refs.extend(statement_refs)
                except Exception as exc:
                    warnings.append(f"Financial statement series unavailable: {exc}")

            if financials.available_count() < 3:
                warnings.append("Dữ liệu cơ bản còn thiếu; Fundamental Agent sẽ giảm confidence.")

        data = MarketData(
            ticker=symbol,
            instrument=instrument,
            source=f"vnstock:{quote_source}",
            access_mode='LIVE_FETCH',
            company_name=company_name,
            bars=bars,
            financials=financials,
            financial_observations=observations,
            financial_statement_observations=statement_observations,
            corporate_actions=corporate_actions,
            issuer_events=issuer_events,
            source_refs=source_refs,
            fetched_at=fetched_at,
            as_of=bars[-1].date,
            price_adjustment=PriceAdjustment.UNKNOWN,
            schema_version="market-data-v4",
            warnings=warnings,
        )
        data = self._apply_quality(data, include_financials)
        if self.cache is not None:
            self.cache.set(cache_key, data, self.cache_ttl_seconds)
        self.last_success_at = datetime.now(timezone.utc)
        self.last_error = None
        return data

    def refresh_market_data(self, ticker: str, include_financials: bool = True) -> MarketData:
        """Bypass only this instrument's cache before one bounded recovery attempt."""
        symbol = ticker.strip().upper()
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('market_provider.refresh')
        if self.cache is not None:
            self.cache.delete(self._cache_key(symbol, include_financials))
        return self.get_market_data(symbol, include_financials=include_financials)

    def health(self) -> dict[str, Any]:
        """In-process operational state for probes, CLI reporting and tests.

        This intentionally exposes no API secrets or raw response payloads.
        Counters reset when the process restarts; durable telemetry is a later
        operations concern.
        """
        upstream: ProviderHealth = self.executor.health()
        total_cache_reads = self.cache_hits + self.cache_misses
        return {
            "provider": f"vnstock:{self.source}",
            "cache_enabled": self.cache is not None,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / total_cache_reads, 4) if total_cache_reads else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error": self.last_error,
            "api_key_configured": self.api_key_configured,
            "request_interval_seconds": self.executor.policy.min_request_interval_seconds,
            "attempts_total": upstream.attempts_total,
            "successful_operations": upstream.successful_operations,
            "failed_operations": upstream.failed_operations,
            "retries_total": upstream.retries_total,
            "circuit_state": upstream.circuit_state,
            "consecutive_failures": upstream.consecutive_failures,
            "rate_limit_wait_seconds": upstream.rate_limit_wait_seconds,
        }

    def _cache_key(self, symbol: str, include_financials: bool) -> str:
        return ":".join((
            self.SCHEMA_VERSION, self.source, symbol, str(self.lookback_days), str(int(include_financials)),
            f"price_fallbacks={','.join(self._price_sources())}",
            f"profile={int(self._feature_enabled('STOCK_AGENT_COMPANY_PROFILE_ENABLED', 'false'))}",
            f"events={int(self._feature_enabled('STOCK_AGENT_ISSUER_EVENTS_ENABLED', 'false'))}",
            f"statements={int(self._feature_enabled('STOCK_AGENT_FINANCIAL_STATEMENTS_ENABLED', 'false'))}",
            f"quarterly={int(self._feature_enabled('STOCK_AGENT_QUARTERLY_STATEMENTS_ENABLED', 'false'))}",
        ))

    def _price_sources(self) -> tuple[str, ...]:
        """Configured OHLCV sources, ordered and deduplicated.

        VCI is intentionally retained as the primary source. Some cloud IP
        ranges receive a legitimate access denial from VCI, in which case KBS
        can provide real OHLCV data. The selected source is always recorded in
        the resulting `MarketData` and `SourceReference`.
        """
        fallback_setting = os.getenv(
            "VNSTOCK_PRICE_FALLBACK_SOURCES",
            "KBS" if self.source == "VCI" else "",
        )
        candidates = [self.source, *(item.strip().upper() for item in fallback_setting.split(","))]
        return tuple(dict.fromkeys(item for item in candidates if item))

    @staticmethod
    def _feature_enabled(name: str, default: str = "true") -> bool:
        return os.getenv(name, default).lower() not in {"0", "false", "off"}

    def _apply_quality(self, data: MarketData, require_financials: bool) -> MarketData:
        quality = self.quality_checker.evaluate(data, require_financials=require_financials)
        quality_warnings = [issue.message for issue in quality.issues if issue.severity is QualitySeverity.WARNING]
        result = data.model_copy(update={
            "quality_report": quality,
            "warnings": list(dict.fromkeys([*data.warnings, *quality_warnings])),
        })
        if quality.status is QualityStatus.FAIL:
            failures = "; ".join(issue.message for issue in quality.issues if issue.severity is QualitySeverity.ERROR)
            raise MarketDataError(f"Data quality validation failed for {data.ticker}: {failures}")
        return result

    @staticmethod
    def _configure_runtime_storage() -> Path:
        """Keep vnstock's config/cache on the same D-drive project as the venv."""
        default_root = Path(__file__).resolve().parents[3] / ".runtime"
        configured_root = os.getenv("STOCK_AGENT_RUNTIME_DIR")
        runtime_root = Path(configured_root if configured_root else default_root).resolve()
        if os.name == "nt" and runtime_root.drive.upper() != "D:":
            raise MarketDataError(
                f"Workspace policy requires runtime/cache/snapshots on drive D:, got {runtime_root}"
            )
        runtime_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("VNSTOCK_DATA_DIR", str(runtime_root / ".vnstock"))
        os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
        os.environ.setdefault("VNSTOCK_TIMEOUT", os.getenv("STOCK_AGENT_PROVIDER_TIMEOUT_SECONDS", "20"))
        # Application-level retry/backoff is observable and bounded; avoid nested retries.
        os.environ.setdefault("VNSTOCK_MAX_RETRIES", "0")
        if os.name == "nt":
            # vnstock/vnai 4.0.9 resolves its config with Path.home(), not
            # VNSTOCK_DATA_DIR in every module. Scope the process profile to D.
            os.environ["USERPROFILE"] = str(runtime_root)
            # vnstock's free-tier notice contains Vietnamese characters. On a
            # legacy Windows code page, its own print() can otherwise raise
            # UnicodeEncodeError and discard an otherwise valid quarterly
            # response. This only adjusts the process text stream; no source
            # payload or provider behaviour is changed.
            reconfigure = getattr(sys.stdout, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="backslashreplace")
                except (OSError, ValueError):
                    pass
        return runtime_root

    @staticmethod
    def _frame_payload(frame: Any) -> Any:
        if isinstance(frame, pd.DataFrame):
            return json.loads(frame.to_json(orient="split", date_format="iso", default_handler=str))
        return frame

    @staticmethod
    def _normalize_bars(frame: Any) -> list[PriceBar]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise MarketDataError("Nguồn dữ liệu trả về bảng giá rỗng.")
        data = frame.copy()
        data.columns = [str(col).strip().lower() for col in data.columns]
        if "time" in data.columns and "date" not in data.columns:
            data = data.rename(columns={"time": "date"})
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise MarketDataError(f"Bảng giá thiếu cột: {', '.join(sorted(missing))}")
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
        for column in ("open", "high", "low", "close", "volume"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=list(required)).drop_duplicates("date").sort_values("date")
        bars: list[PriceBar] = []
        for row in data.itertuples(index=False):
            values = row._asdict()
            try:
                bars.append(PriceBar(**{key: values[key] for key in required}))
            except ValueError:
                continue
        if not bars:
            raise MarketDataError("Không có dòng OHLCV hợp lệ sau khi chuẩn hóa.")
        return bars

    def _fetch_company_profile(
        self,
        ticker: str,
        fetched_at: datetime,
    ) -> tuple[Instrument, str | None, SourceReference | None]:
        """Attach only profile fields explicitly returned by the provider."""
        from vnstock.api.company import Company

        frame = self.executor.run(lambda: Company(symbol=ticker, source=self.source).overview())
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return Instrument(ticker=ticker), None, None
        record = frame.iloc[0]

        def text(*keys: str) -> str | None:
            for key in keys:
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    return re.sub(r"\s+", " ", value).strip()
            return None

        name = text("organ_name", "organ_short_name", "company_name")
        sector = text("sector", "sector_name", "industry")
        exchange = text("exchange", "exchange_name")
        source_id = f"{self.source}:Company.overview:{ticker}"
        digest, snapshot_id = self.snapshot_store.save(
            provider=f"vnstock-{self.source}", data_kind=DataKind.COMPANY_PROFILE.value, ticker=ticker,
            payload=self._frame_payload(frame), fetched_at=fetched_at, request_parameters={"symbol": ticker},
        )
        return (
            Instrument(ticker=ticker, name=name, sector=sector, exchange=exchange),
            name,
            SourceReference(
                provider="vnstock", source_id=source_id, data_kind=DataKind.COMPANY_PROFILE,
                fetched_at=fetched_at, request_parameters={"symbol": ticker}, schema_version=self.SCHEMA_VERSION,
                content_sha256=digest, snapshot_id=snapshot_id,
            ),
        )

    def _fetch_issuer_events(
        self,
        ticker: str,
        fetched_at: datetime,
    ) -> tuple[list[IssuerEvent], list[CorporateAction], SourceReference | None]:
        """Normalise only explicit VCI issuer-event records and retain the raw snapshot."""
        from vnstock.api.company import Company

        frame = self.executor.run(lambda: Company(symbol=ticker, source=self.source, show_log=False).events())
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return [], [], None
        source_id = f"{self.source}:Company.events:{ticker}"
        events: list[IssuerEvent] = []
        actions: list[CorporateAction] = []
        action_types = {
            "DIV": "dividend",
            "ISS": "share_issue",
            "AIS": "additional_listing",
            "SUSP": "trading_suspension",
        }
        for _, row in frame.iterrows():
            code = self._text_value(row, "event_code")
            title = self._text_value(row, "event_title_en", "event_name_en", "event_title_vi", "event_name_vi")
            event_date = self._first_date(row, "display_date1", "public_date", "record_date", "exright_date", "issue_date")
            if title is None or event_date is None:
                continue
            published_date = self._first_date(row, "public_date")
            events.append(IssuerEvent(
                event_code=code, title=title, event_date=event_date, published_date=published_date, source_id=source_id,
            ))
            action_type = action_types.get(code or "")
            if action_type is not None:
                action_date = self._first_date(row, "exright_date", "record_date", "issue_date", "public_date")
                if action_date is not None:
                    ratio = pd.to_numeric(row.get("exercise_ratio"), errors="coerce")
                    amount = pd.to_numeric(row.get("value_per_share"), errors="coerce")
                    actions.append(CorporateAction(
                        action_type=action_type,
                        ex_date=action_date,
                        description=title,
                        ratio_or_amount=float(ratio) if pd.notna(ratio) else float(amount) if pd.notna(amount) else None,
                        source_id=source_id,
                    ))
        digest, snapshot_id = self.snapshot_store.save(
            provider=f"vnstock-{self.source}", data_kind=DataKind.ISSUER_EVENTS.value, ticker=ticker,
            payload=self._frame_payload(frame), fetched_at=fetched_at, request_parameters={"symbol": ticker},
        )
        event_dates = [item.event_date for item in events]
        return events, actions, SourceReference(
            provider="vnstock", source_id=source_id, data_kind=DataKind.ISSUER_EVENTS,
            fetched_at=fetched_at, as_of=max(event_dates) if event_dates else None,
            request_parameters={"symbol": ticker}, schema_version=self.SCHEMA_VERSION,
            content_sha256=digest, snapshot_id=snapshot_id,
        )

    def _fetch_statement_series(
        self,
        finance: Any,
        ticker: str,
        fetched_at: datetime,
    ) -> tuple[list[FinancialStatementObservation], list[SourceReference]]:
        """Fetch annual and, when available, quarterly statement lines with provenance."""
        specifications = (
            ("income_statement", DataKind.INCOME_STATEMENT, finance.income_statement),
            ("balance_sheet", DataKind.BALANCE_SHEET, finance.balance_sheet),
            ("cash_flow", DataKind.CASH_FLOW, finance.cash_flow),
        )
        all_observations: list[FinancialStatementObservation] = []
        refs: list[SourceReference] = []
        frequencies = ["year"]
        if self._feature_enabled("STOCK_AGENT_QUARTERLY_STATEMENTS_ENABLED", "false"):
            frequencies.append("quarter")
        for frequency in frequencies:
            for statement, kind, method in specifications:
                parameters = {"symbol": ticker, "period": frequency, "lang": "en", "dropna": True}
                try:
                    frame = self.executor.run(
                        lambda method=method, frequency=frequency: method(period=frequency, lang="en", dropna=True, show_log=False)
                    )
                except Exception:
                    continue
                source_id = f"{self.source}:Finance.{statement}:{frequency}:{ticker}"
                observations = self._normalize_statement_observations(
                    frame, statement, source_id, frequency="annual" if frequency == "year" else "quarter",
                )
                if not observations:
                    continue
                digest, snapshot_id = self.snapshot_store.save(
                    provider=f"vnstock-{self.source}", data_kind=kind.value, ticker=ticker,
                    payload=self._frame_payload(frame), fetched_at=fetched_at, request_parameters=parameters,
                )
                period_dates = [item.period_end for item in observations if item.period_end]
                refs.append(SourceReference(
                    provider="vnstock", source_id=source_id, data_kind=kind, fetched_at=fetched_at,
                    as_of=max(period_dates) if period_dates else None, request_parameters=parameters,
                    schema_version=self.SCHEMA_VERSION, content_sha256=digest, snapshot_id=snapshot_id,
                ))
                all_observations.extend(observations)
        return all_observations, refs

    @classmethod
    def _normalize_statement_observations(
        cls,
        frame: Any,
        statement: str,
        source_id: str,
        frequency: str = "annual",
    ) -> list[FinancialStatementObservation]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []
        aliases = {
            "income_statement": {
                "revenue": ("revenue", "netsales", "netsalesrevenue", "sales"),
                "net_profit": ("netprofit", "profitaftertax", "netincome"),
            },
            "cash_flow": {
                "operating_cash_flow": ("operatingcashflow", "cashflowfromoperating", "netcashflowsfromoperatingactivities"),
            },
            "balance_sheet": {
                "total_assets": ("totalassets",),
                "total_equity": ("totalequity", "ownersequity"),
                "total_liabilities": ("totalliabilities",),
            },
        }
        statement_aliases = aliases.get(statement, {})
        columns_by_key = {cls._key(str(column)): index for index, column in enumerate(frame.columns)}
        identity_indexes = [
            index for key, index in columns_by_key.items()
            if key in {"itemid", "itemen", "item", "name", "indicator"}
        ]
        value_indexes = [
            index for index, column in enumerate(frame.columns)
            if cls._period_end(str(column)) is not None
        ]
        if not statement_aliases:
            return []
        if not identity_indexes or not value_indexes:
            # Some provider versions return one record per reporting period.
            # Support declared metadata columns instead of assuming row 0 is latest.
            observations = []
            for row_index in range(frame.shape[0]):
                row = frame.iloc[row_index]
                period_label, period_end = cls._record_financial_period(row)
                if period_end is None:
                    continue
                for column, raw_value in row.items():
                    key = cls._key(str(column))
                    metric = next((candidate for candidate, needles in statement_aliases.items()
                        if key in {cls._key(needle) for needle in needles}), None)
                    value = pd.to_numeric(raw_value, errors='coerce')
                    if metric is None or pd.isna(value) or not math.isfinite(float(value)):
                        continue
                    row_frequency = 'quarter' if period_label.startswith('Q') else frequency
                    metadata = cls._financial_frame_metadata(frame, row_frequency)
                    unit = metadata.pop('unit', 'currency')
                    observations.append(FinancialStatementObservation(statement=statement, metric=metric,
                        value=float(value), unit=unit, period_label=period_label, period_end=period_end,
                        frequency=row_frequency, source_id=source_id, **metadata))
            return sorted(observations, key=lambda item: (item.metric, item.period_end or date.min))
        observations: list[FinancialStatementObservation] = []
        seen: set[tuple[str, date | None, float]] = set()
        for row_index in range(frame.shape[0]):
            identity = " ".join(str(frame.iloc[row_index, index]) for index in identity_indexes)
            identity_key = cls._key(identity)
            metric = next((
                candidate for candidate, needles in statement_aliases.items()
                if any(cls._key(needle) in identity_key for needle in needles)
            ), None)
            if metric is None:
                continue
            for value_index in value_indexes:
                value = pd.to_numeric(frame.iloc[row_index, value_index], errors="coerce")
                period_label = str(frame.columns[value_index])
                period_end = cls._period_end(period_label)
                if pd.isna(value) or not math.isfinite(float(value)) or period_end is None or (metric, period_end, float(value)) in seen:
                    continue
                seen.add((metric, period_end, float(value)))
                metadata = cls._financial_frame_metadata(frame, frequency)
                unit_index = columns_by_key.get('unit')
                unit = str(frame.iloc[row_index, unit_index]) if unit_index is not None and pd.notna(frame.iloc[row_index, unit_index]) else metadata.pop('unit', 'currency')
                metadata.pop('unit', None)
                observations.append(FinancialStatementObservation(
                    statement=statement, metric=metric, value=float(value), unit=unit,
                    period_label=period_label, period_end=period_end, frequency=frequency, source_id=source_id,
                    **metadata,
                ))
        return sorted(observations, key=lambda item: (item.metric, item.period_end or date.min))

    @classmethod
    def _normalize_ratios(cls, frame: Any) -> FinancialMetrics:
        metrics, _ = cls._normalize_ratios_with_observations(frame, "unknown")
        return metrics

    @classmethod
    def _normalize_ratios_with_observations(
        cls,
        frame: Any,
        source_id: str,
    ) -> tuple[FinancialMetrics, list[FinancialMetricObservation]]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return FinancialMetrics(), []
        flat: dict[str, list[tuple[float, str]]] = {}
        percentage_keys: set[str] = set()

        # vnstock 4.x VCI commonly returns a report-oriented frame: metrics
        # are rows (`item_id`) and periods are duplicate-labelled columns.
        columns_by_key = {cls._key(str(column)): index for index, column in enumerate(frame.columns)}
        id_index = columns_by_key.get("itemid")
        name_index = columns_by_key.get("itemen")
        if id_index is not None:
            metadata_indexes = {index for index in (id_index, name_index, columns_by_key.get("item")) if index is not None}
            value_indexes = [index for index in range(frame.shape[1]) if index not in metadata_indexes]
            for row_index in range(frame.shape[0]):
                identity = str(frame.iloc[row_index, id_index])
                display = str(frame.iloc[row_index, name_index]) if name_index is not None else ""
                if '%' in display or 'percent' in display.casefold():
                    percentage_keys.add(cls._key(f'{identity} {display}'))
                candidates: list[tuple[float, str, int]] = []
                for value_index in value_indexes:
                    numeric = pd.to_numeric(frame.iloc[row_index, value_index], errors="coerce")
                    if pd.notna(numeric):
                        candidates.append((float(numeric), str(frame.columns[value_index]), value_index))
                if candidates:
                    ordered = sorted(candidates, key=lambda item: (cls._period_end(item[1]) or date.min, item[2]), reverse=True)
                    flat[cls._key(f"{identity} {display}")] = [(value, label) for value, label, _ in ordered]
        else:
            # Also support record-oriented frames returned by other sources.
            for row_index in range(frame.shape[0]):
                row = frame.iloc[row_index]
                label, _ = cls._record_financial_period(row)
                for column, value in row.items():
                    key = " ".join(str(part) for part in column) if isinstance(column, tuple) else str(column)
                    numeric = pd.to_numeric(value, errors="coerce")
                    if pd.notna(numeric) and math.isfinite(float(numeric)):
                        flat.setdefault(cls._key(key), []).append((float(numeric), label))
                        if '%' in key or 'percent' in key.casefold():
                            percentage_keys.add(cls._key(key))
            for items in flat.values():
                items.sort(key=lambda item: cls._period_end(item[1]) or date.min, reverse=True)

        def pick(*needles: str) -> tuple[str, list[tuple[float, str]]] | None:
            for key, observation in flat.items():
                if any(cls._key(needle) in key for needle in needles):
                    return key, observation
            return None

        definitions = {
            "revenue_growth": (("revenuegrowth", "salesgrowth"), "ratio", True),
            "profit_growth": (("netprofitgrowth", "earningsgrowth"), "ratio", True),
            "eps": (("eps",), "currency_per_share", False),
            "pe": (("pe_ratio", "price earning"), "multiple", False),
            "pb": (("pb_ratio", "price to book"), "multiple", False),
            "roe": (("roe",), "ratio", True),
            "roa": (("roa",), "ratio", True),
            "debt_to_equity": (("debtequity", "debttoequity"), "ratio", False),
            "operating_cash_flow": (("operatingcashflow", "cashflowfromoperating"), "currency", False),
            "profit_margin": (("netprofitmargin", "profitmargin"), "ratio", True),
        }
        normalized: dict[str, float | None] = {}
        observations: list[FinancialMetricObservation] = []
        for metric, (needles, unit, is_ratio) in definitions.items():
            selected = pick(*needles)
            if selected is None:
                normalized[metric] = None
                continue
            key, history = selected
            normalized[metric] = None
            seen_values = set()
            for history_index, (raw_value, period_label) in enumerate(history):
                # Explicit percentages alone justify /100. Ambiguous scale is
                # preserved in history and excluded from the aggregate score.
                explicitly_percent = unit == 'ratio' and (key in percentage_keys or frame.attrs.get('ratio_format') == 'percent')
                value = raw_value / 100 if explicitly_percent else raw_value
                ambiguous_scale = is_ratio and not explicitly_percent and frame.attrs.get('ratio_format') != 'decimal' and abs(raw_value) > 2
                if history_index == 0:
                    normalized[metric] = None if ambiguous_scale else value
                if (period_label, value) in seen_values:
                    continue
                seen_values.add((period_label, value))
                frequency = 'quarter' if re.search(r'(?:q|quarter)\s*[1-4]', period_label, re.I) else 'annual'
                metadata = cls._financial_frame_metadata(frame, frequency)
                metadata.pop('unit', None)
                if 'reporting_basis' not in frame.attrs:
                    metadata['reporting_basis'] = 'UNKNOWN'  # ratios may use TTM even in annual-labelled columns
                observation_unit = 'unknown_ratio' if ambiguous_scale else unit
                observation_value = raw_value if observation_unit == 'unknown_ratio' else value
                observations.append(FinancialMetricObservation(metric=metric, value=observation_value, unit=observation_unit,
                    period_label=period_label, period_end=cls._period_end(period_label), source_id=source_id,
                    frequency=frequency,
                    **metadata))
        return FinancialMetrics(**normalized), observations

    @staticmethod
    def _financial_frame_metadata(frame, frequency):
        """Only retain explicit source declarations; endpoint year declares annual, not consolidation."""
        attrs = getattr(frame, 'attrs', {})
        allowed = {'currency', 'scale', 'period_start', 'reporting_basis', 'consolidation', 'assurance', 'unit'}
        metadata = {key: value for key, value in attrs.items() if key in allowed and value is not None}
        if frequency == 'annual':
            metadata.setdefault('reporting_basis', 'ANNUAL')
        return metadata

    @classmethod
    def _record_financial_period(cls, row):
        fields = {cls._key(' '.join(map(str, key)) if isinstance(key, tuple) else str(key)): value for key, value in row.items()}
        for key in ('periodend', 'reportdate', 'enddate', 'date'):
            if key in fields and pd.notna(fields[key]):
                try:
                    value = pd.Timestamp(fields[key]).date()
                    return value.isoformat(), value
                except (ValueError, TypeError):
                    pass
        year = next((value for key, value in fields.items() if key.endswith(('yearreport', 'reportyear')) or key == 'year'), None)
        if year is None or pd.isna(year):
            return 'latest', None
        try:
            year = int(year)
            if not 2000 <= year <= 2100:
                return 'latest', None
            quarter = next((value for key, value in fields.items() if key.endswith(('quarter', 'lengthreport'))), None)
            if quarter is not None and pd.notna(quarter) and int(quarter) in (1, 2, 3, 4):
                label = f'Q{int(quarter)} {year}'
            else:
                label = str(year)
            return label, cls._period_end(label)
        except (ValueError, TypeError, OverflowError):
            return 'latest', None

    @staticmethod
    def _key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    @staticmethod
    def _text_value(row: pd.Series, *keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return re.sub(r"\s+", " ", value).strip()
        return None

    @staticmethod
    def _first_date(row: pd.Series, *keys: str) -> date | None:
        for key in keys:
            value = pd.to_datetime(row.get(key), errors="coerce")
            if not pd.isna(value):
                return value.date()
        return None

    @staticmethod
    def _as_ratio(value: float | None) -> float | None:
        if value is None:
            return None
        return value / 100.0 if abs(value) > 2 else value

    @staticmethod
    def _period_end(label: str) -> date | None:
        exact = re.search(r'\b(20\d{2})-(\d{2})-(\d{2})\b', label)
        if exact:
            try:
                return date(*map(int, exact.groups()))
            except ValueError:
                return None
        year_match = re.search(r"\b(20\d{2})\b", label)
        if not year_match:
            return None
        year = int(year_match.group(1))
        quarter_match = re.search(r"(?:q|quarter)\s*([1-4])", label, flags=re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            month = quarter * 3
            next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
            return next_month - timedelta(days=1)
        return date(year, 12, 31)
