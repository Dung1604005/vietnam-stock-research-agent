from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from conftest import conflict_market_data
from stock_multi_agent.models.chat import ConversationState, ParsedQuery, QueryIntent
from stock_multi_agent.models.data import DataKind, Instrument, SourceReference
from stock_multi_agent.models.screening import ScreenSpec, ScreenStatus
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.providers.universe_provider import InstrumentUniverseSnapshot, VnstockInstrumentUniverseProvider
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.screening_service import ScreeningService
from stock_multi_agent.services.universe_service import UniverseService


class ScreenProvider:
    def __init__(self, tickers: list[str]) -> None:
        self.data = {ticker: conflict_market_data(ticker) for ticker in tickers}

    def get_market_data(self, ticker: str, include_financials: bool = True):
        del include_financials
        if ticker not in self.data:
            raise MarketDataError(f"No fixture for {ticker}")
        return self.data[ticker]


class RateLimitedScreenProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_market_data(self, ticker: str, include_financials: bool = True):
        del ticker, include_financials
        self.calls += 1
        raise MarketDataError("Rate Limit Exceeded: wait 59 seconds")


class CurrentUniverseProvider:
    def fetch_current(self) -> InstrumentUniverseSnapshot:
        source = SourceReference(
            provider="fixture", source_id="fixture:current-universe", data_kind=DataKind.INSTRUMENT_UNIVERSE,
            fetched_at=datetime(2026, 10, 10, tzinfo=timezone.utc),
        )
        return InstrumentUniverseSnapshot(
            instruments=[
                Instrument(ticker="FPT", exchange="HOSE"),
                Instrument(ticker="SHS", exchange="HNX"),
                Instrument(ticker="ABC", exchange="UPCOM"),
            ], source_reference=source,
        )


def test_screen_spec_discloses_bounded_universe_and_missing_data_policy() -> None:
    analysis = AnalysisService(provider=ScreenProvider(["FPT", "CMG", "VNINDEX"]))
    universes = UniverseService({"default": ["FPT", "CMG"]})
    run = ScreeningService(analysis, universes).run(
        ScreenSpec(universe_key="default", minimum_technical_score=9.5), "3 months",
    )
    assert run.universe.point_in_time_supported is False
    assert [candidate.ticker for candidate in run.candidates] == ["FPT", "CMG"]
    assert all(candidate.status is ScreenStatus.FAIL for candidate in run.candidates)
    assert all(candidate.as_of is not None and candidate.source_ids for candidate in run.candidates)
    assert all("below 9.5" in candidate.reasons[0] for candidate in run.candidates)


def test_screen_keeps_provider_failures_visible_and_unranked() -> None:
    analysis = AnalysisService(provider=ScreenProvider(["FPT", "VNINDEX"]))
    run = ScreeningService(analysis, UniverseService({"default": ["FPT", "MISSING"]})).run(
        ScreenSpec(universe_key="default"), "3 months",
    )
    unavailable = next(candidate for candidate in run.candidates if candidate.ticker == "MISSING")
    assert unavailable.status is ScreenStatus.UNAVAILABLE
    assert unavailable.technical_score is None
    assert "No fixture" in unavailable.reasons[0]


def test_screen_stops_after_a_provider_rate_limit_instead_of_burning_more_calls() -> None:
    provider = RateLimitedScreenProvider()
    run = ScreeningService(
        AnalysisService(provider=provider), UniverseService({"default": ["FPT", "CMG", "HPG"]}),
    ).run(ScreenSpec(universe_key="default"), "3 months")
    assert provider.calls == 1
    assert len(run.candidates) == 3
    assert "provider rate limit reached" in run.warnings[0]
    assert "Not evaluated" in run.candidates[-1].reasons[0]


def test_current_exchange_universe_has_retrieval_provenance_but_is_not_historical() -> None:
    universes = UniverseService({"default": ["FPT"]}, current_provider=CurrentUniverseProvider())
    hose = universes.select("hose_current")
    assert hose.tickers == ["FPT"]
    assert hose.source_reference is not None
    assert hose.captured_at == datetime(2026, 10, 10, tzinfo=timezone.utc)
    assert hose.membership_as_of is None
    assert hose.point_in_time_supported is False


def test_current_listing_normalization_excludes_non_equities_and_invalid_symbols() -> None:
    frame = pd.DataFrame([
        {"symbol": "FPT", "exchange": "HOSE", "type": "STOCK", "organ_name": "FPT Corp"},
        {"symbol": "OLD", "exchange": "DELISTED", "type": "STOCK", "organ_name": "Old Corp"},
        {"symbol": "VN30F1M", "exchange": "HNX", "type": "FUTURES", "organ_name": "Future"},
        {"symbol": "?", "exchange": "HOSE", "type": "STOCK", "organ_name": "Invalid"},
    ])
    instruments = VnstockInstrumentUniverseProvider._normalize(frame)
    assert [item.ticker for item in instruments] == ["FPT"]
    assert instruments[0].name == "FPT Corp"


def test_screen_withholds_ranking_when_price_dates_do_not_align() -> None:
    provider = ScreenProvider(["FPT", "CMG", "VNINDEX"])
    provider.data["CMG"] = provider.data["CMG"].model_copy(
        update={"as_of": provider.data["CMG"].as_of - timedelta(days=1)},
    )
    run = ScreeningService(AnalysisService(provider=provider), UniverseService({"default": ["FPT", "CMG"]})).run(
        ScreenSpec(universe_key="default"), "3 months",
    )
    assert run.price_as_of_aligned is False
    assert all(item.status is ScreenStatus.UNAVAILABLE for item in run.candidates)
    assert all(item.rank is None for item in run.candidates)
    assert "ranking is withheld" in run.warnings[0]


def test_comparison_withholds_incompatible_price_and_financial_fields() -> None:
    provider = ScreenProvider(["FPT", "CMG", "VNINDEX"])
    cmg = provider.data["CMG"]
    provider.data["CMG"] = cmg.model_copy(update={"as_of": cmg.as_of - timedelta(days=1)})
    service = ChatService(
        analysis_service=AnalysisService(provider=provider),
        universe={"default": ["FPT", "CMG"]},
    )
    query = ParsedQuery(
        raw_query="So sánh FPT và CMG", intent=QueryIntent.STOCK_COMPARISON,
        tickers=["FPT", "CMG"], confidence=0.9,
    )
    response = service.handle(query.raw_query, ConversationState(), parsed_query=query)
    assert response.metadata["comparison_alignment"]["price_as_of_aligned"] is False
    assert response.metadata["comparison_alignment"]["financial_period_aligned"] is False
    assert all("UNAVAILABLE" in row["Technical"] for row in response.tables[0].rows)
    assert all("UNAVAILABLE" in row["Fundamental"] for row in response.tables[0].rows)
