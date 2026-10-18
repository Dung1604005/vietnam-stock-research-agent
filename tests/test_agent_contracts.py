from datetime import datetime, timezone

from stock_multi_agent.agents import DataAgent, FundamentalAgent, ManagerAgent, ResearchAgent, RiskAgent, TechnicalAgent
from stock_multi_agent.models.agent_contracts import AgentExecutionReport
from stock_multi_agent.models.analysis import FundamentalAnalysis, Recommendation, RiskAnalysis, RiskLevel, Signal, TechnicalAnalysis
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.research import Citation, EvidenceBundle
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.agent_evaluation import AgentEvaluationService
from stock_multi_agent.tools.financial_data import FinancialToolkit
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


class _ResearchFixture:
    def search(self, query: str, ticker: str | None = None, limit: int = 5, **kwargs) -> EvidenceBundle:
        del ticker, limit
        now = datetime.now(timezone.utc)
        citation = Citation(
            document_id="document-1234567890",
            chunk_id="chunk-123456789012",
            title="FPT annual report 2025",
            source_url="https://example.com/fpt-report",
            publisher="FPT",
            published_at=now,
            retrieved_at=now,
            content_sha256="a" * 64,
            quote="FPT reported audited revenue and profit for the period.",
            retrieval_score=0.8,
            retrieval_modes=["vector", "fts5"],
        )
        return EvidenceBundle(query=query, citations=[citation], retrieval_note="fixture")


def test_every_agent_declares_an_explicit_contract_and_only_bounded_roles_can_use_llm() -> None:
    agent_types = [DataAgent, ResearchAgent, TechnicalAgent, FundamentalAgent, RiskAgent, ManagerAgent]
    assert {agent.spec.role for agent in agent_types} == set(AgentName) - {AgentName.USER}
    assert [agent.spec.role for agent in agent_types if agent.spec.can_use_llm] == [
        AgentName.RESEARCH, AgentName.TECHNICAL, AgentName.FUNDAMENTAL, AgentName.RISK, AgentName.MANAGER,
    ]
    assert all(agent.spec.required_context for agent in agent_types)
    assert all(agent.spec.output_type for agent in agent_types)


def test_research_agent_returns_provenance_linked_evidence(market_data) -> None:
    state = AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", market_data=market_data,
    )
    update = ResearchAgent(store=_ResearchFixture()).run(state)

    assert len(update["research_evidence"]) == 1
    assert str(update["research_evidence"][0].source_url) == "https://example.com/fpt-report"
    assert update["agent_reports"][0].evidence_count == 1
    assert {message.receiver for message in update["messages"]} == {AgentName.FUNDAMENTAL, AgentName.RISK}


def test_fundamental_uses_research_as_context_without_changing_numeric_score(market_data) -> None:
    citation = _ResearchFixture().search("FPT").citations[0]
    baseline = AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", market_data=market_data,
    )
    enriched = baseline.model_copy(update={"research_evidence": [citation]})
    agent = FundamentalAgent(FinancialToolkit())

    baseline_result = agent.run(baseline)["fundamental_result"]
    enriched_result = agent.run(enriched)["fundamental_result"]

    assert enriched_result.score == baseline_result.score
    assert enriched_result.research_context_count == 1
    assert any(item.label.startswith("Research context:") for item in enriched_result.evidence)


def test_requested_atr_and_adx_are_computed_by_python(market_data) -> None:
    state = AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT",
        market_data=market_data, requested_indicators=["ATR", "ADX"],
    )

    result = TechnicalAgent(TechnicalIndicatorToolkit()).run(state)["technical_result"]

    assert {item.label[:3] for item in result.evidence} == {"ATR", "ADX"}
    assert set(result.feature_contributions) == {"ATR", "ADX"}


def test_execution_evaluation_detects_contract_and_output_coverage(market_data) -> None:
    report = AgentExecutionReport(
        agent=AgentName.DATA,
        stage="data_acquisition",
        status="COMPLETED",
        summary="Data loaded",
        confidence=0.9,
        output_key="market_data",
        evidence_count=1,
    )
    state = AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT",
        market_data=market_data, agent_reports=[report],
    )

    result = AgentEvaluationService().evaluate(state, [AgentName.DATA])

    assert result.passed is True
    assert result.contract_coverage == 1.0
    assert result.evidence_coverage == 1.0


def test_manager_blocks_buy_when_evidence_completeness_is_too_low(market_data) -> None:
    state = AgentState(
        ticker="FPT",
        investment_horizon="3 months",
        user_goal="Analyze FPT",
        market_data=market_data,
        technical_result=TechnicalAnalysis(
            signal=Signal.BUY, score=8, confidence=0.8, evidence=[], risks=[],
            tools_used=["SMA20"], reasoning_summary="Positive trend", timeframe="3 months",
        ),
        fundamental_result=FundamentalAnalysis(
            signal=Signal.BUY, score=8, confidence=0.8, strengths=[], weaknesses=[],
            evidence=[], reasoning_summary="Positive fundamentals",
        ),
        risk_result=RiskAnalysis(
            risk_level=RiskLevel.LOW, risk_score=1, criticisms=[], warnings=[], veto=False,
            reasoning_summary="Measured risk is low", issuer_risk_level=RiskLevel.LOW,
            issuer_risk_score=1, data_uncertainty_level=RiskLevel.LOW, data_uncertainty_score=1,
        ),
    )

    decision = ManagerAgent().finalize(state)["final_decision"]

    assert decision.recommendation is Recommendation.WATCH
    assert decision.evidence_completeness < 0.75
