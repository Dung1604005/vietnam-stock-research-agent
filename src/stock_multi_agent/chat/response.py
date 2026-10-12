from __future__ import annotations

from stock_multi_agent.models.chat import AssistantResponse, ResponseTable, ResponseType
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.state import AgentState


class ResponseComposer:
    SIGNAL_LABELS = {"BUY": "MUA", "SELL": "BÁN", "HOLD": "NẮM GIỮ", "UNKNOWN": "CHƯA XÁC ĐỊNH"}
    RECOMMENDATION_LABELS = {"BUY": "MUA", "HOLD": "NẮM GIỮ", "WATCH": "THEO DÕI", "AVOID": "TRÁNH"}
    RISK_LABELS = {"LOW": "THẤP", "MEDIUM": "TRUNG BÌNH", "HIGH": "CAO"}

    @classmethod
    def full_analysis(cls, state: AgentState) -> AssistantResponse:
        if not all((state.final_decision, state.technical_result, state.fundamental_result, state.risk_result, state.market_data)):
            raise ValueError("Full analysis result is incomplete")
        final = state.final_decision
        technical = state.technical_result
        fundamental = state.fundamental_result
        risk = state.risk_result
        forecast = state.price_forecast
        forecast_summary = cls._forecast_summary(state)
        message = (
            f"{forecast_summary}\n\n"
            f"### {state.ticker} · Phân tích {cls._horizon_label(state.investment_horizon)}\n\n"
            f"**Kỹ thuật:** {cls._signal_label(technical.signal.value)} · {technical.score:.1f}/10  \n"
            f"**Cơ bản:** {cls._signal_label(fundamental.signal.value)} · {fundamental.score:.1f}/10  \n"
            f"**Rủi ro đo lường:** {cls.RISK_LABELS.get(risk.risk_level.value, risk.risk_level.value)} · {risk.risk_score:.1f}/10  \n"
            f"**Rủi ro riêng doanh nghiệp:** {cls.RISK_LABELS.get((risk.issuer_risk_level or risk.risk_level).value, (risk.issuer_risk_level or risk.risk_level).value)} · "
            f"{risk.issuer_risk_score if risk.issuer_risk_score is not None else risk.risk_score:.1f}/10  \n"
            f"**Độ bất định dữ liệu:** {cls.RISK_LABELS.get(risk.data_uncertainty_level.value, risk.data_uncertainty_level.value)} · "
            f"{risk.data_uncertainty_score:.1f}/10\n\n"
            f"#### Kết luận: {cls.RECOMMENDATION_LABELS.get(final.recommendation.value, final.recommendation.value)}\n"
            f"{final.final_reasoning}"
        )
        return AssistantResponse(
            message=message,
            response_type=ResponseType.ANALYSIS,
            tickers=[state.ticker],
            agent_trace=state.trace,
            sources=cls._sources(state, include_benchmark=True),
            source_details=cls._source_details(state, include_benchmark=True),
            warnings=list(dict.fromkeys([*cls._warnings(state), *(forecast.warnings if forecast else [])])),
            data_timestamp=state.market_data.bars[-1].date,
            data_fetched_at=state.market_data.fetched_at,
            metadata={
                "recommendation": final.recommendation.value,
                "confidence": final.confidence,
                "decision_policy_version": final.policy_version,
                "decision_calibration_status": final.calibration_status,
                "technical_score": technical.score,
                "technical_signal": technical.signal.value,
                "fundamental_score": fundamental.score,
                "fundamental_signal": fundamental.signal.value,
                "risk_score": risk.risk_score,
                "risk_level": risk.risk_level.value,
                "issuer_risk_score": risk.issuer_risk_score,
                "issuer_risk_level": (risk.issuer_risk_level or risk.risk_level).value,
                "data_uncertainty_score": risk.data_uncertainty_score,
                "data_uncertainty_level": risk.data_uncertainty_level.value,
                "evidence_completeness": final.evidence_completeness,
                "price_forecast": forecast.model_dump(mode="json") if forecast else None,
                "conflicts": final.conflicts,
                "agent_messages": cls._agent_messages(state),
                "research_citations": cls._research_citations(state),
                **cls._agent_quality_metadata(state, [
                    AgentName.DATA, AgentName.RESEARCH, AgentName.TECHNICAL,
                    AgentName.FUNDAMENTAL, AgentName.RISK, AgentName.MANAGER,
                ]),
            },
        )

    @staticmethod
    def _forecast_summary(state: AgentState) -> str:
        if state.market_data is None:
            return ""
        latest = state.market_data.bars[-1]
        forecast = state.price_forecast
        if forecast is None or forecast.forecast_price is None:
            return (
                "#### Giá hiện tại và ước tính phiên kế tiếp\n"
                f"**Giá gần nhất ({latest.date.isoformat()}):** {latest.close:,.2f} VND  \n"
                "**Ước tính phiên kế tiếp:** chưa có vì mô hình không tạo được kết quả dự báo."
            )
        bounds = ""
        if forecast.lower_bound is not None and forecast.upper_bound is not None:
            bounds = f" · vùng dao động chẩn đoán {forecast.lower_bound:,.2f}–{forecast.upper_bound:,.2f} VND"
        confidence = "chưa hiệu chỉnh" if forecast.confidence == 0 else f"{forecast.confidence:.0%}"
        return (
            "#### Giá hiện tại và ước tính phiên kế tiếp\n"
            f"**Giá gần nhất ({forecast.as_of.isoformat()}):** {forecast.current_price:,.2f} VND  \n"
            f"**Ước tính phiên kế tiếp:** {forecast.forecast_price:,.2f} VND{bounds}  \n"
            f"**Trạng thái dự báo:** {forecast.status.value} · {forecast.model_version} · độ tin cậy {confidence}\n"
            "Ước tính này độc lập với chính sách kết luận MUA/NẮM GIỮ/THEO DÕI."
        )

    @classmethod
    def _signal_label(cls, value: str) -> str:
        return cls.SIGNAL_LABELS.get(value, value)

    @staticmethod
    def _horizon_label(value: str) -> str:
        parts = value.split(maxsplit=1)
        if len(parts) == 2 and parts[1] == "months":
            return f"{parts[0]} tháng"
        if len(parts) == 2 and parts[1] in {"year", "years"}:
            return f"{parts[0]} năm"
        return value

    @classmethod
    def technical(cls, state: AgentState) -> AssistantResponse:
        if state.technical_result is None or state.market_data is None:
            raise ValueError("Technical result is incomplete")
        result = state.technical_result
        evidence = cls._format_evidence(result.evidence)
        return AssistantResponse(
            message=(
                f"{cls._forecast_summary(state)}\n\n"
                f"### Phân tích kỹ thuật {state.ticker}\n\n"
                f"**Tín hiệu:** {result.signal.value} · **Điểm:** {result.score:.1f}/10 · "
                f"**Độ tin cậy:** {result.confidence:.0%}\n\n{result.reasoning_summary}\n\n{evidence}"
            ),
            response_type=ResponseType.ANALYSIS,
            tickers=[state.ticker], agent_trace=state.trace,
            sources=cls._sources(state, include_benchmark=True),
            source_details=cls._source_details(state, include_benchmark=True),
            warnings=cls._warnings_with_forecast(state), data_timestamp=state.market_data.bars[-1].date,
            data_fetched_at=state.market_data.fetched_at,
            metadata={
                "technical_score": result.score, "technical_signal": result.signal.value,
                "price_forecast": cls._forecast_metadata(state), "agent_messages": cls._agent_messages(state),
                **cls._agent_quality_metadata(state, [AgentName.DATA, AgentName.TECHNICAL]),
            },
        )

    @classmethod
    def fundamental(cls, state: AgentState) -> AssistantResponse:
        if state.fundamental_result is None or state.market_data is None:
            raise ValueError("Fundamental result is incomplete")
        result = state.fundamental_result
        evidence = cls._format_evidence(result.evidence)
        return AssistantResponse(
            message=(
                f"{cls._forecast_summary(state)}\n\n"
                f"### Phân tích cơ bản {state.ticker}\n\n"
                f"**Tín hiệu:** {result.signal.value} · **Điểm:** {result.score:.1f}/10 · "
                f"**Độ tin cậy:** {result.confidence:.0%}\n\n{result.reasoning_summary}\n\n{evidence}"
            ),
            response_type=ResponseType.ANALYSIS,
            tickers=[state.ticker], agent_trace=state.trace, sources=[state.market_data.source],
            source_details=state.market_data.source_refs,
            warnings=cls._warnings_with_forecast(state), data_timestamp=state.market_data.bars[-1].date,
            data_fetched_at=state.market_data.fetched_at,
            metadata={
                "fundamental_score": result.score, "fundamental_signal": result.signal.value,
                "price_forecast": cls._forecast_metadata(state), "agent_messages": cls._agent_messages(state),
                "research_citations": cls._research_citations(state),
                **cls._agent_quality_metadata(
                    state, [AgentName.DATA, AgentName.RESEARCH, AgentName.FUNDAMENTAL],
                ),
            },
        )

    @classmethod
    def risk(cls, state: AgentState) -> AssistantResponse:
        if state.risk_result is None or state.market_data is None:
            raise ValueError("Risk result is incomplete")
        result = state.risk_result
        critiques = "\n".join(f"- {item}" for item in result.criticisms) or "- Không có phản biện trọng yếu."
        return AssistantResponse(
            message=(
                f"{cls._forecast_summary(state)}\n\n"
                f"### Đánh giá rủi ro {state.ticker}\n\n"
                f"**Rủi ro đo lường:** {result.risk_level.value} · **Điểm:** {result.risk_score:.1f}/10  \n"
                f"**Rủi ro riêng doanh nghiệp:** {(result.issuer_risk_level or result.risk_level).value} · "
                f"{result.issuer_risk_score if result.issuer_risk_score is not None else result.risk_score:.1f}/10  \n"
                f"**Độ bất định dữ liệu:** {result.data_uncertainty_level.value} · {result.data_uncertainty_score:.1f}/10 · "
                f"**Veto:** {'Có' if result.veto else 'Không'}\n\n{result.reasoning_summary}\n\n{critiques}"
            ),
            response_type=ResponseType.ANALYSIS,
            tickers=[state.ticker], agent_trace=state.trace, sources=[state.market_data.source],
            source_details=state.market_data.source_refs,
            warnings=list(dict.fromkeys([*cls._warnings_with_forecast(state), *result.warnings])), data_timestamp=state.market_data.bars[-1].date,
            data_fetched_at=state.market_data.fetched_at,
            metadata={
                "risk_score": result.risk_score,
                "risk_level": result.risk_level.value,
                "issuer_risk_score": result.issuer_risk_score,
                "issuer_risk_level": (result.issuer_risk_level or result.risk_level).value,
                "data_uncertainty_score": result.data_uncertainty_score,
                "data_uncertainty_level": result.data_uncertainty_level.value,
                "veto": result.veto,
                "price_forecast": cls._forecast_metadata(state),
                "risk_sections": [section.model_dump(mode="json") for section in result.sections],
                "agent_messages": cls._agent_messages(state),
                "research_citations": cls._research_citations(state),
                **cls._agent_quality_metadata(state, [
                    AgentName.DATA, AgentName.RESEARCH, AgentName.TECHNICAL,
                    AgentName.FUNDAMENTAL, AgentName.RISK,
                ]),
            },
        )

    @staticmethod
    def _agent_quality_metadata(state: AgentState, expected_agents: list[AgentName]) -> dict:
        from stock_multi_agent.services.agent_evaluation import AgentEvaluationService

        evaluation = AgentEvaluationService().evaluate(state, expected_agents)
        return {
            "agent_reports": [report.model_dump(mode="json") for report in state.agent_reports],
            "agent_evaluation": evaluation.model_dump(mode="json"),
        }

    @staticmethod
    def _research_citations(state: AgentState) -> list[dict]:
        return [citation.model_dump(mode="json") for citation in state.research_evidence]

    @staticmethod
    def _forecast_metadata(state: AgentState) -> dict | None:
        return state.price_forecast.model_dump(mode="json") if state.price_forecast else None

    @classmethod
    def _warnings_with_forecast(cls, state: AgentState) -> list[str]:
        forecast_warnings = state.price_forecast.warnings if state.price_forecast else []
        return list(dict.fromkeys([*cls._warnings(state), *forecast_warnings]))

    @staticmethod
    def _warnings(state: AgentState) -> list[str]:
        warnings = list(state.market_data.warnings) if state.market_data else []
        return list(dict.fromkeys(warnings))

    @staticmethod
    def _sources(state: AgentState, *, include_benchmark: bool = False) -> list[str]:
        sources = [state.market_data.source] if state.market_data else []
        if include_benchmark and state.benchmark_data is not None:
            sources.append(f"{state.benchmark_data.source} (VNINDEX benchmark)")
        return list(dict.fromkeys(sources))

    @staticmethod
    def _source_details(state: AgentState, *, include_benchmark: bool = False):
        refs = list(state.market_data.source_refs) if state.market_data else []
        if include_benchmark and state.benchmark_data is not None:
            refs.extend(state.benchmark_data.source_refs)
        unique = {}
        for ref in refs:
            unique[(ref.source_id, ref.content_sha256, ref.data_kind.value)] = ref
        return list(unique.values())

    @staticmethod
    def _agent_messages(state: AgentState) -> list[dict[str, str]]:
        return [
            {
                "sender": message.sender.value,
                "receiver": message.receiver.value,
                "type": message.message_type.value,
                "task": message.task,
            }
            for message in state.messages
        ]

    @staticmethod
    def _format_evidence(items) -> str:
        return "\n".join(f"- **{item.label}:** {item.value}" for item in items)
