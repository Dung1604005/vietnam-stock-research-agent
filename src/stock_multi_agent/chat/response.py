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
        response = AssistantResponse(
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
                "technical_policy_version": technical.policy_version,
                "technical_feature_contributions": technical.feature_contributions,
                "technical_diagnostics": technical.diagnostics,
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
                "financial_fact_proposals": state.financial_fact_proposals,
                **cls._agent_quality_metadata(state, [
                    AgentName.DATA, AgentName.RESEARCH, AgentName.TECHNICAL,
                    AgentName.FUNDAMENTAL, AgentName.RISK, AgentName.MANAGER,
                ]),
            },
        )

        return cls._enhance(response, state, ['technical', 'fundamental', 'risk'])

    @staticmethod
    def _forecast_summary(state: AgentState) -> str:
        if state.market_data is None:
            return ""
        latest = state.market_data.bars[-1]
        unit = state.market_data.price_unit
        unit_label = {'VND': 'VND', 'THOUSAND_VND': 'nghìn VND'}.get(unit, 'đơn vị nguồn (chưa xác minh thang giá)')
        forecast = state.price_forecast
        if forecast is None or forecast.forecast_price is None:
            return (
                "#### Giá hiện tại và ước tính phiên kế tiếp\n"
                f"**Giá gần nhất ({latest.date.isoformat()}):** {latest.close:,.2f} {unit_label}  \n"
                "**Ước tính phiên kế tiếp:** chưa có vì mô hình không tạo được kết quả dự báo."
            )
        bounds = ""
        if forecast.lower_bound is not None and forecast.upper_bound is not None:
            bounds = f" · vùng dao động chẩn đoán {forecast.lower_bound:,.2f}–{forecast.upper_bound:,.2f} {unit_label}"
        weight = state.technical_result.diagnostics.get('lstm_weight', 0) if state.technical_result else 0
        if weight:
            score_role = f"Ước tính LSTM này đóng góp {weight:.0%} vào điểm Technical; {1-weight:.0%} còn lại đến từ các chỉ báo kỹ thuật."
        else:
            score_role = 'Dự báo này chưa được áp dụng vào điểm Technical trong kết quả này.'
        if forecast.experimental_blend_enabled:
            score_role += ' V4 thử nghiệm do người dùng chọn; chưa qua forward validation, không phải model đã được phê duyệt.'
        if forecast.status.value == 'BASELINE_UNTRAINED':
            score_role = 'Ước tính baseline chưa huấn luyện không tham gia vào điểm Technical.'
        return (
            "#### Giá hiện tại và ước tính phiên kế tiếp\n"
            f"**Giá gần nhất ({forecast.as_of.isoformat()}):** {forecast.current_price:,.2f} {unit_label}  \n"
            f"**Ước tính phiên kế tiếp:** {forecast.forecast_price:,.2f} {unit_label}{bounds}  \n"
            f"**Trạng thái dự báo:** {forecast.status.value} · {forecast.model_version}\n"
            f"{score_role}"
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
        response = AssistantResponse(
            message=(
                f"{cls._forecast_summary(state)}\n\n"
                f"### Phân tích kỹ thuật {state.ticker}\n\n"
                f"**Tín hiệu:** {result.signal.value} · **Điểm:** {result.score:.1f}/10\n\n"
                f"{result.reasoning_summary}\n\n{evidence}"
            ),
            response_type=ResponseType.ANALYSIS,
            tickers=[state.ticker], agent_trace=state.trace,
            sources=cls._sources(state, include_benchmark=True),
            source_details=cls._source_details(state, include_benchmark=True),
            warnings=cls._warnings_with_forecast(state), data_timestamp=state.market_data.bars[-1].date,
            data_fetched_at=state.market_data.fetched_at,
            metadata={
                "technical_score": result.score, "technical_signal": result.signal.value,
                "technical_policy_version": result.policy_version,
                "technical_feature_contributions": result.feature_contributions,
                "technical_diagnostics": result.diagnostics,
                "price_forecast": cls._forecast_metadata(state), "agent_messages": cls._agent_messages(state),
                **cls._agent_quality_metadata(state, [AgentName.DATA, AgentName.TECHNICAL]),
            },
        )

        return cls._enhance(response, state, ['technical'])

    @classmethod
    def fundamental(cls, state: AgentState) -> AssistantResponse:
        if state.fundamental_result is None or state.market_data is None:
            raise ValueError("Fundamental result is incomplete")
        result = state.fundamental_result
        evidence = cls._format_evidence(result.evidence)
        response = AssistantResponse(
            message=(
                f"{cls._forecast_summary(state)}\n\n"
                f"### Phân tích cơ bản {state.ticker}\n\n"
                f"**Tín hiệu:** {result.signal.value} · **Điểm:** {result.score:.1f}/10\n\n"
                f"{result.reasoning_summary}\n\n{evidence}"
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

        return cls._enhance(response, state, ['fundamental'])

    @classmethod
    def risk(cls, state: AgentState) -> AssistantResponse:
        if state.risk_result is None or state.market_data is None:
            raise ValueError("Risk result is incomplete")
        result = state.risk_result
        critiques = "\n".join(f"- {item}" for item in result.criticisms) or "- Không có phản biện trọng yếu."
        response = AssistantResponse(
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

        return cls._enhance(response, state, ['risk'])

    @staticmethod
    def _enhance(response, state, sections):
        from stock_multi_agent.services.analysis_explainer import AnalysisExplainer
        from stock_multi_agent.services.chart_service import ChartService
        from stock_multi_agent.models.chart import ChartRequest, ChartType
        AnalysisExplainer().attach(response, state, sections)
        from stock_multi_agent.services.financial_facts import build_financial_evidence
        response.metadata['canonical_financial_evidence'] = build_financial_evidence(state.market_data)
        response.metadata['evidence_follow_up_outcomes'] = state.evidence_follow_up_outcomes
        data = state.market_data
        response.metadata['data_origins'] = [{'group': 'market_and_financial_data',
            'mode': data.access_mode, 'as_of': str(data.as_of), 'fetched_at': data.fetched_at.isoformat(),
            'source_ids': [item.source_id for item in data.source_refs]}]
        if state.price_forecast:
            forecast = state.price_forecast
            response.metadata['data_origins'].append({'group': 'forecast', 'mode': 'MODEL_ESTIMATE',
                'model_version': forecast.model_version, 'as_of': str(forecast.as_of),
                'horizon_sessions': forecast.horizon_sessions, 'validation_status': forecast.validation_status,
                'policy_eligibility': forecast.policy_eligibility,
                'actual_technical_weight': state.technical_result.diagnostics.get('lstm_weight', 0) if state.technical_result else 0})
        kinds = []
        if 'technical' in sections:
            kinds.append(ChartType.PRICE_MA)
        if 'risk' in sections:
            kinds.append(ChartType.DRAWDOWN)
        if 'fundamental' in sections:
            kinds.append(ChartType.REVENUE_PROFIT)
        charts = ChartService()
        unavailable = []
        for kind in kinds:
            try:
                response.charts.append(charts.build_artifact(ChartRequest(ticker=state.ticker, chart_type=kind), state.market_data))
            except ValueError as exc:
                unavailable.append({'chart': kind.value, 'reason': str(exc)})
        response.metadata['unavailable_charts'] = unavailable
        if unavailable:
            response.message += '\n\n**Biểu đồ chưa có dữ liệu:** ' + '; '.join(f'{item["chart"]}: {item["reason"]}' for item in unavailable)
        return response

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
