from __future__ import annotations

import re
import json
import logging
import sqlite3
import time
import hashlib
from collections import OrderedDict
from datetime import date
from typing import Any

from stock_multi_agent.models.chat import ConversationState, ParsedQuery, QueryIntent
from stock_multi_agent.nlu.context_resolver import ContextResolver
from stock_multi_agent.nlu.entity_extractor import EntityExtractor
from stock_multi_agent.nlu.intent_classifier import SemanticSimilarityClassifier
from stock_multi_agent.nlu.models import SemanticQuery
from stock_multi_agent.nlu.normalizer import VietnameseTextNormalizer
from stock_multi_agent.nlu.prompts import EXAMPLE_BANK_VERSION, SEMANTIC_PROMPT_VERSION, SEMANTIC_SYSTEM_PROMPT
from stock_multi_agent.nlu.telemetry import SemanticTelemetryEvent, SemanticTelemetryStore, configured_semantic_telemetry_store
from stock_multi_agent.providers.llm_provider import SemanticLLMProvider, configured_semantic_provider
from stock_multi_agent.services.request_telemetry import current_metrics, observed


class HybridSemanticParser:
    """Deterministic entities + optional structured LLM + local semantic fallback."""

    def __init__(
        self,
        llm_provider: SemanticLLMProvider | None = None,
        use_configured_provider: bool = True,
        telemetry_store: SemanticTelemetryStore | None = None,
    ) -> None:
        self.normalizer = VietnameseTextNormalizer()
        self.entities = EntityExtractor()
        self.context = ContextResolver()
        self.classifier = SemanticSimilarityClassifier(self.normalizer)
        self.llm_provider = llm_provider if llm_provider is not None else (
            configured_semantic_provider() if use_configured_provider else None
        )
        self.telemetry_store = telemetry_store if telemetry_store is not None else configured_semantic_telemetry_store()
        self._cache: OrderedDict[str, tuple[float, ParsedQuery]] = OrderedDict()
        self._force_llm = llm_provider is not None

    @observed("semantic_parse")
    def parse(self, raw_query: str, context: ConversationState | None = None) -> ParsedQuery:
        started_at = time.perf_counter()
        raw = raw_query.strip()
        normalized = self.normalizer.normalize(raw)
        signature = {"query": raw, "context": context.model_dump(mode="json", exclude={
            "messages", "conversation_id", "updated_at", "last_analysis"}) if context else None}
        cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached and started_at - cached[0] < 300:
            self._cache.move_to_end(cache_key)
            metrics = current_metrics()
            if metrics:
                metrics.cache_hits += 1
            return cached[1].model_copy(deep=True, update={"raw_query": raw_query, "cache_hit": True,
                "llm_attempted": False, "llm_latency_ms": None,
                "parser_latency_ms": round((time.perf_counter() - started_at) * 1000)})
        extracted = self.entities.extract(raw, normalized)
        resolved = self.context.resolve(normalized, extracted.tickers, context)

        local = self._fallback(raw, normalized, extracted, resolved, context)
        local = self._enforce_knowledge_contract(local, normalized)
        local = self._extended_intents(local, normalized, context, extracted)
        use_llm = self.llm_provider is not None and (self._force_llm or local.confidence < 0.85
            or bool(local.secondary_intents) or local.intent == QueryIntent.UNKNOWN)
        semantic: SemanticQuery | None = None
        llm_attempted = use_llm
        llm_latency_ms: int | None = None
        fallback_reason: str | None = "no_configured_llm" if self.llm_provider is None else None
        if use_llm:
            llm_started_at = time.perf_counter()
            try:
                semantic = self._parse_with_llm(raw, normalized, extracted.model_dump(), resolved.model_dump(), context)
                semantic = self._merge_deterministic_entities(semantic, extracted.model_dump(), resolved.model_dump())
                if extracted.duration:
                    semantic = semantic.model_copy(update={"time_horizon": local.time_horizon, "chart_period": local.chart_period})
            except Exception as exc:
                # Network/model failures must not break local mock mode behavior.
                semantic = None
                fallback_reason = f"llm_error:{type(exc).__name__}"
            finally:
                llm_latency_ms = round((time.perf_counter() - llm_started_at) * 1000)
        if semantic is None:
            semantic = local
        semantic = self._enforce_recommendation_contract(semantic, normalized)
        semantic = self._enforce_discovery_contract(semantic)
        semantic = self._enforce_knowledge_contract(semantic, normalized)
        semantic = self._extended_intents(semantic, normalized, context, extracted)
        semantic = self._enforce_required_entity_contract(semantic)
        parsed = semantic.to_parsed_query().model_copy(update={
            "parser_latency_ms": round((time.perf_counter() - started_at) * 1000),
            "llm_attempted": llm_attempted,
            "llm_latency_ms": llm_latency_ms,
            "fallback_reason": fallback_reason,
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "example_bank_version": EXAMPLE_BANK_VERSION,
        })
        exchange = re.search(r"\b(hose|hsx|hnx|upcom)\b", normalized)
        date_match = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", raw)
        if date_match:
            try:
                parsed.as_of = date(*map(int, date_match.groups()))
            except ValueError:
                parsed.needs_clarification = True
                parsed.clarification_question = "Ngày yêu cầu không hợp lệ; bạn nhập lại theo YYYY-MM-DD nhé."
        parsed.exchange = exchange.group(1).upper().replace("HSX", "HOSE") if exchange else None
        raw_dates = re.findall(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", raw)
        if len(raw_dates) >= 2 and re.search(r"\btu\b.+\bden\b", normalized):
            try:
                parsed.published_after = date(*map(int, raw_dates[0]))
                parsed.as_of = date(*map(int, raw_dates[-1]))
                if parsed.published_after > parsed.as_of:
                    raise ValueError("inverted date range")
            except ValueError:
                parsed.needs_clarification = True
                parsed.clarification_question = "Khoảng ngày không hợp lệ; ngày bắt đầu phải trước ngày kết thúc."
        if re.search(r"\b(?:an toan|rui ro thap|bao thu)\b", normalized):
            parsed.risk_preference = "LOW"
        elif re.search(r"\b(?:rui ro cao|mao hiem|chap nhan rui ro)\b", normalized):
            parsed.risk_preference = "HIGH"
        for dimension, pattern in {"valuation": r"\b(?:dinh gia|pe|pb)\b", "risk": r"\brui ro\b",
                "growth": r"\btang truong\b", "liquidity": r"\bthanh khoan\b"}.items():
            if re.search(pattern, normalized):
                parsed.comparison_dimensions.append(dimension)
        for indicator, operator, number in re.findall(r"\b(pe|pb|roe|rsi)\s*(duoi|tren|<|>)\s*(\d+(?:[.,]\d+)?)", normalized):
            parsed.filters[f"{indicator}_{'lt' if operator in {'duoi', '<'} else 'gt'}"] = float(number.replace(",", "."))
        if parsed.intent == QueryIntent.PORTFOLIO_RISK:
            pairs = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*%', raw)
            parsed.portfolio_weights = {ticker: float(value.replace(',', '.')) / 100 for ticker, value in pairs}
            if len(pairs) == len(parsed.portfolio_weights) and parsed.portfolio_weights and all(0 < value <= 1 for value in parsed.portfolio_weights.values()) and sum(parsed.portfolio_weights.values()) <= 1.000001:
                parsed.tickers = list(parsed.portfolio_weights)
                parsed.needs_clarification, parsed.clarification_question = False, None
            else:
                parsed.needs_clarification = True
                parsed.clarification_question = 'Nhập mã và tỷ trọng không trùng, tổng không quá 100%, ví dụ: rủi ro danh mục FPT 60% HPG 40%.'
        if parsed.intent == QueryIntent.SCENARIO_ANALYSIS:
            shocks = re.findall(r'(?:(giảm|giam|tăng|tang)\s+)?([+-]?\d+(?:[.,]\d+)?)\s*%', raw.casefold())
            parsed.scenario_shocks = [(-abs(float(value.replace(',', '.'))) if direction in {'giảm', 'giam'} else float(value.replace(',', '.'))) / 100 for direction, value in shocks][:5]
        self._cache[cache_key] = (started_at, parsed.model_copy(deep=True))
        while len(self._cache) > 256:
            self._cache.popitem(last=False)
        logging.getLogger(__name__).info(
            "semantic_parse %s",
            json.dumps({
                "parser_source": parsed.parser_source,
                "parser_latency_ms": parsed.parser_latency_ms,
                "llm_attempted": parsed.llm_attempted,
                "llm_latency_ms": parsed.llm_latency_ms,
                "fallback_reason": parsed.fallback_reason,
                "ambiguity_count": len(parsed.ambiguities),
                "needs_clarification": parsed.needs_clarification,
                "prompt_version": parsed.prompt_version,
                "example_bank_version": parsed.example_bank_version,
            }, sort_keys=True),
        )
        if self.telemetry_store is not None:
            try:
                self.telemetry_store.record(SemanticTelemetryEvent(
                    parser_source=parsed.parser_source,
                    parser_latency_ms=parsed.parser_latency_ms,
                    llm_attempted=parsed.llm_attempted,
                    llm_latency_ms=parsed.llm_latency_ms,
                    fallback_reason=parsed.fallback_reason,
                    ambiguity_count=len(parsed.ambiguities),
                    needs_clarification=parsed.needs_clarification,
                    prompt_version=parsed.prompt_version,
                    example_bank_version=parsed.example_bank_version,
                ))
            except (OSError, ValueError, sqlite3.Error) as exc:
                logging.getLogger(__name__).warning("semantic_telemetry_write_failed %s", type(exc).__name__)
        return parsed

    @staticmethod
    def _extended_intents(semantic, normalized, context, extracted):
        updates = {}
        if re.search(r"\b(?:danh muc|portfolio)\b", normalized) and re.search(r"\b(?:rui ro|risk|bien dong)\b", normalized):
            updates = {"intent": QueryIntent.PORTFOLIO_RISK, "confidence": 0.95,
                "needs_clarification": True,
                "clarification_question": "Nhập mã và tỷ trọng, ví dụ: rủi ro danh mục FPT 60% HPG 40%."}
        elif re.search(r'\b(?:xuat|export)\b.*\b(?:bao cao|report|html|markdown)\b', normalized):
            updates = {'intent': QueryIntent.REPORT_EXPORT, 'confidence': .98, 'needs_clarification': False, 'clarification_question': None}
        elif re.search(r'\b(?:ve|bieu do|chart|dashboard)\b', normalized) and not re.match(r'^(?:phan tich|danh gia|nhan xet|so\b|so sanh|compare)', normalized) and not re.search(r'\b(?:la gi|cong thuc)\b', normalized):
            updates = {'intent': QueryIntent.STOCK_CHART, 'confidence': .97}
        elif re.search(r'\b(?:vi sao|tai sao)\b.*\b(?:ket luan|danh gia)\b.*\b(?:doi|thay doi)\b', normalized):
            updates = {'intent': QueryIntent.CHANGE_EXPLANATION, 'confidence': .97}
        elif re.search(r'\b(?:kich ban|sensitivity|scenario)\b', normalized):
            updates = {'intent': QueryIntent.SCENARIO_ANALYSIS, 'confidence': .97}
        elif re.search(r'\b(?:tac dong|anh huong)\b.*\b(?:su kien|tin tuc)\b', normalized):
            updates = {'intent': QueryIntent.EVENT_IMPACT, 'confidence': .97}
        elif re.search(r'\b(?:so sanh|thay doi)\b.*\b(?:bao cao|loi nhuan|doanh thu)\b.*\b(?:hai ky|2 ky|giua cac ky)\b', normalized):
            updates = {'intent': QueryIntent.REPORT_CHANGE, 'confidence': .97}
        elif re.search(r'\b(?:percentile|peer comparison|so sanh cung nganh)\b', normalized):
            updates = {'intent': QueryIntent.PEER_COMPARISON, 'confidence': .97}
        elif semantic.tickers and re.search(r"\b(?:du doan|du bao|forecast|uoc tinh gia)\b", normalized):
            updates = {"intent": QueryIntent.FORECAST, "confidence": 0.95}
        elif re.search(r"\b(?:nguon du lieu|link nguon|trich dan|source)\b", normalized):
            updates = {"intent": QueryIntent.SOURCE_REQUEST, "confidence": 0.95}
        elif re.search(r"\b(?:luan diem|thesis)\b", normalized):
            updates = {"intent": QueryIntent.THESIS_REVIEW, "confidence": 0.92}
        elif re.search(r"\b(?:trong|dua tren|doc|tom tat)\b", normalized) and re.search(r"\b(?:bao cao|tai lieu|filing|document)\b", normalized):
            updates = {"intent": QueryIntent.DOCUMENT_QA, "confidence": 0.94}
        if context and extracted.duration and not extracted.tickers and re.match(r"^(?:doi thanh|doi sang|chuyen sang)", normalized):
            updates = {"intent": context.last_intent or QueryIntent.STOCK_ANALYSIS,
                       "tickers": list(context.active_tickers), "time_horizon": HybridSemanticParser._human_horizon(extracted.duration),
                       "references_previous_context": True, "needs_clarification": False, "clarification_question": None}
        result = semantic.model_copy(update=updates) if updates else semantic
        if context and not result.tickers and len(context.active_tickers) == 1 and result.intent in {
            QueryIntent.CHANGE_EXPLANATION, QueryIntent.SCENARIO_ANALYSIS, QueryIntent.EVENT_IMPACT, QueryIntent.REPORT_CHANGE}:
            result = result.model_copy(update={'tickers': list(context.active_tickers), 'references_previous_context': True})
        if result.intent in {QueryIntent.DOCUMENT_QA, QueryIntent.SOURCE_REQUEST, QueryIntent.THESIS_REVIEW, QueryIntent.FORECAST,
                QueryIntent.CHANGE_EXPLANATION, QueryIntent.SCENARIO_ANALYSIS, QueryIntent.EVENT_IMPACT,
                QueryIntent.REPORT_CHANGE, QueryIntent.PEER_COMPARISON}:
            if result.tickers or result.intent != QueryIntent.FORECAST:
                result = result.model_copy(update={"needs_clarification": False, "clarification_question": None})
        secondary = list(result.secondary_intents)
        if result.tickers and re.search(r"\b(?:va|dong thoi|kem)\b", normalized):
            if re.search(r"\b(?:tim|doc|tom tat) .{0,40}(?:bao cao|tai lieu)\b", normalized) and result.intent not in {
                QueryIntent.RESEARCH, QueryIntent.DOCUMENT_QA, QueryIntent.SOURCE_REQUEST, QueryIntent.THESIS_REVIEW}:
                secondary.append(QueryIntent.RESEARCH)
            if re.search(r"\bphan tich ky thuat\b", normalized) and result.intent in {
                QueryIntent.RESEARCH, QueryIntent.DOCUMENT_QA, QueryIntent.SOURCE_REQUEST, QueryIntent.THESIS_REVIEW}:
                secondary.append(QueryIntent.TECHNICAL_ANALYSIS)
        return result.model_copy(update={"secondary_intents": list(dict.fromkeys(secondary))})

    def _parse_with_llm(
        self,
        raw: str,
        normalized: str,
        entities: dict[str, Any],
        resolution: dict[str, Any],
        context: ConversationState | None,
    ) -> SemanticQuery:
        context_payload = {
            "active_tickers": context.active_tickers if context else [],
            "last_stock_list": context.last_stock_list if context else [],
            "last_intent": context.last_intent.value if context and context.last_intent else None,
            "last_chart_period": context.last_chart_period if context else None,
            "summary": context.summary if context else "",
        }
        payload = {
            "query": raw,
            "normalized_query": normalized,
            "deterministic_entities": entities,
            "context_resolution": resolution,
            "conversation_context": context_payload,
        }
        result = self.llm_provider.parse_structured(SEMANTIC_SYSTEM_PROMPT, payload, SemanticQuery)  # type: ignore[union-attr]
        return result.model_copy(update={
            "original_query": raw,
            "normalized_query": normalized,
            "parser_source": "llm_structured",
        })

    def _fallback(self, raw, normalized, extracted, resolved, context) -> SemanticQuery:
        intent, confidence, scores = self.classifier.classify(normalized, extracted, context)
        secondary = self.classifier.secondary_intents(normalized, intent, extracted)

        theory_terms = re.search(
            r"\b(?:co phieu|von hoa|thanh khoan|chi so|pe|pb|roe|roa|eps|rsi|macd|atr|sma|ema|"
            r"bollinger|dcf|ev ebitda|margin of safety|capm|sharpe|beta|alpha|backtesting?|look ahead|"
            r"survivorship|data leakage|co tuc|chia tach|phat hanh|quyen mua|khong huong quyen|"
            r"bao cao tai chinh|bang can doi|ket qua kinh doanh|luu chuyen tien|debt to equity|"
            r"current ratio|free cash flow|dividend yield|bien loi nhuan|khop lenh|thoa thuan|"
            r"volatility|drawdown|da dang hoa|position sizing|rui ro thanh khoan|"
            r"thi truong hieu qua|factor investing|phan tich ky thuat|ho tro|khang cu|xu huong)\b",
            normalized,
        )
        asks_theory = re.search(
            r"\b(?:la gi|nghia|cong thuc|duoc tinh|tinh the nao|giai thich|phan biet|tai sao|vi sao|"
            r"dung de lam gi|cho biet dieu gi|noi len dieu gi|gom gi|do huong|do bien dong|"
            r"khac .{0,30}the nao|do .{0,20}(?:gi|huong|bien dong)|co lam .{0,30}khong|lam .{0,20}sai the nao|the nao)\b",
            normalized,
        )
        asks_current_rule = re.search(
            r"\b(?:quy dinh|hien hanh|bien do|ato|atc|chu ky thanh toan|t\+?2)\b",
            normalized,
        )
        exchange_scope = re.search(r"\b(?:hose|hsx|hnx|upcom)\b", normalized) and re.search(
            r"\b(?:hien tai|khac nhau|quy dinh|bao nhieu)\b", normalized,
        )
        applies_to_ticker = resolved.tickers and re.search(
            r"\b(?:ap dung|danh gia|tot khong|dat khong|y nghia gi voi)\b", normalized,
        )
        if theory_terms and applies_to_ticker and (asks_theory or extracted.indicators):
            intent = QueryIntent.MIXED
            confidence = max(confidence, 0.88)
        elif (asks_current_rule or exchange_scope) and not resolved.tickers:
            intent = QueryIntent.CURRENT_RULE
            confidence = max(confidence, 0.88)
        elif theory_terms and asks_theory and not resolved.tickers:
            intent = QueryIntent.THEORY
            confidence = max(confidence, 0.88)

        recommendation_request = (
            not resolved.tickers
            and re.search(r"\b(?:ma|co phieu|ticker)\b", normalized)
            and re.search(r"\b(?:nen mua|dang mua|de (?:mua|buy)|de dau tu|dau tu ma nao|top .{0,20} de mua|lai cao|an toan)\b", normalized)
        )
        if recommendation_request:
            intent = QueryIntent.STOCK_RECOMMENDATION
            confidence = max(confidence, 0.9)

        # Strong compositional precedence: comparison coordinates multiple entities;
        # an opening broad assessment remains primary when chart/risk is added later.
        if re.search(r"\b(?:so sanh|so voi|compare|versus| vs |hon|hay)\b", f" {normalized} ") and (
            len(resolved.tickers) >= 2 or "comparison_base" in resolved.resolved_references
        ):
            intent = QueryIntent.STOCK_COMPARISON
        elif extracted.sector in {"market_current", "hose_current", "hnx_current", "upcom_current"} and not resolved.tickers:
            # Exchange words are universe selectors, never equity tickers.
            # Selecting one is an explicit request for the E2 current-data screen.
            intent = QueryIntent.STOCK_DISCOVERY
            confidence = max(confidence, 0.85)
        elif re.match(r"^(?:phan tich|danh gia|nhan xet)", normalized):
            intent = (
                QueryIntent.TECHNICAL_ANALYSIS
                if re.search(r"\b(?:ky thuat|technical|trend|xu huong|momentum|dong luong|price action)\b", normalized)
                else QueryIntent.FUNDAMENTAL_ANALYSIS if re.match(r'^(?:phan tich|danh gia|nhan xet)\s+(?:co ban|fundamental)\b', normalized)
                else QueryIntent.RISK_ANALYSIS if re.match(r'^(?:phan tich|danh gia|nhan xet)\s+(?:rui ro|risk)\b', normalized)
                else QueryIntent.STOCK_ANALYSIS
            )
        elif extracted.indicators and re.search(r"\b(?:the nao|ra sao|bao nhieu|o dau|thi sao|cho xem)\b", normalized):
            if not re.search(r"\b(?:ky thuat|technical|bieu do|chart|ve|giai thich)\b", normalized):
                intent = QueryIntent.INDICATOR_QUERY
        elif resolved.tickers and not extracted.indicators and intent == QueryIntent.INDICATOR_QUERY:
            intent = QueryIntent.STOCK_ANALYSIS

        ordinal_resolution = any(
            re.search(r"\b(?:dau|dau tien|mot|hai|2|ba|3)\b", key)
            for key in resolved.resolved_references
        )
        if ordinal_resolution and not re.search(
            r"\b(?:chart|ve|bieu do|rsi|macd|ky thuat|technical|rui ro|nguy hiem|so|compare)\b",
            normalized,
        ):
            intent = QueryIntent.STOCK_ANALYSIS

        if context and extracted.duration and not extracted.tickers and re.fullmatch(
            r"(?:6 thang|3 thang|1 nam)(?: thoi)?", normalized
        ) and context.last_intent:
            intent = context.last_intent
            resolved.references_previous_context = True
            resolved.tickers = list(context.active_tickers)

        ambiguities = list(resolved.ambiguities)
        if intent in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION} and len(extracted.requested_actions) > 1:
            ambiguities.append("buy_or_sell_request")
        if re.search(r"\b(?:con kia|ma kia|cai kia)\b", normalized) and not re.search(r"\b(?:thu 2|thu hai|dau tien)\b", normalized):
            if not re.search(r"\b(?:so|so sanh|so voi|compare)\b", normalized):
                intent = QueryIntent.UNKNOWN
        if intent == QueryIntent.FOLLOW_UP and context and context.last_intent:
            intent = context.last_intent

        ticker_required = {
            QueryIntent.STOCK_ANALYSIS, QueryIntent.TECHNICAL_ANALYSIS,
            QueryIntent.FUNDAMENTAL_ANALYSIS, QueryIntent.RISK_ANALYSIS,
            QueryIntent.STOCK_CHART, QueryIntent.INDICATOR_QUERY, QueryIntent.EXPLANATION, QueryIntent.RESEARCH,
            QueryIntent.MIXED,
        }
        if not resolved.tickers and context and len(context.active_tickers) == 1 and intent in ticker_required:
            resolved.tickers = list(context.active_tickers)
            resolved.references_previous_context = True
            resolved.resolved_references["omitted subject"] = context.active_tickers[0]

        chart_period = None
        time_horizon = self._human_horizon(extracted.duration) if extracted.temporal_direction == "future" else None
        if extracted.duration and (
            intent == QueryIntent.STOCK_CHART
            or QueryIntent.STOCK_CHART in secondary
            or extracted.temporal_direction == "historical"
        ):
            chart_period = extracted.duration
        elif extracted.duration and extracted.temporal_direction is None:
            if intent == QueryIntent.STOCK_CHART:
                chart_period = extracted.duration
            else:
                time_horizon = self._human_horizon(extracted.duration)
        if time_horizon is None:
            time_horizon = context.investment_horizon if context else "3 months"
        if intent == QueryIntent.STOCK_CHART and chart_period is None:
            chart_period = context.last_chart_period if context and context.last_chart_period else "6M"

        needs_clarification, question = self._clarification(intent, resolved.tickers, ambiguities)
        if confidence < 0.48 and intent not in {QueryIntent.GENERAL_CHAT, QueryIntent.UNKNOWN}:
            needs_clarification = True
            question = question or "Bạn muốn mình thực hiện loại phân tích nào?"
        if intent == QueryIntent.UNKNOWN:
            needs_clarification = True
            question = question or "Bạn có thể nói rõ mã cổ phiếu và nội dung muốn xem không?"

        if resolved.references_previous_context:
            confidence = min(0.97, confidence + (0.06 if resolved.resolved_references else 0.0))
        interpretation = self._interpretation(intent, resolved.tickers, chart_period, extracted.indicators)
        return SemanticQuery(
            original_query=raw,
            normalized_query=normalized,
            intent=intent,
            secondary_intents=secondary,
            tickers=resolved.tickers,
            sector=extracted.sector,
            time_horizon=time_horizon,
            chart_period=chart_period,
            indicators=extracted.indicators,
            requested_result_count=extracted.requested_result_count,
            requested_actions=extracted.requested_actions,
            comparison_targets=resolved.tickers[1:] if intent == QueryIntent.STOCK_COMPARISON else [],
            requested_information=extracted.requested_information,
            references_previous_context=resolved.references_previous_context,
            resolved_references=resolved.resolved_references,
            confidence=confidence,
            ambiguities=ambiguities,
            needs_clarification=needs_clarification,
            clarification_question=question,
            interpretation=interpretation,
            parser_source="semantic_similarity",
            intent_scores=scores,
        )

    @staticmethod
    def _merge_deterministic_entities(semantic: SemanticQuery, entities: dict, resolution: dict) -> SemanticQuery:
        tickers = resolution["tickers"] or semantic.tickers
        indicators = entities["indicators"] or semantic.indicators
        return semantic.model_copy(update={
            "tickers": tickers,
            "indicators": indicators,
            "sector": entities["sector"] or semantic.sector,
            "requested_result_count": entities["requested_result_count"] or semantic.requested_result_count,
            "requested_actions": entities["requested_actions"] or semantic.requested_actions,
            "requested_information": list(dict.fromkeys([*semantic.requested_information, *entities["requested_information"]])),
            "references_previous_context": resolution["references_previous_context"] or semantic.references_previous_context,
            "resolved_references": {**semantic.resolved_references, **resolution["resolved_references"]},
            "ambiguities": list(dict.fromkeys([*semantic.ambiguities, *resolution["ambiguities"]])),
        })

    @classmethod
    def _enforce_discovery_contract(cls, semantic: SemanticQuery) -> SemanticQuery:
        """Keep result count/action semantics deterministic even when an LLM parsed the intent."""
        if semantic.intent not in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION} or len(semantic.requested_actions) <= 1:
            return semantic
        ambiguities = list(dict.fromkeys([*semantic.ambiguities, "buy_or_sell_request"]))
        needs_clarification, question = cls._clarification(semantic.intent, semantic.tickers, ambiguities)
        return semantic.model_copy(update={
            "ambiguities": ambiguities,
            "needs_clarification": semantic.needs_clarification or needs_clarification,
            "clarification_question": question or semantic.clarification_question,
        })

    @staticmethod
    def _enforce_recommendation_contract(semantic: SemanticQuery, normalized: str) -> SemanticQuery:
        """Make explicit buy/investment shortlist requests deterministic across LLM and fallback NLU."""
        if semantic.tickers:
            return semantic
        has_stock_subject = re.search(r"\b(?:ma|co phieu|ticker)\b", normalized)
        asks_recommendation = re.search(
            r"\b(?:nen mua|dang mua|de (?:mua|buy)|de dau tu|dau tu ma nao|top .{0,20} de mua|lai cao|an toan)\b",
            normalized,
        )
        if not (has_stock_subject and asks_recommendation):
            return semantic
        return semantic.model_copy(update={
            "intent": QueryIntent.STOCK_RECOMMENDATION,
            "confidence": max(semantic.confidence, 0.9),
        })

    @staticmethod
    def _enforce_knowledge_contract(semantic: SemanticQuery, normalized: str) -> SemanticQuery:
        """Keep theory/current-rule/MIXED routing deterministic even with an LLM parser."""
        concept = re.search(
            r"\b(?:co phieu|von hoa|thanh khoan|chi so|pe|pb|roe|roa|eps|rsi|macd|atr|sma|ema|"
            r"bollinger|dcf|ev ebitda|margin of safety|capm|sharpe|beta|alpha|backtesting?|look ahead|"
            r"survivorship|data leakage|co tuc|chia tach|phat hanh|quyen mua|khong huong quyen|"
            r"bao cao tai chinh|bang can doi|ket qua kinh doanh|luu chuyen tien|debt to equity|"
            r"current ratio|free cash flow|dividend yield|bien loi nhuan|khop lenh|thoa thuan|"
            r"volatility|drawdown|da dang hoa|position sizing|rui ro thanh khoan|"
            r"thi truong hieu qua|factor investing|phan tich ky thuat|ho tro|khang cu|xu huong)\b",
            normalized,
        )
        theory_request = re.search(
            r"\b(?:la gi|nghia|cong thuc|duoc tinh|tinh the nao|giai thich|phan biet|tai sao|vi sao|"
            r"dung de lam gi|cho biet dieu gi|noi len dieu gi|gom gi|do huong|do bien dong|"
            r"khac .{0,30}the nao|do .{0,20}(?:gi|huong|bien dong)|co lam .{0,30}khong|lam .{0,20}sai the nao|the nao)\b",
            normalized,
        )
        current_rule = re.search(
            r"\b(?:quy dinh|hien hanh|bien do|ato|atc|chu ky thanh toan|t\+?2)\b",
            normalized,
        )
        exchange_scope = re.search(r"\b(?:hose|hsx|hnx|upcom)\b", normalized) and re.search(
            r"\b(?:hien tai|khac nhau|quy dinh|bao nhieu)\b", normalized,
        )
        apply = re.search(r"\b(?:ap dung|danh gia|tot khong|dat khong|y nghia gi voi)\b", normalized)
        updates: dict[str, object] = {}
        if concept and semantic.tickers and apply and (theory_request or semantic.indicators):
            updates = {"intent": QueryIntent.MIXED, "confidence": max(semantic.confidence, 0.88)}
        elif (current_rule or exchange_scope) and not semantic.tickers:
            updates = {
                "intent": QueryIntent.CURRENT_RULE, "confidence": max(semantic.confidence, 0.88),
                "needs_clarification": False, "clarification_question": None,
            }
        elif concept and theory_request and not semantic.tickers:
            updates = {
                "intent": QueryIntent.THEORY, "confidence": max(semantic.confidence, 0.88),
                "needs_clarification": False, "clarification_question": None,
            }
        elif semantic.intent is QueryIntent.MIXED:
            # Similarity examples may weakly resemble ordinary ticker analysis;
            # MIXED is allowed only when the deterministic concept/application
            # contract above is satisfied.
            updates = {"intent": QueryIntent.STOCK_ANALYSIS}
        return semantic.model_copy(update=updates) if updates else semantic

    @classmethod
    def _enforce_required_entity_contract(cls, semantic: SemanticQuery) -> SemanticQuery:
        """Do not let an LLM route ticker work without the required ticker(s)."""
        needs_clarification, question = cls._clarification(
            semantic.intent,
            semantic.tickers,
            semantic.ambiguities,
        )
        if not needs_clarification:
            return semantic
        return semantic.model_copy(update={
            "needs_clarification": True,
            "clarification_question": question or semantic.clarification_question,
        })

    @staticmethod
    def _human_horizon(duration: str | None) -> str:
        if not duration:
            return "3 months"
        if duration.endswith("M"):
            return f"{duration[:-1]} months"
        if duration.endswith("Y"):
            count = duration[:-1]
            return f"{count} year" if count == "1" else f"{count} years"
        return duration

    @staticmethod
    def _clarification(intent: QueryIntent, tickers: list[str], ambiguities: list[str]) -> tuple[bool, str | None]:
        if ambiguities:
            if intent in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION} and "buy_or_sell_request" in ambiguities:
                return True, "Báº¡n muá»‘n tÃ¬m mÃ£ cÃ³ tÃ­n hiá»‡u BUY hay cÃ¡c mÃ£ cÃ³ tÃ­n hiá»‡u SELL? Hai hÆ°á»›ng nÃ y khÃ´ng thá»ƒ xáº¿p chung thÃ nh má»™t danh sÃ¡ch 'tá»‘t nháº¥t'."
            if intent == QueryIntent.STOCK_COMPARISON and len(tickers) == 1:
                return True, f"Bạn muốn so sánh {tickers[0]} với mã nào?"
            return True, "Bạn đang nhắc tới mã nào?"
        single_ticker_intents = {
            QueryIntent.STOCK_ANALYSIS, QueryIntent.TECHNICAL_ANALYSIS,
            QueryIntent.FUNDAMENTAL_ANALYSIS, QueryIntent.RISK_ANALYSIS,
            QueryIntent.STOCK_CHART, QueryIntent.INDICATOR_QUERY, QueryIntent.MIXED,
        }
        if intent in single_ticker_intents and not tickers:
            action = "xem biểu đồ" if intent == QueryIntent.STOCK_CHART else "phân tích"
            return True, f"Bạn muốn {action} của mã nào?"
        if intent == QueryIntent.STOCK_COMPARISON and len(tickers) < 2:
            base = tickers[0] if tickers else "mã hiện tại"
            return True, f"Bạn muốn so sánh {base} với mã nào?"
        return False, None

    @staticmethod
    def _interpretation(intent, tickers, period, indicators) -> str:
        target = ", ".join(tickers) if tickers else "chưa xác định mã"
        details = []
        if period:
            details.append(f"giai đoạn {period}")
        if indicators:
            details.append("chỉ báo " + ", ".join(indicators))
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Người dùng muốn {intent.value} cho {target}{suffix}."
