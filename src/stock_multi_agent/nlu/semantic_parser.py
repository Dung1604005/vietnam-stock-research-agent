from __future__ import annotations

import re
import json
import logging
import sqlite3
import time
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

    def parse(self, raw_query: str, context: ConversationState | None = None) -> ParsedQuery:
        started_at = time.perf_counter()
        raw = raw_query.strip()
        normalized = self.normalizer.normalize(raw)
        extracted = self.entities.extract(raw, normalized)
        resolved = self.context.resolve(normalized, extracted.tickers, context)

        semantic: SemanticQuery | None = None
        llm_attempted = self.llm_provider is not None
        llm_latency_ms: int | None = None
        fallback_reason: str | None = "no_configured_llm" if self.llm_provider is None else None
        if self.llm_provider is not None:
            llm_started_at = time.perf_counter()
            try:
                semantic = self._parse_with_llm(raw, normalized, extracted.model_dump(), resolved.model_dump(), context)
                semantic = self._merge_deterministic_entities(semantic, extracted.model_dump(), resolved.model_dump())
            except Exception as exc:
                # Network/model failures must not break local mock mode behavior.
                semantic = None
                fallback_reason = f"llm_error:{type(exc).__name__}"
            finally:
                llm_latency_ms = round((time.perf_counter() - llm_started_at) * 1000)
        if semantic is None:
            semantic = self._fallback(raw, normalized, extracted, resolved, context)
        semantic = self._enforce_recommendation_contract(semantic, normalized)
        semantic = self._enforce_discovery_contract(semantic)
        semantic = self._enforce_knowledge_contract(semantic, normalized)
        semantic = self._enforce_required_entity_contract(semantic)
        parsed = semantic.to_parsed_query().model_copy(update={
            "parser_latency_ms": round((time.perf_counter() - started_at) * 1000),
            "llm_attempted": llm_attempted,
            "llm_latency_ms": llm_latency_ms,
            "fallback_reason": fallback_reason,
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "example_bank_version": EXAMPLE_BANK_VERSION,
        })
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
