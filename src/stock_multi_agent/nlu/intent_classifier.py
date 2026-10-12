from __future__ import annotations

import math
import re
from collections import Counter

from stock_multi_agent.models.chat import ConversationState, QueryIntent
from stock_multi_agent.nlu.intent_examples import INTENT_EXAMPLES
from stock_multi_agent.nlu.models import EntityExtraction
from stock_multi_agent.nlu.normalizer import VietnameseTextNormalizer


def _ngrams(value: str, size: int = 3) -> Counter[str]:
    padded = f"  {value} "
    return Counter(padded[index:index + size] for index in range(max(0, len(padded) - size + 1)))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(count * right.get(key, 0) for key, count in left.items())
    if not numerator:
        return 0.0
    lnorm = math.sqrt(sum(value * value for value in left.values()))
    rnorm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (lnorm * rnorm) if lnorm and rnorm else 0.0


class SemanticSimilarityClassifier:
    """Local semantic fallback based on example similarity plus bounded domain evidence."""

    def __init__(self, normalizer: VietnameseTextNormalizer | None = None) -> None:
        self.normalizer = normalizer or VietnameseTextNormalizer()
        self._vectors = {
            intent: [_ngrams(self.normalizer.normalize(example)) for example in examples]
            for intent, examples in INTENT_EXAMPLES.items()
        }

    def classify(
        self,
        text: str,
        entities: EntityExtraction,
        context: ConversationState | None,
    ) -> tuple[QueryIntent, float, dict[str, float]]:
        query_vector = _ngrams(text)
        scores = {
            intent: max((_cosine(query_vector, example) for example in vectors), default=0.0)
            for intent, vectors in self._vectors.items()
        }
        self._add_domain_evidence(scores, text, entities, context)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        intent, top = ordered[0]
        second = ordered[1][1]
        confidence = min(0.98, max(0.25, 0.43 + 0.45 * top + 0.65 * max(0.0, top - second)))
        return intent, confidence, {key.value: round(value, 4) for key, value in ordered}

    @staticmethod
    def _add(scores: dict[QueryIntent, float], intent: QueryIntent, amount: float) -> None:
        scores[intent] = scores.get(intent, 0.0) + amount

    def _add_domain_evidence(
        self,
        scores: dict[QueryIntent, float],
        text: str,
        entities: EntityExtraction,
        context: ConversationState | None,
    ) -> None:
        # These are supporting signals. The primary baseline score always comes from
        # similarity to diverse utterances rather than exact sentence matching.
        patterns: tuple[tuple[QueryIntent, str, float], ...] = (
            (QueryIntent.HELP, r"\b(?:help|huong dan|lam duoc gi|co the hoi gi|tro giup)\b", 0.42),
            (QueryIntent.EXPLANATION, r"\b(?:nghia la gi|giai thich|tai sao|vi sao|dung de lam gi)\b", 0.48),
            (QueryIntent.THEORY, r"\b(?:la gi|khai niem|cong thuc|duoc tinh|tinh the nao|phan biet|y nghia gi|giai thich)\b", 0.58),
            (QueryIntent.CURRENT_RULE, r"\b(?:quy dinh|hien hanh|bien do|gia tran|gia san|gia tham chieu|ato|atc|chu ky thanh toan|t\+?2)\b", 0.62),
            (QueryIntent.STOCK_COMPARISON, r"\b(?:so sanh|so voi|compare|versus|hon|hay)\b", 0.34),
            (QueryIntent.STOCK_CHART, r"\b(?:bieu do|chart|ve|duong gia|lich su gia|show)\b", 0.32),
            (QueryIntent.TECHNICAL_ANALYSIS, r"\b(?:ky thuat|technical|uptrend|downtrend|momentum|dong luong|price action|xu huong|trend)\b", 0.35),
            (QueryIntent.FUNDAMENTAL_ANALYSIS, r"\b(?:co ban|fundamental|dinh gia|loi nhuan|doanh thu|tai chinh|lam an|doanh nghiep)\b", 0.38),
            (QueryIntent.RISK_ANALYSIS, r"\b(?:rui ro|nguy hiem|diem xau|case xau|truong hop te|worst case|downside|mat von|can than|can trong|so cai gi|co van de gi|cang)\b", 0.38),
            (QueryIntent.MARKET_OVERVIEW, r"\b(?:thi truong|chung khoan viet nam|vnindex|market|do hay xanh|chi so chung)\b", 0.43),
            (QueryIntent.STOCK_RECOMMENDATION, r"\b(?:nen mua|dang mua|de (?:mua|buy)|de dau tu|dau tu ma nao|top co phieu de mua|lai cao|an toan)\b", 0.58),
            (QueryIntent.STOCK_DISCOVERY, r"\b(?:goi y|tim|scan|ma nao|con nao|co phieu nao|vai ma|may ma|ma khac)\b", 0.36),
            (QueryIntent.RESEARCH, r"\b(?:bao cao|filing|cong bo thong tin|nghi quyet|annual report|tai lieu|ban lanh dao|investor relations)\b", 0.46),
            (QueryIntent.STOCK_ANALYSIS, r"\b(?:phan tich|danh gia|nhan xet|goc nhin|on khong|ngon khong|dang de y|xem giup)\b", 0.28),
            (QueryIntent.GENERAL_CHAT, r"\b(?:xin chao|hello|cam on|chao buoi|ban khoe)\b", 0.48),
        )
        for intent, pattern, weight in patterns:
            if re.search(pattern, text):
                self._add(scores, intent, weight)

        if len(entities.tickers) >= 2:
            self._add(scores, QueryIntent.STOCK_COMPARISON, 0.28)
        elif entities.tickers:
            self._add(scores, QueryIntent.STOCK_ANALYSIS, 0.11)
        if "VNINDEX" in entities.tickers:
            self._add(scores, QueryIntent.MARKET_OVERVIEW, 0.55)
        if re.search(r"\b(?:neu )?thi truong xau\b", text):
            self._add(scores, QueryIntent.RISK_ANALYSIS, 0.55)
        if entities.indicators:
            if any(item in entities.indicators for item in ("PE", "PB", "ROE", "ROA", "EPS")):
                self._add(scores, QueryIntent.FUNDAMENTAL_ANALYSIS, 0.18)
            self._add(scores, QueryIntent.INDICATOR_QUERY, 0.32)
        if entities.sector and not entities.tickers:
            self._add(scores, QueryIntent.STOCK_DISCOVERY, 0.18)
        if entities.duration and entities.temporal_direction == "historical":
            self._add(scores, QueryIntent.STOCK_CHART, 0.20)
        if entities.duration and entities.temporal_direction == "future":
            self._add(scores, QueryIntent.STOCK_ANALYSIS, 0.18)
        if context and re.fullmatch(r"(?:6 thang|3 thang|1 nam|nua nam)(?: thoi)?", text):
            if context.last_intent:
                self._add(scores, context.last_intent, 0.70)

        # Explicit indicator names are safe deterministic entities. Decide whether
        # the user asks for a value, a chart, a broad domain analysis, or a definition.
        if entities.indicators:
            if re.search(r"\b(?:nghia la gi|giai thich|dung de lam gi)\b", text):
                self._add(scores, QueryIntent.THEORY, 0.85)
            elif re.search(r"\b(?:bieu do|chart|ve|show)\b", text):
                self._add(scores, QueryIntent.STOCK_CHART, 0.45)
            elif re.search(r"\b(?:ky thuat|technical|xu huong|trend)\b", text):
                self._add(scores, QueryIntent.TECHNICAL_ANALYSIS, 0.36)
            elif re.search(r"\b(?:the nao|ra sao|bao nhieu|o dau|cho xem|thi sao|hien tai)\b", text):
                self._add(scores, QueryIntent.INDICATOR_QUERY, 0.42)

        theory_terms = re.search(
            r"\b(?:pe|pb|roe|roa|eps|rsi|macd|atr|dcf|capm|sharpe|beta|alpha|"
            r"backtest|look ahead|survivorship|data leakage|co tuc|bao cao tai chinh)\b",
            text,
        )
        asks_concept = re.search(r"\b(?:la gi|cong thuc|duoc tinh|tinh the nao|giai thich|phan biet|tai sao|vi sao)\b", text)
        if theory_terms and asks_concept:
            self._add(scores, QueryIntent.THEORY, 0.85)
        if entities.tickers and theory_terms and re.search(r"\b(?:ap dung|danh gia|tot khong|dat khong|y nghia gi voi)\b", text):
            self._add(scores, QueryIntent.MIXED, 1.05)

    def secondary_intents(
        self,
        text: str,
        primary: QueryIntent,
        entities: EntityExtraction,
    ) -> list[QueryIntent]:
        result: list[QueryIntent] = []
        joined = bool(re.search(r"\b(?:va|roi|tien|kem|luon)\b", text))
        if primary != QueryIntent.STOCK_CHART and joined and re.search(r"\b(?:bieu do|chart|ve|duong gia)\b", text):
            result.append(QueryIntent.STOCK_CHART)
        if primary != QueryIntent.INDICATOR_QUERY and entities.indicators and joined:
            result.append(QueryIntent.INDICATOR_QUERY)
        if primary != QueryIntent.RISK_ANALYSIS and joined and re.search(r"\b(?:rui ro|diem xau|can than)\b", text):
            result.append(QueryIntent.RISK_ANALYSIS)
        if primary != QueryIntent.FUNDAMENTAL_ANALYSIS and joined and re.search(r"\b(?:co ban|dinh gia|fundamental)\b", text):
            result.append(QueryIntent.FUNDAMENTAL_ANALYSIS)
        return list(dict.fromkeys(result))
