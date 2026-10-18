from __future__ import annotations

import re

from stock_multi_agent.models.chat import ConversationState
from stock_multi_agent.nlu.models import ContextResolution


class ContextResolver:
    _generic_reference = re.compile(
        r"\b(?:no|ma nay|con nay|thang nay|ma vua roi|con vua roi|cai dau|cai nay|it|this one|previous one)\b"
    )
    _ambiguous_other = re.compile(r"\b(?:con kia|ma kia|cai kia|the other one)\b")
    _ordinal_words = {"dau": 0, "dau tien": 0, "mot": 0, "hai": 1, "2": 1, "ba": 2, "3": 2}

    def resolve(
        self,
        text: str,
        explicit_tickers: list[str],
        context: ConversationState | None,
    ) -> ContextResolution:
        tickers = list(explicit_tickers)
        if context is None:
            ambiguities = ["Tham chiếu 'mã kia' không có ngữ cảnh."] if self._ambiguous_other.search(text) else []
            return ContextResolution(tickers=tickers, ambiguities=ambiguities)

        resolved: dict[str, object] = {}
        ambiguities: list[str] = []
        referenced = False
        stock_list = context.last_stock_list
        active = context.active_tickers

        ordinal = re.search(r"\b(?:con|ma|cai)\s+(?:thu\s+)?(dau tien|dau|mot|hai|2|ba|3)\b", text)
        if ordinal:
            referenced = True
            index = self._ordinal_words[ordinal.group(1)]
            if index < len(stock_list):
                selected = stock_list[index]
                resolved[ordinal.group(0)] = selected
                if re.search(r"\b(?:so|so sanh|so voi|hon|compare)\b", text) and active:
                    tickers = [active[0], selected]
                else:
                    tickers = [selected]
            else:
                ambiguities.append(f"Danh sách gần nhất không có mã ở vị trí {index + 1}.")

        both_reference = re.search(r"\b(?:ca hai|hai ma tren|2 ma tren|both|hai con nay)\b", text)
        if both_reference:
            referenced = True
            candidates = active if len(active) >= 2 else stock_list[:2]
            if len(candidates) >= 2:
                tickers = list(candidates[:2])
                resolved[both_reference.group(0)] = tickers
            else:
                ambiguities.append("Chưa xác định được hai mã đang được nhắc tới.")

        if self._ambiguous_other.search(text):
            referenced = True
            ambiguities.append("'Mã kia' có thể chỉ nhiều mã khác nhau.")
            if not re.search(r"\b(?:so|so sanh|so voi|compare)\b", text):
                tickers = []

        generic = self._generic_reference.search(text)
        if generic:
            referenced = True
            if len(active) == 1:
                if not tickers:
                    tickers = [active[0]]
                resolved[generic.group(0)] = active[0]
            elif len(active) > 1:
                ambiguities.append("Tham chiếu số ít nhưng đang có nhiều mã hoạt động.")
            else:
                ambiguities.append("Không có mã trước đó để giải tham chiếu.")

        comparison_language = re.search(r"\b(?:so|so sanh|so voi|compare|versus|hon)\b", text)
        if comparison_language and len(tickers) == 1 and active:
            previous = next((item for item in active if item != tickers[0]), active[0])
            tickers = [previous, tickers[0]]
            referenced = True
            resolved["comparison_base"] = previous
        elif comparison_language and not tickers and len(active) >= 2:
            tickers = list(active[:2])
            referenced = True
            resolved["two active stocks"] = tickers

        implicit_action = re.search(
            r"\b(?:chart dau|ve thu|cho chart|cho rsi|cho macd|rsi thi sao|macd thi sao|nua nam thoi|dai han thi sao|doi thanh|doi sang|ve bieu do|bieu do doanh thu|bieu do dong tien|bieu do tang truong|them chart|them bieu do)\b",
            text,
        )
        if implicit_action and not tickers and active:
            tickers = list(active)
            referenced = True
            resolved["implicit active ticker"] = tickers

        if re.search(r"\b(?:con|the|van|tiep|vua roi|tren|khac)\b", text) and (active or stock_list):
            referenced = True
        return ContextResolution(
            tickers=list(dict.fromkeys(tickers)),
            references_previous_context=referenced,
            resolved_references=resolved,
            ambiguities=list(dict.fromkeys(ambiguities)),
        )
