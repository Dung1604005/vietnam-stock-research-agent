from __future__ import annotations

import re

from stock_multi_agent.nlu.models import EntityExtraction


class EntityExtractor:
    _not_tickers = {
        "AI", "API", "BUY", "SELL", "HOLD", "WATCH", "AVOID", "RSI", "MACD",
        "MA", "SMA", "EMA", "PE", "PB", "ROE", "ROA", "EPS", "VN", "USD",
        "USER", "CHART", "STOCK", "VOLUME", "RISK", "HIGH", "LOW", "MEDIUM",
        "TTCK", "CEO", "CFO", "EBITDA", "YOY", "QOQ", "HOSE", "HSX", "HNX", "UPCOM",
        "ATO", "ATC", "LO", "MOK", "MAK", "MTL", "DCF", "CAPM", "ATR", "ADX", "EV", "FCF",
    }
    _known_tickers = {
        "FPT", "CMG", "CTR", "ELC", "VNM", "HPG", "HSG", "VCB", "TCB", "MBB",
        "ACB", "BID", "CTG", "VPB", "STB", "SSI", "VND", "HCM", "MWG", "PNJ",
        "MSN", "VIC", "VHM", "NVL", "KDH", "DXG", "GAS", "PLX", "POW", "REE",
        "SAB", "BVH", "GVR", "DGC", "DCM", "DPM", "VRE", "VJC", "HVN", "VNINDEX",
    }
    _aliases = {
        "fpt corporation": "FPT", "cong ty fpt": "FPT", "hoa phat": "HPG",
        "vietcombank": "VCB", "vinamilk": "VNM", "the gioi di dong": "MWG",
        "vn index": "VNINDEX", "vnindex": "VNINDEX",
    }

    def extract(self, raw: str, normalized: str) -> EntityExtraction:
        tickers = self._tickers(raw, normalized)
        duration, direction = self._period(normalized)
        indicators = self._indicators(normalized)
        return EntityExtraction(
            tickers=tickers,
            sector=self._sector(normalized),
            duration=duration,
            temporal_direction=direction,
            indicators=indicators,
            requested_information=self._requested_information(normalized, indicators),
            requested_result_count=self._result_count(normalized),
            requested_actions=self._requested_actions(normalized),
        )

    @staticmethod
    def _result_count(text: str) -> int | None:
        """Extract an explicit result cardinality, not a chart/time duration."""
        patterns = (
            r"\btop\s+(\d{1,3})\b",
            r"\b(\d{1,3})\s*(?:ma|co phieu|ticker|stocks?)\b",
            r"\b(?:tim|loc|goi y|cho toi|lay)\s+(\d{1,3})\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                return value if 1 <= value <= 100 else None
        if re.search(r"\b(?:mot|1)\s*(?:ma|co phieu)\b", text):
            return 1
        if re.search(r"\b(?:tot nhat|best|cao nhat)\b", text) and re.search(r"\b(?:ma|co phieu)\b", text):
            return 1
        return None

    @staticmethod
    def _requested_actions(text: str) -> list[str]:
        actions: list[str] = []
        if re.search(r"\b(?:buy|mua)\b", text):
            actions.append("BUY")
        if re.search(r"\b(?:sell|ban)\b", text):
            actions.append("SELL")
        return actions

    def _tickers(self, raw: str, text: str) -> list[str]:
        prepared = re.sub(r"VN[-\s]?INDEX", "VNINDEX", raw, flags=re.IGNORECASE)
        candidates = re.findall(r"(?<![A-Za-z0-9])[A-Z]{2,10}(?![A-Za-z0-9])", prepared)
        result = [value for value in candidates if value not in self._not_tickers]
        for alias, ticker in self._aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", text):
                result.append(ticker)
        for word in text.split():
            upper = word.upper()
            if upper in self._known_tickers:
                result.append(upper)
        # Unknown tickers are accepted when users preserve the conventional
        # uppercase form. We intentionally do not fuzzy-correct ticker symbols.
        return list(dict.fromkeys(result))

    @staticmethod
    def _period(text: str) -> tuple[str | None, str | None]:
        aliases = (
            (r"\b(?:nua nam|half year|sau thang)\b", "6M"),
            (r"\b(?:mot quy|quarter|ba thang)\b", "3M"),
            (r"\b(?:mot nam|one year)\b", "1Y"),
            (r"\b(?:hai nam|two years)\b", "2Y"),
        )
        duration = next((value for pattern, value in aliases if re.search(pattern, text)), None)
        if duration is None:
            month = re.search(r"\b(1|3|6|9|12|18|24)\s*(?:thang|months?|m)\b", text)
            year = re.search(r"\b(1|2|3|5)\s*(?:nam|years?|y)\b", text)
            if month:
                duration = f"{month.group(1)}M"
            elif year:
                duration = f"{year.group(1)}Y"
        future = re.search(r"\b(?:toi|sap toi|tuong lai|next|sap|phia truoc)\b", text)
        past = re.search(r"\b(?:qua|vua qua|gan day|nay|lich su|historical|past)\b", text)
        direction = "future" if future else "historical" if past else None
        if duration is None and re.search(r"\b(?:dai han|long term)\b", text):
            return "1Y", "future"
        if duration is None and re.search(r"\b(?:ngan han|short term)\b", text):
            return "1M", "future"
        return duration, direction

    @staticmethod
    def _indicators(text: str) -> list[str]:
        indicators: list[str] = []
        for name in ("RSI", "MACD", "ATR", "ADX", "PE", "PB", "ROE", "ROA", "EPS"):
            if re.search(rf"\b{name.lower()}\b", text):
                indicators.append(name)
        for period in re.findall(r"\b(?:s?ma)\s*(20|50|100|200)\b", text):
            indicators.append(f"MA{period}")
        if re.search(r"\b(?:volume|khoi luong|thanh khoan)\b", text):
            indicators.append("VOLUME")
        if re.search(r"\b(?:nen|candlestick)\b", text):
            indicators.append("CANDLESTICK")
        return list(dict.fromkeys(indicators))

    @staticmethod
    def _sector(text: str) -> str | None:
        sectors = {
            "market_current": ("toan thi truong", "toan san", "tat ca san", "all market"),
            "hose_current": ("san hose", "hose", "hsx"),
            "hnx_current": ("san hnx", "hnx"),
            "upcom_current": ("san upcom", "upcom"),
            "technology": ("cong nghe", "technology", "tech"),
            "banking": ("ngan hang", "banking", "bank"),
            "steel": ("thep", "steel"),
            "real_estate": ("bat dong san", "real estate"),
            "retail": ("ban le", "retail"),
        }
        return next((sector for sector, terms in sectors.items() if any(term in text for term in terms)), None)

    @staticmethod
    def _requested_information(text: str, indicators: list[str]) -> list[str]:
        result: list[str] = []
        if any(term in text for term in ("dinh gia", "dat khong", "valuation")) or any(item in indicators for item in ("PE", "PB")):
            result.append("valuation")
        if any(term in text for term in ("loi nhuan", "profit", "lai rong")):
            result.append("profit")
        if any(term in text for term in ("doanh thu", "revenue")):
            result.append("revenue")
        if any(term in text for term in ("rui ro", "downside", "worst case", "diem xau")):
            result.append("risk")
        return result
