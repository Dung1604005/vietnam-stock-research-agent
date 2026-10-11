from __future__ import annotations

import re
import unicodedata


class VietnameseTextNormalizer:
    """Conservative Vietnamese normalization that never fuzzy-corrects tickers."""

    _abbreviations = {
        "ko": "khong",
        "k": "khong",
        "kh": "khong",
        "ntn": "nhu the nao",
        "sao bro": "sao",
        "ttck": "thi truong chung khoan",
        "bđs": "bat dong san",
        "bds": "bat dong san",
        "half year": "6 thang",
        "nua nam": "6 thang",
        "fundametal": "fundamental",
        "fudamental": "fundamental",
        "technial": "technical",
        "bieu do": "bieu do",
    }
    def normalize(self, value: str) -> str:
        lowered = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
        ascii_text = "".join(char for char in lowered if unicodedata.category(char) != "Mn")
        ascii_text = re.sub(r"(?<=\d)\s*(m|y)\b", r"\1", ascii_text)
        ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
        text = re.sub(r"\s+", " ", ascii_text).strip()
        # Slash-separated valuation abbreviations become separate tokens after
        # punctuation cleanup; join only these known deterministic indicators.
        text = re.sub(r"\bp\s+e\b", "pe", text)
        text = re.sub(r"\bp\s+b\b", "pb", text)
        for source, target in sorted(self._abbreviations.items(), key=lambda item: len(item[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)
        return text
