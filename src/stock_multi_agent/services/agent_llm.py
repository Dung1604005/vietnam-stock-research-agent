from __future__ import annotations

import os
import re
from typing import Any

from pydantic import BaseModel, Field

from stock_multi_agent.models.llm import AgentLLMReview
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.providers.llm_provider import SemanticLLMProvider, configured_semantic_provider


class _StructuredReview(BaseModel):
    assessment: str = Field(min_length=1, max_length=1400)
    supporting_observations: list[str] = Field(default_factory=list, max_length=3)
    counterarguments: list[str] = Field(default_factory=list, max_length=2)
    missing_information: list[str] = Field(default_factory=list, max_length=2)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=10)


class _CompactVietnameseReview(BaseModel):
    lines: list[str] = Field(min_length=8, max_length=8)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=10)


class AgentLLMAdvisor:
    """Shared structured-reasoning capability for specialist agents.

    The advisor may interpret, challenge, and identify missing information. It
    never mutates measured values, scores, signals, vetoes, or recommendations.
    """

    SYSTEM_PROMPT = """Bạn là chuyên gia trong hệ Multi-Agent nghiên cứu cổ phiếu Việt Nam.
Chỉ sử dụng dữ kiện có cấu trúc được cung cấp. Không bịa giá, ngày, tỷ số, tin tức,
thông tin doanh nghiệp, độ chính xác mô hình hoặc nguồn. Phân biệt rõ quan sát với
điểm chưa chắc chắn và phản biện các giả định yếu. Chỉ trích dẫn mã nguồn nằm trong
allowed_source_ids. Không tính lại hoặc thay đổi điểm số, tín hiệu, quyền phủ quyết
rủi ro, dự báo hay kết luận cuối cùng. Viết assessment bằng 2–3 câu tiếng Việt tự
nhiên. Sau đó cung cấp đúng 3 luận cứ chính, đúng 2 phản biện/rủi ro và 1–2 thông
tin còn thiếu. Mỗi ý phải là một câu cụ thể, đề cập bằng chứng, mâu thuẫn, độ bất
định dự báo hoặc tác động rủi ro. Tuyệt đối không dùng tiếng Trung, tiếng Nhật hoặc
tiếng Anh trong nội dung nhận định. Khi nhắc tới nhãn hệ thống, hãy diễn đạt BUY là
MUA, SELL là BÁN, HOLD là NẮM GIỮ, WATCH là THEO DÕI, AVOID là TRÁNH, UNKNOWN là
CHƯA XÁC ĐỊNH và drawdown là mức sụt giảm. Phản hồi đúng JSON schema."""

    _CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")
    _VIETNAMESE_PATTERN = re.compile(
        r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
        r"|\b(?:và|của|không|rủi ro|dữ liệu|giá|cổ phiếu|tín hiệu|dự báo|đánh giá)\b",
        re.IGNORECASE,
    )

    def __init__(self, provider: SemanticLLMProvider | None = None, enabled: bool | None = None) -> None:
        self.provider = provider if provider is not None else configured_semantic_provider()
        configured = os.getenv("STOCK_AGENT_LLM_SPECIALISTS_ENABLED", "false").strip().lower()
        self.enabled = enabled if enabled is not None else configured in {"1", "true", "yes", "on"}
        roles = os.getenv(
            "STOCK_AGENT_LLM_SPECIALIST_ROLES",
            "data,technical,fundamental,risk,manager",
        )
        self.roles = {item.strip().lower() for item in roles.split(",") if item.strip()}

    def review(
        self,
        agent: AgentName,
        task: str,
        facts: dict[str, Any],
        allowed_source_ids: list[str],
    ) -> AgentLLMReview | None:
        if not self.enabled or self.provider is None or agent.value not in self.roles:
            return None
        payload = {
            "specialist_role": agent.value,
            "task": task,
            "facts": facts,
            "allowed_source_ids": list(dict.fromkeys(allowed_source_ids)),
        }
        try:
            try:
                raw = self.provider.parse_structured(self.SYSTEM_PROMPT, payload, _StructuredReview)
            except Exception:
                raw = self._compact_retry(payload)
            raw = self._localize_terms(raw)
            if not self._is_vietnamese(raw) or not self._has_required_detail(raw):
                raw = self._localize_terms(self._compact_retry(payload))
            if not self._is_vietnamese(raw) or not self._has_required_detail(raw):
                raise ValueError("LLM response is not Vietnamese")
        except Exception as exc:
            return AgentLLMReview(
                agent=agent,
                task=task,
                status="UNAVAILABLE",
                assessment=f"LLM specialist unavailable ({type(exc).__name__}); deterministic result preserved.",
            )
        allowed = set(allowed_source_ids)
        return AgentLLMReview(
            agent=agent,
            task=task,
            status="GROUNDED",
            assessment=raw.assessment,
            supporting_observations=raw.supporting_observations,
            counterarguments=raw.counterarguments,
            missing_information=raw.missing_information,
            cited_source_ids=[item for item in raw.cited_source_ids if item in allowed],
        )

    def _compact_retry(self, payload: dict[str, Any]) -> _StructuredReview:
        prompt = self.SYSTEM_PROMPT + """
Schema đầy đủ trước đó không hợp lệ. Hãy trả đúng 8 dòng tiếng Việt trong trường lines:
2 dòng kết luận tổng quát, 3 dòng luận cứ chính, 2 dòng phản biện/rủi ro và 1 dòng
thông tin còn thiếu. Mỗi phần tử là một câu độc lập, không đánh số và không dùng
tiếng Trung, tiếng Nhật hoặc thuật ngữ tiếng Anh nếu có cách diễn đạt tiếng Việt."""
        compact = self.provider.parse_structured(prompt, payload, _CompactVietnameseReview)  # type: ignore[union-attr]
        return _StructuredReview(
            assessment=" ".join(compact.lines[:2]),
            supporting_observations=compact.lines[2:5],
            counterarguments=compact.lines[5:7],
            missing_information=compact.lines[7:8],
            cited_source_ids=compact.cited_source_ids,
        )

    @classmethod
    def _is_vietnamese(cls, review: _StructuredReview) -> bool:
        content = " ".join([
            review.assessment,
            *review.supporting_observations,
            *review.counterarguments,
            *review.missing_information,
        ])
        return not cls._CJK_PATTERN.search(content) and bool(cls._VIETNAMESE_PATTERN.search(content))

    @staticmethod
    def _has_required_detail(review: _StructuredReview) -> bool:
        return (
            len(review.supporting_observations) == 3
            and len(review.counterarguments) == 2
            and len(review.missing_information) >= 1
        )

    @staticmethod
    def _localize_terms(review: _StructuredReview) -> _StructuredReview:
        replacements = {
            r"\bAVOID\b": "TRÁNH",
            r"\bWATCH\b": "THEO DÕI",
            r"\bHOLD\b": "NẮM GIỮ",
            r"\bBUY\b": "MUA",
            r"\bSELL\b": "BÁN",
            r"\bUNKNOWN\b": "CHƯA XÁC ĐỊNH",
            r"\bdrawdown\b": "mức sụt giảm",
            r"\btrend\b": "xu hướng",
        }

        def localize(text: str) -> str:
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            return text

        return review.model_copy(update={
            "assessment": localize(review.assessment),
            "supporting_observations": [localize(item) for item in review.supporting_observations],
            "counterarguments": [localize(item) for item in review.counterarguments],
            "missing_information": [localize(item) for item in review.missing_information],
        })

