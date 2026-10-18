from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from stock_multi_agent.models.llm import AgentLLMReview, GroundedClaim, AgentFollowUpProposal
from stock_multi_agent.services.claim_grounding import audit_review, build_evidence_pack
from stock_multi_agent.services.financial_facts import ResearchFinancialFactProposal
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.providers.llm_provider import SemanticLLMProvider, configured_semantic_provider


class _StructuredReview(BaseModel):
    assessment: str = Field(min_length=1, max_length=1400)
    supporting_observations: list[str] = Field(default_factory=list, max_length=3)
    counterarguments: list[str] = Field(default_factory=list, max_length=2)
    missing_information: list[str] = Field(default_factory=list, max_length=2)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=10)
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=2)
    follow_up_proposals: list[AgentFollowUpProposal] = Field(default_factory=list, max_length=2)


class _CompactVietnameseReview(BaseModel):
    lines: list[str] = Field(min_length=8, max_length=8)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=10)


class _TechnicalIndicatorPlan(BaseModel):
    selected_indicators: list[
        Literal["BOLLINGER_BANDS", "MAX_DRAWDOWN", "SUPPORT_RESISTANCE", "ATR", "ADX"]
    ] = Field(min_length=1, max_length=2)
    rationale: str = Field(min_length=1, max_length=240)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=10)


class _FinancialFact(BaseModel):
    metric: Literal['revenue', 'net_profit', 'operating_cash_flow', 'eps', 'roe', 'debt_to_equity']
    value: float = Field(allow_inf_nan=False)
    unit: Literal['VND', 'VND_per_share', 'ratio', 'percent']
    period_end: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    source_id: str
    exact_quote: str = Field(min_length=10, max_length=400)


class _FinancialFacts(BaseModel):
    facts: list[_FinancialFact] = Field(default_factory=list, max_length=5)


class _StatementFacts(BaseModel):
    facts: list[ResearchFinancialFactProposal] = Field(default_factory=list, max_length=4)


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
CHƯA XÁC ĐỊNH và drawdown là mức sụt giảm. Khi nhắc số, giữ nguyên định dạng và
đơn vị từ bằng chứng; không tự đổi tỷ lệ hoặc làm tròn. Phản hồi đúng JSON schema."""

    _CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")
    _VIETNAMESE_PATTERN = re.compile(
        r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
        r"|\b(?:và|của|không|rủi ro|dữ liệu|giá|cổ phiếu|tín hiệu|dự báo|đánh giá)\b",
        re.IGNORECASE,
    )
    _ENGLISH_SENTENCE_PATTERN = re.compile(
        r"\b(?:forecast|score|indicators?|signal|relative return|versus|based on|sessions?|"
        r"suggests?|strong|month(?:ly)?|technical analysis)\b",
        re.IGNORECASE,
    )

    ROLE_PROMPTS = {
        AgentName.FUNDAMENTAL: '''Vai trò Fundamental: đánh giá tăng trưởng doanh thu/lợi nhuận,
khả năng sinh lời, định giá, đòn bẩy và chất lượng dòng tiền từ reported_metrics,
statements và deterministic_result. Nêu số liệu nào hỗ trợ nhận xét, kỳ dữ liệu
và hạn chế của phép so sánh. P/E thấp không tự chứng minh cổ phiếu rẻ; ROE cao có
thể đi cùng đòn bẩy. Không áp ngưỡng chung để kết luận ngân hàng như doanh nghiệp
phi tài chính. Thiếu lịch sử thì không tuyên bố tăng trưởng bền vững. Mỗi nhận định
phải dựa trên evidence pack; không chỉ lặp lại điểm số. Trích ít nhất một mã nguồn
được cấp nếu danh sách nguồn không rỗng.''',
        AgentName.RISK: '''Vai trò Risk/Critic: giải thích rủi ro đến từ biến động giá,
sụt giảm, thanh khoản, nợ, sự kiện, tập trung danh mục hay dữ liệu thiếu/cũ.
Phân biệt rủi ro đo được với độ bất định dữ liệu; không gọi dữ liệu thiếu là bằng
chứng doanh nghiệp xấu. Đối chiếu kết quả Technical và Fundamental, nêu một kịch
bản bất lợi có điều kiện và thông tin nào có thể bác bỏ kịch bản đó. Không suy
ra tin xấu chỉ từ giá giảm, không bịa sự kiện hoặc nguyên nhân kinh doanh. Không
đưa ra xác suất/khoảng lỗ mới. Trích ít nhất một mã nguồn được cấp nếu có.''',
    }

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
            "evidence_pack": build_evidence_pack(facts, allowed_source_ids),
        }
        system = self.SYSTEM_PROMPT + '\n' + self.ROLE_PROMPTS.get(agent, '') + '''
Chỉ viết tối đa hai claims ngắn. Mỗi nhận định có số liệu phải có một phần tử claims: claim_vi, source_id, exact_quote,
metric, period_end, unit và ticker nếu biết. Sao chép exact_quote từ đúng một mục
evidence_pack, chỉ trích đoạn cần thiết, ưu tiên dưới 180 ký tự; không mượn số của chỉ tiêu, công ty hoặc kỳ khác. Không đổi tỷ lệ thành
phần trăm, không tự tính số. Có thể đề xuất tối đa hai follow_up_proposals nếu thực sự
thiếu bằng chứng; chỉ chọn task_kind trong schema, không gọi công cụ tùy ý.
Các câu không có bằng chứng phải diễn đạt là giới hạn hoặc kịch bản có điều kiện.'''
        audit = {'status': 'NOT_AUDITED', 'semantic_truth_verified': False}
        try:
            try:
                raw = self.provider.parse_structured(system, payload, _StructuredReview)
            except Exception:
                raw = self._compact_retry(payload)
            raw = self._localize_terms(raw)
            if not self._is_vietnamese(raw) or not self._has_required_detail(raw):
                raw = self._localize_terms(self._compact_retry(payload))
            if not self._is_vietnamese(raw) or not self._has_required_detail(raw):
                raise ValueError("LLM response is not Vietnamese")
            if raw.claims:
                raw = raw.model_copy(update={'cited_source_ids': list(dict.fromkeys([
                    *raw.cited_source_ids, *[claim.source_id for claim in raw.claims]]))[:10]})
            if agent in self.ROLE_PROMPTS and allowed_source_ids and not set(raw.cited_source_ids).intersection(allowed_source_ids):
                raise ValueError('Specialist review has no valid evidence citation')
            audit = audit_review(raw, payload['evidence_pack'], facts.get('ticker'))
            if audit['status'] == 'REJECTED':
                raise ValueError('Claim anchors failed source-scoped audit')
        except Exception as exc:
            return AgentLLMReview(
                agent=agent,
                task=task,
                status="UNAVAILABLE",
                assessment=f"LLM specialist unavailable ({type(exc).__name__}); deterministic result preserved.",
                grounding_status='REJECTED' if audit['status'] == 'REJECTED' else 'NOT_AUDITED',
                grounding_audit=audit,
            )
        allowed = set(allowed_source_ids)
        proposals = []
        for proposal in raw.follow_up_proposals:
            invalid = set(proposal.source_ids).difference(allowed)
            if invalid or (not proposal.source_ids and proposal.task_kind not in {'FIND_RECENT_REPORT', 'FETCH_FINANCIAL_HISTORY'}):
                audit.setdefault('proposal_issues', []).append('Unknown or missing source reference in follow-up proposal')
                continue
            proposals.append(proposal)
        return AgentLLMReview(
            agent=agent,
            task=task,
            status="GROUNDED",
            assessment=raw.assessment,
            supporting_observations=raw.supporting_observations,
            counterarguments=raw.counterarguments,
            missing_information=raw.missing_information,
            cited_source_ids=[item for item in raw.cited_source_ids if item in allowed],
            claims=raw.claims,
            grounding_status=audit['status'],
            grounding_audit=audit,
            follow_up_proposals=proposals,
        )

    def plan_technical_indicators(
        self,
        facts: dict[str, Any],
        allowed_source_ids: list[str],
    ) -> _TechnicalIndicatorPlan | None:
        """Let the LLM choose tools; Python remains responsible for every calculation."""
        if not self.enabled or self.provider is None or AgentName.TECHNICAL.value not in self.roles:
            return None
        system = """Bạn là bộ lập kế hoạch công cụ cho Technical Agent chứng khoán Việt Nam.
Đọc dữ liệu tóm tắt từ Data Agent và dự báo định lượng, sau đó chọn 1 hoặc 2 phép kiểm tra
bổ sung phù hợp nhất trong BOLLINGER_BANDS, MAX_DRAWDOWN, SUPPORT_RESISTANCE, ATR, ADX. Không tự tính
giá trị chỉ báo, không đưa ra tín hiệu MUA/BÁN và không viết phân tích dài. Rationale chỉ gồm
một câu tiếng Việt ngắn. Chỉ dùng mã nguồn nằm trong allowed_source_ids và trả đúng JSON schema."""
        payload = {
            "task": "Chọn phép tính kỹ thuật bổ sung",
            "facts": facts,
            "allowed_source_ids": list(dict.fromkeys(allowed_source_ids)),
        }
        try:
            raw = self.provider.parse_structured(system, payload, _TechnicalIndicatorPlan)
        except Exception:
            return None
        selected = list(dict.fromkeys(raw.selected_indicators))[:2]
        if not selected:
            return None
        rationale = raw.rationale.strip()
        if self._CJK_PATTERN.search(rationale) or not self._VIETNAMESE_PATTERN.search(rationale):
            rationale = "LLM đã chọn các phép kiểm tra bổ sung dựa trên dữ liệu thị trường và dự báo hiện có."
        allowed = set(allowed_source_ids)
        return raw.model_copy(update={
            "selected_indicators": selected,
            "rationale": rationale,
            "cited_source_ids": [item for item in raw.cited_source_ids if item in allowed],
        })

    def map_statement_facts(self, citations, ticker: str) -> list[dict]:
        """Propose explicit statement rows; caller MUST validate before use.

        This method neither changes scored metrics nor asserts that a proposal
        is verified. Missing basis, currency, scale or issuer means no proposal.
        """
        if not self.enabled or self.provider is None or 'fundamental' not in self.roles:
            return []
        selected = [item for item in citations if ticker in item.tickers][:4]
        if not selected:
            return []
        allowed = {item.chunk_id for item in selected}
        try:
            result = self.provider.parse_structured(
                '''Trích xuất tối đa bốn đề xuất số liệu báo cáo cho đúng ticker được yêu cầu.
Chỉ lấy số liệu được viết rõ trong cùng một trích đoạn nguồn: ticker, statement,
metric, value, unit, currency, scale, period_label, period_end, reporting_basis,
consolidation, source_id, evidence_quote. Giá trị value giữ nguyên như nguồn;
scale là hệ số đơn vị được ghi rõ (đồng=1, triệu đồng=1000000, tỷ đồng=1000000000).
Không tự nhân, đổi tiền, suy ra kỳ, đoán hợp nhất/riêng lẻ hoặc quý riêng/lũy kế.
evidence_quote sao chép nguyên văn, đủ chứa chỉ tiêu, số, đơn vị, kỳ và cơ sở báo cáo.
Thiếu bất kỳ khai báo bắt buộc nào thì bỏ đề xuất đó. Không dùng kiến thức bên ngoài.
Đây chỉ là đề xuất chờ Python kiểm chứng, không phải số liệu đã được xác nhận.
Chỉ dùng source_id trong allowed_source_ids. Trả JSON schema, facts=[] nếu không đủ.''',
                {'ticker': ticker, 'allowed_source_ids': sorted(allowed),
                 'sources': [{'source_id': item.chunk_id, 'quote': item.quote[:2500],
                    'tickers': item.tickers, 'period_end': str(item.period_end) if item.period_end else None,
                    'publisher': item.publisher, 'source_url': item.source_url,
                    'published_at': item.published_at.isoformat()} for item in selected]},
                _StatementFacts)
            return [item.model_dump(mode='json') for item in result.facts
                    if item.ticker == ticker and item.source_id in allowed]
        except Exception:
            return []

    def map_financial_facts(self, citations):
        """Quote/period/unit checked candidates; never fill missing scored metrics."""
        if not self.enabled or self.provider is None or 'fundamental' not in self.roles or not citations:
            return []
        from stock_multi_agent.services.evidence_audit import _numbers
        allowed = {item.chunk_id: item for item in citations[:3] if item.period_end is not None}
        if not allowed:
            return []
        try:
            result = self.provider.parse_structured(
                'Ánh xạ tối đa năm số liệu tài chính được viết rõ trong bằng chứng. Giữ nguyên số, đơn vị, kỳ và exact_quote; không tính lại, không đoán trường thiếu. Trả JSON schema.',
                {'sources': [{'source_id': key, 'quote': item.quote, 'period_end': str(item.period_end)} for key, item in allowed.items()],
                 'allowed_source_ids': list(allowed)}, _FinancialFacts)
            accepted = []
            for fact in result.facts:
                citation = allowed.get(fact.source_id)
                if not citation or fact.exact_quote not in citation.quote or fact.period_end != str(citation.period_end):
                    continue
                number = str(int(fact.value)) if fact.value.is_integer() else str(fact.value)
                if not _numbers(number).issubset(_numbers(fact.exact_quote.replace('%', ''))):
                    continue
                if fact.unit.startswith('VND') and re.search(r'\b(?:nghìn|ngàn|triệu|tỷ|million|billion)\b', fact.exact_quote, re.I):
                    continue  # Scale is not represented by the proposal schema; do not guess.
                anchors = {'revenue': r'(?:doanh thu|revenue)', 'net_profit': r'(?:lợi nhuận sau thuế|net profit)',
                    'operating_cash_flow': r'(?:dòng tiền.*kinh doanh|operating cash)', 'eps': r'\bEPS\b',
                    'roe': r'(?:\bROE\b|lợi nhuận.*vốn chủ)', 'debt_to_equity': r'(?:nợ.*vốn chủ|debt.*equity)'}
                if not re.search(anchors[fact.metric], fact.exact_quote, re.I):
                    continue
                units = {'VND': r'\b(?:VND|đồng)\b', 'VND_per_share': r'(?:đồng|VND)\s*/\s*(?:cổ phiếu|cp|share)',
                         'ratio': r'\b(?:lần|ratio)\b', 'percent': r'%'}
                if not re.search(units[fact.unit], fact.exact_quote, re.I):
                    continue
                accepted.append({**fact.model_dump(), 'status': 'PROPOSAL_NOT_SCORED'})
            return accepted
        except Exception:
            return []

    def _compact_retry(self, payload: dict[str, Any]) -> _StructuredReview:
        role = AgentName(payload['specialist_role'])
        prompt = self.SYSTEM_PROMPT + '\n' + self.ROLE_PROMPTS.get(role, '') + """
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
            *[claim.claim_vi for claim in review.claims],
            *[proposal.reason for proposal in review.follow_up_proposals],
        ])
        english_markers = cls._ENGLISH_SENTENCE_PATTERN.findall(content)
        return (
            not cls._CJK_PATTERN.search(content)
            and bool(cls._VIETNAMESE_PATTERN.search(content))
            and len(english_markers) < 2
        )

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

