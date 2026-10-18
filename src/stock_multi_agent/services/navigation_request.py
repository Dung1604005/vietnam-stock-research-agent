"""Pure fast path for explicit navigation requests; no browsing or model calls."""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from urllib.parse import urlsplit

from stock_multi_agent.models.chat import AssistantResponse, ConversationState
from stock_multi_agent.nlu.entity_extractor import EntityExtractor
from stock_multi_agent.services.interaction_actions import build_interaction_actions, validate_ticker


def _normalize(value: str) -> str:
    value = value.replace('đ', 'd').replace('Đ', 'D')
    return ' '.join(''.join(char for char in unicodedata.normalize('NFD', value)
                            if unicodedata.category(char) != 'Mn').lower().split())


_WORDS = set('mo open hay giup toi vui long ban muon cho trang web website cua co phieu ma gia bang '
             'xem lich su bieu do bao cao tai lieu pdf url vndirect dstock dboard de mua ban vao '
             'cong giao dich tren va voi cac nhung nay do no nhe ngay duoc khong xin'.split())
_REPORT_TYPES = {'financial_statement', 'annual_report', 'regulatory_disclosure', 'earnings_release', 'investor_presentation'}


def _tickers(text: str, normalized: str) -> list[str]:
    # Command/provider words must not become fictional upper-case ticker symbols.
    tokens = re.findall(r'[^\W_]+', text, flags=re.UNICODE)
    cleaned = ' '.join(word for word in tokens if _normalize(word) not in _WORDS)
    extracted = EntityExtractor().extract(cleaned, _normalize(cleaned)).tickers
    result = []
    for value in extracted:
        try:
            value = validate_ticker(value)
        except ValueError:
            continue
        if value not in result:
            result.append(value)
    return result


def _clarify() -> AssistantResponse:
    return AssistantResponse(message='Bạn muốn mở mã cổ phiếu nào? Hãy nêu một mã cụ thể, ví dụ “Mở bảng giá FPT”.',
                             metadata={'navigation_clarification': True})


def _association(record: dict, context: AssistantResponse, targets: list[str]) -> bool:
    explicit = record.get('tickers') or record.get('ticker') or record.get('symbol')
    if explicit:
        values = explicit if isinstance(explicit, list) else [explicit]
        return bool(set(value for value in values if isinstance(value, str)) & set(targets))
    return len(context.tickers) == 1 and context.tickers[0] in targets


def _pdf(url) -> bool:
    try:
        return isinstance(url, str) and urlsplit(url).path.lower().endswith('.pdf')
    except ValueError:
        return False


def _report_from_context(conversation: ConversationState, targets: list[str]) -> AssistantResponse | None:
    for message in reversed(conversation.messages):
        context = message.response
        if context is None:
            continue
        metadata = {}
        citations = context.metadata.get('research_citations', [])
        metadata['research_citations'] = [deepcopy(item) for item in citations
            if isinstance(item, dict) and _association(item, context, targets)] if isinstance(citations, list) else []
        bundle = context.metadata.get('evidence_bundle', {})
        metadata['evidence_bundle'] = {}
        if isinstance(bundle, dict):
            for key in ('citations', 'external_citations', 'historical_citations', 'web_citations'):
                items = bundle.get(key, [])
                if isinstance(items, list):
                    metadata['evidence_bundle'][key] = [deepcopy(item) for item in items
                        if isinstance(item, dict) and _association(item, context, targets)]
        references = []
        for reference in context.source_details:
            if not _association(reference.request_parameters, context, targets):
                continue
            references.append(reference.model_copy(deep=True))
            if _pdf(reference.source_url):
                explicit = reference.request_parameters.get('ticker') or reference.request_parameters.get('symbol')
                metadata['research_citations'].append({'source_url': reference.source_url,
                    'title': f'Tài liệu nguồn · {reference.provider}',
                    'tickers': [explicit] if explicit else list(context.tickers),
                    'retrieved_at': reference.fetched_at.isoformat()})
        tables = []
        for table in context.tables:
            rows = [deepcopy(row) for row in table.rows if _association(row, context, targets)]
            if rows:
                tables.append(table.model_copy(update={'rows': rows}, deep=True))
                for row in rows:
                    url = row.get('canonical_url') or row.get('source_url') or row.get('url')
                    if _pdf(url) or row.get('document_type') in _REPORT_TYPES:
                        metadata['research_citations'].append({**row, 'source_url': url})
        if len(context.tickers) == 1 and context.tickers[0] in targets:
            for url in context.sources:
                if _pdf(url):
                    metadata['research_citations'].append({'source_url': url,
                        'title': 'URL báo cáo từ phản hồi trước', 'tickers': list(context.tickers)})
        candidate = AssistantResponse(message='', tickers=list(targets), metadata=metadata,
                                      source_details=references, tables=tables,
                                      warnings=list(context.warnings), data_timestamp=context.data_timestamp,
                                      data_fetched_at=context.data_fetched_at)
        reports = [item for item in build_interaction_actions(candidate) if item.kind == 'REPORT']
        if not reports or not set(targets).issubset({item.ticker for item in reports}):
            continue
        candidate.message = ('Mở các URL báo cáo đã có trong cuộc trò chuyện cho ' + ', '.join(targets)
                             + '. Đây là tài liệu đã truy xuất trước đó, không khẳng định là báo cáo mới nhất; hãy kiểm tra kỳ dữ liệu và ngày công bố tại nguồn.')
        candidate.metadata['navigation_request'] = {'kind': 'REPORT', 'automatic': True}
        candidate.metadata['report_context_as_of'] = str(context.data_timestamp) if context.data_timestamp else None
        return candidate
    return None


def navigation_response(text: str, conversation: ConversationState) -> AssistantResponse | None:
    """Return navigation-only response, or None to retain the normal research flow."""
    if not isinstance(text, str) or len(text) > 350:
        return None
    # Never treat arbitrary pasted URLs, scripts, paths, or multi-line text as navigation.
    if any(char in text for char in ('\n', '\r', '`', ';', '{', '}', '<', '>', '\\', '/', ':', '=', '[', ']')):
        return None
    if re.search(r'\b[a-z0-9-]+\.[a-z]{2,}\b', text, flags=re.I):
        return None
    normalized = _normalize(text)
    if not re.match(r'^(?:(?:hay|vui long|giup toi|ban hay|toi muon|cho toi|xin)\s+)*(?:mo|open)\b', normalized):
        return None
    if re.search(r'\b(?:tai khoan|vi the|phan tich|danh gia|du doan|so sanh|nen mua|nen ban|tai sao|vi sao|chi so|co tot|co nen|khuyen nghi|tinh toan|doanh thu|loi nhuan|tang truong|rui ro)\b', normalized):
        return None
    if re.search(r'\b(?:www|https?|javascript|powershell|python|cmd|exec|import)\b', normalized):
        return None
    report = bool(re.search(r'\b(?:bao cao|tai lieu|pdf)\b', normalized))
    if report and re.search(r'\b(?:moi nhat|moi|gan nhat|hien tai|cap nhat|latest|newest)\b', normalized):
        return None  # Existing conversation evidence cannot prove latest publication.
    if report and re.search(r'\b(?:20\d{2}|quy|q[1-4]|nam|thang)\b', normalized):
        return None  # A period-qualified request needs normal evidence/date matching.
    if report:
        kind = 'REPORT'
    elif re.search(r'\b(?:giao dich|dat lenh|de mua|de ban|dboard|vndirect)\b', normalized):
        kind = 'BROKER'
    elif re.search(r'\b(?:bieu do|chart)\b', normalized):
        kind = 'CHART'
    elif re.search(r'\b(?:bang gia|gia|lich su)\b', normalized):
        kind = 'QUOTE'
    else:
        kind = 'STOCK'
    tickers = _tickers(text, normalized)
    if not tickers:
        tickers = list(dict.fromkeys(conversation.active_tickers)) if len(set(conversation.active_tickers)) == 1 else []
    if not tickers or (kind == 'BROKER' and len(tickers) != 1):
        return _clarify()
    if kind == 'REPORT':
        return _report_from_context(conversation, tickers)
    labels = {'STOCK': 'trang cổ phiếu', 'QUOTE': 'trang xem giá', 'CHART': 'trang biểu đồ',
              'BROKER': 'website giao dịch do bạn cấu hình'}
    return AssistantResponse(message=f'Mở {labels[kind]} cho {", ".join(tickers)}. '
                             'Ứng dụng chỉ điều hướng, không đăng nhập hoặc đặt lệnh; hãy kiểm tra mã và thời điểm giá tại website.',
                             tickers=tickers, metadata={'navigation_request': {'kind': kind, 'automatic': True}})
