"""Construct browser links from structured evidence without network or trading APIs.

Stockbiz URL patterns checked against its own FPT overview navigation:
https://web.stockbiz.vn/Stocks/FPT/Overview.aspx
https://web.stockbiz.vn/Stocks/FPT/HistoricalQuotes.aspx
https://web.stockbiz.vn/Stocks/FPT/Chart.aspx
TradingView exchange-qualified pattern: https://vn.tradingview.com/symbols/HOSE-FPT/
Portal availability and destination contents are not verified on each render.
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
from collections.abc import Mapping
from datetime import date, datetime
from string import Formatter
from urllib.parse import parse_qsl, unquote, urlsplit

from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.models.interaction import InteractionAction

_TICKER = re.compile(r'^[A-Z][A-Z0-9]{1,9}$')
_SECRET = re.compile(r'(?:password|passwd|secret|token|api.?key|authorization|credential|session|signature|sig$)', re.I)
_REPORT_TYPES = {'financial_statement', 'annual_report', 'regulatory_disclosure',
                 'earnings_release', 'investor_presentation'}
_BROKER_QUERY_KEYS = {'ticker', 'symbol', 'code', 'stock', 'lang', 'language', 'exchange'}
_BLOCKED_HOST_SUFFIXES = ('.localhost', '.local', '.internal', '.test', '.invalid')


def validate_ticker(ticker: str) -> str:
    if not isinstance(ticker, str) or not _TICKER.fullmatch(ticker.strip().upper()):
        raise ValueError('Mã cổ phiếu phải gồm 2–10 ký tự chữ/số, bắt đầu bằng chữ.')
    return ticker.strip().upper()


def validate_external_url(url: str) -> str:
    """Syntactic navigation guard; no DNS lookup or HTTP requests are performed.

    Public DNS names can still redirect/change resolution later. This is not an
    SSRF fetch policy and must never be used to authorize server-side fetching.
    """
    if not isinstance(url, str) or not url or len(url) > 4096:
        raise ValueError('URL không hợp lệ.')
    decoded = unquote(unquote(url))
    if any(char.isspace() for char in url) or any(ord(char) < 32 or ord(char) == 127 for char in url + decoded):
        raise ValueError('URL không được chứa khoảng trắng/ký tự điều khiển.')
    if any(char in decoded for char in ('\\', '<', '>', '"', "'", '`', '{', '}')):
        raise ValueError('URL chứa ký tự không an toàn.')
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or '').rstrip('.').lower()
        if parsed.scheme != 'https' or not host or parsed.username is not None or parsed.password is not None:
            raise ValueError('Chỉ chấp nhận URL HTTPS công khai, không có tài khoản/mật khẩu.')
        if parsed.port not in (None, 443):
            raise ValueError('Chỉ chấp nhận cổng HTTPS mặc định.')
        if host in {'localhost', 'localhost.localdomain'} or host.endswith(_BLOCKED_HOST_SUFFIXES):
            raise ValueError('Không mở địa chỉ nội bộ.')
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if not host.isascii() or '.' not in host or not re.fullmatch(r'[a-z0-9.-]+', host):
                raise ValueError('Tên miền công khai không hợp lệ.')
            labels = host.split('.')
            if any(not label or label.startswith('-') or label.endswith('-') for label in labels):
                raise ValueError('Tên miền không hợp lệ.')
            if labels[-1].isdigit() or all(re.fullmatch(r'(?:0x[0-9a-f]+|[0-9]+)', part) for part in labels):
                raise ValueError('Không chấp nhận địa chỉ IP dạng rút gọn/mã hóa.')
        else:
            if not address.is_global:
                raise ValueError('Không mở địa chỉ IP nội bộ.')
        query_parts = [parsed.query]
        if '?' in parsed.fragment:
            query_parts.append(parsed.fragment.split('?', 1)[1])
        elif '=' in parsed.fragment:
            query_parts.append(parsed.fragment)
        if any(_SECRET.search(key) for part in query_parts for key, _ in parse_qsl(part, keep_blank_values=True)):
            raise ValueError('Không đưa thông tin đăng nhập/API key/token vào liên kết.')
    except (ValueError, UnicodeError) as exc:
        raise ValueError('URL HTTPS không hợp lệ hoặc không an toàn.') from exc
    return url


def validate_broker_template(template: str) -> str:
    """Only a manually supplied navigation address, never an order request."""
    try:
        fields = list(Formatter().parse(template))
        if any(name not in (None, 'ticker') or spec or conversion for _, name, spec, conversion in fields):
            raise ValueError('Chỉ hỗ trợ biến {ticker}.')
        sample = template.format(ticker='FPT')
        validate_external_url(sample)
        parsed = urlsplit(sample)
        if '{ticker}' in urlsplit(template).netloc:
            raise ValueError('Không được thay đổi tên miền bằng mã cổ phiếu.')
        parts = [parsed.query]
        if '?' in parsed.fragment:
            parts.append(parsed.fragment.split('?', 1)[1])
        elif '=' in parsed.fragment:
            parts.append(parsed.fragment)
        if any(key.lower() not in _BROKER_QUERY_KEYS for part in parts for key, _ in parse_qsl(part, keep_blank_values=True)):
            raise ValueError('Chỉ cho phép tham số điều hướng mã/ngôn ngữ/sàn; không có lệnh mua bán.')
        path = unquote(parsed.path + '/' + parsed.fragment.split('?', 1)[0]).lower()
        if re.search(r'(?:^|[/_.-])(?:submit|execute|placeorder|place-order|cancelorder|buy|sell)(?:$|[/_.-])', path):
            raise ValueError('Không dùng URL thực thi lệnh.')
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError('Đường dẫn cổng giao dịch không an toàn; chỉ dùng URL điều hướng công khai.') from exc
    return template


def _action(kind: str, label: str, url: str, ticker: str | None = None, **kwargs) -> InteractionAction:
    validate_external_url(url)
    action_id = hashlib.sha256(f'{kind}|{ticker}|{url}'.encode()).hexdigest()[:20]
    return InteractionAction(id=action_id, kind=kind, label=label[:240], url=url, ticker=ticker, **kwargs)


def actions_for_ticker(ticker: str, exchange: str | None = None) -> list[InteractionAction]:
    ticker = validate_ticker(ticker)
    stockbiz = f'https://web.stockbiz.vn/Stocks/{ticker}'
    # Syntactic validity does not establish that a symbol is listed or active.
    note = 'Mở trang theo mã; hãy đối chiếu doanh nghiệp, sàn và thời điểm giá tại website. Không đảm bảo giá realtime.'
    chart_url = f'{stockbiz}/Chart.aspx'
    chart_name = 'Stockbiz'
    if isinstance(exchange, str) and exchange.upper() in {'HOSE', 'HSX', 'HNX', 'UPCOM'}:
        venue = 'HOSE' if exchange.upper() == 'HSX' else exchange.upper()
        chart_url = f'https://vn.tradingview.com/symbols/{venue}-{ticker}/'
        chart_name = 'TradingView'
    return [
        _action('STOCK', f'{ticker} · VNDIRECT DStock', f'https://dstock.vndirect.com.vn/suc-khoe-co-phieu/{ticker}', ticker,
                description='Trang phân tích đúng mã của VNDIRECT DStock, không phải phiếu đặt lệnh. ' + note),
        _action('STOCK', f'{ticker} · Trang doanh nghiệp', f'{stockbiz}/Overview.aspx', ticker, description=note),
        _action('QUOTE', f'{ticker} · Xem giá và lịch sử', f'{stockbiz}/HistoricalQuotes.aspx', ticker, description=note),
        _action('CHART', f'{ticker} · Biểu đồ {chart_name}', chart_url, ticker, description=note),
    ]


def _records(value):
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _text(value):
    return value if isinstance(value, str) else None


def _timestamp(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).isoformat()
        except ValueError:
            pass
    return None


def _citation_tickers(item: Mapping, response_tickers: list[str]) -> list[str | None]:
    explicit = item.get('tickers')
    # Some adapters expose singular ticker/symbol instead of Citation.tickers.
    # Respect that explicit association rather than assigning another company.
    if not isinstance(explicit, list) or not explicit:
        singular = item.get('ticker') or item.get('symbol')
        if isinstance(singular, str) and singular:
            explicit = [singular]
        elif isinstance(explicit, str) and explicit:
            explicit = [explicit]
    if isinstance(explicit, list) and explicit:
        matched = []
        for value in explicit:
            try:
                ticker = validate_ticker(value)
            except ValueError:
                continue
            if ticker in response_tickers and ticker not in matched:
                matched.append(ticker)
        return matched  # Explicit other-company evidence is not relabelled.
    return [response_tickers[0]] if len(response_tickers) == 1 else [None]


def build_interaction_actions(response: AssistantResponse, *, broker_url_template: str | None = None,
                              broker_name: str = 'Cổng giao dịch cá nhân') -> list[InteractionAction]:
    tickers = []
    for value in response.tickers:
        try:
            ticker = validate_ticker(value)
        except ValueError:
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    actions = []
    metadata = response.metadata if isinstance(response.metadata, Mapping) else {}
    exchanges = metadata.get('exchanges', {})
    for ticker in tickers:
        exchange = exchanges.get(ticker) if isinstance(exchanges, Mapping) else None
        if not exchange and len(tickers) == 1:
            exchange = metadata.get('exchange')
        actions.extend(actions_for_ticker(ticker, exchange))
    bundle = metadata.get('evidence_bundle')
    bundle = bundle if isinstance(bundle, Mapping) else {}
    groups = [(_records(metadata.get('research_citations')), 'evidence')]
    groups.extend((_records(bundle.get(key)), mode) for key, mode in (
        ('citations', 'evidence'), ('external_citations', 'external'),
        ('historical_citations', 'historical'), ('web_citations', 'discovery')))
    for records, mode in groups:
        for item in records:
            url = _text(item.get('canonical_url')) or _text(item.get('source_url'))
            if not url:
                continue
            try:
                validate_external_url(url)
            except ValueError:
                continue
            document_type = item.get('document_type')
            kind = 'REPORT' if mode != 'discovery' and ((isinstance(document_type, str) and document_type in _REPORT_TYPES) or urlsplit(url).path.lower().endswith('.pdf')) else 'SOURCE'
            title = _text(item.get('title')) or 'Tài liệu nguồn'
            period = item.get('reporting_period')
            period_label = _text(period.get('label')) if isinstance(period, Mapping) else None
            if not period_label:
                period_end = (period.get('end') if isinstance(period, Mapping) else None) or item.get('period_end')
                period_label = f'Dữ liệu đến kỳ {period_end}' if isinstance(period_end, (str, date)) else None
            description = {'discovery': 'Kết quả tìm kiếm, chưa xác minh nội dung; không phải báo cáo đã kiểm chứng.',
                           'historical': 'Tài liệu lịch sử; không mặc định là kỳ mới nhất.',
                           'external': 'URL đã truy xuất trong phiên research; kiểm tra nội dung và kỳ báo cáo.',
                           'evidence': 'Mở đúng URL được trích dẫn; không khẳng định đây là báo cáo mới nhất.'}[mode]
            publisher = _text(item.get('publisher')) or _text(item.get('source_domain'))
            if publisher:
                description += f' Nguồn: {publisher[:200]}.'
            for ticker in _citation_tickers(item, tickers):
                try:
                    actions.append(_action(kind, f'{"Báo cáo" if kind == "REPORT" else "Nguồn"} · {title}', url,
                        ticker, description=description, published_at=_timestamp(item.get('published_at')), period_label=period_label))
                except ValueError:
                    continue
    for reference in response.source_details:
        if not reference.source_url:
            continue
        parameters = reference.request_parameters
        explicit = parameters.get('ticker') or parameters.get('symbol')
        source_tickers = _citation_tickers({'tickers': [explicit]} if explicit else {}, tickers)
        for ticker in source_tickers:
            try:
                actions.append(_action('SOURCE', f'Nguồn dữ liệu · {reference.provider}', reference.source_url, ticker,
                    description=f'{reference.data_kind.value}; dữ liệu đến {reference.as_of or "chưa xác định"}; lấy lúc {reference.fetched_at.isoformat()}.'))
            except ValueError:
                continue
    # Legacy source lists can contain URLs or descriptive provider names. Never parse URLs from prose.
    for url in response.sources:
        try:
            actions.append(_action('SOURCE', 'Mở URL nguồn tham khảo', url, description='URL nguồn; chưa xác định loại tài liệu hoặc mã liên quan.'))
        except ValueError:
            continue
    if broker_url_template:
        validate_broker_template(broker_url_template)
        has_symbol = '{ticker}' in broker_url_template
        for ticker in tickers if has_symbol else [None]:
            url = broker_url_template.format(ticker=ticker)
            suffix = f' · {ticker}' if ticker else ''
            actions.append(_action('BROKER', f'Mở {broker_name}{suffix}', url, ticker,
                description='Chỉ mở website do bạn cấu hình; tự kiểm tra mã sau đăng nhập. Không điền giá/khối lượng, không đặt lệnh, không đảm bảo website giữ mã qua đăng nhập.'))
    # Same cited URL may have many chunks. Preserve company association and report priority.
    unique = {}
    for action in actions:
        key = (action.url, action.ticker)
        previous = unique.get(key)
        if previous is None or (action.kind == 'REPORT' and previous.kind == 'SOURCE'):
            unique[key] = action
    return list(unique.values())
