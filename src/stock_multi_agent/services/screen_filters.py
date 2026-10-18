"""Deterministic numeric filters. Preserve operators lost by text normalization."""
from __future__ import annotations

import math
import operator
import re
import unicodedata

COMPARATORS = {'lt': operator.lt, 'gt': operator.gt, 'lte': operator.le,
               'gte': operator.ge, 'eq': operator.eq}
ALLOWED_FILTERS = {f'{metric}_{op}' for metric in ('rsi', 'pe', 'pb', 'roe') for op in COMPARATORS}
_FIELD = r'(?P<field>rsi(?:\s*\(?\s*14\s*\)?)?|p\s*/?\s*e|p\s*/?\s*b|roe)'
_OP = r'(?P<op>>=|<=|==|>|<|=|khong nho hon|khong thap hon|khong duoi|it nhat|toi thieu|khong lon hon|khong cao hon|khong vuot qua|toi da|lon hon|cao hon|nho hon|thap hon|tren|duoi|bang)'
_NUMBER = r'(?P<number>[+-]?\d+(?:[.,]\d+)?)'
_OPERATORS = {'>': 'gt', '<': 'lt', '>=': 'gte', '<=': 'lte', '=': 'eq', '==': 'eq',
    'tren': 'gt', 'lon hon': 'gt', 'cao hon': 'gt', 'duoi': 'lt', 'nho hon': 'lt', 'thap hon': 'lt',
    'khong nho hon': 'gte', 'khong thap hon': 'gte', 'khong duoi': 'gte', 'it nhat': 'gte', 'toi thieu': 'gte',
    'khong lon hon': 'lte', 'khong cao hon': 'lte', 'khong vuot qua': 'lte', 'toi da': 'lte', 'bang': 'eq'}


def _plain(text):
    text = text.casefold().replace('đ', 'd').replace('≥', '>=').replace('≤', '<=')
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)).split())


def parse_screen_filters(text: str) -> tuple[dict[str, float], list[str]]:
    plain = _plain(text)
    filters, errors = {}, []
    def record(field, op, value):
        field = re.sub(r'[^a-z]', '', field)
        metric = 'rsi' if field.startswith('rsi') else field
        value = float(value.replace(',', '.'))
        if metric == 'rsi' and not 0 <= value <= 100:
            errors.append('Ngưỡng RSI phải nằm trong khoảng 0–100.')
            return
        key = f'{metric}_{op}'
        if key in filters:
            if op in {'gt', 'gte'}:
                value = max(value, filters[key])
            elif op in {'lt', 'lte'}:
                value = min(value, filters[key])
            elif value != filters[key]:
                errors.append('Các điều kiện bằng nhau đang mâu thuẫn.')
        filters[key] = value
    for match in re.finditer(r'\b' + _FIELD + r'\s*(?:la\s+)?' + _OP + r'\s*' + _NUMBER, plain):
        record(match['field'], _OPERATORS[match['op']], match['number'])
        # Support "RSI > 30 và < 70" without silently dropping its second bound.
        tail = plain[match.end():]
        continuation = re.match(r'\s*(?:va|and)\s*' + _OP + r'\s*' + _NUMBER, tail)
        if continuation:
            record(match['field'], _OPERATORS[continuation['op']], continuation['number'])
    for match in re.finditer(r'\b' + _FIELD + r'\s*(?:nam\s+)?(?:trong\s+khoang\s+|tu\s+)' + _NUMBER + r'\s*(?:den|toi|-)\s*(?P<upper>\d+(?:[.,]\d+)?)', plain):
        record(match['field'], 'gte', match['number'])
        record(match['field'], 'lte', match['upper'])
    if (filters or errors) and re.search(r'\b(?:hoac|or)\b', plain):
        errors.append('Hiện hỗ trợ kết hợp điều kiện bằng “và”; hãy tách truy vấn “hoặc” thành các lượt riêng.')
    if re.search(r'\brsi\s*\(?\s*(?!14\b)\d+\s*\)?\s*(?:>=|<=|>|<|=)', plain):
        errors.append('Bộ lọc hiện dùng RSI(14); bạn hãy dùng RSI hoặc RSI14.')
    for metric in ('rsi', 'pe', 'pb', 'roe'):
        bounds = [(op, value) for key, value in filters.items() if key.startswith(metric + '_') for op in [key.rsplit('_', 1)[1]]]
        if not bounds:
            continue
        # Intersection of thresholds must be nonempty (strict endpoints matter).
        lower = max([(v, op == 'gt') for op, v in bounds if op in {'gt', 'gte', 'eq'}], default=(-math.inf, False))
        upper = min([(v, op != 'lt') for op, v in bounds if op in {'lt', 'lte', 'eq'}], default=(math.inf, True))
        if lower[0] > upper[0] or (lower[0] == upper[0] and (lower[1] or not upper[1])):
            errors.append(f'Các điều kiện {metric.upper()} không có giao nhau.')
    return filters, list(dict.fromkeys(errors))


def is_numeric_screen(text: str, filters: dict, errors: list, explicit_tickers: list[str]) -> bool:
    # Uppercase command words are not explicit tickers (e.g. TIM CHO TOI).
    command_words = {'tim', 'cho', 'toi', 'co', 'phieu', 'ma', 'loc', 'hay', 'liet', 'ke',
                     'chon', 'scan', 'screen', 'top', 'stocks', 'va', 'and', 'tren', 'duoi'}
    tickers = [ticker for ticker in explicit_tickers if _plain(ticker) not in command_words]
    if tickers or not (filters or errors):
        return False
    plain = _plain(text)
    if re.search(r'\b(?:la gi|nghia la gi|cong thuc|giai thich)\b', plain) and not re.search(r'\b(?:tim|loc|liet ke)\b', plain):
        return False
    return bool(re.search(r'\b(?:tim|loc|liet ke|chon|scan|screen|top)\b', plain)
                or re.search(r'\b\d+\s+(?:ma|co phieu|stocks?)\b', plain))


def validate_filter_keys(filters):
    if set(filters).difference(ALLOWED_FILTERS):
        raise ValueError('Supported filters: RSI, PE, PB, ROE with <, >, <=, >=, =')
    if any(not math.isfinite(float(value)) for value in filters.values()):
        raise ValueError('Filter thresholds must be finite')


def matches_filter(value, op, threshold):
    return value is not None and math.isfinite(float(value)) and COMPARATORS[op](value, threshold)
