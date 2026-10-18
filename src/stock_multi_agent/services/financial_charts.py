"""Period-aware financial charts: reported facts and compatible derived growth."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import re

from stock_multi_agent.models.chart import ChartArtifact, ChartType
from stock_multi_agent.services.financial_facts import build_financial_evidence

FINANCIAL_TYPES = {ChartType.REVENUE_PROFIT, ChartType.CASH_FLOW, ChartType.GROWTH,
                   ChartType.VALUATION, ChartType.PROFITABILITY}
METRICS = {
    ChartType.REVENUE_PROFIT: {'revenue': 'Doanh thu', 'net_profit': 'Lợi nhuận sau thuế'},
    ChartType.CASH_FLOW: {'operating_cash_flow': 'Dòng tiền kinh doanh'},
    ChartType.GROWTH: {'revenue_growth': 'Tăng trưởng doanh thu', 'profit_growth': 'Tăng trưởng lợi nhuận'},
    ChartType.VALUATION: {'pe': 'P/E', 'pb': 'P/B'},
    ChartType.PROFITABILITY: {'roe': 'ROE', 'roa': 'ROA', 'profit_margin': 'Biên lợi nhuận'},
}
EXPLANATIONS = {
    ChartType.REVENUE_PROFIT: 'Đối chiếu doanh thu và lợi nhuận: doanh thu tăng nhưng lợi nhuận không tăng tương ứng có thể gợi ý áp lực biên. Biểu đồ không tự chứng minh nguyên nhân.',
    ChartType.CASH_FLOW: 'Dòng tiền kinh doanh cho biết tiền từ hoạt động chính. Một kỳ âm chưa chứng minh kinh doanh xấu; cần xem vốn lưu động, mùa vụ và dòng tiền cùng kỳ.',
    ChartType.GROWTH: 'Tăng trưởng dẫn xuất dùng cùng kỳ năm trước, cùng cơ sở báo cáo, hợp nhất và đơn vị; không so quý riêng lẻ với lũy kế. Tăng trưởng không đồng nghĩa cổ phiếu rẻ.',
    ChartType.VALUATION: 'P/E và P/B là bội số định giá, không phải giá tiền hoặc phần trăm tăng trưởng. Mức thấp không tự chứng minh hấp dẫn nếu lợi nhuận hoặc tài sản có vấn đề.',
    ChartType.PROFITABILITY: 'ROE, ROA và biên lợi nhuận mô tả hiệu quả sinh lời. ROE cao có thể đi cùng đòn bẩy cao; cần đối chiếu nợ và chất lượng lợi nhuận.',
}


def _panel_key(row):
    return (row['frequency'], row['unit'], row['reporting_basis'], row['consolidation'],
        row['assurance'], '' if row['normalized'] else row['source_id'])


def financial_artifact(request, data):
    metrics = METRICS[request.chart_type]
    evidence = build_financial_evidence(data)
    buckets = defaultdict(list)
    controls = {str(value).upper() for value in request.indicators}
    frequency_filter = 'annual' if 'ANNUAL' in controls else 'quarter' if 'QUARTER' in controls else None
    for fact in evidence['normalized_facts']:
        if fact['metric'] not in metrics or frequency_filter and fact['frequency'] != frequency_filter:
            continue
        if fact['unit'] == 'unknown_ratio':
            continue
        unit = fact['unit'] if fact['normalized_value'] is not None else f"{fact['unit']} (thang nguồn chưa xác minh)"
        key = (fact['metric'], fact['period_end'], unit, fact['frequency'], fact['reporting_basis'],
            fact['consolidation'], fact['assurance'])
        buckets[key].append(fact)
    records, warnings = [], list(data.warnings)
    for (metric, period, unit, frequency, basis, consolidation, assurance), items in sorted(buckets.items(), key=lambda pair: pair[0][1]):
        values = {item['normalized_value'] if item['normalized_value'] is not None else item['value'] for item in items}
        if len(values) != 1:
            warnings.append(f'Bỏ {metric} kỳ {period}: nguồn báo cáo các giá trị xung đột.')
            continue
        item = items[0]
        records.append({'date': period, 'metric': metric, 'label': metrics[metric], 'value': values.pop(),
            'unit': unit, 'frequency': frequency, 'source_id': item['source_id'],
            'source_ids': sorted({fact['source_id'] for fact in items}),
            'published_at': item['published_at'], 'reporting_basis': basis, 'consolidation': consolidation,
            'assurance': assurance, 'normalized': item['normalized_value'] is not None,
            'period_start': item['period_start'], 'formula': 'reported', 'status': item['status']})
    if request.chart_type == ChartType.GROWTH:
        for derived in evidence['derived_metrics']:
            if derived['metric'] not in metrics:
                continue
            if any(row['metric'] == derived['metric'] and row['date'] == derived['period_end'] for row in records):
                continue
            records.append({'date': derived['period_end'], 'metric': derived['metric'], 'label': metrics[derived['metric']],
                'value': derived['value'], 'unit': 'ratio', 'frequency': 'derived_yoy',
                'source_id': ' + '.join(derived['source_ids']), 'source_ids': derived['source_ids'],
                'published_at': None, 'reporting_basis': 'SAME_BASIS_YOY', 'consolidation': 'VERIFIED_MATCH',
                'assurance': 'VERIFIED_MATCH', 'normalized': True, 'period_start': None,
                'formula': derived['formula'], 'status': 'DERIVED'})
    if not records:
        raise ValueError('Không có quan sát tài chính đúng chỉ tiêu kèm kỳ và đơn vị; không dựng chuỗi từ một tỷ số hiện tại. ' +
            (' '.join(evidence['limitations'][:3]) if request.chart_type == ChartType.GROWTH else ''))
    newest = max(date.fromisoformat(item['date']) for item in records)
    requested = request.period.upper()
    yearly = re.fullmatch(r'(\d+)Y', requested)
    quarterly = re.fullmatch(r'(\d+)Q', requested)
    if yearly:
        records = [row for row in records if date.fromisoformat(row['date']).year >= newest.year - int(yearly[1]) + 1]
    elif quarterly:
        periods = sorted({row['date'] for row in records if row['frequency'] == 'quarter'}, reverse=True)[:int(quarterly[1])]
        records = [row for row in records if row['frequency'] == 'quarter' and row['date'] in periods]
    elif requested != 'ALL':
        retained = defaultdict(list)
        for row in records:
            retained[(row['metric'], _panel_key(row))].append(row)
        records = [row for rows in retained.values() for row in sorted(rows, key=lambda row: row['date'])[-8:]]
    if not records:
        raise ValueError('Không có quan sát tài chính phù hợp bộ lọc kỳ/tần suất; không thay bằng dữ liệu năm.')
    newest = max(date.fromisoformat(item['date']) for item in records)
    if any(row['reporting_basis'] == 'UNKNOWN' or row['consolidation'] == 'UNKNOWN' for row in records):
        warnings.append('Một số nguồn chưa xác minh quý riêng lẻ/lũy kế hoặc hợp nhất/riêng lẻ. Chỉ hiển thị số báo cáo, không suy ra tăng trưởng từ các số này.')
    if any(not row['normalized'] for row in records):
        warnings.append('Thang tiền tệ chưa xác minh: giữ nguyên số nguồn trên panel riêng, không gắn nhãn VND hay so chéo nguồn.')
    if (data.as_of - newest).days > 220:
        warnings.append(f'Dữ liệu tài chính chỉ đến kỳ {newest}; không coi là tình hình cập nhật hôm nay.')
    used = {identifier for item in records for identifier in item['source_ids']}
    return ChartArtifact(request=request, title=f'{request.ticker} · {request.chart_type.value} · dữ liệu đến kỳ {newest}',
        records=sorted(records, key=lambda row: (row['date'], row['metric'])), data_timestamp=newest,
        data_fetched_at=data.fetched_at, source=data.source,
        source_details=[ref for ref in data.source_refs if ref.source_id in used], warnings=list(dict.fromkeys(warnings)),
        explanation=EXPLANATIONS[request.chart_type] + ' Mỗi cột là kỳ đã có dữ liệu; không nối qua kỳ thiếu. '
            'Panel tách theo đơn vị, tần suất, cơ sở kế toán và hợp nhất. Bộ lọc: ANNUAL/QUARTER, 3Y/5Y/8Q/ALL.',
        axis_unit='Đơn vị chuẩn hóa nếu nguồn khai báo rõ; nếu không giữ nguyên thang nguồn')


def render_financial(artifact):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    panels = sorted({_panel_key(row) for row in artifact.records})
    figure = make_subplots(rows=len(panels), cols=1,
        subplot_titles=[' · '.join(value for value in panel if value) for panel in panels])
    for index, panel in enumerate(panels, 1):
        rows = [row for row in artifact.records if _panel_key(row) == panel]
        unit = panel[1]
        for metric in sorted({row['metric'] for row in rows}):
            values = [row for row in rows if row['metric'] == metric]
            figure.add_trace(go.Bar(x=[row['date'] for row in values], y=[row['value'] for row in values],
                name=values[0]['label'], customdata=[[row['source_id'], row['unit'], row['formula']] for row in values],
                hovertemplate='%{x}<br>%{y} %{customdata[1]}<br>Nguồn: %{customdata[0]}<br>Công thức: %{customdata[2]}<extra>%{fullData.name}</extra>'), row=index, col=1)
        figure.update_yaxes(title_text=unit, tickformat='.1%' if unit == 'ratio' else None, row=index, col=1)
    figure.update_layout(title=artifact.title, barmode='group', height=max(430, len(panels) * 300))
    return figure
