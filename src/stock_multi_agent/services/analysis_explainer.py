"""Vietnamese presentation of computed evidence, not a second scoring policy."""
from __future__ import annotations
import math

from stock_multi_agent.models.chat import ResponseTable


class AnalysisExplainer:
    METRICS = {
        'revenue_growth': ('Tăng trưởng doanh thu', 'Mức thay đổi doanh thu theo kỳ so sánh của nguồn; không tự suy ra chất lượng lợi nhuận.', True),
        'profit_growth': ('Tăng trưởng lợi nhuận', 'Lợi nhuận tăng/giảm theo kỳ nguồn; cần kiểm tra khoản bất thường và nền so sánh.', True),
        'roe': ('ROE', 'Lợi nhuận/vốn chủ: hiệu quả dùng vốn; ROE cao có thể đi kèm đòn bẩy cao.', True),
        'roa': ('ROA', 'Lợi nhuận/tổng tài sản: hiệu quả tài sản; khác biệt ngành rất quan trọng.', True),
        'profit_margin': ('Biên lợi nhuận', 'Lợi nhuận/doanh thu: khả năng giữ lại lợi nhuận trên doanh thu.', True),
        'pe': ('P/E', 'Giá/EPS: cần đối chiếu tăng trưởng và peer; P/E thấp không tự có nghĩa rẻ.', False),
        'pb': ('P/B', 'Giá/giá trị sổ sách mỗi cổ phiếu: nên đọc cùng ROE và chất lượng tài sản.', False),
        'eps': ('EPS', 'Lợi nhuận trên mỗi cổ phiếu; cần kiểm tra kỳ, pha loãng và đơn vị nguồn.', False),
        'debt_to_equity': ('Nợ/vốn chủ', 'Mức đòn bẩy; không áp cùng cách đọc cho ngân hàng và doanh nghiệp phi tài chính.', False),
        'operating_cash_flow': ('Dòng tiền kinh doanh', 'Tiền từ hoạt động kinh doanh; dương/âm không đủ để kết luận khi thiếu lịch sử và mùa vụ.', False),
    }
    FACTORS = {
        'price_vs_fast_sma': 'Giá so với MA nhanh', 'fast_vs_slow_sma': 'MA nhanh so với MA chậm',
        'rsi': 'Động lượng RSI', 'macd': 'Động lượng MACD', 'volume_confirmation': 'Xác nhận khối lượng',
        'volatility': 'Biến động giá', 'liquidity': 'Thanh khoản', 'price_adjustment_quality': 'Trạng thái điều chỉnh giá',
        'relative_benchmark': 'Lợi suất tương đối VN-Index', 'lstm_blend_delta': 'Thay đổi điểm do phối hợp LSTM 20%',
        'Revenue growth': 'Tăng trưởng doanh thu', 'Profit growth': 'Tăng trưởng lợi nhuận',
        'Profit margin': 'Biên lợi nhuận', 'Debt/equity': 'Đòn bẩy nợ/vốn chủ', 'Operating cash flow': 'Dòng tiền kinh doanh',
        'reported_revenue_growth': 'Tăng trưởng tính từ báo cáo', 'reported_profit_growth': 'Tăng trưởng lợi nhuận từ báo cáo',
        'cash_conversion': 'Chuyển đổi lợi nhuận thành tiền', 'data_quality_penalty': 'Chiết giảm do chất lượng dữ liệu',
    }

    def attach(self, response, state, sections):
        data = state.market_data
        if data is None:
            return response
        paragraphs = ['\n\n### Vì sao hệ thống đưa ra nhận định này?']
        drivers = []
        if 'technical' in sections and state.technical_result:
            result = state.technical_result
            direction = {'UPTREND': 'tăng', 'DOWNTREND': 'giảm', 'MIXED': 'chưa đồng nhất'}.get(result.diagnostics.get('regime'), 'chưa đủ dữ liệu xác định')
            paragraphs.append(f'**Kỹ thuật:** xu hướng {direction}. Điểm {result.score:.2f}/10 đến từ các đóng góp đã tính bên dưới, không phải xác suất tăng giá. MA mô tả xu hướng; RSI/MACD mô tả động lượng; khối lượng kiểm tra mức xác nhận. Các chỉ báo có thể mâu thuẫn.')
            self._drivers(drivers, 'Kỹ thuật', result.feature_contributions, [source.source_id for source in data.source_refs])
            response.tables.append(ResponseTable(title='Bằng chứng kỹ thuật đã tính',
                columns=['Chỉ báo', 'Giá trị quan sát', 'Nguồn'], rows=[{'Chỉ báo': item.label,
                    'Giá trị quan sát': item.value, 'Nguồn': ', '.join(item.source_ids) or 'Chưa có lineage nguồn'} for item in result.evidence]))
        if 'fundamental' in sections:
            from stock_multi_agent.services.financial_facts import build_financial_evidence
            from stock_multi_agent.services.financial_freshness import fundamental_inputs
            selected = fundamental_inputs(data)
            pack = build_financial_evidence(data)
            fact_rows = [{'Chỉ tiêu': item['metric'], 'Giá trị gốc': item['value'],
                'Đơn vị gốc': item['reported_unit'], 'Giá trị chuẩn hóa': item['normalized_value'],
                'Tiền tệ': item['currency'], 'Dữ liệu đến kỳ': item['period_end'],
                'Cơ sở kỳ': item['reporting_basis'], 'Hợp nhất': item['consolidation'],
                'Assurance': item['assurance'], 'Nguồn': item['source_id']}
                for item in sorted(pack['normalized_facts'], key=lambda item: item.get('period_end') or '', reverse=True)[:32]]
            if fact_rows:
                response.tables.append(ResponseTable(title='Lịch sử chỉ tiêu có kỳ và cơ sở báo cáo',
                    columns=list(fact_rows[0]), rows=fact_rows,
                    note='Giữ riêng giá trị gốc và chuẩn hóa; UNKNOWN không được tự suy đoán là hợp nhất hay kiểm toán.'))
            derived_rows = [{'Chỉ tiêu': item['metric'], 'Giá trị': item['value'],
                'Kỳ hiện tại': item['period_end'], 'Kỳ so sánh': item['previous_period_end'],
                'Công thức': item['formula'], 'Nguồn': ', '.join(item['source_ids'])}
                for item in pack['derived_metrics']]
            if derived_rows:
                response.tables.append(ResponseTable(title='Chỉ tiêu tính từ các kỳ tương thích',
                    columns=list(derived_rows[0]), rows=derived_rows))
            response.metadata['financial_comparability_limitations'] = pack['limitations']
            rows = []
            for metric, (label, meaning, percent) in self.METRICS.items():
                value = selected.statements.get(metric, getattr(selected.metrics, metric))
                period = selected.metric_periods.get(metric)
                observations = [item for item in data.financial_observations if item.metric == metric
                    and item.period_end == period]
                matched = [item for item in observations if value is not None and math.isclose(item.value, value, rel_tol=1e-6, abs_tol=1e-8)]
                periods = [item.period_end for item in matched if item.period_end]
                sources = list(dict.fromkeys(item.source_id for item in matched))
                if metric in selected.statements:
                    sources = next((item['source_ids'] for item in pack['derived_metrics'] if item['metric'] == metric), [])
                    periods = [period] if period else []
                if not sources:
                    sources = [ref.source_id for ref in data.source_refs if ref.data_kind.value == 'FINANCIAL_RATIOS']
                rows.append({'Chỉ tiêu': label, 'Giá trị': 'Chưa có' if value is None else f'{value:.2%}' if percent else f'{value:,.2f}',
                    'Ý nghĩa/cách đọc': meaning, 'Kỳ dữ liệu': str(max(periods)) if periods else 'Chưa đối chiếu được kỳ của giá trị này',
                    'Nguồn': ', '.join(sources) or 'Chưa có lineage cho chỉ tiêu',
                    'Phạm vi': 'Kỳ gần nhất (lịch sử)' if metric in selected.fallback_metrics else
                              'Kỳ hiện có' if period else 'Chưa xác minh kỳ'})
            response.tables.append(ResponseTable(title='Chỉ số doanh nghiệp và cách đọc', columns=list(rows[0]), rows=rows,
                note='Giá trị normalized do Data/Fundamental cung cấp; không coi tỷ số thiếu là 0, không dùng ngày giá làm kỳ tài chính.'))
            result = state.fundamental_result
            if result:
                self._drivers(drivers, 'Cơ bản', result.feature_contributions, [source.source_id for source in data.source_refs])
            growth = []
            for key, label in [('revenue_growth', 'doanh thu'), ('profit_growth', 'lợi nhuận')]:
                value = selected.statements.get(key, getattr(selected.metrics, key))
                growth.append(f'{label} {value:+.2%}' if value is not None else f'{label} chưa có tỷ lệ tăng trưởng')
            paragraphs.append('**Doanh nghiệp và tăng trưởng:** ' + '; '.join(growth) + '. Đây là tỷ lệ do nguồn cung cấp, chưa tự gán YoY/QoQ khi thiếu kỳ so sánh. Cần đối chiếu doanh thu, biên lợi nhuận, dòng tiền và nợ; một tỷ số tốt không đủ chứng minh doanh nghiệp tốt.')
            if selected.fallback_metrics:
                paragraphs.append('**Kỳ đánh giá cơ bản:** dùng dữ liệu gần nhất còn hợp lệ cho các chỉ số thiếu kỳ mới. '
                    'Xem kỳ của từng chỉ số trong bảng; dữ liệu lịch sử đã làm giảm độ tin cậy, không xác nhận tình hình hiện tại.')
        if 'risk' in sections and state.risk_result:
            risk = state.risk_result
            self._drivers(drivers, 'Rủi ro đo lường', risk.feature_contributions,
                [source.source_id for source in data.source_refs])
            coverage_rows = [{'Nhóm': section.name, 'Trạng thái': section.status,
                'Phạm vi kiểm tra': section.coverage,
                'Còn thiếu': '; '.join(section.missing_information) or 'Không ghi nhận thêm trong checklist hiện tại'}
                for section in risk.sections]
            if coverage_rows:
                response.tables.append(ResponseTable(title='Phạm vi đánh giá rủi ro theo ngành',
                    columns=list(coverage_rows[0]), rows=coverage_rows,
                    note='Coverage là mức đầy đủ của checklist đầu vào, không phải xác suất an toàn.'))
            labels = {'market': 'Biến động/sụt giảm giá', 'liquidity': 'Thanh khoản', 'data': 'Độ bất định dữ liệu',
                'business_and_accounting': 'Tài chính doanh nghiệp', 'event_news': 'Sự kiện', 'portfolio': 'Tập trung danh mục',
                'research_context': 'Báo cáo/tin tức'}
            rows = [{'Nhóm': labels.get(section.name, section.name),
                'Đo được': 'Có dữ liệu' if section.status == 'ASSESSED' else 'Chưa đủ dữ liệu',
                'Bằng chứng': '; '.join(f'{item.label}: {item.value}' for item in section.evidence) or 'Chưa có bằng chứng định lượng',
                'Nguồn': ', '.join(dict.fromkeys(source for item in section.evidence for source in item.source_ids)) or 'Người dùng nhập hoặc chưa có nguồn'} for section in risk.sections]
            response.tables.append(ResponseTable(title='Rủi ro đến từ đâu?', columns=['Nhóm', 'Đo được', 'Bằng chứng', 'Nguồn'], rows=rows))
            paragraphs.append(f'**Rủi ro:** điểm đo lường {risk.risk_score:.2f}/10 được tách khỏi độ bất định dữ liệu {risk.data_uncertainty_score:.2f}/10. Biến động và sụt giảm mô tả lịch sử giá; nợ/thanh khoản phản ánh các mặt rủi ro khác. Dữ liệu cũ/thiếu làm giảm mức chắc chắn, không tự chứng minh có tin xấu. '
                + ('Có veto trong chính sách nên Manager không được đưa MUA.' if risk.veto else 'Không có veto không đồng nghĩa an toàn hoặc chắc chắn sinh lời.'))
        if state.final_decision:
            decision = state.final_decision.recommendation.value
            reasons = {'BUY': 'Technical và Fundamental cùng MUA, bằng chứng đạt ngưỡng và các giới hạn rủi ro cho phép.',
                'HOLD': 'Technical và Fundamental cùng NẮM GIỮ, rủi ro chưa ở mức cao.',
                'WATCH': 'Chưa đồng thuận hoặc bằng chứng/chất lượng dữ liệu chưa đủ điều kiện MUA; nên theo dõi xác nhận.',
                'AVOID': 'Có veto, hoặc rủi ro cao đi cùng tín hiệu BÁN từ ít nhất một chuyên gia.'}
            paragraphs.append('**Vì sao kết luận cuối cùng?** ' + reasons.get(decision, 'Chưa có kết luận.'))
        if drivers:
            response.tables.append(ResponseTable(title='Yếu tố cộng/trừ điểm theo chính sách', columns=list(drivers[0]), rows=drivers,
                note='Đây là đóng góp rule/policy, không phải causal attribution. Điểm cơ bản còn qua quality weighting; Technical có thể clipping và phối hợp LSTM. Không cộng cơ học mọi hàng để suy ra xác suất.'))
        response.message += '\n\n'.join(paragraphs)
        response.metadata['analysis_explanation'] = {'tool': 'AnalysisExplainer', 'sections': sections, 'drivers': drivers,
            'as_of': str(data.as_of), 'numeric_owner': 'Python', 'llm_owner': 'evidence_review'}
        return response

    def _drivers(self, rows, section, values, sources):
        for key, value in values.items():
            rows.append({'Phần': section, 'Yếu tố': self.FACTORS.get(key, key), 'Đóng góp điểm': value,
                'Hướng': 'Hỗ trợ' if value > 0 else 'Bất lợi' if value < 0 else 'Trung tính', 'Nguồn': ', '.join(sources) or 'Chưa có lineage'})
