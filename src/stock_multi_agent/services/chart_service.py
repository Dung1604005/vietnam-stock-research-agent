from __future__ import annotations

import numpy as np
import pandas as pd
import re
import unicodedata

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.chart import ChartArtifact, ChartRequest, ChartType
from stock_multi_agent.models.chat import ParsedQuery
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit
from stock_multi_agent.services.financial_charts import FINANCIAL_TYPES, financial_artifact, render_financial


class ChartService:
    """Turns safe chart specifications into data and Plotly figures."""

    def __init__(self, toolkit: TechnicalIndicatorToolkit | None = None) -> None:
        self.tools = toolkit or TechnicalIndicatorToolkit()

    @staticmethod
    def _chart_matches(raw):
        text = ''.join(char for char in unicodedata.normalize('NFD', raw.casefold().replace('đ', 'd')) if unicodedata.category(char) != 'Mn')
        patterns = {
            ChartType.REVENUE_PROFIT: r'\b(?:doanh thu|loi nhuan|revenue|net profit)\b',
            ChartType.CASH_FLOW: r'\b(?:dong tien|cash flow)\b',
            ChartType.GROWTH: r'\b(?:tang truong|growth)\b',
            ChartType.VALUATION: r'\b(?:p/?e|p/?b|dinh gia|valuation)\b',
            ChartType.PROFITABILITY: r'\b(?:roe|roa|bien loi nhuan|sinh loi)\b',
            ChartType.BOLLINGER_BANDS: r'\bbollinger\b',
            ChartType.DRAWDOWN: r'\b(?:drawdown|sut giam|giam tu dinh)\b',
            ChartType.VOLATILITY: r'\b(?:volatility|bien dong)\b',
            ChartType.RETURNS: r'\b(?:loi suat|returns?)\b',
            ChartType.RSI: r'\brsi\b', ChartType.MACD: r'\bmacd\b',
            ChartType.PRICE_MA: r'\b(?:s?ma\s*(?:20|50|100|200)|duong trung binh)\b',
        }
        matches = [kind for kind, pattern in patterns.items() if re.search(pattern, text)]
        if ChartType.GROWTH in matches:
            matches = [kind for kind in matches if kind != ChartType.REVENUE_PROFIT]
        return matches

    def requests_from_query(self, query, ticker=None):
        if re.search(r'\bdashboard\b', query.raw_query, re.I):
            kinds = [ChartType.PRICE_MA, ChartType.PRICE_VOLUME, ChartType.RSI, ChartType.DRAWDOWN]
        else:
            kinds = self._chart_matches(query.raw_query)
        if not kinds:
            return [self.request_from_query(query, ticker)]
        requests = [ChartRequest(ticker=ticker or query.tickers[0], chart_type=kind, period=query.chart_period or '6M',
            indicators=list(query.requested_indicators)) for kind in kinds[:6]]
        text = ''.join(char for char in unicodedata.normalize('NFD', query.raw_query.casefold().replace('đ', 'd')) if unicodedata.category(char) != 'Mn')
        for request in requests:
            if request.chart_type not in FINANCIAL_TYPES:
                continue
            if re.search(r'\b(?:theo quy|hang quy|quarterly|quarter)\b', text):
                request.indicators.append('QUARTER')
            elif re.search(r'\b(?:theo nam|hang nam|annual|yearly)\b', text):
                request.indicators.append('ANNUAL')
            duration = re.search(r'\b(\d{1,2})\s*(nam|quy|years?|quarters?)\b', text)
            if duration:
                request.period = f'{max(1, min(20, int(duration[1])))}' + ('Q' if duration[2] in {'quy', 'quarter', 'quarters'} else 'Y')
            elif re.search(r'\b(?:toan bo|tat ca lich su|all history)\b', text):
                request.period = 'ALL'
        return requests

    def request_from_query(self, query: ParsedQuery, ticker: str | None = None) -> ChartRequest:
        if not query.tickers and ticker is None:
            raise ValueError("A ticker is required for a chart")
        indicators = query.requested_indicators
        raw = query.raw_query.lower()
        explicit = self._chart_matches(query.raw_query)
        if explicit:
            chart_type = explicit[0]
        elif "RSI" in indicators:
            chart_type = ChartType.RSI
        elif "MACD" in indicators:
            chart_type = ChartType.MACD
        elif "CANDLESTICK" in indicators:
            chart_type = ChartType.CANDLESTICK
        elif any(item.startswith("MA") for item in indicators):
            chart_type = ChartType.PRICE_MA
        elif "VOLUME" in indicators and any(word in raw for word in ("giá", "gia", "price")):
            chart_type = ChartType.PRICE_VOLUME
        elif "VOLUME" in indicators:
            chart_type = ChartType.VOLUME
        else:
            chart_type = ChartType.PRICE
        return ChartRequest(
            ticker=ticker or query.tickers[0],
            chart_type=chart_type,
            period=query.chart_period or "6M",
            indicators=indicators,
        )

    def build_artifact(self, request: ChartRequest, market_data: MarketData) -> ChartArtifact:
        if request.ticker != market_data.ticker:
            raise ValueError('Chart ticker and data ticker differ')
        if request.chart_type in FINANCIAL_TYPES:
            return financial_artifact(request, market_data)
        frame = pd.DataFrame([bar.model_dump() for bar in market_data.bars])
        frame["date"] = pd.to_datetime(frame["date"])
        closes = frame["close"].to_numpy(dtype=float)
        for window in (20, 50, 100, 200):
            frame[f'ma{window}'] = self.tools.sma_series(closes, window) if len(closes) >= window else np.nan
        frame["rsi"] = self.tools.rsi_series(closes) if len(closes) >= 15 else np.nan
        frame.loc[:13, 'rsi'] = np.nan
        macd, signal, histogram = self.tools.macd_series(closes) if len(closes) >= 35 else (np.full(len(closes), np.nan),) * 3
        frame["macd"] = macd
        frame["macd_signal"] = signal
        frame["macd_histogram"] = histogram
        average = frame['close'].rolling(20, min_periods=20).mean()
        deviation = frame['close'].rolling(20, min_periods=20).std(ddof=0)
        frame['bb_middle'], frame['bb_upper'], frame['bb_lower'] = average, average + 2 * deviation, average - 2 * deviation
        frame['drawdown'] = (frame['close'] / frame['close'].cummax() - 1) * 100
        frame['return_pct'] = frame['close'].pct_change(fill_method=None) * 100
        frame['volatility'] = np.log(frame['close']).diff().rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252) * 100
        frame = frame.tail(self._period_sessions(request.period)).copy()
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        frame = frame.replace({np.nan: None})
        warnings = list(market_data.warnings)
        if len(closes) < 50:
            warnings.append('Lịch sử ngắn: chỉ báo chưa đủ warm-up được để trống, không điền số trung tính.')
        last_date = market_data.bars[-1].date
        return ChartArtifact(
            request=request,
            title=f"{request.ticker} · {request.period} · {request.chart_type.value}",
            records=frame.to_dict(orient="records"),
            data_timestamp=last_date,
            data_fetched_at=market_data.fetched_at,
            source=market_data.source,
            source_details=market_data.source_refs,
            warnings=warnings,
            axis_unit=market_data.price_unit,
            explanation={ChartType.BOLLINGER_BANDS: 'MA20 ± 2 độ lệch chuẩn giá; chạm dải không tự là tín hiệu mua/bán.',
                ChartType.DRAWDOWN: 'close / đỉnh lũy kế - 1 (%); đỉnh tính từ đầu lịch sử được tải, không phải đỉnh mọi thời đại.',
                ChartType.VOLATILITY: 'Độ lệch chuẩn lợi suất log 20 phiên × √252 (% năm hóa); là biến động lịch sử.',
                ChartType.RETURNS: 'Lợi suất phiên close_t / close_t-1 - 1 (%), không phải dự báo.'}.get(request.chart_type, 'Chỉ báo tính bằng Python trên dữ liệu giá có nguồn.'),
        )

    def render_figure(self, artifact: ChartArtifact):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        if artifact.request.chart_type in FINANCIAL_TYPES:
            return render_financial(artifact)

        frame = pd.DataFrame(artifact.records)
        request = artifact.request
        if request.chart_type == ChartType.PRICE_VOLUME:
            figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
            figure.add_trace(go.Scatter(x=frame["date"], y=frame["close"], name="Close", line={"width": 2}), row=1, col=1)
            figure.add_trace(go.Bar(x=frame["date"], y=frame["volume"], name="Volume", opacity=0.65), row=2, col=1)
        else:
            figure = go.Figure()
            if request.chart_type == ChartType.CANDLESTICK:
                figure.add_trace(go.Candlestick(
                    x=frame["date"], open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"], name=request.ticker
                ))
            elif request.chart_type == ChartType.PRICE_MA:
                figure.add_trace(go.Scatter(x=frame["date"], y=frame["close"], name="Close"))
                requested = request.indicators or ["MA20", "MA50"]
                if not any(item.startswith('MA') for item in requested):
                    requested = ['MA20', 'MA50']
                for window in (20, 50, 100, 200):
                    if f'MA{window}' in requested:
                        figure.add_trace(go.Scatter(x=frame['date'], y=frame[f'ma{window}'], name=f'MA{window}'))
            elif request.chart_type == ChartType.VOLUME:
                figure.add_trace(go.Bar(x=frame["date"], y=frame["volume"], name="Volume"))
            elif request.chart_type == ChartType.RSI:
                figure.add_trace(go.Scatter(x=frame["date"], y=frame["rsi"], name="RSI14"))
                figure.add_hline(y=70, line_dash="dash", line_color="#ef4444")
                figure.add_hline(y=30, line_dash="dash", line_color="#22c55e")
                figure.update_yaxes(range=[0, 100])
            elif request.chart_type == ChartType.MACD:
                figure.add_trace(go.Scatter(x=frame["date"], y=frame["macd"], name="MACD"))
                figure.add_trace(go.Scatter(x=frame["date"], y=frame["macd_signal"], name="Signal"))
                figure.add_trace(go.Bar(x=frame["date"], y=frame["macd_histogram"], name="Histogram", opacity=0.55))
            elif request.chart_type == ChartType.BOLLINGER_BANDS:
                for column, label in [('bb_upper', 'Dải trên'), ('bb_lower', 'Dải dưới'), ('bb_middle', 'MA20'), ('close', 'Giá đóng cửa')]:
                    figure.add_trace(go.Scatter(x=frame['date'], y=frame[column], name=label))
            elif request.chart_type in {ChartType.DRAWDOWN, ChartType.VOLATILITY, ChartType.RETURNS}:
                column, label = {ChartType.DRAWDOWN: ('drawdown', 'Sụt giảm từ đỉnh (%)'),
                    ChartType.VOLATILITY: ('volatility', 'Biến động năm hóa (%)'), ChartType.RETURNS: ('return_pct', 'Lợi suất phiên (%)')}[request.chart_type]
                figure.add_trace(go.Scatter(x=frame['date'], y=frame[column], name=label, fill='tozeroy' if request.chart_type == ChartType.DRAWDOWN else None))
                figure.update_yaxes(title_text='%')
            else:
                figure.add_trace(go.Scatter(x=frame["date"], y=frame["close"], name="Close", line={"width": 2}))
        figure.update_layout(
            title=artifact.title,
            height=430,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.05},
        )
        if request.chart_type in {ChartType.PRICE, ChartType.PRICE_MA, ChartType.CANDLESTICK, ChartType.BOLLINGER_BANDS}:
            figure.update_yaxes(title_text=artifact.axis_unit if artifact.axis_unit != 'UNKNOWN' else 'Đơn vị nguồn (chưa xác minh thang giá)')
        return figure

    @staticmethod
    def _period_sessions(period: str) -> int:
        value = period.upper()
        if value.endswith("M") and value[:-1].isdigit():
            return max(22, int(value[:-1]) * 22)
        if value.endswith("Y") and value[:-1].isdigit():
            return int(value[:-1]) * 252
        return 132

