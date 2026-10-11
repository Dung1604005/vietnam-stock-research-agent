from __future__ import annotations

import numpy as np
import pandas as pd

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.chart import ChartArtifact, ChartRequest, ChartType
from stock_multi_agent.models.chat import ParsedQuery
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


class ChartService:
    """Turns safe chart specifications into data and Plotly figures."""

    def __init__(self, toolkit: TechnicalIndicatorToolkit | None = None) -> None:
        self.tools = toolkit or TechnicalIndicatorToolkit()

    def request_from_query(self, query: ParsedQuery, ticker: str | None = None) -> ChartRequest:
        if not query.tickers and ticker is None:
            raise ValueError("A ticker is required for a chart")
        indicators = query.requested_indicators
        raw = query.raw_query.lower()
        if "RSI" in indicators:
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
        frame = pd.DataFrame([bar.model_dump() for bar in market_data.bars])
        frame["date"] = pd.to_datetime(frame["date"])
        closes = frame["close"].to_numpy(dtype=float)
        frame["ma20"] = self.tools.sma_series(closes, 20)
        frame["ma50"] = self.tools.sma_series(closes, 50)
        frame["rsi"] = self.tools.rsi_series(closes)
        macd, signal, histogram = self.tools.macd_series(closes)
        frame["macd"] = macd
        frame["macd_signal"] = signal
        frame["macd_histogram"] = histogram
        frame = frame.tail(self._period_sessions(request.period)).copy()
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        frame = frame.replace({np.nan: None})
        warnings = list(market_data.warnings)
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
        )

    def render_figure(self, artifact: ChartArtifact):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

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
                if "MA20" in requested:
                    figure.add_trace(go.Scatter(x=frame["date"], y=frame["ma20"], name="MA20"))
                if "MA50" in requested:
                    figure.add_trace(go.Scatter(x=frame["date"], y=frame["ma50"], name="MA50"))
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
            else:
                figure.add_trace(go.Scatter(x=frame["date"], y=frame["close"], name="Close", line={"width": 2}))
        figure.update_layout(
            title=artifact.title,
            height=430,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.05},
        )
        return figure

    @staticmethod
    def _period_sessions(period: str) -> int:
        value = period.upper()
        if value.endswith("M") and value[:-1].isdigit():
            return max(22, int(value[:-1]) * 22)
        if value.endswith("Y") and value[:-1].isdigit():
            return int(value[:-1]) * 252
        return 132

