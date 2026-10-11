from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.price_forecast import PriceForecastService


T = TypeVar("T")


def load_workspace_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def timed(label: str, operation: Callable[[], T]) -> T:
    started = time.perf_counter()
    result = operation()
    print(f"{label}={time.perf_counter() - started:.3f}s", flush=True)
    return result


def main() -> None:
    load_workspace_env()
    ticker = sys.argv[1].strip().upper() if len(sys.argv) > 1 else "FPT"
    forecast = timed(
        "forecast_service_init",
        lambda: PriceForecastService.from_environment(default_enabled=True),
    )
    service = timed(
        "analysis_service_init",
        lambda: AnalysisService(price_forecast_service=forecast),
    )
    state = service._initial_state(  # noqa: SLF001 - explicit diagnostic script
        ticker,
        "3 months",
        max_iterations=3,
        required_data=["ohlcv", "financials", "benchmark"],
    )
    state = service._merge(state, timed("data_agent", lambda: service.data_agent.run(state)))  # noqa: SLF001
    state = service._merge(state, timed("forecast", lambda: service.graph._forecast(state)))  # noqa: SLF001
    state = service._merge(  # noqa: SLF001
        state,
        timed("technical_agent", lambda: service.technical_agent.run(state, extended=False)),
    )
    state = service._merge(  # noqa: SLF001
        state,
        timed("fundamental_agent", lambda: service.fundamental_agent.run(state)),
    )
    state = service._merge(state, timed("risk_agent", lambda: service.risk_agent.run(state)))  # noqa: SLF001
    state = service._merge(  # noqa: SLF001
        state,
        timed("manager_review", lambda: service.manager_agent.review(state)),
    )
    if state.reanalysis_requested:
        state = service._merge(  # noqa: SLF001
            state,
            timed("technical_reanalysis", lambda: service.technical_agent.run(state, extended=True)),
        )
        state = service._merge(  # noqa: SLF001
            state,
            timed("risk_reanalysis", lambda: service.risk_agent.run(state)),
        )
        state = service._merge(  # noqa: SLF001
            state,
            timed("manager_review_2", lambda: service.manager_agent.review(state)),
        )
    state = service._merge(  # noqa: SLF001
        state,
        timed("manager_finalize", lambda: service.manager_agent.finalize(state)),
    )
    print(
        "summary=",
        {
            "iteration": state.iteration,
            "ticker": ticker,
            "llm_reviews": [
                {
                    "agent": review.agent.value,
                    "status": review.status,
                    "assessment": review.assessment,
                }
                for review in state.llm_reviews
            ],
            "provider_health": service.data_agent.market_tool.provider.health(),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
