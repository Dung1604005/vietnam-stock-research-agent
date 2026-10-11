from __future__ import annotations

import argparse
import json

from stock_multi_agent.services.analysis_service import AnalysisService


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-agent Vietnamese stock analysis demo")
    parser.add_argument("ticker", help="Vietnamese stock ticker, e.g. FPT")
    parser.add_argument("--horizon", default="3 months")
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()
    state = AnalysisService().analyze(args.ticker, args.horizon, args.max_iterations)
    if state.final_decision is None:
        raise RuntimeError("Graph finished without a final decision")
    print(json.dumps(state.final_decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print("\nExecution trace")
    for event in state.trace:
        tool = f" [{event.tool}]" if event.tool else ""
        print(f"- {event.agent.value:11} {event.phase:7} {event.action}{tool}: {event.detail}")


if __name__ == "__main__":
    main()

