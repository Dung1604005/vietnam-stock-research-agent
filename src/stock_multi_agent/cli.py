from __future__ import annotations

import argparse
import json

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.services.analysis_service import AnalysisService


def main() -> None:
    load_workspace_env()
    parser = argparse.ArgumentParser(description="Six-agent Vietnamese stock analysis demo")
    parser.add_argument("ticker", nargs='?', help="Vietnamese stock ticker, e.g. FPT")
    parser.add_argument('--resume-run', help='Continue a saved analysis run UUID without replaying completed nodes')
    parser.add_argument("--horizon", default="3 months")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum evidence-producing re-analysis rounds (default: 2)",
    )
    args = parser.parse_args()
    if bool(args.ticker) == bool(args.resume_run):
        parser.error('Provide a ticker or --resume-run UUID')
    try:
        service = AnalysisService()
        state = service.resume_analysis(args.resume_run) if args.resume_run else service.analyze(args.ticker, args.horizon, args.max_iterations)
    except Exception as exc:
        run_id = getattr(exc, 'analysis_run_id', None)
        continuation = f' Resume with: stock-agents --resume-run {run_id}' if run_id else ''
        parser.exit(1, f'Analysis unavailable ({type(exc).__name__}).{continuation}\n')
    if state.final_decision is None:
        raise RuntimeError("Graph finished without a final decision")
    print(json.dumps(state.final_decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f'\nAnalysis run: {state.run_id}')
    print("\nExecution trace")
    for event in state.trace:
        tool = f" [{event.tool}]" if event.tool else ""
        print(f"- {event.agent.value:11} {event.phase:7} {event.action}{tool}: {event.detail}")


if __name__ == "__main__":
    main()

