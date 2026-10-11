from __future__ import annotations

import argparse

from stock_multi_agent.providers.universe_provider import VnstockInstrumentUniverseProvider
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.forward_evaluation import ForwardEvaluationRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real current data for a bounded forward-only evaluation.")
    parser.add_argument("tickers", nargs="+", help="Bounded list, for example: FPT VNM HPG MWG VCB SSI")
    parser.add_argument("--horizon", default="3 months")
    args = parser.parse_args()
    # A fresh listing capture is required; a stale current universe must not be
    # retroactively labelled as today's membership observation.
    provider = VnstockInstrumentUniverseProvider(cache_ttl_seconds=0)
    capture, destination = ForwardEvaluationRecorder(AnalysisService(), provider).capture(args.tickers, args.horizon)
    print(f"Forward capture written: {destination}")
    print(f"Signals: {len(capture.signals)}; price observations: {len(capture.prices)}; failures: {len(capture.failures)}")
    for failure in capture.failures:
        print(f"- {failure}")


if __name__ == "__main__":
    main()
