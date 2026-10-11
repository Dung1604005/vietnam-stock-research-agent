from __future__ import annotations

import json

from stock_multi_agent.nlu.telemetry import SQLiteSemanticTelemetryStore


def main() -> None:
    print(json.dumps(SQLiteSemanticTelemetryStore().summary(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
