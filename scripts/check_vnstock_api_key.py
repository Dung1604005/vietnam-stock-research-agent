from __future__ import annotations

import os

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.providers.market_provider import VnstockMarketDataProvider


def main() -> None:
    load_workspace_env()
    if not os.getenv("VNSTOCK_API_KEY", "").strip():
        raise SystemExit("Chưa có VNSTOCK_API_KEY trong .env. Hãy dán key vào rồi chạy lại.")

    # Configure vnstock/vnai storage under the D-drive project before importing vnai.
    provider = VnstockMarketDataProvider()
    from vnai import get_user_tier

    tier = get_user_tier()
    limits = tier.get("limits", {})
    print(
        "Vnstock API key đã được nhận: "
        f"tier={tier.get('tier', 'unknown')}, "
        f"limit={limits.get('per_minute', 'unknown')}/phút, "
        f"app_interval={provider.executor.policy.min_request_interval_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
