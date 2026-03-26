"""Cassandra — Autonomous Multi-Strategy Crypto Trading Bot.

main.py is the startup daemon. In Phase 1 it performs a subsystem healthcheck:
verifies all modules import correctly and the testnet exchange is reachable.
Full swarm orchestration is wired in Phase 6.
"""
import sys

from dotenv import load_dotenv

load_dotenv()


def healthcheck() -> bool:
    """Verify all Phase 1 subsystems import and the testnet exchange is reachable.

    Returns:
        True if all checks pass, False otherwise.
    """
    passed = True

    print("[1/3] Checking ccxt client...")
    try:
        from execution.ccxt_client import build_exchange, fetch_balance
        exchange = build_exchange()
        balance = fetch_balance(exchange)
        print(f"      OK — testnet USDT balance: {balance:.2f}")
    except Exception as exc:
        print(f"      FAIL — {exc}")
        passed = False

    print("[2/3] Checking VectorBT engine...")
    try:
        from backtest.vectorbt_engine import load_ohlcv_from_df, run_buy_and_hold  # noqa: F401
        print("      OK — VectorBT engine imported.")
    except Exception as exc:
        print(f"      FAIL — {exc}")
        passed = False

    print("[3/3] Checking agent memory files...")
    import pathlib
    for path in ["agents/memory/failure_log.md", "agents/memory/PENDING_UPGRADES.md"]:
        if pathlib.Path(path).exists():
            print(f"      OK — {path}")
        else:
            print(f"      FAIL — {path} missing")
            passed = False

    return passed


if __name__ == "__main__":
    print("=== Cassandra Phase 1 Healthcheck ===\n")
    ok = healthcheck()
    print(f"\n{'=== ALL SYSTEMS GO ===' if ok else '=== HEALTHCHECK FAILED ==='}")
    sys.exit(0 if ok else 1)
