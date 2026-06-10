"""Phase 0 smoke test.

Verifies the environment by:
  1. Initializing the FastF1 cache.
  2. Pulling a single past session (2024 Bahrain Race).
  3. Loading lap data and printing a tiny summary.

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fastf1

from src.config import init_cache


def main() -> int:
    init_cache()
    print("FastF1 version:", fastf1.__version__)

    session = fastf1.get_session(2024, "Bahrain", "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps
    if laps.empty:
        print("FAIL: no laps returned")
        return 1

    winner = laps[laps["LapNumber"] == laps["LapNumber"].max()].sort_values("Position").iloc[0]
    print(f"Loaded {len(laps)} laps across {laps['Driver'].nunique()} drivers")
    print(f"2024 Bahrain race winner per loaded data: {winner['Driver']}")
    print("Smoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
