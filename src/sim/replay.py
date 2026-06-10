"""Phase 5: replay harness.

Feeds a past race to the simulator lap by lap, exactly as the live runner will
during an actual GP -- same ``state -> MonteCarlo.simulate`` path, only the source
of the state differs (Parquet here, FastF1 live timing in Phase 6). This lets us
validate the whole pipeline without waiting for a race weekend.
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from src.models.predictors import Predictors
from src.sim.monte_carlo import MonteCarlo, SimResult
from src.sim.state import actual_winner, slice_event, state_at_lap


def replay_race(
    laps: pd.DataFrame,
    year: int,
    rnd: int,
    mc: MonteCarlo,
    step: int = 1,
    start_lap: int = 1,
) -> Iterator[tuple[int, SimResult]]:
    """Yield ``(lap, SimResult)`` for each simulated lap of one race."""
    ev = slice_event(laps, year, rnd)
    if ev.empty:
        return
    total = int(ev["total_laps"].iloc[0])
    for lap in range(start_lap, total + 1, step):
        yield lap, mc.simulate(state_at_lap(ev, lap))


def replay_to_frame(
    laps: pd.DataFrame, year: int, rnd: int, mc: MonteCarlo, step: int = 1
) -> pd.DataFrame:
    """Full P(win) trajectory for a race as a tidy frame: lap x driver."""
    rows = []
    for lap, res in replay_race(laps, year, rnd, mc, step=step):
        for drv, p in res.p_win.items():
            rows.append({"lap": lap, "driver": drv, "p_win": p})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["winner"] = actual_winner(slice_event(laps, year, rnd))
    return df


def predicted_winner_at(res: SimResult) -> tuple[str, float]:
    ranked = res.ranked()
    return ranked[0] if ranked else ("", 0.0)


def main() -> int:
    import argparse

    from src.config import LAP_FEATURES_PATH

    p = argparse.ArgumentParser(description="Replay one race lap-by-lap.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--runs", type=int, default=3000)
    args = p.parse_args()

    laps = pd.read_parquet(LAP_FEATURES_PATH)
    mc = MonteCarlo(Predictors.load(), n_runs=args.runs)
    ev = slice_event(laps, args.year, args.round)
    if ev.empty:
        print(f"No data for {args.year} round {args.round}.")
        return 1
    truth = actual_winner(ev)
    print(f"Replaying {int(ev['year'].iloc[0])} {ev['event'].iloc[0]} -- actual winner: {truth}\n")
    print(f"{'lap':>4}  {'predicted top-3':40s}  leader")
    for lap, res in replay_race(laps, args.year, args.round, mc, step=args.step):
        top = res.ranked()[:3]
        s = ", ".join(f"{d} {p:.0%}" for d, p in top)
        hit = "<-- correct" if top and top[0][0] == truth else ""
        print(f"{lap:>4}  {s:40s}  {hit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
