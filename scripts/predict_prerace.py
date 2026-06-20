"""Pre-race win-probability forecast from the qualifying grid only.

The replay / backtest / live paths all consume *race* laps. This script fills the
remaining gap: a genuine **pre-race** prediction that uses only information
available once qualifying is over — the starting grid, each driver's leak-free
``elo_pre`` rating going into the race, and the circuit's historical overtaking
character. No lap of the race itself is read, so it is a true forecast.

It assembles a lap-0 :class:`RaceState` (everyone on the grid, fresh tyres) and
hands it to the same Monte Carlo simulator used everywhere else, so the P(win)
numbers are produced by identical machinery to the live/replay paths.

Usage:
    python scripts/predict_prerace.py --year 2026 --round 7
    python scripts/predict_prerace.py --year 2026 --round 7 --runs 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import ELO_HISTORY_PATH, ELO_INITIAL, LAP_FEATURES_PATH, init_cache  # noqa: E402
from src.features.raw_io import Event, load  # noqa: E402
from src.models.predictors import Predictors  # noqa: E402
from src.sim.monte_carlo import DriverState, MonteCarlo, RaceState  # noqa: E402
from src.sim.state import track_overtake_prob  # noqa: E402

# Same circuit, different event name across eras (e.g. Barcelona-Catalunya was the
# "Spanish Grand Prix" 2019-2025 and the "Barcelona Grand Prix" in 2026, while the
# 2026 "Spanish Grand Prix" is the new Madrid track). Used only to source the
# historical overtaking estimate and lap count from the right circuit.
CIRCUIT_ALIASES = {
    "Barcelona_Grand_Prix": ["Spanish_Grand_Prix", "Barcelona_Grand_Prix"],
}


def _event_dir(year: int, rnd: int) -> Event:
    from src.config import RAW_DIR

    matches = sorted(RAW_DIR.glob(f"{year}_{rnd:02d}_*"))
    if not matches:
        raise FileNotFoundError(f"No raw session dir for {year} round {rnd}. Fetch it first.")
    d = matches[0]
    slug = d.name.split("_", 2)[2]
    return Event(year=year, round=rnd, slug=slug, path=d)


def _grid_from_qualifying(ev: Event) -> pd.DataFrame:
    """Starting grid as (driver, team, grid_pos), ordered front to back.

    Uses the official ``GridPosition`` when present (it folds in penalties); else
    falls back to the qualifying classification order, which is the best pre-race
    information available when grid penalties are not yet published.
    """
    q = load(ev, "Q", "results")
    if q is None or q.empty:
        raise FileNotFoundError(f"No qualifying results for {ev.key}.")
    q = q.copy()
    grid = pd.to_numeric(q.get("GridPosition"), errors="coerce")
    qpos = pd.to_numeric(q.get("Position"), errors="coerce")
    # GridPosition is often unset (0/NaN) in a freshly-fetched quali file; prefer
    # it only when it carries real, distinct positions.
    use_grid = grid.notna().sum() == len(q) and grid.gt(0).all() and grid.nunique() == len(q)
    q["_grid"] = grid if use_grid else qpos
    q = q.dropna(subset=["_grid"]).sort_values("_grid")
    team_col = "TeamName" if "TeamName" in q.columns else None
    return pd.DataFrame({
        "driver": q["Abbreviation"].astype(str).values,
        "team": (q[team_col].astype(str).values if team_col else ""),
        "grid_pos": q["_grid"].astype(int).values,
    })


def _elo_pre(year: int, rnd: int) -> dict[str, float]:
    """Leak-free rating entering this race.

    Prefers the ``elo_pre`` recorded for exactly this (year, round). If the race
    is not yet in the Elo history (true pre-race), falls back to each driver's
    most recent ``elo_post`` from any earlier race.
    """
    elo = pd.read_parquet(ELO_HISTORY_PATH)
    here = elo[(elo.year == year) & (elo["round"] == rnd)]
    if not here.empty:
        return dict(zip(here["driver"], here["elo_pre"]))
    prior = elo[(elo.year < year) | ((elo.year == year) & (elo["round"] < rnd))]
    latest = prior.sort_values(["year", "round"]).groupby("driver").tail(1)
    return dict(zip(latest["driver"], latest["elo_post"]))


def _circuit_history(laps: pd.DataFrame, slug: str, year: int) -> pd.DataFrame:
    """All prior runnings of this circuit (handles cross-era event renames)."""
    names = CIRCUIT_ALIASES.get(slug, [slug])
    hist = laps[laps["event"].isin(names) & (laps["year"] < year)]
    return hist


def _qualifying_weather(ev: Event) -> dict[str, float]:
    w = load(ev, "Q", "weather")
    out = {"track_temp": float("nan"), "air_temp": float("nan"), "rainfall": 0.0}
    if w is not None and not w.empty:
        if "TrackTemp" in w:
            out["track_temp"] = float(pd.to_numeric(w["TrackTemp"], errors="coerce").mean())
        if "AirTemp" in w:
            out["air_temp"] = float(pd.to_numeric(w["AirTemp"], errors="coerce").mean())
        if "Rainfall" in w:
            out["rainfall"] = float(pd.to_numeric(w["Rainfall"], errors="coerce").fillna(0).mean())
    return out


def build_prerace_state(
    year: int,
    rnd: int,
    spacing: float = 1.0,
    start_compound: str = "MEDIUM",
) -> RaceState:
    ev = _event_dir(year, rnd)
    grid = _grid_from_qualifying(ev)
    elo = _elo_pre(year, rnd)
    laps = pd.read_parquet(LAP_FEATURES_PATH)

    hist = _circuit_history(laps, ev.slug, year)
    if not hist.empty:
        total_laps = int(hist.sort_values("year").groupby(["year", "round"])
                         ["total_laps"].first().iloc[-1])
        overtake_prob = float(
            pd.Series([track_overtake_prob(g) for _, g in hist.groupby(["year", "round"])]).mean()
        )
    else:  # no circuit history — fall back to this event's own (scheduled) lap count
        own = laps[(laps.year == year) & (laps["round"] == rnd)]
        total_laps = int(own["total_laps"].iloc[0]) if not own.empty else 0
        overtake_prob = 0.30

    wx = _qualifying_weather(ev)

    drivers = [
        DriverState(
            driver=r.driver,
            position=float(r.grid_pos),
            gap_to_leader=float((r.grid_pos - 1) * spacing),
            compound=start_compound,
            tire_age=0,
            stint_number=1,
            pit_count=0,
            elo_pre=float(elo.get(r.driver, ELO_INITIAL)),
            is_running=True,
            team=r.team,
        )
        for r in grid.itertuples(index=False)
    ]
    return RaceState(
        year=year, round=rnd, event=ev.slug, current_lap=0, total_laps=total_laps,
        drivers=drivers, track_temp=wx["track_temp"], air_temp=wx["air_temp"],
        rainfall=wx["rainfall"], meta={"overtake_prob": overtake_prob},
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-race win forecast from the qualifying grid.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--runs", type=int, default=10000)
    p.add_argument("--spacing", type=float, default=1.0,
                   help="grid time gap per position (s); sets starting track order")
    p.add_argument("--start-compound", default="MEDIUM")
    p.add_argument("--top", type=int, default=22)
    args = p.parse_args()

    init_cache()
    pred = Predictors.load()
    if not pred.is_ready:
        print("Models not trained — run `python -m src.models.train` first.")
        return 1

    state = build_prerace_state(args.year, args.round, args.spacing, args.start_compound)
    mc = MonteCarlo(pred, n_runs=args.runs)
    res = mc.simulate(state)

    print(f"\nPre-race forecast — {args.year} round {args.round} ({state.event})")
    print(f"{state.total_laps} laps · {len(state.drivers)} cars · {args.runs:,} simulations · "
          f"overtake_prob={state.meta['overtake_prob']:.2f} · "
          f"track {state.track_temp:.0f}°C air {state.air_temp:.0f}°C rain {state.rainfall:.2f}")
    grid_pos = {d.driver: int(d.position) for d in state.drivers}
    print(f"\n{'#':>2}  {'drv':<4} {'grid':>4}  {'P(win)':>7}  {'P(podium)':>9}  {'P(finish)':>9}")
    for i, (drv, pw) in enumerate(res.ranked(), 1):
        print(f"{i:>2}  {drv:<4} {grid_pos.get(drv,0):>4}  {pw:>6.1%}  "
              f"{res.p_podium.get(drv,0):>8.1%}  {res.p_finish.get(drv,0):>8.1%}")
        if i >= args.top:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
