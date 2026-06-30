"""Build a :class:`RaceState` snapshot from the lap-feature table.

Shared by the replay harness (Phase 5), the backtest, and the Streamlit UI so
they all assemble the simulator's input the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ELO_INITIAL
from src.sim.monte_carlo import DriverState, RaceState


# The measured on-track pass rate (genuine order-flips per green car-lap) is itself
# a sound per-lap pass probability: a car running close to the one ahead converts
# at roughly this rate, so over a full stint it matches the real number of passes.
# Hence scale 1.0. The floor is deliberately tiny so a near-unpassable circuit
# (Monaco ~0.007) stays sticky over a full race instead of compounding to a
# certain pass; easy circuits (Spa ~0.05) remain freely passable.
_OVERTAKE_SCALE = 1.0
_OVERTAKE_FALLBACK = 0.06
_OVERTAKE_CLIP = (0.005, 0.6)


def track_overtake_prob(event_laps: pd.DataFrame) -> float:
    """Estimate how easy on-track passing is at this circuit, from real races.

    Counts genuine on-track overtakes: pairs of cars whose running order flips
    between consecutive laps while *both* are on track (not pitting, not under a
    safety car / VSC). The earlier version counted any position a car gained
    versus the previous lap, which also captured cars inheriting positions when a
    rival pitted or retired -- at a street circuit like Monaco that pit/DNF churn
    is most of the apparent "overtaking", so it badly overstated how passable the
    track is. Restricting to order-flips between cars that both stayed out removes
    that confound. The rate is scaled into a per-lap pass probability.
    """
    df = event_laps.copy()
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    in_pit = pd.to_numeric(df.get("in_pit"), errors="coerce").fillna(0).astype(int)
    neut = (pd.to_numeric(df.get("sc_active"), errors="coerce").fillna(0).astype(int)
            | pd.to_numeric(df.get("vsc_active"), errors="coerce").fillna(0).astype(int))
    df["_block"] = (in_pit > 0) | (neut > 0)

    overtakes = 0
    car_laps = 0
    prev: dict[str, float] | None = None
    for _, cur_df in df.groupby("lap"):
        cur = dict(zip(cur_df["driver"], cur_df["position"]))
        blocked = dict(zip(cur_df["driver"], cur_df["_block"]))
        if prev is not None:
            elig = [d for d in cur
                    if d in prev and not blocked.get(d, True)
                    and pd.notna(cur[d]) and pd.notna(prev[d])]
            if len(elig) >= 2:
                pp = np.array([prev[d] for d in elig])
                cp = np.array([cur[d] for d in elig])
                # a pair (i, j) flips when i was behind j last lap and ahead now
                overtakes += int(((pp[:, None] > pp[None, :]) & (cp[:, None] < cp[None, :])).sum())
                car_laps += len(elig)
        prev = cur

    if car_laps == 0:
        return _OVERTAKE_FALLBACK
    rate = overtakes / car_laps
    return float(np.clip(rate * _OVERTAKE_SCALE, *_OVERTAKE_CLIP))


def event_keys(laps: pd.DataFrame) -> pd.DataFrame:
    """Distinct races in the feature table, chronological."""
    return (
        laps[["year", "round", "event", "total_laps"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )


def slice_event(laps: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    return laps[(laps["year"] == year) & (laps["round"] == rnd)].copy()


def state_at_lap(event_laps: pd.DataFrame, lap: int) -> RaceState:
    """Snapshot of one race at the end of ``lap``.

    A driver is *running* if they have a recorded lap at or beyond ``lap``.
    Drivers whose final lap is before ``lap`` are carried as retired so they are
    excluded from the win tally but still reported with P(win)=0.
    """
    ev = event_laps.iloc[0]
    total_laps = int(ev["total_laps"])
    drivers: list[DriverState] = []

    def _num(value, default: float) -> float:
        """Coerce to float, treating NaN/None/missing as ``default``.

        Plain ``x or default`` is unsafe here: NaN is truthy, so it slips through
        and crashes ``int(NaN)``. Older seasons have sporadic missing TyreLife,
        position, etc., so every numeric pull goes through this.
        """
        v = pd.to_numeric(value, errors="coerce")
        return float(default) if pd.isna(v) else float(v)

    for drv, g in event_laps.groupby("driver"):
        g = g.sort_values("lap")
        last_lap = int(g["lap"].max())
        running = last_lap >= lap
        # the row describing this driver at (or just before) the snapshot lap
        upto = g[g["lap"] <= lap]
        row = upto.iloc[-1] if not upto.empty else g.iloc[0]
        drivers.append(
            DriverState(
                driver=str(drv),
                position=_num(row.get("position"), 0.0),
                gap_to_leader=_num(row.get("gap_to_leader"), 0.0),
                compound=str(row.get("compound") or "MEDIUM"),
                tire_age=int(_num(row.get("tire_age_laps"), 0)),
                stint_number=int(_num(row.get("stint_number"), 1)),
                pit_count=int(_num(row.get("pit_count"), 0)),
                elo_pre=_num(row.get("elo_pre"), ELO_INITIAL),
                is_running=bool(running),
                team=str(row.get("team", "")),
            )
        )

    # weather at the snapshot lap (mean across cars on that lap)
    onlap = event_laps[event_laps["lap"] == lap]
    src = onlap if not onlap.empty else event_laps
    return RaceState(
        year=int(ev["year"]),
        round=int(ev["round"]),
        event=str(ev["event"]),
        current_lap=lap,
        total_laps=total_laps,
        drivers=drivers,
        track_temp=float(pd.to_numeric(src["track_temp"], errors="coerce").mean()),
        air_temp=float(pd.to_numeric(src["air_temp"], errors="coerce").mean()),
        rainfall=float(pd.to_numeric(src["rainfall"], errors="coerce").mean()),
        meta={"overtake_prob": track_overtake_prob(event_laps)},
    )


def actual_winner(event_laps: pd.DataFrame) -> str | None:
    """Driver classified P1: the runner who completed the final lap in P1."""
    final = event_laps[event_laps["lap"] == event_laps["lap"].max()]
    finishers = final[final["did_finish"]]
    pool = finishers if not finishers.empty else final
    pool = pool.sort_values("position")
    return str(pool.iloc[0]["driver"]) if not pool.empty else None
