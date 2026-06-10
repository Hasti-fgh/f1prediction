"""Build a :class:`RaceState` snapshot from the lap-feature table.

Shared by the replay harness (Phase 5), the backtest, and the Streamlit UI so
they all assemble the simulator's input the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ELO_INITIAL
from src.sim.monte_carlo import DriverState, RaceState


def track_overtake_prob(event_laps: pd.DataFrame) -> float:
    """Estimate how easy on-track passing is at this circuit, from the race itself.

    We count green-flag, non-pit laps where a driver *gained* a track position
    versus the previous lap, as a fraction of all car-laps. Processional circuits
    (Monaco) yield a tiny rate; high-overtaking ones (Spa, Monza) a large one.
    The rate is scaled into a per-lap pass probability used by the simulator.
    """
    df = event_laps.sort_values(["driver", "lap"]).copy()
    df["prev_pos"] = df.groupby("driver")["position"].shift(1)
    gained = (
        (pd.to_numeric(df["position"], errors="coerce") < pd.to_numeric(df["prev_pos"], errors="coerce"))
        & (df["in_pit"] == 0)
        & (df["sc_active"] == 0)
        & (df["vsc_active"] == 0)
    )
    rate = float(gained.sum()) / max(len(df), 1)  # overtakes per car-lap
    return float(np.clip(rate * 6.0, 0.03, 0.6))


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
