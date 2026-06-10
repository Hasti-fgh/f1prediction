"""Phase 2: build the lap-level feature table.

Reads the raw Parquet lake (Phase 1) and produces a single tidy table with one
row per (race, driver, lap), carrying exactly the race-situation features the
Monte Carlo simulator and ML estimators consume. Driver identity never enters as
a feature -- skill is folded in as the continuous ``elo_pre`` rating from
``src.features.elo``.

Run:  python -m src.features.build_features
Output: data/features/lap_features.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import COMPOUNDS, ELO_INITIAL, LAP_FEATURES_PATH  # noqa: E402
from src.features.elo import build_and_save as build_elo  # noqa: E402
from src.features.raw_io import Event, has_race, list_events, load  # noqa: E402

# TrackStatus is a string of concatenated single-digit flags seen during a lap:
#   1=clear 2=yellow 4=safety-car 5=red-flag 6=VSC-deployed 7=VSC-ending


def _norm_compound(c: object) -> str:
    s = str(c).upper().strip() if c is not None and str(c) != "nan" else "UNKNOWN"
    return s if s in COMPOUNDS else ("UNKNOWN" if s in ("", "NAN", "NONE") else "OTHER")


def _track_flags(ts: pd.Series) -> pd.DataFrame:
    s = ts.fillna("").astype(str)
    return pd.DataFrame(
        {
            "sc_active": s.str.contains("4").astype("int8"),
            "vsc_active": (s.str.contains("6") | s.str.contains("7")).astype("int8"),
            "red_flag": s.str.contains("5").astype("int8"),
            "yellow_flag": s.str.contains("2").astype("int8"),
        },
        index=ts.index,
    )


def _is_finisher(classified_position: object) -> bool:
    """A driver is classified (finished or lapped) iff ClassifiedPosition is numeric.

    FastF1 puts a number there for everyone who took the flag and a letter
    (R=retired, D=disqualified, W=withdrew, ...) for those who did not. This is
    more reliable than ``Status`` text, where "Lapped" is a *finisher* and only
    "Retired"/"Did not start"/"Disqualified" are genuine non-classifications.
    """
    s = str(classified_position).strip()
    try:
        float(s)
        return True
    except ValueError:
        return False


def _attach_weather(laps: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    cols = {"track_temp": np.nan, "air_temp": np.nan, "rainfall": 0.0, "humidity": np.nan}
    if weather is None or weather.empty or "Time_s" not in weather.columns:
        for c, v in cols.items():
            laps[c] = v
        return laps
    w = weather.sort_values("Time_s")[
        [c for c in ("Time_s", "TrackTemp", "AirTemp", "Rainfall", "Humidity") if c in weather.columns]
    ].copy()
    left = laps.sort_values("Time_s")
    merged = pd.merge_asof(
        left, w, on="Time_s", direction="nearest", tolerance=600.0
    )
    merged["track_temp"] = merged.get("TrackTemp", np.nan)
    merged["air_temp"] = merged.get("AirTemp", np.nan)
    merged["rainfall"] = pd.to_numeric(merged.get("Rainfall"), errors="coerce").fillna(0.0)
    merged["humidity"] = merged.get("Humidity", np.nan)
    return merged.drop(columns=[c for c in ("TrackTemp", "AirTemp", "Rainfall", "Humidity") if c in merged.columns])


def features_from_frames(
    laps: pd.DataFrame | None,
    results: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    year: int,
    rnd: int,
    slug: str,
    elo_lookup: dict[str, float],
) -> pd.DataFrame | None:
    """Core feature builder over already-loaded frames.

    Shared by the offline lap-table builder and the live runner so a race scored
    live produces byte-for-byte the same columns as one scored from the lake.
    """
    if laps is None or laps.empty:
        return None
    ev = Event(year=year, round=rnd, slug=slug, path=Path("."))
    return _compute_features(laps, results, weather, ev, elo_lookup)


def _build_one_race(ev: Event, elo_lookup: dict[str, float]) -> pd.DataFrame | None:
    laps = load(ev, "R", "laps")
    if laps is None or laps.empty:
        return None
    return _compute_features(laps, load(ev, "R", "results"), load(ev, "R", "weather"), ev, elo_lookup)


def _compute_features(
    laps: pd.DataFrame,
    results: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    ev: Event,
    elo_lookup: dict[str, float],
) -> pd.DataFrame | None:
    if laps is None or laps.empty:
        return None

    laps = laps.copy()
    laps = laps[laps["LapNumber"].notna()]
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    total_laps = int(laps["LapNumber"].max())

    # --- gaps: session time relative to the lap leader -----------------------
    laps["pos"] = pd.to_numeric(laps["Position"], errors="coerce")
    leader_time = laps[laps["pos"] == 1].set_index("LapNumber")["Time_s"]
    laps["leader_time_s"] = laps["LapNumber"].map(leader_time)
    laps["gap_to_leader"] = (laps["Time_s"] - laps["leader_time_s"]).clip(lower=0)

    # gap to the car immediately ahead, within each lap by session time
    laps = laps.sort_values(["LapNumber", "Time_s"])
    laps["gap_to_ahead"] = laps.groupby("LapNumber")["Time_s"].diff().fillna(0.0).clip(lower=0)
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    # --- pit / stint ---------------------------------------------------------
    laps["in_pit"] = ((laps["PitInTime_s"].notna()) | (laps["PitOutTime_s"].notna())).astype("int8")
    laps["pit_count"] = laps.groupby("Driver")["PitInTime_s"].apply(
        lambda s: s.notna().cumsum()
    ).reset_index(level=0, drop=True).astype("int16")
    laps["stint_number"] = pd.to_numeric(laps["Stint"], errors="coerce").fillna(1).astype("int16")
    laps["tire_age_laps"] = pd.to_numeric(laps["TyreLife"], errors="coerce")
    laps["compound"] = laps["Compound"].map(_norm_compound)

    # --- track-status flags + laps since last neutralisation -----------------
    flags = _track_flags(laps["TrackStatus"])
    laps = pd.concat([laps, flags], axis=1)
    neutralised = (laps["sc_active"] | laps["vsc_active"]).astype("int8")

    def _laps_since(group_neut: pd.Series) -> pd.Series:
        out = np.empty(len(group_neut), dtype="int32")
        counter = 9999
        for i, n in enumerate(group_neut.to_numpy()):
            counter = 0 if n else counter + 1
            out[i] = counter
        return pd.Series(out, index=group_neut.index)

    laps["_neut"] = neutralised
    laps["laps_since_sc"] = (
        laps.groupby("Driver", group_keys=False)["_neut"].apply(_laps_since)
    )

    # --- pace + race-state context ------------------------------------------
    laps["lap_time_s"] = pd.to_numeric(laps["LapTime_s"], errors="coerce")
    laps["sector1_s"] = pd.to_numeric(laps.get("Sector1Time_s"), errors="coerce")
    laps["sector2_s"] = pd.to_numeric(laps.get("Sector2Time_s"), errors="coerce")
    laps["sector3_s"] = pd.to_numeric(laps.get("Sector3Time_s"), errors="coerce")
    laps["laps_remaining"] = total_laps - laps["LapNumber"]
    laps["race_progress"] = laps["LapNumber"] / total_laps

    # --- weather -------------------------------------------------------------
    laps = _attach_weather(laps, weather)

    # --- DNF / is_running ----------------------------------------------------
    last_lap = laps.groupby("Driver")["LapNumber"].transform("max")
    finisher = {}
    grid = {}
    if results is not None and not results.empty and "Abbreviation" in results.columns:
        for _, r in results.iterrows():
            finisher[r["Abbreviation"]] = _is_finisher(r.get("ClassifiedPosition"))
            grid[r["Abbreviation"]] = pd.to_numeric(pd.Series([r.get("GridPosition")]), errors="coerce").iloc[0]
    laps["did_finish"] = laps["Driver"].map(finisher).fillna(False).astype(bool)
    laps["grid_position"] = laps["Driver"].map(grid)
    laps["is_running"] = 1  # every observed lap was actually run
    # The retirement lap for a DNF driver is their final observed lap.
    laps["dnf_this_lap"] = (
        (~laps["did_finish"]) & (laps["LapNumber"] == last_lap)
    ).astype("int8")

    # --- driver skill --------------------------------------------------------
    laps["elo_pre"] = laps["Driver"].map(elo_lookup).fillna(ELO_INITIAL)

    # --- identifiers ---------------------------------------------------------
    laps["year"] = ev.year
    laps["round"] = ev.round
    laps["event"] = ev.slug
    laps["total_laps"] = total_laps

    keep = [
        "year", "round", "event", "total_laps",
        "Driver", "Team", "DriverNumber",
        "LapNumber", "laps_remaining", "race_progress",
        "Position", "grid_position", "gap_to_leader", "gap_to_ahead",
        "lap_time_s", "sector1_s", "sector2_s", "sector3_s",
        "compound", "tire_age_laps", "stint_number",
        "in_pit", "pit_count",
        "sc_active", "vsc_active", "red_flag", "yellow_flag", "laps_since_sc",
        "track_temp", "air_temp", "rainfall", "humidity",
        "is_running", "did_finish", "dnf_this_lap",
        "elo_pre",
    ]
    out = laps[[c for c in keep if c in laps.columns]].copy()
    out = out.rename(columns={"Driver": "driver", "Team": "team", "DriverNumber": "driver_number",
                              "LapNumber": "lap", "Position": "position"})
    return out


def build(years: list[int] | None = None, refresh_elo: bool = True) -> pd.DataFrame:
    if refresh_elo:
        elo_df = build_elo()
    else:
        from src.config import ELO_HISTORY_PATH
        elo_df = pd.read_parquet(ELO_HISTORY_PATH) if ELO_HISTORY_PATH.exists() else build_elo()

    events = [e for e in list_events(years) if has_race(e)]
    if not events:
        return pd.DataFrame()

    # pre-race Elo lookup keyed by (year, round, driver)
    elo_key = {
        (int(r.year), int(r.round), r.driver): float(r.elo_pre)
        for r in elo_df.itertuples(index=False)
    }

    frames = []
    for ev in tqdm(events, desc="features", unit="race"):
        # driver -> pre-race Elo for this specific event
        ev_lookup = {
            k[2]: v for k, v in elo_key.items() if k[0] == ev.year and k[1] == ev.round
        }
        try:
            df = _build_one_race(ev, ev_lookup)
        except Exception as e:
            # A session still being fetched can leave a half-written parquet;
            # skip it rather than aborting the whole build.
            tqdm.write(f"  skip {ev.key}: {type(e).__name__}: {str(e)[:80]}")
            continue
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_and_save(years: list[int] | None = None) -> pd.DataFrame:
    df = build(years)
    LAP_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(LAP_FEATURES_PATH, index=False)
    return df


def main() -> int:
    df = build_and_save()
    if df.empty:
        print("No race data found under data/raw ---- run the Phase 1 fetcher first.")
        return 1
    n_races = df[["year", "round"]].drop_duplicates().shape[0]
    print(f"\nWrote {len(df):,} lap-rows ({n_races} races) to {LAP_FEATURES_PATH}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nDNF laps flagged: {int(df['dnf_this_lap'].sum())}")
    print(f"SC laps: {int(df['sc_active'].sum())}  VSC laps: {int(df['vsc_active'].sum())}")
    print(f"Compounds: {df['compound'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
