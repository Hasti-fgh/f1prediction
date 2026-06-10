"""Feature specifications shared by training (Phase 3) and the simulator (Phase 4).

Keeping these lists in one place guarantees the columns a model was trained on
are exactly the columns the Monte Carlo simulator feeds back at inference time.
None of these features is driver identity — skill enters only via ``elo_pre``
(see PROGRESS.md decision #2).
"""
from __future__ import annotations

import pandas as pd

# Lap-pace regressor: predicts a single clean green-flag lap time (seconds).
PACE_FEATURES = [
    "compound",          # categorical
    "tire_age_laps",
    "stint_number",
    "race_progress",     # proxy for fuel burn-off
    "laps_remaining",
    "position",
    "elo_pre",
    "track_temp",
    "air_temp",
    "rainfall",
]

# Safety-car / VSC onset hazard: P(a neutralisation begins on this lap). One row
# per (event, lap) — a race-level event, not per driver.
SC_FEATURES = [
    "race_progress",
    "lap",
    "total_laps",
    "laps_since_neut",
]

# Retirement hazard: P(a running driver retires on this lap).
DNF_FEATURES = [
    "compound",          # categorical
    "tire_age_laps",
    "stint_number",
    "position",
    "gap_to_ahead",
    "race_progress",
    "sc_active",
    "vsc_active",
    "pit_count",
    "elo_pre",
]

# Pit-stop time loss (seconds, in-lap PitIn to out-lap PitOut).
PIT_FEATURES = [
    "race_progress",
    "stint_number",
    "total_laps",
]

CATEGORICAL = {"compound"}

# Stable integer encoding so a model saved as a plain-text LightGBM booster keeps
# the same categorical mapping at inference time (no sklearn pickle required).
COMPOUND_CODES = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
    "OTHER": 5,
    "UNKNOWN": 6,
}


def encode_compound(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().map(COMPOUND_CODES).fillna(COMPOUND_CODES["UNKNOWN"]).astype("int32")


def categorical_indices(features: list[str]) -> list[int]:
    """Positional indices of categorical columns, for LightGBM's API."""
    return [i for i, f in enumerate(features) if f in CATEGORICAL]


def as_model_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a numeric frame with exactly ``features`` columns in order.

    Categorical columns are integer-encoded; everything else is coerced numeric.
    Missing columns are created as NaN so inference frames assembled by the
    simulator never crash on a column the training frame happened to have.
    """
    out = pd.DataFrame(index=df.index)
    for col in features:
        src = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)
        if col in CATEGORICAL:
            out[col] = encode_compound(src)
        else:
            out[col] = pd.to_numeric(src, errors="coerce")
    return out
