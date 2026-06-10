"""Phase 3: train the four LightGBM estimators the simulator needs.

The estimators predict *parameters*, never race outcomes (PROGRESS.md decision #1):

  1. pace        regressor   clean green-flag lap time (s)
  2. sc_hazard   classifier  P(a safety-car/VSC begins on this lap)   [race-level]
  3. dnf_hazard  classifier  P(a running driver retires this lap)     [driver-level]
  4. pit_dur     regressor   pit-stop time loss (s)

Each is saved as a plain-text LightGBM booster in ``models/`` plus a
``model_meta.json`` holding residual spreads and base rates the Monte Carlo
simulator uses for noise and as graceful fallbacks.

Run:  python -m src.models.train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    LAP_FEATURES_PATH,
    MODEL_DNF,
    MODEL_META,
    MODEL_PACE,
    MODEL_PIT,
    MODEL_SC,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.features.raw_io import has_race, list_events, load  # noqa: E402
from src.models import spec  # noqa: E402

_COMMON = dict(verbosity=-1, seed=RANDOM_SEED, deterministic=True, force_col_wise=True)


def _fit(X: pd.DataFrame, y: pd.Series, features: list[str], params: dict, rounds: int) -> lgb.Booster:
    dset = lgb.Dataset(
        X, label=y, categorical_feature=spec.categorical_indices(features), free_raw_data=False
    )
    return lgb.train(params, dset, num_boost_round=rounds)


# --------------------------------------------------------------------------- #
# 1. Pace                                                                       #
# --------------------------------------------------------------------------- #
def train_pace(laps: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    df = laps[
        (laps["sc_active"] == 0)
        & (laps["vsc_active"] == 0)
        & (laps["red_flag"] == 0)
        & (laps["in_pit"] == 0)
        & (laps["lap"] > 1)
        & (laps["lap_time_s"].notna())
        & (laps["is_running"] == 1)
    ].copy()
    # Drop in/out-of-range and per-race slow outliers (traffic, mistakes).
    df = df[(df["lap_time_s"] > 50) & (df["lap_time_s"] < 200)]
    med = df.groupby("event")["lap_time_s"].transform("median")
    df = df[df["lap_time_s"] <= 1.10 * med]

    X = spec.as_model_frame(df, spec.PACE_FEATURES)
    y = df["lap_time_s"].astype(float)
    params = {"objective": "regression", "metric": "l2", "learning_rate": 0.05,
              "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9, **_COMMON}
    booster = _fit(X, y, spec.PACE_FEATURES, params, rounds=400)
    resid = y.to_numpy() - booster.predict(X)
    meta = {
        "n_train": int(len(df)),
        "resid_std": float(np.std(resid)),
        "resid_std_by_compound": {
            str(c): float(np.std((y - booster.predict(X))[df["compound"] == c]))
            for c in df["compound"].unique()
        },
        "mae": float(np.mean(np.abs(resid))),
    }
    return booster, meta


# --------------------------------------------------------------------------- #
# 2. Safety-car onset hazard (per event-lap)                                    #
# --------------------------------------------------------------------------- #
def _race_lap_table(laps: pd.DataFrame) -> pd.DataFrame:
    g = laps.groupby(["event", "lap"])
    tab = g.agg(
        sc_active=("sc_active", "max"),
        vsc_active=("vsc_active", "max"),
        total_laps=("total_laps", "max"),
        race_progress=("race_progress", "max"),
    ).reset_index()
    tab["neut"] = ((tab["sc_active"] == 1) | (tab["vsc_active"] == 1)).astype(int)
    tab = tab.sort_values(["event", "lap"])
    prev = tab.groupby("event")["neut"].shift(1).fillna(0)
    tab["onset"] = ((tab["neut"] == 1) & (prev == 0)).astype(int)

    def _since(neut: pd.Series) -> pd.Series:
        out, c = [], 9999
        for n in neut.to_numpy():
            c = 0 if n else c + 1
            out.append(c)
        return pd.Series(out, index=neut.index)

    tab["laps_since_neut"] = tab.groupby("event")["neut"].transform(_since).shift(1).fillna(9999)
    return tab


def train_sc(laps: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    tab = _race_lap_table(laps)
    X = spec.as_model_frame(tab, spec.SC_FEATURES)
    y = tab["onset"].astype(int)
    pos = int(y.sum())
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
              "num_leaves": 15, "min_data_in_leaf": 50, "scale_pos_weight": max(1.0, (len(y) - pos) / max(pos, 1)),
              **_COMMON}
    booster = _fit(X, y, spec.SC_FEATURES, params, rounds=120)
    meta = {"n_train": int(len(tab)), "n_onset": pos, "base_rate": float(y.mean())}
    return booster, meta


# --------------------------------------------------------------------------- #
# 3. DNF hazard (per driver-lap)                                                #
# --------------------------------------------------------------------------- #
def train_dnf(laps: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    df = laps[laps["is_running"] == 1].copy()
    X = spec.as_model_frame(df, spec.DNF_FEATURES)
    y = df["dnf_this_lap"].astype(int)
    pos = int(y.sum())
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
              "num_leaves": 15, "min_data_in_leaf": 50,
              "scale_pos_weight": max(1.0, (len(y) - pos) / max(pos, 1)), **_COMMON}
    booster = _fit(X, y, spec.DNF_FEATURES, params, rounds=120)
    meta = {"n_train": int(len(df)), "n_dnf": pos, "base_rate": float(y.mean())}
    return booster, meta


# --------------------------------------------------------------------------- #
# 4. Pit-stop duration (recomputed from raw laps)                               #
# --------------------------------------------------------------------------- #
def _pit_durations(min_year: int | None = None) -> pd.DataFrame:
    rows = []
    events = [e for e in list_events() if has_race(e)]
    if min_year is not None:
        events = [e for e in events if e.year >= min_year]
    for ev in events:
        laps = load(ev, "R", "laps")
        if laps is None or laps.empty:
            continue
        laps = laps.sort_values(["Driver", "LapNumber"])
        total = float(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
        for _, g in laps.groupby("Driver"):
            pin = g["PitInTime_s"].to_numpy()
            pout = g["PitOutTime_s"].to_numpy()
            lap = pd.to_numeric(g["LapNumber"], errors="coerce").to_numpy()
            stint = pd.to_numeric(g["Stint"], errors="coerce").to_numpy()
            for i in range(len(g) - 1):
                if not np.isnan(pin[i]) and not np.isnan(pout[i + 1]):
                    dur = pout[i + 1] - pin[i]
                    if 10.0 < dur < 60.0:  # plausible pit-lane transit window
                        rows.append({
                            "duration": dur,
                            "race_progress": lap[i] / total if total else 0.0,
                            "stint_number": stint[i] if not np.isnan(stint[i]) else 1,
                            "total_laps": total,
                        })
    return pd.DataFrame(rows)


def train_pit(min_year: int | None = None) -> tuple[lgb.Booster | None, dict]:
    df = _pit_durations(min_year)
    if df.empty:
        return None, {"n_train": 0, "mean": 24.0, "std": 3.0, "resid_std": 3.0}
    X = spec.as_model_frame(df, spec.PIT_FEATURES)
    y = df["duration"].astype(float)
    params = {"objective": "regression", "metric": "l2", "learning_rate": 0.05,
              "num_leaves": 15, "min_data_in_leaf": 30, **_COMMON}
    booster = _fit(X, y, spec.PIT_FEATURES, params, rounds=120)
    resid = y.to_numpy() - booster.predict(X)
    meta = {"n_train": int(len(df)), "mean": float(y.mean()), "std": float(y.std()),
            "resid_std": float(np.std(resid))}
    return booster, meta


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-year", type=int, default=None,
                    help="Train only on races from this year onward (e.g. 2022 for the ground-effect era).")
    args = ap.parse_args()

    if not LAP_FEATURES_PATH.exists():
        print("Missing lap features -- run `python -m src.features.build_features` first.")
        return 1
    laps = pd.read_parquet(LAP_FEATURES_PATH)
    if args.min_year is not None:
        before = laps["year"].nunique()
        laps = laps[laps["year"] >= args.min_year].copy()
        print(f"Filtered to year >= {args.min_year}: {laps['year'].nunique()}/{before} seasons, {len(laps):,} lap-rows")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {"features": {
        "pace": spec.PACE_FEATURES, "sc": spec.SC_FEATURES,
        "dnf": spec.DNF_FEATURES, "pit": spec.PIT_FEATURES,
    }, "compound_codes": spec.COMPOUND_CODES}

    print("Training pace regressor ...")
    pace, meta["pace"] = train_pace(laps)
    pace.save_model(str(MODELS_DIR / MODEL_PACE))
    print(f"  n={meta['pace']['n_train']:,}  MAE={meta['pace']['mae']:.3f}s  resid_std={meta['pace']['resid_std']:.3f}s")

    print("Training safety-car hazard ...")
    sc, meta["sc"] = train_sc(laps)
    sc.save_model(str(MODELS_DIR / MODEL_SC))
    print(f"  n={meta['sc']['n_train']:,}  onsets={meta['sc']['n_onset']}  base_rate={meta['sc']['base_rate']:.4f}")

    print("Training DNF hazard ...")
    dnf, meta["dnf"] = train_dnf(laps)
    dnf.save_model(str(MODELS_DIR / MODEL_DNF))
    print(f"  n={meta['dnf']['n_train']:,}  dnfs={meta['dnf']['n_dnf']}  base_rate={meta['dnf']['base_rate']:.5f}")

    print("Training pit-duration regressor ...")
    pit, meta["pit"] = train_pit(args.min_year)
    if pit is not None:
        pit.save_model(str(MODELS_DIR / MODEL_PIT))
    print(f"  n={meta['pit']['n_train']:,}  mean={meta['pit']['mean']:.2f}s  resid_std={meta['pit']['resid_std']:.2f}s")

    (MODELS_DIR / MODEL_META).write_text(json.dumps(meta, indent=2))
    print(f"\nSaved 4 models + {MODEL_META} to {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
