"""Loads the trained Phase-3 boosters and exposes typed prediction helpers.

The Monte Carlo simulator (Phase 4) talks to the models only through this class,
so the simulator never needs to know about LightGBM, file paths, or the
categorical encoding. ``Predictors.load()`` returns ``None`` for any model that
has not been trained yet, and the helpers fall back to the base rates / means
stored in ``model_meta.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    MODEL_DNF,
    MODEL_META,
    MODEL_PACE,
    MODEL_PIT,
    MODEL_SC,
    MODELS_DIR,
)
from src.models import spec


def _maybe_load(name: str) -> lgb.Booster | None:
    fp = MODELS_DIR / name
    if not fp.exists():
        return None
    return lgb.Booster(model_file=str(fp))


@dataclass
class Predictors:
    pace: lgb.Booster | None
    sc: lgb.Booster | None
    dnf: lgb.Booster | None
    pit: lgb.Booster | None
    meta: dict

    @classmethod
    def load(cls) -> "Predictors":
        meta_fp = MODELS_DIR / MODEL_META
        meta = json.loads(meta_fp.read_text()) if meta_fp.exists() else {}
        return cls(
            pace=_maybe_load(MODEL_PACE),
            sc=_maybe_load(MODEL_SC),
            dnf=_maybe_load(MODEL_DNF),
            pit=_maybe_load(MODEL_PIT),
            meta=meta,
        )

    @property
    def is_ready(self) -> bool:
        return self.pace is not None

    # --- pace ------------------------------------------------------------- #
    def predict_pace(self, df: pd.DataFrame) -> np.ndarray:
        X = spec.as_model_frame(df, spec.PACE_FEATURES)
        if self.pace is None:
            return np.full(len(df), 90.0)
        return self.pace.predict(X)

    @property
    def pace_resid_std(self) -> float:
        return float(self.meta.get("pace", {}).get("resid_std", 0.8))

    # --- safety car ------------------------------------------------------- #
    def predict_sc_onset(self, df: pd.DataFrame) -> np.ndarray:
        X = spec.as_model_frame(df, spec.SC_FEATURES)
        if self.sc is None:
            return np.full(len(df), self.meta.get("sc", {}).get("base_rate", 0.01))
        return np.clip(self.sc.predict(X), 0.0, 1.0)

    # --- dnf -------------------------------------------------------------- #
    def predict_dnf(self, df: pd.DataFrame) -> np.ndarray:
        X = spec.as_model_frame(df, spec.DNF_FEATURES)
        if self.dnf is None:
            return np.full(len(df), self.meta.get("dnf", {}).get("base_rate", 0.0015))
        return np.clip(self.dnf.predict(X), 0.0, 1.0)

    # --- pit duration ----------------------------------------------------- #
    def predict_pit(self, df: pd.DataFrame) -> np.ndarray:
        if self.pit is None:
            return np.full(len(df), self.meta.get("pit", {}).get("mean", 24.0))
        X = spec.as_model_frame(df, spec.PIT_FEATURES)
        return self.pit.predict(X)

    @property
    def pit_mean(self) -> float:
        return float(self.meta.get("pit", {}).get("mean", 24.0))

    @property
    def pit_std(self) -> float:
        return float(self.meta.get("pit", {}).get("resid_std", 3.0))
