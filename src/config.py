"""Project-wide paths, constants, and FastF1 cache bootstrap."""
from pathlib import Path
import fastf1

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
LIVE_DIR = DATA_DIR / "live"
MODELS_DIR = ROOT / "models"

TRAIN_SEASONS = list(range(2019, 2027))

REG_RESET_YEAR = 2026

SESSION_TYPES = ("Q", "R")

# --- Feature / model artifacts ---------------------------------------------
LAP_FEATURES_PATH = FEATURES_DIR / "lap_features.parquet"
ELO_HISTORY_PATH = FEATURES_DIR / "elo_history.parquet"

# Tyre compounds we model explicitly; anything else folds into "OTHER".
COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")

# Elo configuration
ELO_INITIAL = 1500.0
ELO_K = 24.0            # update step size
ELO_REGRESS = 0.10      # fraction pulled back toward mean at a regulation reset

# Monte Carlo defaults
MC_RUNS = 10_000
RANDOM_SEED = 42

# Trained model filenames (inside MODELS_DIR)
MODEL_PACE = "pace_lgbm.txt"
MODEL_SC = "sc_hazard_lgbm.txt"
MODEL_DNF = "dnf_hazard_lgbm.txt"
MODEL_PIT = "pit_duration_lgbm.txt"
MODEL_META = "model_meta.json"


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RAW_DIR, FEATURES_DIR, LIVE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def init_cache() -> None:
    ensure_dirs()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
