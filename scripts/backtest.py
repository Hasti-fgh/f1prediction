"""Phase 5: backtest the predictor across a whole season.

For every race we replay it at several race-distance checkpoints (e.g. 25/50/75/
90%) and score the prediction against the actual winner:

  * top-1 accuracy   how often the highest-P(win) driver is the real winner
  * Brier score      mean squared error of P(win) vs the winner indicator
  * log loss         penalises confident wrong calls
  * calibration      do drivers given ~p% actually win ~p% of the time

Lower Brier/log-loss and higher accuracy at later checkpoints is the expected
shape -- uncertainty should shrink as the race unfolds.

Run:  python scripts/backtest.py --year 2024 --runs 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import LAP_FEATURES_PATH  # noqa: E402
from src.models.predictors import Predictors  # noqa: E402
from src.sim.monte_carlo import MonteCarlo  # noqa: E402
from src.sim.state import actual_winner, event_keys, slice_event, state_at_lap  # noqa: E402

CHECKPOINTS = (0.25, 0.50, 0.75, 0.90)


def _brier(p_win: dict[str, float], winner: str) -> float:
    return float(np.mean([(p - (1.0 if d == winner else 0.0)) ** 2 for d, p in p_win.items()]))


def _logloss(p_win: dict[str, float], winner: str) -> float:
    p = min(max(p_win.get(winner, 0.0), 1e-6), 1 - 1e-6)
    return float(-np.log(p))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--year", type=int, default=None, help="Limit to one season")
    p.add_argument("--runs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--form-sigma", type=float, default=None,
                   help="Override the per-sim form offset (s/lap) for calibration sweeps.")
    args = p.parse_args()

    if not LAP_FEATURES_PATH.exists():
        print("Missing lap features -- run `python -m src.features.build_features` first.")
        return 1
    laps = pd.read_parquet(LAP_FEATURES_PATH)
    if args.year is not None:
        laps = laps[laps["year"] == args.year]
    if laps.empty:
        print("No data for the requested filter.")
        return 1

    mc_kwargs = {} if args.form_sigma is None else {"form_sigma": args.form_sigma}
    mc = MonteCarlo(Predictors.load(), n_runs=args.runs, seed=args.seed, **mc_kwargs)
    races = event_keys(laps)
    print(f"Backtesting {len(races)} races x {len(CHECKPOINTS)} checkpoints "
          f"({args.runs} sims each)...\n")

    rows = []
    cal = []  # (predicted_p, hit) pairs for calibration
    for _, r in races.iterrows():
        ev = slice_event(laps, int(r["year"]), int(r["round"]))
        truth = actual_winner(ev)
        total = int(r["total_laps"])
        for frac in CHECKPOINTS:
            lap = max(1, int(round(frac * total)))
            res = mc.simulate(state_at_lap(ev, lap))
            ranked = res.ranked()
            pred = ranked[0][0] if ranked else ""
            rows.append({
                "year": int(r["year"]), "round": int(r["round"]), "event": r["event"],
                "frac": frac, "lap": lap, "winner": truth, "pred": pred,
                "correct": int(pred == truth),
                "p_winner": res.p_win.get(truth, 0.0),
                "brier": _brier(res.p_win, truth),
                "logloss": _logloss(res.p_win, truth),
            })
            for d, pw in res.p_win.items():
                cal.append((pw, 1 if d == truth else 0))

    df = pd.DataFrame(rows)
    print("=== By race-distance checkpoint ===")
    summary = df.groupby("frac").agg(
        races=("correct", "size"),
        top1_acc=("correct", "mean"),
        mean_brier=("brier", "mean"),
        mean_logloss=("logloss", "mean"),
        mean_p_on_winner=("p_winner", "mean"),
    ).round(3)
    print(summary.to_string())

    # calibration table
    cdf = pd.DataFrame(cal, columns=["p", "hit"])
    cdf["bin"] = pd.cut(cdf["p"], bins=[0, .05, .1, .2, .4, .6, .8, 1.0])
    print("\n=== Calibration (predicted P(win) vs realised win rate) ===")
    ctab = cdf.groupby("bin", observed=True).agg(
        n=("hit", "size"), predicted=("p", "mean"), realised=("hit", "mean")
    ).round(3)
    print(ctab.to_string())

    miss = df[(df["frac"] == 0.90) & (df["correct"] == 0)]
    if not miss.empty:
        print("\n=== Wrong top-pick at 90% distance ===")
        print(miss[["event", "winner", "pred", "p_winner"]].to_string(index=False))

    out = LAP_FEATURES_PATH.parent / "backtest_results.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved per-checkpoint results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
