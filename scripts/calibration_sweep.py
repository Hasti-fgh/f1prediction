"""Sweep the Monte Carlo `form_sigma` to fix P(win) overconfidence.

For each candidate per-sim form offset we backtest the given seasons at the
standard checkpoints and report:
  * top1@90      -- top-1 accuracy at 90% race distance (must not collapse)
  * ECE          -- expected calibration error (lower = better calibrated)
  * hi_pred/hi_real -- predicted vs realised win-rate in the confident region (p>0.4)

The best sigma keeps top1@90 healthy while driving hi_pred ~= hi_real and ECE low.

Run:  python scripts/calibration_sweep.py --years 2024 2025 --runs 2000
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
BINS = [0, .05, .1, .2, .4, .6, .8, 1.0]


def _eval(mc: MonteCarlo, races: pd.DataFrame, laps: pd.DataFrame) -> dict:
    cal_p, cal_hit, correct90 = [], [], []
    for _, r in races.iterrows():
        ev = slice_event(laps, int(r["year"]), int(r["round"]))
        truth = actual_winner(ev)
        total = int(r["total_laps"])
        for frac in CHECKPOINTS:
            lap = max(1, int(round(frac * total)))
            res = mc.simulate(state_at_lap(ev, lap))
            ranked = res.ranked()
            pred = ranked[0][0] if ranked else ""
            if frac == 0.90:
                correct90.append(int(pred == truth))
            for d, pw in res.p_win.items():
                cal_p.append(pw)
                cal_hit.append(1 if d == truth else 0)
    cdf = pd.DataFrame({"p": cal_p, "hit": cal_hit})
    cdf["bin"] = pd.cut(cdf["p"], bins=BINS)
    grp = cdf.groupby("bin", observed=True).agg(n=("hit", "size"),
                                                pred=("p", "mean"), real=("hit", "mean"))
    ece = float((grp["n"] / grp["n"].sum() * (grp["pred"] - grp["real"]).abs()).sum())
    hi = cdf[cdf["p"] > 0.4]
    return {
        "top1@90": float(np.mean(correct90)),
        "ECE": ece,
        "hi_pred": float(hi["p"].mean()) if len(hi) else float("nan"),
        "hi_real": float(hi["hit"].mean()) if len(hi) else float("nan"),
        "hi_n": int(len(hi)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.10, 0.15, 0.20, 0.25, 0.30])
    args = ap.parse_args()

    laps = pd.read_parquet(LAP_FEATURES_PATH)
    laps = laps[laps["year"].isin(args.years)]
    races = event_keys(laps)
    pred = Predictors.load()
    print(f"Sweeping form_sigma on {len(races)} races ({args.years}), {args.runs} sims each\n", flush=True)
    print(f"{'sigma':>6} {'top1@90':>8} {'ECE':>7} {'hi_pred':>8} {'hi_real':>8} {'hi_n':>6}", flush=True)
    for s in args.sigmas:
        mc = MonteCarlo(pred, n_runs=args.runs, seed=args.seed, form_sigma=s)
        m = _eval(mc, races, laps)
        print(f"{s:>6.2f} {m['top1@90']:>8.3f} {m['ECE']:>7.3f} "
              f"{m['hi_pred']:>8.3f} {m['hi_real']:>8.3f} {m['hi_n']:>6d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
