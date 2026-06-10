"""Render model-performance charts from saved backtest result parquets.

Each input parquet is one season's per-(race, checkpoint) rows produced by
scripts/backtest.py (columns: year, frac, correct, p_winner, brier, logloss,
winner, pred). Produces a 2x2 PNG:

  A  top-1 accuracy by race distance (per season)
  B  reliability / calibration curve (pooled, predicted vs realised P(win))
  C  mean Brier score by race distance (lower is better)
  D  distribution of P(win) the model placed on the actual winner @ 90%

Run:  python scripts/plot_performance.py data/features/bt_2024_weather.parquet \
          data/features/bt_2025_weather.parquet -o data/features/performance.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CHECKPOINTS = [0.25, 0.50, 0.75, 0.90]
CAL_BINS = [0, .05, .1, .2, .4, .6, .8, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquets", nargs="+", help="backtest result parquet file(s)")
    ap.add_argument("-o", "--out", default="data/features/performance.png")
    args = ap.parse_args()

    frames = []
    for p in args.parquets:
        df = pd.read_parquet(p)
        df["season"] = str(int(df["year"].iloc[0]))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    seasons = sorted(data["season"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("F1 winner-prediction model — backtest performance", fontsize=15, fontweight="bold")
    colors = {"2024": "#1f77b4", "2025": "#d62728", "all": "#2ca02c"}

    # --- A: accuracy by race distance ---------------------------------- #
    ax = axes[0, 0]
    width = 0.8 / len(seasons)
    x = np.arange(len(CHECKPOINTS))
    for i, s in enumerate(seasons):
        acc = [data[(data.season == s) & (np.isclose(data.frac, f))]["correct"].mean()
               for f in CHECKPOINTS]
        bars = ax.bar(x + i * width, acc, width, label=s, color=colors.get(s, None))
        for b, a in zip(bars, acc):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x + width * (len(seasons) - 1) / 2)
    ax.set_xticklabels([f"{int(f*100)}%" for f in CHECKPOINTS])
    ax.set_xlabel("race distance completed"); ax.set_ylabel("top-1 accuracy")
    ax.set_title("A. How often the top pick is the real winner"); ax.set_ylim(0, 1)
    ax.legend(title="season"); ax.grid(axis="y", alpha=0.3)

    # --- B: calibration curve ------------------------------------------ #
    ax = axes[0, 1]
    data["bin"] = pd.cut(data["p_winner"].clip(0, 1), bins=CAL_BINS)
    # use ALL (driver,checkpoint) prob mass: rebuild from p_winner is winner-only,
    # so this curve reflects calibration on the eventual winner specifically.
    cal = data.groupby("bin", observed=True).agg(pred=("p_winner", "mean"),
                                                  real=("correct", "mean"), n=("correct", "size"))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(cal["pred"], cal["real"], "o-", color=colors["all"], label="model")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.pred, r.real), fontsize=7,
                    textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel("predicted P(win) on the winner"); ax.set_ylabel("realised win rate")
    ax.set_title("B. Calibration (closer to dashed = more honest)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    # --- C: Brier by race distance ------------------------------------- #
    ax = axes[1, 0]
    for i, s in enumerate(seasons):
        br = [data[(data.season == s) & (np.isclose(data.frac, f))]["brier"].mean()
              for f in CHECKPOINTS]
        ax.bar(x + i * width, br, width, label=s, color=colors.get(s, None))
    ax.set_xticks(x + width * (len(seasons) - 1) / 2)
    ax.set_xticklabels([f"{int(f*100)}%" for f in CHECKPOINTS])
    ax.set_xlabel("race distance completed"); ax.set_ylabel("mean Brier (lower = better)")
    ax.set_title("C. Probabilistic error by race distance"); ax.legend(title="season")
    ax.grid(axis="y", alpha=0.3)

    # --- D: P(win) placed on the actual winner @ 90% ------------------- #
    ax = axes[1, 1]
    at90 = data[np.isclose(data.frac, 0.90)]
    ax.hist(at90["p_winner"], bins=np.linspace(0, 1, 21), color=colors["all"], alpha=0.8)
    mean_p = at90["p_winner"].mean()
    ax.axvline(mean_p, color="black", linestyle="--", label=f"mean = {mean_p:.2f}")
    ax.set_xlabel("P(win) the model gave the eventual winner")
    ax.set_ylabel("number of races")
    ax.set_title("D. Confidence in the actual winner @ 90% distance")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}  ({len(data)} rows across seasons {seasons})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
