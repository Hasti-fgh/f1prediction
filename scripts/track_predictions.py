"""Record pre-race predictions and build the predicted vs actual scoreboard.

This is the "publish it properly" layer on top of scripts/predict_prerace.py.
For each race it writes a small JSON file holding the pre-race forecast (and the
real result once the race has run), then regenerates PREDICTIONS.md from every
JSON on disk. Keeping each prediction as its own committed file means the git
history is a tamper-proof log: the commit timestamp proves a call was made before
the race, which is the whole point of publishing predictions.

A prediction is tagged:
  * "live"     if the race result is not available yet when it is generated
               (a genuine before-the-race call), or
  * "backfill" if the race had already run (validation only, not a real call).

Usage:
    python scripts/track_predictions.py --year 2026 --round 8     # one race
    python scripts/track_predictions.py --year 2026 --all         # every fetched race
    python scripts/track_predictions.py --rebuild                 # only redraw the table
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from src.config import RAW_DIR, ROOT, init_cache  # noqa: E402
from src.features.raw_io import load  # noqa: E402
from src.models.predictors import Predictors  # noqa: E402
from src.sim.monte_carlo import MonteCarlo  # noqa: E402

from predict_prerace import _event_dir, build_prerace_state  # noqa: E402

PRED_DIR = ROOT / "predictions"
SCOREBOARD = ROOT / "PREDICTIONS.md"
FULL_NAMES = {
    "RUS": "Russell", "HAM": "Hamilton", "ANT": "Antonelli", "NOR": "Norris",
    "VER": "Verstappen", "HAD": "Hadjar", "PIA": "Piastri", "LAW": "Lawson",
    "HUL": "Hulkenberg", "LEC": "Leclerc", "LIN": "Lindblad", "BOR": "Bortoleto",
    "COL": "Colapinto", "GAS": "Gasly", "BEA": "Bearman", "SAI": "Sainz",
    "OCO": "Ocon", "ALB": "Albon", "PER": "Perez", "BOT": "Bottas",
    "STR": "Stroll", "ALO": "Alonso",
}


def _actual_result(ev) -> dict | None:
    """Final classification for a race, or None if it has not run yet."""
    r = load(ev, "R", "results")
    if r is None or r.empty or "Abbreviation" not in r.columns:
        return None
    r = r.copy()
    r["_pos"] = pd.to_numeric(r.get("Position"), errors="coerce")
    fin = r.dropna(subset=["_pos"]).sort_values("_pos")
    if fin.empty:
        return None
    finish_pos = {row["Abbreviation"]: int(row["_pos"]) for _, row in fin.iterrows()}
    return {
        "winner": str(fin.iloc[0]["Abbreviation"]),
        "podium": [str(x) for x in fin.head(3)["Abbreviation"].tolist()],
        "finish_pos": finish_pos,
    }


def predict_one(year: int, rnd: int, runs: int = 10000, top_n: int = 5) -> dict:
    ev = _event_dir(year, rnd)
    state = build_prerace_state(year, rnd)
    mc = MonteCarlo(Predictors.load(), n_runs=runs)
    res = mc.simulate(state)
    ranked = res.ranked()
    grid = {d.driver: int(d.position) for d in state.drivers}

    prediction = [
        {"pos": i, "driver": drv, "grid": grid.get(drv, 0), "p_win": round(p, 4)}
        for i, (drv, p) in enumerate(ranked[:top_n], 1)
    ]

    actual = _actual_result(ev)
    kind = "backfill" if actual is not None else "live"
    out = {
        "year": year,
        "round": rnd,
        "event": state.event,
        "total_laps": state.total_laps,
        "predicted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "runs": runs,
        "prediction": prediction,
        "actual": None,
    }
    if actual is not None:
        top_pick = prediction[0]["driver"]
        out["actual"] = {
            "winner": actual["winner"],
            "podium": actual["podium"],
            "top_pick_correct": bool(top_pick == actual["winner"]),
            "predicted_winner_finish": actual["finish_pos"].get(top_pick),
        }

    PRED_DIR.joinpath(str(year)).mkdir(parents=True, exist_ok=True)
    fp = PRED_DIR / str(year) / f"{year}_{rnd:02d}_{state.event}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _fetched_rounds(year: int) -> list[int]:
    rounds = []
    for d in sorted(RAW_DIR.glob(f"{year}_*")):
        if (d / "Q_results.parquet").exists():
            try:
                rounds.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(set(rounds))


def _load_all_predictions() -> list[dict]:
    preds = []
    for fp in sorted(PRED_DIR.glob("*/*.json")):
        try:
            preds.append(json.loads(fp.read_text()))
        except json.JSONDecodeError:
            continue
    return sorted(preds, key=lambda p: (p["year"], p["round"]))


def _pretty(drv: str) -> str:
    return FULL_NAMES.get(drv, drv)


def _row(p: dict) -> str:
    top = p["prediction"][0]
    conf = f"{top['p_win']:.0%}"
    pick = f"{_pretty(top['driver'])} (P{top['grid']})"
    event = p["event"].replace("_", " ")
    if p["actual"] is None:
        return f"| {p['round']} | {event} | {pick} | {conf} | _not run yet_ | ⏳ |"
    a = p["actual"]
    won = _pretty(a["winner"])
    if a["top_pick_correct"]:
        result = "✅ hit"
    else:
        fin = a["predicted_winner_finish"]
        where = f"P{fin}" if fin else "DNF"
        result = f"❌ pick finished {where}"
    return f"| {p['round']} | {event} | {pick} | {conf} | {won} | {result} |"


def _section(title: str, note: str, preds: list[dict]) -> str:
    header = (
        "| Round | Race | Predicted winner | P(win) | Actual winner | Result |\n"
        "|------|------|------------------|--------|---------------|--------|"
    )
    if not preds:
        return f"## {title}\n\n{note}\n\n_None yet._\n"
    scored = [p for p in preds if p["actual"] is not None]
    hits = sum(1 for p in scored if p["actual"]["top_pick_correct"])
    rate = f"{hits}/{len(scored)} ({hits / len(scored):.0%})" if scored else "n/a"
    rows = "\n".join(_row(p) for p in preds)
    return (
        f"## {title}\n\n{note}\n\n"
        f"**Top-pick hit rate: {rate}**\n\n"
        f"{header}\n{rows}\n"
    )


def rebuild_scoreboard() -> None:
    preds = _load_all_predictions()
    live = [p for p in preds if p["kind"] == "live"]
    backfill = [p for p in preds if p["kind"] == "backfill"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = (
        "# 2026 Predictions vs Actual\n\n"
        f"_Auto-generated by `scripts/track_predictions.py` on {stamp}. Do not edit by hand._\n\n"
        "Each prediction is made from the qualifying grid only (a lap-0 Monte Carlo "
        "forecast), before any race lap is seen. The P(win) column is how sure the "
        "model was about its top pick. Early-season accuracy is expected to be modest; "
        "the model is honest about uncertainty rather than confidently wrong.\n\n"
        + _section(
            "Live predictions",
            "Made before the race ran, so the result was genuinely unknown at the time. "
            "This is the real track record.",
            live,
        )
        + "\n"
        + _section(
            "Backfill / validation",
            "The race had already run when these were generated, and those races were "
            "in the training data, so they are validation only, not live calls.",
            backfill,
        )
    )
    SCOREBOARD.write_text(body, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--all", action="store_true", help="every fetched race of the year")
    p.add_argument("--rebuild", action="store_true", help="only regenerate PREDICTIONS.md")
    p.add_argument("--runs", type=int, default=10000)
    args = p.parse_args()

    init_cache()

    if args.rebuild:
        rebuild_scoreboard()
        print(f"Rebuilt {SCOREBOARD}")
        return 0

    if args.all:
        rounds = _fetched_rounds(args.year)
    elif args.round is not None:
        rounds = [args.round]
    else:
        print("Pass --round N, --all, or --rebuild.")
        return 1

    for rnd in rounds:
        out = predict_one(args.year, rnd, runs=args.runs)
        top = out["prediction"][0]
        tag = out["kind"]
        actual = f" | actual {out['actual']['winner']}" if out["actual"] else ""
        print(f"  R{rnd:>2} {out['event']:<26} -> {top['driver']} {top['p_win']:.0%} ({tag}){actual}")

    rebuild_scoreboard()
    print(f"\nWrote {len(rounds)} prediction file(s) and rebuilt {SCOREBOARD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
