"""Phase 6: live race runner.

During an actual GP this polls FastF1 for the latest lap data, rebuilds the same
feature snapshot the offline pipeline produces, runs the Monte Carlo simulator,
and writes a fresh P(win) table to ``data/live/`` every poll. It deliberately
shares ``features_from_frames`` and ``state_at_lap`` with the offline path, so a
race scored live is scored identically to a race scored from the lake -- the only
difference is where the laps come from.

Three sub-commands::

    # 1. Record the raw live-timing stream to disk during the session
    python -m src.live.runner record --year 2026 --round 7

    # 2. Drive the predictor from a recorded/loadable session, polling for laps
    python -m src.live.runner live --year 2026 --round 7 --poll 45

    # 3. Dry-run the live loop against a past race from the Parquet lake
    python -m src.live.runner replay --year 2024 --round 1 --poll 0

FastF1 only exposes lap data once it has been published (a ~30s-2min lag, which
PROGRESS.md accepts). The ``record`` step captures the SignalR stream so nothing
is lost; ``live`` reconstructs laps from whatever is available each poll.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from src.config import ELO_HISTORY_PATH, LIVE_DIR, init_cache  # noqa: E402
from src.fetch.bulk_history import _clean_for_parquet  # noqa: E402
from src.features.build_features import features_from_frames  # noqa: E402
from src.models.predictors import Predictors  # noqa: E402
from src.sim.monte_carlo import MonteCarlo  # noqa: E402
from src.sim.state import state_at_lap  # noqa: E402


def _elo_lookup(year: int) -> dict[str, float]:
    """Latest pre-race Elo for each driver as of the most recent prior race."""
    if not ELO_HISTORY_PATH.exists():
        return {}
    elo = pd.read_parquet(ELO_HISTORY_PATH)
    elo = elo[elo["year"] <= year].sort_values(["year", "round"])
    latest = elo.groupby("driver").tail(1)
    # use elo_post of the last completed race as the rating entering this one
    return dict(zip(latest["driver"], latest["elo_post"]))


def _emit(res, event: str, year: int, rnd: int) -> Path:
    """Write the current prediction snapshot to data/live/ as parquet + json."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df = pd.DataFrame(
        [{"driver": d, "p_win": res.p_win[d], "p_podium": res.p_podium[d],
          "p_finish": res.p_finish[d]} for d in res.p_win]
    ).sort_values("p_win", ascending=False)
    df["lap"] = res.current_lap
    df["total_laps"] = res.total_laps
    df["updated"] = stamp
    base = LIVE_DIR / f"{year}_{rnd:02d}_{event}"
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    base.with_suffix(".json").write_text(
        json.dumps({"event": event, "year": year, "round": rnd, "lap": res.current_lap,
                    "total_laps": res.total_laps, "updated": stamp,
                    "p_win": res.p_win}, indent=2)
    )
    return base.with_suffix(".parquet")


def _print_top(res, truth: str | None = None) -> None:
    line = "  ".join(f"{d} {p:.0%}" for d, p in res.ranked()[:5])
    tag = ""
    if truth:
        top = res.ranked()[0][0] if res.ranked() else ""
        tag = "  <-- correct" if top == truth else f"  (actual: {truth})"
    print(f"  lap {res.current_lap}/{res.total_laps}:  {line}{tag}")


# --------------------------------------------------------------------------- #
# record                                                                        #
# --------------------------------------------------------------------------- #
def cmd_record(args) -> int:
    from fastf1.livetiming.client import SignalRClient

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = LIVE_DIR / f"{args.year}_{args.round:02d}_livetiming.txt"
    print(f"Recording live timing to {out}\nPress Ctrl-C to stop.")
    client = SignalRClient(filename=str(out))
    try:
        client.start()
    except KeyboardInterrupt:
        print("\nStopped recording.")
    return 0


# --------------------------------------------------------------------------- #
# live                                                                          #
# --------------------------------------------------------------------------- #
def _load_session_frames(year: int, rnd: int):
    """Return (laps, results, weather, event_name) from FastF1 for a session."""
    import fastf1

    sess = fastf1.get_session(year, rnd, "R")
    sess.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = _clean_for_parquet(sess.laps) if sess.laps is not None else None
    try:
        results = _clean_for_parquet(sess.results)
    except Exception:
        results = None
    try:
        weather = _clean_for_parquet(sess.weather_data)
    except Exception:
        weather = None
    name = getattr(sess, "event", {}).get("EventName", f"Round {rnd}") if hasattr(sess, "event") else f"Round {rnd}"
    return laps, results, weather, str(name).replace(" ", "_")


def cmd_live(args) -> int:
    init_cache()
    pred = Predictors.load()
    if not pred.is_ready:
        print("Models not trained -- run `python -m src.models.train` first.")
        return 1
    mc = MonteCarlo(pred, n_runs=args.runs)
    elo = _elo_lookup(args.year)

    print(f"Live predictor for {args.year} round {args.round}  (poll {args.poll}s)\n")
    last_lap = -1
    while True:
        try:
            laps, results, weather, name = _load_session_frames(args.year, args.round)
            feats = features_from_frames(laps, results, weather, args.year, args.round, name, elo)
            if feats is not None and not feats.empty:
                cur = int(feats["lap"].max())
                if cur != last_lap:
                    st = state_at_lap(feats, cur)
                    res = mc.simulate(st)
                    _print_top(res)
                    _emit(res, name, args.year, args.round)
                    last_lap = cur
        except Exception as e:  # network blips, partial data early in a session
            print(f"  (waiting for data: {type(e).__name__}: {str(e)[:80]})")
        if args.poll <= 0:
            break
        time.sleep(args.poll)
    return 0


# --------------------------------------------------------------------------- #
# replay (offline dry-run of the live loop)                                     #
# --------------------------------------------------------------------------- #
def cmd_replay(args) -> int:
    from src.config import LAP_FEATURES_PATH
    from src.sim.state import actual_winner, slice_event

    pred = Predictors.load()
    if not pred.is_ready:
        print("Models not trained -- run `python -m src.models.train` first.")
        return 1
    laps = pd.read_parquet(LAP_FEATURES_PATH)
    ev = slice_event(laps, args.year, args.round)
    if ev.empty:
        print(f"No data for {args.year} round {args.round}.")
        return 1
    mc = MonteCarlo(pred, n_runs=args.runs)
    truth = actual_winner(ev)
    total = int(ev["total_laps"].iloc[0])
    name = str(ev["event"].iloc[0])
    print(f"Replay-driving the live loop for {args.year} {name} -- actual winner {truth}\n")
    for lap in range(args.step, total + 1, args.step):
        res = mc.simulate(state_at_lap(ev, lap))
        _print_top(res, truth)
        _emit(res, name, args.year, args.round)
        if args.poll > 0:
            time.sleep(args.poll)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="record raw live-timing stream")
    pr.add_argument("--year", type=int, required=True)
    pr.add_argument("--round", type=int, required=True)

    pl = sub.add_parser("live", help="poll FastF1 and predict during a session")
    pl.add_argument("--year", type=int, required=True)
    pl.add_argument("--round", type=int, required=True)
    pl.add_argument("--poll", type=int, default=45, help="seconds between polls (0 = once)")
    pl.add_argument("--runs", type=int, default=5000)

    px = sub.add_parser("replay", help="dry-run the live loop on a past race")
    px.add_argument("--year", type=int, required=True)
    px.add_argument("--round", type=int, required=True)
    px.add_argument("--step", type=int, default=5)
    px.add_argument("--poll", type=int, default=0)
    px.add_argument("--runs", type=int, default=3000)

    args = p.parse_args()
    return {"record": cmd_record, "live": cmd_live, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
