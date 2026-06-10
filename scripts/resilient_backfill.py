"""Rate-limit-aware historical backfill.

FastF1's public API allows ~500 calls/hour. A naive bulk fetch of several seasons
blows through that and the remaining years silently fail. This runner paces the
work: it fetches missing (year, round, session) one at a time and, whenever it
hits ``RateLimitExceededError``, sleeps for a cool-down before retrying the same
session — so the job makes steady forward progress as the rolling quota frees up.

It is idempotent (sessions already on disk are skipped) and processes seasons in
priority order, newest meaningful data first, so the most-wanted races land soon.

Run (in the background; it may take a few hours):
    python scripts/resilient_backfill.py
    python scripts/resilient_backfill.py --years 2026 2025
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fastf1  # noqa: E402
from fastf1.exceptions import RateLimitExceededError  # noqa: E402

from src.fetch.bulk_history import (  # noqa: E402
    _already_fetched,
    _events_for_year,
    _session_dir,
    fetch_one,
    update_manifest,
)
from src.config import SESSION_TYPES, init_cache  # noqa: E402

# Newest-first so "current" data (and the user-requested 2026 through Monaco)
# arrives before the older training backfill.
DEFAULT_ORDER = [2026, 2025, 2023, 2022, 2021, 2020, 2019, 2024]
COOLDOWN_S = 420  # ~7 min; the rolling-hour quota frees up while we wait


def _patient(fn, *, label: str, max_waits: int = 30):
    """Call ``fn``; on a rate-limit error, sleep and retry up to ``max_waits``."""
    for attempt in range(max_waits + 1):
        try:
            return fn()
        except RateLimitExceededError:
            if attempt >= max_waits:
                raise
            wait = COOLDOWN_S
            print(f"  [rate-limit] {label}: cooling down {wait}s "
                  f"(retry {attempt + 1}/{max_waits})", flush=True)
            time.sleep(wait)
    return None


def backfill(years: list[int], force: bool = False) -> None:
    init_cache()
    for year in years:
        sched = _patient(lambda y=year: _events_for_year(y), label=f"{year} schedule")
        if sched is None or sched.empty:
            print(f"[{year}] no events to fetch", flush=True)
            continue

        missing = []
        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            name = str(ev["EventName"])
            for st in SESSION_TYPES:
                if force or not _already_fetched(_session_dir(year, rnd, name), st):
                    missing.append((rnd, name, st))

        print(f"[{year}] {len(missing)} sessions to fetch", flush=True)
        records = []
        for rnd, name, st in missing:
            rec = _patient(
                lambda r=rnd, n=name, s=st: fetch_one(
                    year, r, n, s, force=force, raise_on_rate_limit=True
                ),
                label=f"{year} R{rnd} {name} {st}",
            )
            if rec is not None:
                records.append(rec)
                flag = rec["status"]
                print(f"  {year} R{rnd:02d} {name[:28]:28s} {st}: {flag}", flush=True)
            # gentle spacing between sessions to avoid bursting the quota
            time.sleep(2.0)
        update_manifest(records)
        ok = sum(1 for r in records if r["status"] == "ok")
        print(f"[{year}] done: {ok}/{len(records)} fetched OK", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_ORDER)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    backfill(args.years, force=args.force)
    print("\nBackfill complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
