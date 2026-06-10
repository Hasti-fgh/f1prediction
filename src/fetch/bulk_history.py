"""Phase 1: Bulk-fetch historical F1 sessions (qualifying + race) to a Parquet lake.

For each event in each season, this saves:
    data/raw/{year}_{round:02d}_{event_slug}/
        Q_laps.parquet           qualifying lap-level data
        Q_results.parquet        qualifying final classification
        Q_weather.parquet        timestamped weather samples
        R_laps.parquet           race lap-level data
        R_results.parquet        race final classification
        R_weather.parquet        timestamped weather samples
        R_race_control.parquet   race control messages (SC, VSC, red flag, ...)

A manifest at data/raw/_manifest.parquet tracks fetch status per (year, round, session).
Re-running is idempotent: sessions already on disk are skipped unless --force.

Usage:
    python -m src.fetch.bulk_history                       # all seasons 2019-2026
    python -m src.fetch.bulk_history --years 2024          # one season
    python -m src.fetch.bulk_history --years 2025 2026     # specific seasons
    python -m src.fetch.bulk_history --force               # re-fetch existing
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fastf1  # noqa: E402
from fastf1.exceptions import RateLimitExceededError  # noqa: E402

from src.config import RAW_DIR, SESSION_TYPES, TRAIN_SEASONS, init_cache  # noqa: E402

MANIFEST_PATH = RAW_DIR / "_manifest.parquet"

warnings.filterwarnings("ignore", category=FutureWarning)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _session_dir(year: int, round_num: int, event_name: str) -> Path:
    return RAW_DIR / f"{year}_{round_num:02d}_{_slug(event_name)}"


def _already_fetched(out_dir: Path, session_type: str) -> bool:
    return (
        (out_dir / f"{session_type}_laps.parquet").exists()
        and (out_dir / f"{session_type}_results.parquet").exists()
    )


def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Cast FastF1 DataFrame subclass to plain DataFrame; expand timedeltas to seconds.

    Parquet handles timedelta64[ns] natively, but downstream tooling reads floats more
    cleanly. We keep the original column AND add a `_s` suffix copy in seconds.
    """
    out = pd.DataFrame(df).copy()
    td_cols = out.select_dtypes(include=["timedelta64[ns]"]).columns
    for col in td_cols:
        out[f"{col}_s"] = out[col].dt.total_seconds()
    return out


def _save_session(session, out_dir: Path, session_type: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"n_laps": 0, "n_drivers": 0}

    try:
        laps = session.laps
    except Exception:
        laps = None
    if laps is not None and len(laps) > 0:
        laps_clean = _clean_for_parquet(laps)
        laps_clean.to_parquet(out_dir / f"{session_type}_laps.parquet", index=False)
        info["n_laps"] = len(laps_clean)
        info["n_drivers"] = int(laps_clean["Driver"].nunique()) if "Driver" in laps_clean else 0

    try:
        results = session.results
    except Exception:
        results = None
    if results is not None and len(results) > 0:
        _clean_for_parquet(results).to_parquet(out_dir / f"{session_type}_results.parquet", index=False)

    try:
        weather = session.weather_data
    except Exception:
        weather = None
    if weather is not None and len(weather) > 0:
        _clean_for_parquet(weather).to_parquet(out_dir / f"{session_type}_weather.parquet", index=False)

    if session_type == "R":
        try:
            rcm = session.race_control_messages
            if rcm is not None and len(rcm) > 0:
                _clean_for_parquet(rcm).to_parquet(out_dir / "R_race_control.parquet", index=False)
        except Exception:
            pass

    return info


def fetch_one(
    year: int,
    round_num: int,
    event_name: str,
    session_type: str,
    force: bool = False,
    max_retries: int = 3,
    raise_on_rate_limit: bool = False,
) -> dict:
    out_dir = _session_dir(year, round_num, event_name)
    record = {
        "year": year,
        "round": round_num,
        "event": event_name,
        "session": session_type,
        "status": "skipped",
        "n_laps": 0,
        "n_drivers": 0,
        "error": "",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not force and _already_fetched(out_dir, session_type):
        return record

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            sess = fastf1.get_session(year, round_num, session_type)
            sess.load(
                laps=True,
                telemetry=False,
                weather=True,
                messages=(session_type == "R"),
            )
            info = _save_session(sess, out_dir, session_type)
            record.update({"status": "ok", **info})
            return record
        except RateLimitExceededError:
            # Let a rate-limit-aware caller (resilient_backfill) cool down and
            # retry instead of burning the remaining fast retries — those just
            # waste more API calls against an already-exhausted quota.
            if raise_on_rate_limit:
                raise
            last_err = "RateLimitExceededError: any API quota exhausted"
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
                continue
            record.update({"status": "error", "error": last_err})
            return record
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
                continue
            record.update({"status": "error", "error": last_err})
            return record
    return record


def _events_for_year(year: int) -> pd.DataFrame:
    sched = fastf1.get_event_schedule(year, include_testing=False)
    if sched is None or sched.empty:
        print(f"[{year}] empty schedule returned by FastF1")
        return pd.DataFrame()

    total = len(sched)

    current_year = pd.Timestamp.now().year
    if year < current_year:
        print(f"[{year}] schedule: {total} events (past season, no date filter)")
        return sched

    date_col = next((c for c in ("Session5Date", "EventDate") if c in sched.columns), None)
    if date_col is None:
        print(f"[{year}] schedule: {total} events (no date column to filter on)")
        return sched

    dates = pd.to_datetime(sched[date_col], errors="coerce", utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")
    mask = (dates <= now_utc).fillna(False)
    kept = sched.loc[mask].copy()
    print(f"[{year}] schedule: {total} events, {len(kept)} completed as of {now_utc.date()}")
    return kept


def fetch_year(year: int, force: bool = False) -> list[dict]:
    records: list[dict] = []
    try:
        sched = _events_for_year(year)
    except Exception as e:
        print(f"[{year}] schedule fetch failed: {e}")
        return records

    if sched.empty:
        print(f"[{year}] no completed events to fetch")
        return records

    pbar = tqdm(sched.iterrows(), total=len(sched), desc=f"{year}", unit="event")
    for _, ev in pbar:
        round_num = int(ev["RoundNumber"])
        event_name = str(ev["EventName"])
        pbar.set_postfix_str(event_name[:30])
        for st in SESSION_TYPES:
            rec = fetch_one(year, round_num, event_name, st, force=force)
            records.append(rec)
    return records


def update_manifest(new_records: Iterable[dict]) -> None:
    df_new = pd.DataFrame(list(new_records))
    if df_new.empty:
        return
    key = ["year", "round", "session"]
    if MANIFEST_PATH.exists():
        df_old = pd.read_parquet(MANIFEST_PATH)
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=key, keep="last")
    else:
        df_combined = df_new
    df_combined.sort_values(key).to_parquet(MANIFEST_PATH, index=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=TRAIN_SEASONS,
        help=f"Seasons to fetch (default: {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]})",
    )
    p.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = p.parse_args()

    init_cache()

    all_records: list[dict] = []
    for year in args.years:
        recs = fetch_year(year, force=args.force)
        all_records.extend(recs)
        update_manifest(recs)

    df = pd.DataFrame(all_records)
    if df.empty:
        print("Nothing fetched.")
        return 0

    print("\n--- Summary by year x status ---")
    print(df.groupby(["year", "status"]).size().unstack(fill_value=0))

    errs = df[df["status"] == "error"]
    if not errs.empty:
        print(f"\n{len(errs)} errors. First few:")
        print(errs[["year", "round", "event", "session", "error"]].head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
