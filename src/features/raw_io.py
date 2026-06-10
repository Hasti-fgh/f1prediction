"""Helpers for reading the Phase-1 raw Parquet lake.

The lake lives at ``data/raw/{year}_{round:02d}_{slug}/`` with one Parquet per
(session, kind). These helpers enumerate events in chronological order and load
individual frames, returning ``None`` when a file is absent rather than raising.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

_DIR_RE = re.compile(r"^(?P<year>\d{4})_(?P<round>\d{2})_(?P<slug>.+)$")


@dataclass(frozen=True)
class Event:
    year: int
    round: int
    slug: str
    path: Path

    @property
    def name(self) -> str:
        return self.slug.replace("_", " ").strip()

    @property
    def key(self) -> str:
        return f"{self.year}_{self.round:02d}_{self.slug}"


def list_events(years: list[int] | None = None) -> list[Event]:
    """Return events on disk, sorted chronologically (year, then round)."""
    events: list[Event] = []
    if not RAW_DIR.exists():
        return events
    for d in RAW_DIR.iterdir():
        if not d.is_dir():
            continue
        m = _DIR_RE.match(d.name)
        if not m:
            continue
        year = int(m.group("year"))
        if years is not None and year not in years:
            continue
        events.append(Event(year=year, round=int(m.group("round")), slug=m.group("slug"), path=d))
    events.sort(key=lambda e: (e.year, e.round))
    return events


def load(event: Event, session: str, kind: str) -> pd.DataFrame | None:
    """Load one frame, e.g. ``load(ev, "R", "laps")``. Returns None if missing."""
    fp = event.path / f"{session}_{kind}.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


def has_race(event: Event) -> bool:
    return (event.path / "R_laps.parquet").exists() and (event.path / "R_results.parquet").exists()
