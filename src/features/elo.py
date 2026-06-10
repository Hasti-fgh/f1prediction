"""Phase 2: per-driver skill ratings via multiplayer Elo.

Driver identity is deliberately kept *out* of the ML feature matrix (see PROGRESS.md
decision #2). Skill instead enters through a single continuous rating that updates
after every race, so a strong driver stuck in a bad race state does not auto-win.

We compute a race-by-race Elo where each race is decomposed into pairwise
comparisons between every pair of classified finishers. A driver who beats a
much higher-rated rival gains more; beating a back-marker gains little. The
rating *going into* a race (``elo_pre``) is leak-free and is what downstream
features consume; ``elo_post`` is the rating after the race is scored.

At a regulation reset (e.g. 2026) ratings are partially regressed toward the
mean, reflecting that prior-era form transfers only weakly to new machinery.

Run:  python -m src.features.elo
Output: data/features/elo_history.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    ELO_HISTORY_PATH,
    ELO_INITIAL,
    ELO_K,
    ELO_REGRESS,
    REG_RESET_YEAR,
)
from src.features.raw_io import Event, list_events, load  # noqa: E402


def _expected(r_a: float, r_b: float) -> float:
    """Logistic expected score of A vs B on the standard 400-point Elo scale."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def _race_order(results: pd.DataFrame) -> list[str]:
    """Classified finishing order (best first) as driver abbreviations.

    ``ClassifiedPosition`` carries letters (R=retired, D=disqualified, ...) for
    non-finishers; numeric entries are the real classification. Non-numeric
    entries are appended after the finishers in results-table order so that DNFs
    still rank below everyone who saw the flag.
    """
    if results is None or results.empty or "Abbreviation" not in results.columns:
        return []
    df = results.copy()
    pos = pd.to_numeric(df.get("Position"), errors="coerce")
    df = df.assign(_pos=pos)
    finishers = df[df["_pos"].notna()].sort_values("_pos")
    dnf = df[df["_pos"].isna()]
    return [*finishers["Abbreviation"].tolist(), *dnf["Abbreviation"].tolist()]


def _update_one_race(ratings: dict[str, float], order: list[str]) -> dict[str, float]:
    """Return per-driver rating deltas from one race's finishing order.

    Every ordered pair contributes: the better-placed driver scores 1, the worse
    scores 0, each compared against the logistic expectation. Deltas are averaged
    over a driver's (n-1) comparisons so K stays interpretable regardless of grid
    size.
    """
    n = len(order)
    if n < 2:
        return {}
    deltas = {d: 0.0 for d in order}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                continue
            score = 1.0 if i < j else 0.0
            exp = _expected(ratings.get(a, ELO_INITIAL), ratings.get(b, ELO_INITIAL))
            deltas[a] += score - exp
    return {d: ELO_K * (deltas[d] / (n - 1)) for d in order}


def compute_elo(years: list[int] | None = None) -> pd.DataFrame:
    """Walk every race chronologically and emit pre/post ratings per driver."""
    events: list[Event] = list_events(years)
    ratings: dict[str, float] = {}
    rows: list[dict] = []
    last_year: int | None = None

    for ev in events:
        results = load(ev, "R", "results")
        order = _race_order(results)
        if not order:
            continue

        # Regulation reset: regress everyone partway to the mean once per new era.
        if ev.year >= REG_RESET_YEAR and last_year is not None and ev.year != last_year:
            for d in list(ratings):
                ratings[d] = ratings[d] + ELO_REGRESS * (ELO_INITIAL - ratings[d])
        last_year = ev.year

        pre = {d: ratings.get(d, ELO_INITIAL) for d in order}
        deltas = _update_one_race(ratings, order)
        for d in order:
            ratings[d] = pre[d] + deltas.get(d, 0.0)
            rows.append(
                {
                    "year": ev.year,
                    "round": ev.round,
                    "event": ev.slug,
                    "driver": d,
                    "elo_pre": pre[d],
                    "elo_post": ratings[d],
                }
            )

    return pd.DataFrame(rows)


def build_and_save(years: list[int] | None = None) -> pd.DataFrame:
    df = compute_elo(years)
    ELO_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ELO_HISTORY_PATH, index=False)
    return df


def main() -> int:
    df = build_and_save()
    if df.empty:
        print("No race results found under data/raw -- run the Phase 1 fetcher first.")
        return 1
    print(f"Wrote {len(df)} driver-race rating rows to {ELO_HISTORY_PATH}")
    latest = (
        df.sort_values(["year", "round"]).groupby("driver").tail(1).sort_values("elo_post", ascending=False)
    )
    print("\nTop 10 by latest Elo:")
    print(latest[["driver", "year", "round", "elo_post"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
