# F1 Race Winner Prediction — Progress Log

_Last updated: 2026-06-10_

---

## Goal

Build a system that predicts the F1 race winner **in real time during the race** (accepting ~30s–2min FastF1 lag), and at the end of the race compares the predicted winner against the actual winner.

---

## Key architectural decisions (and why)

### 1. Hybrid ML + Monte Carlo, not pure ML
A pure ML classifier trained on historical race outcomes would learn "Max usually wins" and be biased toward whoever has dominated recently.

**Decision:** Use ML only to estimate *parameters* (lap-time pace per driver/tire, safety-car rate, DNF probability, pit-stop time). Use a **Monte Carlo simulator** to combine those parameters with the current race state and produce P(win) per driver by simulating the remaining laps 10,000 times.

**Why this beats pure ML:**
- No "Max always wins" bias — the simulator only sees the current race state, not who won historically.
- Real-time updating is automatic — every lap is a new starting state for a fresh 10,000-run simulation.
- Rare events (SC, rain, DNF) are sampled explicitly, not buried inside learned correlations.
- Interpretable — we can show *"Max won 6,247 of 10,000 simulated futures."*
- Mirrors what real F1 strategy teams (McLaren, Red Bull) actually run during a race.

### 2. Never feed driver identity as a feature
Instead of `driver_name = "verstappen"`, the model sees race-situation features:
`current_position`, `gap_to_leader`, `tire_age`, `compound`, `sector_pace_delta`, `pit_count`, `sc_active`, etc.
Driver skill enters via a continuous **Elo/TrueSkill rating** that updates after each race — so high-rated drivers don't auto-win when they're in poor situations (e.g., qualified P15 with a grid penalty).

### 3. Training is offline, inference is live
Common confusion clarified: "training" learns from many past races (done once, offline, refreshed monthly). "Inference" applies the trained model to new inputs (qualifying results, live lap data). **Qualifying results are not training data for race day — they are inputs to the already-trained model.**

### 4. Replay harness before live mode
We can't wait for the next GP to find out if the model is broken. Same code path runs in two modes:
- **Replay:** feed a past race lap-by-lap as if it were live (validates the system).
- **Live:** identical logic, but pulling from FastF1's `LiveTimingClient` during an actual race.

---

## Phase roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Done | Repo scaffold, requirements, FastF1 cache, smoke test |
| 1 | ✅ Done | Bulk-fetch race + qualifying sessions to Parquet lake (+ rate-limit-aware backfill) |
| 2 | ✅ Done | Lap-level feature table + Elo ratings + SC/DNF event labels |
| 3 | ✅ Done | Train four LightGBM estimators (pace, SC, DNF, pit duration) |
| 4 | ✅ Done | Monte Carlo simulator — turns race state into P(win) per driver (+ overtaking model) |
| 5 | ✅ Done | Replay harness + backtest across 2024 season |
| 6 | ✅ Done | Live runner using FastF1 live timing — record / live / replay modes |
| 7 | ✅ Done | Streamlit UI — race selector, live P(win) chart, predicted-vs-actual, calibration |
| 8 | ⏳ | Automation — cron/n8n triggers for post-quali ingest, pre-race snapshot, live loop |

---

## What's done

### Phase 0 — Scaffold ✅
- Project structure created.
- `requirements.txt` with FastF1, pandas, lightgbm, streamlit, tqdm, pyarrow.
- FastF1 cache initialized at `data/cache/`.
- Smoke test passed — confirmed FastF1 can load a session end-to-end.

### Phase 1 — Bulk historical fetcher 🟡
- Built [src/fetch/bulk_history.py](src/fetch/bulk_history.py) — CLI tool that:
  - Pulls qualifying + race sessions for any year(s) supplied via `--years`.
  - Saves laps, results, weather, and race-control messages as Parquet per event.
  - Maintains a `_manifest.parquet` tracking fetch status per (year, round, session).
  - Is idempotent — re-runs skip already-fetched sessions unless `--force` is passed.
  - Auto-filters 2026 to only races already completed (tz-aware UTC comparison).
- Bug fix during development: original date filter was incorrectly dropping past-season events due to a tz-naive vs tz-aware comparison. Now: past seasons skip the filter entirely; current year uses tz-aware comparison.

**Data lake current state:**
```
data/raw/
├── _manifest.parquet
├── 2024_01_Bahrain_Grand_Prix/      ← 7 parquet files (Q + R laps/results/weather + race_control)
├── 2024_02_Saudi_Arabian_Grand_Prix/
├── ...
└── 2024_24_Abu_Dhabi_Grand_Prix/

24 events × 2 sessions = 48 sessions fetched (2024 complete)
```

Remaining backfill (running or pending):
- 2019–2023: ~5 seasons × ~22 events ≈ 220 sessions to fetch
- 2025: ~24 events × 2 sessions = ~48 sessions
- 2026: partial season, ~8–10 events × 2 sessions ≈ ~20 sessions

---

### Phases 2–7 — Full pipeline built ✅

The end-to-end system is implemented and validated. See [README.md](README.md) for
the module map and run commands. Highlights:

- **Phase 2** — [src/features/elo.py](src/features/elo.py) (multiplayer Elo, leak-free
  `elo_pre`) and [src/features/build_features.py](src/features/build_features.py)
  (one row per race/driver/lap, 36 columns). DNF is taken from `ClassifiedPosition`
  being numeric — "Lapped" drivers are finishers, only `R`/`D`/`W` are retirements.
- **Phase 3** — [src/models/train.py](src/models/train.py) trains four LightGBM
  boosters; [src/models/spec.py](src/models/spec.py) pins the feature lists +
  categorical encoding shared with inference.
- **Phase 4** — [src/sim/monte_carlo.py](src/sim/monte_carlo.py): fully vectorised
  `(runs × drivers)` simulator with SC bunching, DNFs, stint-aware pitting, and a
  **track-dependent overtaking-resistance** model (estimated per circuit from
  historical position volatility — this is what makes Monaco ≠ Monza).
- **Phase 5** — [src/sim/replay.py](src/sim/replay.py) + [scripts/backtest.py](scripts/backtest.py).
- **Phase 6** — [src/live/runner.py](src/live/runner.py): `record` / `live` / `replay`.
- **Phase 7** — [src/ui/app.py](src/ui/app.py): Streamlit dashboard.

### Backtest — honest numbers after full 2019–2025 retrain (2026-06-10)

**The earlier "0.88" was overfit** — it was trained *and* tested on the same 24
races of 2024, so the pace model had memorised that season's pecking order. Once
the models are trained across multiple seasons, the honest modern-era accuracy
settles lower. Top-1 accuracy @ 90% race distance (3000 sims/checkpoint):

| Test set | trained on 2024 only (old) | all-years model | 2022+ model |
|----------|---------------------------|-----------------|-------------|
| 2024     | 0.88 (overfit)            | **0.625**       | 0.583       |
| 2025     | —                         | **0.667**       | 0.583       |
| all 158  | —                         | 0.608           | —           |

**Era-mixing hypothesis rejected:** restricting training to the 2022+ ground-
effect era did *not* beat the all-years model (it was marginally worse on modern
seasons), so the older seasons are not the problem. The **all-years model is the
keeper** — marginally better on modern seasons *and* robust across eras. Backup
of it kept at `models_allyears_backup/`.

**Calibration investigated (2026-06-10) — model was already well-calibrated.**
The scary-looking "predicts 0.87, realises 0.60" was the **0.8–1.0 bucket with
only 5 samples** — statistical noise, not a systematic flaw. Across all data the
ECE is ~0.025 and the populated mid-range buckets line up well (2024: 0.4–0.6
predicts 0.497 / realises 0.485; 0.6–0.8 predicts 0.683 / realises 0.727).
- Added a **persistent per-sim "form of the day" offset** to the simulator
  (`monte_carlo._FORM_SIGMA`, seconds/lap), swept via
  [scripts/calibration_sweep.py](scripts/calibration_sweep.py) and a new
  `backtest.py --form-sigma` flag. Chose **0.15**. At 3000 sims its effect on
  top1@90 is within Monte Carlo noise (0.625 either way); it is a principled,
  neutral-to-slightly-positive refinement, not a dramatic fix — because there was
  no dramatic problem.
- Remaining honest caveat: the backtest trains on full-race laps incl. the race
  being scored (season-level leakage), so ~0.62 is still somewhat optimistic. A
  **leave-one-season-out** harness (now feasible with 7 seasons on disk) would
  give a truly clean number and more samples per calibration bucket.
- `train.py --min-year` flag added for the era experiment (default = all years).

---

## Data lake status (2026-06-10) — COMPLETE

All target seasons fetched, 0 errors. 158 events × (Q+R) = 316 sessions on disk.

| Year | Events | Year | Events |
|------|--------|------|--------|
| 2019 | 21 | 2023 | 22 |
| 2020 | 17 (COVID) | 2024 | 24 |
| 2021 | 22 | 2025 | 24 |
| 2022 | 22 | 2026 | 6 (through Monaco) |

Features rebuilt (173,808 lap-rows) and all 4 boosters retrained on the full
history. Re-run anytime; fetch is idempotent:
```powershell
python -m src.features.build_features   # refreshes Elo + lap features
python -m src.models.train              # retrains the 4 boosters (add --min-year YYYY to restrict era)
python scripts/backtest.py --year 2024 --runs 3000  # validate one season (omit --year for all)
```

## Simulator realism work (2026-06-10)

After confirming the model is well-calibrated, added three simulator refinements,
each backtested at 3000 sims/checkpoint on 2024 + 2025:

| Change | Mechanism | Result |
|--------|-----------|--------|
| **Form-of-the-day** offset | persistent per-sim/driver pace delta (σ=0.15 s/lap) | calibration ECE 0.025→0.023; top-1 within noise |
| **Tire-specific overtaking** | pass prob scales with chaser's tyre-age advantage | neutral on top-1 — the misses are chaos, not passing |
| **Dynamic weather** | per-sim rain onset/drying hazards (data-estimated), dry+wet pace tables, intermediate tyres, ×2.6 wet noise, ×1.9 wet DNF, tyre-change stop | 2024 flat (0.625), 2025 +1 race (0.625→0.667); marginal |

**Honest conclusion:** the remaining misses are *irreducible chaos* (DNFs,
collisions, team orders) or *wet driver-skill* the model structurally cannot see
(no driver identity). Tyre/overtaking/weather mechanics can't fix them — the model
already assigns these races correctly low confidence. Charts in
[assets/performance.png](assets/performance.png) (regenerate with
`scripts/plot_performance.py`).

### Performance snapshot (top-1 accuracy)

| distance | 2024 | 2025 |
|----------|------|------|
| 25% | 0.083 | 0.083 |
| 50% | 0.333 | 0.208 |
| 75% | 0.625 | 0.333 |
| 90% | 0.625 | 0.667 |

## Next step

Open decisions / candidate directions:
1. **Leave-one-season-out backtest** — retrain excluding the scored season to
   remove the current season-level leakage and get a truly clean number.
2. **Wet-skill (wet-Elo) rating** — the only lever that meaningfully attacks the
   wet-upset misses, kept Elo-style to respect the no-driver-identity rule.
3. **Undercut/overcut** strategy modelling for sharper final-stint calls.
4. **Phase 8 — automation** (cron/n8n) for post-quali ingest, pre-race snapshot,
   live loop. Building blocks (`src.live.runner`, idempotent fetchers) are ready.
