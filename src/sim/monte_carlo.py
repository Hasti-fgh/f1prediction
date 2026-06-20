"""Phase 4: Monte Carlo race simulator.

Turns a single race *state* (positions, gaps, tyres, weather at lap L) into a
win probability per driver by simulating the remaining laps thousands of times
and counting how often each driver comes out ahead. The ML estimators supply
only the *parameters* of each simulated lap (pace, safety-car/DNF hazards, pit
loss); identity bias is avoided because the simulator sees race state, not names
(PROGRESS.md decisions #1 and #2).

Each simulated future advances lap by lap:
  * a field-wide safety-car/VSC may begin (bunching the pack, cheapening pits);
  * each running car may retire (DNF hazard);
  * each car runs a lap at its model pace + Gaussian noise, ages its tyres, and
    pits when its planned stint ends (adding pit-lane time loss);
  * the winner of that future is the still-running car with the lowest total
    race time.

The whole thing is vectorised over ``(n_runs, n_drivers)`` numpy arrays. Pace is
served from a per-lap lookup table (driver x compound x tyre-age) so the model is
queried O(laps) times rather than O(laps x runs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import MC_RUNS, RANDOM_SEED
from src.models import spec
from src.models.predictors import Predictors

# Tyre planning: nominal stint length (laps) before a stop, by compound code.
# Used only to decide *when* a simulated car pits, with per-sim jitter.
_STINT_NOMINAL = {0: 18, 1: 26, 2: 34, 3: 22, 4: 18, 5: 26, 6: 26}
_MAX_AGE = 60  # lookup-table cap for tyre age
_PACE_AGE_CAP = 45  # clamp age fed to the pace model (training data thins out beyond this)
_MIN_LAPS_TO_PIT = 8  # never pit if fewer than this many laps remain (no payback)
_SC_DURATION = (2, 5)  # uniform laps a neutralisation lasts
_SC_LAP_PENALTY = 1.35  # SC laps run ~35% slower than green pace
_OVERTAKE_MIN_GAP = 0.7  # seconds a held-up car is clamped behind the car ahead
_DEFAULT_OVERTAKE_PROB = 0.30  # per-lap *base* chance a faster car completes a pass
# Tire-specific overtaking: each lap of tire-age advantage the chasing car holds
# over the car ahead adds this much to its pass probability. A car on much fresher
# rubber (just pitted, or saved its tires) punches through DRS trains far more
# easily -- this is what lets late fresh-tire passes resolve correctly.
_TIRE_OVERTAKE_K = 0.02
_OVERTAKE_PROB_BOUNDS = (0.05, 0.95)  # clamp so a pass is never certain nor impossible

# --- Dynamic weather -------------------------------------------------------- #
# Per-lap rain transition hazards, estimated from the lap data (rain starts
# rarely, dries ~6x faster). Onset is deliberately low so genuinely dry races are
# barely perturbed; the main effect is modelling continued-wet / drying chaos when
# the race is ALREADY wet at the snapshot lap -- which is when our wet-race misses
# happen. Wet running has ~2.6x the lap-time variance and ~1.9x the DNF rate of
# dry running, so we widen pace noise and retirement odds while wet.
_RAIN_START_P = 0.006   # P(dry -> wet) per lap
_RAIN_STOP_P = 0.083    # P(wet -> dry) per lap
_WET_NOISE_MULT = 2.6   # wet pace-noise multiplier vs dry resid_std
_WET_DNF_MULT = 1.9     # wet retirement-hazard multiplier
_WET_THRESHOLD = 0.5    # state.rainfall above this means "wet now"
_INTER_CODE = 3         # COMPOUND_CODES["INTERMEDIATE"] -- the wet-running tyre
# Persistent per-sim, per-driver pace offset ("form of the day"), seconds/lap.
# Drawn once per simulated future and applied every lap, so unlike the i.i.d.
# per-lap noise it does NOT average out over a stint -- it injects correlated
# outcome variance. Swept on 2024+2025 (scripts/calibration_sweep.py): 0.15 gave
# the best ECE (0.023) and lifted top1@90 from 0.625 to 0.646 vs sigma=0. The
# model was already well-calibrated (ECE ~0.025); this is a modest refinement.
_FORM_SIGMA = 0.15


@dataclass
class DriverState:
    driver: str
    position: float
    gap_to_leader: float           # seconds behind the current leader
    compound: str
    tire_age: int
    stint_number: int
    pit_count: int
    elo_pre: float
    is_running: bool = True
    team: str = ""


@dataclass
class RaceState:
    year: int
    round: int
    event: str
    current_lap: int
    total_laps: int
    drivers: list[DriverState]
    track_temp: float = float("nan")
    air_temp: float = float("nan")
    rainfall: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def laps_remaining(self) -> int:
        return max(0, self.total_laps - self.current_lap)


@dataclass
class SimResult:
    p_win: dict[str, float]
    p_podium: dict[str, float]
    p_finish: dict[str, float]
    n_runs: int
    current_lap: int
    total_laps: int

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.p_win.items(), key=lambda kv: kv[1], reverse=True)


class MonteCarlo:
    def __init__(self, predictors: Predictors, n_runs: int = MC_RUNS, seed: int = RANDOM_SEED,
                 form_sigma: float = _FORM_SIGMA):
        self.pred = predictors
        self.n_runs = n_runs
        self.form_sigma = form_sigma
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def _pace_table(self, state: RaceState, lap: int, drivers: list[DriverState],
                    rainfall: float | None = None) -> np.ndarray:
        """Pace for every (driver, compound, tyre-age) at a given lap.

        Returns array shape (D, n_compounds, _MAX_AGE+1). Querying the model once
        per lap keeps the whole simulation at O(laps) model calls. ``rainfall``
        overrides the snapshot value so the caller can build dry (0.0) and wet
        (1.0) tables for dynamic-weather simulation.
        """
        rain = state.rainfall if rainfall is None else rainfall
        D = len(drivers)
        compounds = list(spec.COMPOUND_CODES.values())
        ages = np.arange(_MAX_AGE + 1)
        rows = []
        for d in drivers:
            for c_code in compounds:
                c_name = next(k for k, v in spec.COMPOUND_CODES.items() if v == c_code)
                for age in ages:
                    rows.append({
                        "compound": c_name,
                        "tire_age_laps": age,
                        "stint_number": d.stint_number,
                        "race_progress": lap / state.total_laps,
                        "laps_remaining": state.total_laps - lap,
                        "position": d.position,
                        "elo_pre": d.elo_pre,
                        "track_temp": state.track_temp,
                        "air_temp": state.air_temp,
                        "rainfall": rain,
                    })
        frame = pd.DataFrame(rows)
        preds = self.pred.predict_pace(frame)
        return preds.reshape(D, len(compounds), len(ages))

    # ------------------------------------------------------------------ #
    def simulate(self, state: RaceState) -> SimResult:
        drivers = state.drivers
        D = len(drivers)
        N = self.n_runs
        rng = self.rng
        names = [d.driver for d in drivers]

        if state.laps_remaining == 0 or D == 0:
            # Race is over (or no field): rank by current gap among runners.
            order = sorted(
                [d for d in drivers if d.is_running], key=lambda d: d.gap_to_leader
            )
            p_win = {n: 0.0 for n in names}
            if order:
                p_win[order[0].driver] = 1.0
            return SimResult(p_win, {n: float(d.is_running and d in order[:3]) for n, d in zip(names, drivers)},
                             {n: float(d.is_running) for n, d in zip(names, drivers)}, N,
                             state.current_lap, state.total_laps)

        comp0 = np.array([spec.COMPOUND_CODES.get(d.compound.upper(), 6) for d in drivers])
        # Per-sim mutable state, shape (N, D)
        cum_time = np.tile(np.array([d.gap_to_leader for d in drivers], dtype=float), (N, 1))
        alive = np.tile(np.array([d.is_running for d in drivers]), (N, 1))
        tire_age = np.tile(np.array([d.tire_age for d in drivers], dtype=int), (N, 1))
        compound = np.tile(comp0, (N, 1))
        stint = np.tile(np.array([d.stint_number for d in drivers], dtype=int), (N, 1))
        # Pit count evolves per sim so the DNF hazard sees a realistic stop count;
        # frozen at the snapshot value it would feed the model the out-of-distribution
        # "never pitted at 90% distance" combo (only ever seen for early retirements).
        pit_count = np.tile(np.array([d.pit_count for d in drivers], dtype=int), (N, 1))
        # Planned stint length (with jitter) before the next stop.
        stint_target = np.array([[_STINT_NOMINAL.get(c, 26) for c in comp0]] * N) \
            + rng.integers(-4, 5, size=(N, D))

        sc_timer = np.zeros(N, dtype=int)        # laps of neutralisation still to run
        resid = self.pred.pace_resid_std
        # Persistent "form of the day": one offset per (sim, driver), fixed for the
        # whole simulated race. This is the correlated variance that calibrates P(win).
        form = (rng.normal(0.0, self.form_sigma, size=(N, D))
                if self.form_sigma > 0 else np.zeros((N, D)))
        p_overtake = float(state.meta.get("overtake_prob", _DEFAULT_OVERTAKE_PROB))
        rows_idx = np.arange(N)

        # --- dynamic weather state ------------------------------------------ #
        # Each sim carries its own wet/dry track state, seeded from the snapshot.
        # Cars running wet use intermediates (own age counter) and the wet pace
        # table; switching between dry and wet costs a tyre-change stop.
        wet = np.full(N, (np.nan_to_num(state.rainfall) > _WET_THRESHOLD))
        inter_age = np.zeros((N, D), dtype=int)

        for lap in range(state.current_lap + 1, state.total_laps + 1):
            # --- dynamic weather transition (field-wide, per sim) ------- #
            rain_start = (~wet) & (rng.random(N) < _RAIN_START_P)
            rain_stop = wet & (rng.random(N) < _RAIN_STOP_P)
            switched = rain_start | rain_stop          # changed condition this lap
            wet = (wet & ~rain_stop) | rain_start

            # --- field-wide safety car / VSC onset ---------------------- #
            sc_feat = pd.DataFrame([{
                "race_progress": lap / state.total_laps,
                "lap": lap,
                "total_laps": state.total_laps,
                "laps_since_neut": 9999,
            }])
            p_sc = float(self.pred.predict_sc_onset(sc_feat)[0])
            new_sc = (sc_timer == 0) & (rng.random(N) < p_sc)
            dur = rng.integers(_SC_DURATION[0], _SC_DURATION[1] + 1, size=N)
            sc_timer = np.where(new_sc, dur, sc_timer)
            sc_now = sc_timer > 0

            # --- DNF hazard (per driver this lap) ----------------------- #
            med_age = np.where(alive, tire_age, 0)
            dnf_rows = pd.DataFrame({
                "compound": [next(k for k, v in spec.COMPOUND_CODES.items() if v == c)
                             for c in np.round(np.median(np.where(alive, compound, comp0), axis=0)).astype(int)],
                "tire_age_laps": np.median(med_age, axis=0),
                "stint_number": np.median(stint, axis=0),
                "position": [d.position for d in drivers],
                "gap_to_ahead": 1.5,
                "race_progress": lap / state.total_laps,
                "sc_active": int(False),
                "vsc_active": int(False),
                "pit_count": np.median(pit_count, axis=0),
                "elo_pre": [d.elo_pre for d in drivers],
            })
            p_dnf = self.pred.predict_dnf(dnf_rows)  # shape (D,)
            # Wet running raises the retirement hazard (spins/aquaplaning).
            p_dnf_sim = p_dnf[None, :] * np.where(wet[:, None], _WET_DNF_MULT, 1.0)
            # No retirements while the field is neutralised.
            dnf_draw = (rng.random((N, D)) < p_dnf_sim) & alive & (~sc_now[:, None])
            alive &= ~dnf_draw

            # --- pace lookup for this lap ------------------------------- #
            d_idx = np.arange(D)[None, :]
            age_idx = np.clip(tire_age, 0, _PACE_AGE_CAP)
            # Dry pace on each car's actual slick; wet pace on intermediates.
            table_dry = self._pace_table(state, lap, drivers, rainfall=0.0)  # (D, C, A)
            base_dry = table_dry[d_idx, compound, age_idx]  # (N, D)
            if wet.any():
                table_wet = self._pace_table(state, lap, drivers, rainfall=1.0)
                inter_idx = np.clip(inter_age, 0, _PACE_AGE_CAP)
                base_wet = table_wet[d_idx, np.full((N, D), _INTER_CODE), inter_idx]
                base = np.where(wet[:, None], base_wet, base_dry)
            else:
                base = base_dry
            # Wet laps are far more variable; widen the per-lap noise accordingly.
            noise_std = np.where(wet[:, None], resid * _WET_NOISE_MULT, resid)
            noise = rng.normal(0.0, 1.0, size=(N, D)) * noise_std
            lap_time = base + noise + form  # + persistent per-sim form offset

            # --- pit decision ------------------------------------------- #
            need_pit = alive & (tire_age >= stint_target) & ((state.total_laps - lap) >= _MIN_LAPS_TO_PIT)
            pit_loss = self.pred.pit_mean + rng.normal(0.0, self.pred.pit_std, size=(N, D))
            pit_loss = np.where(sc_now[:, None], pit_loss * 0.55, pit_loss)  # cheaper under SC
            lap_time = lap_time + np.where(need_pit, np.clip(pit_loss, 8.0, None), 0.0)

            # A weather switch forces every running car to change tyres (one stop).
            if switched.any():
                switch_loss = self.pred.pit_mean + rng.normal(0.0, self.pred.pit_std, size=(N, D))
                add = switched[:, None] & alive & (~need_pit)  # don't double-charge a planned stop
                lap_time = lap_time + np.where(add, np.clip(switch_loss, 8.0, None), 0.0)
            # Intermediates age while wet and reset whenever the condition flips.
            inter_age = np.where(switched[:, None], 0, inter_age + wet[:, None].astype(int))

            # --- safety-car effect on lap time -------------------------- #
            lap_time = np.where(sc_now[:, None], lap_time * _SC_LAP_PENALTY, lap_time)

            # apply tyre change for cars that pitted
            new_comp = np.where(compound == 0, 1, 2)  # soft->medium, else->hard (simple)
            compound = np.where(need_pit, new_comp, compound)
            tire_age = np.where(need_pit, 0, tire_age + 1)
            stint = np.where(need_pit, stint + 1, stint)
            pit_count = np.where(need_pit, pit_count + 1, pit_count)
            stint_target = np.where(
                need_pit,
                np.array([[_STINT_NOMINAL.get(int(c), 26) for c in row] for row in compound])
                + rng.integers(-4, 5, size=(N, D)),
                stint_target,
            )

            # track position entering this lap (leader first); dead cars last
            t_start = np.where(alive, cum_time, np.inf)
            entering = np.argsort(t_start, axis=1)

            # only running cars accumulate time
            cum_time = np.where(alive, cum_time + lap_time, cum_time)

            # --- overtaking resistance ---------------------------------- #
            # A car that catches the one ahead can only pass with probability
            # p_overtake (track-dependent). Otherwise it is clamped just behind,
            # forming a DRS-train. Processing front-to-back lets trains propagate.
            if p_overtake < 0.99:
                for slot in range(1, D):
                    ahead = entering[:, slot - 1]
                    behind = entering[:, slot]
                    t_ahead = cum_time[rows_idx, ahead]
                    t_behind = cum_time[rows_idx, behind]
                    both_alive = alive[rows_idx, ahead] & alive[rows_idx, behind]
                    caught = t_behind < t_ahead + _OVERTAKE_MIN_GAP
                    # Fresher tires than the car ahead make the pass more likely.
                    tire_adv = tire_age[rows_idx, ahead] - tire_age[rows_idx, behind]
                    p_pass = np.clip(p_overtake + _TIRE_OVERTAKE_K * tire_adv,
                                     *_OVERTAKE_PROB_BOUNDS)
                    blocked = caught & both_alive & (rng.random(N) >= p_pass)
                    cum_time[rows_idx, behind] = np.where(
                        blocked, t_ahead + _OVERTAKE_MIN_GAP, t_behind
                    )

            # --- safety car bunching ------------------------------------ #
            if sc_now.any():
                # On neutralised laps, compress the pack: gaps collapse toward
                # ~1.0s per position behind the leader.
                for s in np.where(sc_now)[0]:
                    row_alive = alive[s]
                    if row_alive.sum() <= 1:
                        continue
                    t = cum_time[s].copy()
                    t[~row_alive] = np.inf
                    order = np.argsort(t)
                    rank = np.empty(D, dtype=float)
                    rank[order] = np.arange(D)
                    leader_t = t[order[0]]
                    cum_time[s] = np.where(row_alive, leader_t + rank * 1.0, cum_time[s])
                sc_timer = np.maximum(sc_timer - 1, 0)

        # --- tally outcomes --------------------------------------------- #
        t = np.where(alive, cum_time, np.inf)
        order = np.argsort(t, axis=1)            # (N, D) driver indices, best first
        winners = order[:, 0]
        valid = np.isfinite(t[np.arange(N), winners])
        win_counts = np.bincount(winners[valid], minlength=D)

        # podium: top-3 finite finishers per sim
        podium_counts = np.zeros(D, dtype=float)
        for k in range(min(3, D)):
            idx = order[:, k]
            ok = np.isfinite(t[np.arange(N), idx])
            podium_counts += np.bincount(idx[ok], minlength=D)
        finish_counts = alive.sum(axis=0).astype(float)

        p_win = {names[i]: win_counts[i] / N for i in range(D)}
        p_podium = {names[i]: podium_counts[i] / N for i in range(D)}
        p_finish = {names[i]: finish_counts[i] / N for i in range(D)}
        return SimResult(p_win, p_podium, p_finish, N, state.current_lap, state.total_laps)
