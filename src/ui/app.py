"""Phase 7: Streamlit dashboard.

Pick a race, scrub to any lap, and the app rebuilds that race state and runs the
Monte Carlo simulator live to show P(win) per driver — plus the full P(win)
trajectory across the race and a predicted-vs-actual verdict at the chequered
flag. A calibration tab reads the backtest output to show how trustworthy the
probabilities are.

Run:  streamlit run src/ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import LAP_FEATURES_PATH  # noqa: E402
from src.models.predictors import Predictors  # noqa: E402
from src.sim.monte_carlo import MonteCarlo  # noqa: E402
from src.sim.replay import replay_to_frame  # noqa: E402
from src.sim.state import actual_winner, event_keys, slice_event, state_at_lap  # noqa: E402

st.set_page_config(page_title="F1 Live Winner Predictor", page_icon="🏁", layout="wide")


@st.cache_data(show_spinner=False)
def _load_features() -> pd.DataFrame:
    if not LAP_FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(LAP_FEATURES_PATH)


@st.cache_resource(show_spinner=False)
def _load_predictors() -> Predictors:
    return Predictors.load()


@st.cache_data(show_spinner="Simulating full race trajectory...")
def _trajectory(year: int, rnd: int, runs: int, step: int) -> pd.DataFrame:
    laps = _load_features()
    mc = MonteCarlo(_load_predictors(), n_runs=runs)
    return replay_to_frame(laps, year, rnd, mc, step=step)


def main() -> None:
    st.title("🏁 F1 Race Winner Predictor")
    st.caption(
        "Hybrid ML + Monte Carlo. ML estimates per-lap parameters (pace, "
        "safety-car / DNF hazards, pit loss); the simulator rolls the remaining "
        "laps thousands of times to turn race *state* into P(win) — no "
        "\"who-usually-wins\" bias."
    )

    laps = _load_features()
    if laps.empty:
        st.error("No feature table found. Run `python -m src.features.build_features` first.")
        return
    pred = _load_predictors()
    if not pred.is_ready:
        st.warning("Models not trained yet — run `python -m src.models.train`. "
                   "Falling back to base-rate priors.")

    races = event_keys(laps)
    races["label"] = races.apply(
        lambda r: f"{int(r['year'])} R{int(r['round']):02d} — {str(r['event']).replace('_', ' ')}", axis=1
    )

    with st.sidebar:
        st.header("Race")
        label = st.selectbox("Select a race", races["label"].tolist(),
                             index=len(races) - 1)
        sel = races[races["label"] == label].iloc[0]
        year, rnd, total_laps = int(sel["year"]), int(sel["round"]), int(sel["total_laps"])
        st.header("Simulation")
        runs = st.select_slider("Monte Carlo runs", [1000, 2000, 5000, 10000], value=2000)
        lap = st.slider("Lap (snapshot)", 1, total_laps, min(total_laps, max(1, total_laps // 2)))

    ev = slice_event(laps, year, rnd)
    truth = actual_winner(ev)
    mc = MonteCarlo(pred, n_runs=runs)
    snapshot_state = state_at_lap(ev, lap)
    res = mc.simulate(snapshot_state)
    ranked = res.ranked()

    tab_live, tab_traj, tab_verdict, tab_calib = st.tabs(
        ["Live snapshot", "P(win) trajectory", "Predicted vs actual", "Calibration"]
    )

    # --- live snapshot ---------------------------------------------------- #
    with tab_live:
        c1, c2 = st.columns([2, 3])
        with c1:
            top_d, top_p = ranked[0]
            st.metric(f"Most likely winner @ lap {lap}/{total_laps}", top_d, f"{top_p:.1%}")
            st.caption(f"Estimated on-track overtake prob for this circuit: "
                       f"{snapshot_state.meta.get('overtake_prob', 0):.2f}")
        with c2:
            snap = pd.DataFrame(
                [{"driver": d, "P(win)": res.p_win[d], "P(podium)": res.p_podium[d],
                  "P(finish)": res.p_finish[d]} for d, _ in ranked]
            ).head(12).set_index("driver")
            st.bar_chart(snap["P(win)"])
        st.dataframe(
            snap.style.format("{:.1%}"), use_container_width=True
        )

    # --- trajectory ------------------------------------------------------- #
    with tab_traj:
        step = 1 if total_laps <= 40 else 2
        traj = _trajectory(year, rnd, min(runs, 2000), step)
        if traj.empty:
            st.info("No trajectory available.")
        else:
            top_drivers = (
                traj.groupby("driver")["p_win"].max().sort_values(ascending=False).head(8).index
            )
            wide = traj[traj["driver"].isin(top_drivers)].pivot(
                index="lap", columns="driver", values="p_win"
            )
            st.line_chart(wide)
            st.caption(f"P(win) per lap for the 8 most-contending drivers. "
                       f"Actual winner: **{truth}**.")

    # --- verdict ---------------------------------------------------------- #
    with tab_verdict:
        final = mc.simulate(state_at_lap(ev, total_laps - 1))
        pred_winner = final.ranked()[0][0] if final.ranked() else "?"
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted winner (final lap)", pred_winner)
        col2.metric("Actual winner", truth or "?")
        hit = pred_winner == truth
        col3.metric("Result", "✅ Correct" if hit else "❌ Missed")
        st.caption("The prediction at the final lap should match the actual "
                   "winner once the race state is fully resolved.")

    # --- calibration ------------------------------------------------------ #
    with tab_calib:
        bt_path = LAP_FEATURES_PATH.parent / "backtest_results.parquet"
        if not bt_path.exists():
            st.info("Run `python scripts/backtest.py` to populate calibration data.")
        else:
            bt = pd.read_parquet(bt_path)
            summary = bt.groupby("frac").agg(
                top1_accuracy=("correct", "mean"),
                mean_brier=("brier", "mean"),
                mean_p_on_winner=("p_winner", "mean"),
            )
            st.subheader("Accuracy by race distance")
            st.line_chart(summary["top1_accuracy"])
            st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)


if __name__ == "__main__":
    main()
