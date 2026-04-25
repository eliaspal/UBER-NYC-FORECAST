"""Streamlit app: NYC hourly pickup forecast.

Demo framing for logistics/scheduling roles:
  - "Same-day / next-day" forecast with shift-level breakdown
  - Peak alerts in plain business language
  - Model accuracy reported alongside (RMSE / MAE / MAPE)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model.pkl"

SHIFTS = {
    "Morning (06-14)":   range(6, 14),
    "Afternoon (14-22)": range(14, 22),
    "Night (22-06)":     list(range(22, 24)) + list(range(0, 6)),
}


st.set_page_config(
    page_title="NYC Hourly Demand Forecast",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def build_forecast_chart(history_window, forecast, actual_future):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_window.index, y=history_window.values,
        mode="lines", name="History",
        line=dict(color="#888", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        mode="lines", name="Forecast",
        line=dict(color="#d62728", width=2.5),
    ))
    if actual_future is not None and len(actual_future):
        fig.add_trace(go.Scatter(
            x=actual_future.index, y=actual_future.values,
            mode="lines", name="Actual",
            line=dict(color="#1f77b4", width=2, dash="dot"),
        ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05, x=0.01),
        xaxis_title=None, yaxis_title="Pickups per hour",
        hovermode="x unified",
    )
    return fig


def shift_volume(series: pd.Series, hours: range) -> float:
    mask = series.index.hour.isin(list(hours))
    return float(series.loc[mask].sum())


def recent_baseline(history: pd.Series, anchor: pd.Timestamp, weeks: int = 4) -> pd.Series:
    """Hour-of-day average over the last `weeks` weeks before the anchor.

    A scheduler cares about deviations from the recent regime, not from a 6-month
    flat average that masks trend.
    """
    window = history.loc[anchor - pd.Timedelta(weeks=weeks): anchor]
    return window.groupby(window.index.hour).mean()


def detect_peaks(forecast: pd.Series, baseline: pd.Series, threshold: float = 1.25):
    """Flag forecast hours where demand exceeds the recent hour-of-day average * threshold."""
    alerts = []
    for ts, value in forecast.items():
        base = baseline.loc[ts.hour]
        ratio = value / base if base > 0 else 1.0
        if ratio >= threshold:
            alerts.append({
                "time": ts, "forecast": value,
                "baseline": base, "ratio": ratio,
            })
    return alerts


def main():
    bundle = load_bundle()
    forecaster = bundle["forecaster"]
    history = bundle["history"]

    st.title("NYC Hourly Pickup Forecast")
    st.caption(
        "Hybrid LinearRegression + XGBoost · "
        f"trained on {len(history):,} hourly observations "
        f"({history.index[0].date()} -> {history.index[-1].date()})"
    )

    # ── Sidebar controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("Forecast settings")
        min_anchor = (history.index[0] + pd.Timedelta(hours=200)).date()
        max_anchor = (history.index[-1] - pd.Timedelta(hours=49)).date()
        default_anchor = pd.Timestamp("2014-09-23").date()

        anchor_date = st.date_input(
            "Forecast anchor (today)",
            value=default_anchor,
            min_value=min_anchor, max_value=max_anchor,
        )
        anchor_hour = st.slider("Anchor hour", 0, 23, 23)
        horizon = st.radio("Horizon", [24, 48], horizontal=True)

        anchor = pd.Timestamp(anchor_date) + pd.Timedelta(hours=int(anchor_hour))

        st.divider()
        st.markdown(
            "**How to read this app**  \n"
            "1. Pick an anchor timestamp (the 'now')  \n"
            "2. The model forecasts the next 24-48h  \n"
            "3. Compare against the actual future to validate accuracy"
        )

    # ── Run forecast ──────────────────────────────────────────────────
    history_slice = history.loc[:anchor]
    forecast = forecaster.predict_recursive(history_slice, horizon=int(horizon))

    future_end = anchor + pd.Timedelta(hours=int(horizon))
    actual_future = history.loc[anchor + pd.Timedelta("1h"): future_end]

    # ── Section 1: forecast chart ─────────────────────────────────────
    st.subheader(f"Forecast — next {horizon}h after {anchor}")
    history_window = history.loc[anchor - pd.Timedelta(hours=72): anchor]
    st.plotly_chart(
        build_forecast_chart(history_window, forecast, actual_future),
        width="stretch",
    )

    # Accuracy on this specific window
    if len(actual_future) == len(forecast):
        err = actual_future.values - forecast.values
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        mask = actual_future.values > 50
        mape = float(
            np.mean(np.abs(err[mask] / actual_future.values[mask])) * 100
        ) if mask.any() else float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.0f} pickups")
        c2.metric("MAE", f"{mae:.0f} pickups")
        c3.metric("MAPE", f"{mape:.1f}%")

    st.divider()

    # ── Section 2: shift breakdown ────────────────────────────────────
    st.subheader("Shift breakdown")
    st.caption("Baseline = average of the same hour-of-day over the last 4 weeks before the anchor.")
    baseline = recent_baseline(history, anchor, weeks=4)
    cols = st.columns(3)

    for col, (shift_name, shift_hours) in zip(cols, SHIFTS.items()):
        forecast_volume = shift_volume(forecast, shift_hours)
        baseline_per_hour = baseline.loc[list(shift_hours)].mean()
        baseline_volume = baseline_per_hour * len(list(shift_hours))
        delta_pct = (forecast_volume / baseline_volume - 1) * 100

        col.metric(
            shift_name,
            f"{forecast_volume:,.0f} pickups",
            f"{delta_pct:+.1f}% vs 4w avg",
        )

    st.divider()

    # ── Section 3: peak alerts ────────────────────────────────────────
    st.subheader("Peak alerts")
    alerts = detect_peaks(forecast, baseline, threshold=1.25)

    if not alerts:
        st.info("No demand peaks above 25% over the historical baseline.")
    else:
        peak_df = pd.DataFrame(alerts)
        peak_df["delta_%"] = ((peak_df["ratio"] - 1) * 100).round(0)
        peak_df["forecast"] = peak_df["forecast"].round(0).astype(int)
        peak_df["baseline"] = peak_df["baseline"].round(0).astype(int)
        peak_df["time"] = peak_df["time"].dt.strftime("%a %d %b %H:%M")
        peak_df = peak_df[["time", "forecast", "baseline", "delta_%"]]
        peak_df.columns = ["When", "Forecast", "Hist. avg", "Delta %"]
        st.dataframe(peak_df, hide_index=True, width="stretch")

        st.markdown(
            f"**Action**: {len(alerts)} hour(s) flagged with demand "
            "more than 25% above the historical baseline. "
            "Consider adding capacity to absorb the surge."
        )

    st.divider()

    # ── Section 4: model accuracy context ─────────────────────────────
    with st.expander("How accurate is this model? (cross-validation)"):
        st.markdown(
            "**Walk-forward CV** on 5 separate 1-week test windows from the "
            "training period (no data leakage):"
        )
        cv_data = pd.DataFrame({
            "Test week":    ["2014-08-26", "2014-09-02", "2014-09-09",
                             "2014-09-16", "2014-09-23"],
            "RMSE":         [130.2, 178.2, 150.0, 134.4, 170.3],
            "MAE":          [98.0, 142.6, 111.3, 94.8, 142.8],
            "MAPE %":       [15.07, 15.41, 10.55, 8.97, 15.64],
            "R squared":    [0.936, 0.949, 0.965, 0.970, 0.947],
        })
        st.dataframe(cv_data, hide_index=True, width="stretch")
        st.caption(
            "Mean RMSE 152.6 / R squared 0.953. "
            "Baseline (linear trend only) RMSE was 746.7 -> hybrid reduces error 80%."
        )


if __name__ == "__main__":
    main()
