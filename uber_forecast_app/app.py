"""Streamlit app: NYC hourly pickup forecast.

Demo framing for logistics/scheduling roles:
  - "Same-day / next-day" forecast with shift-level breakdown
  - Accuracy reported on the visible window (RMSE / MAE / MAPE)
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

        accuracy = 100 - mape
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{rmse:.0f} pickups")
        c2.metric("MAE", f"{mae:.0f} pickups")
        c3.metric("MAPE", f"{mape:.1f}%")
        c4.metric("Model accuracy", f"{accuracy:.1f}%")

        with st.expander("What do these metrics mean?"):
            st.markdown(
                f"""
**MAE — Mean Absolute Error → {mae:.0f} pickups**
On average, the forecast is off by ±{mae:.0f} pickups per hour (in absolute value).

**RMSE — Root Mean Squared Error → {rmse:.0f} pickups**
Same idea as MAE, but larger errors are penalised more heavily (each error is
squared before averaging). One miss of 500 weighs more than five misses of 100,
even though the totals match.

*Why it matters in logistics:* a single big miss is much worse than many small
ones. Falling 100 packages short for 5 hours → reorganisation. Falling 500
short in one hour → operational collapse.

**MAPE — Mean Absolute Percentage Error →** on average the forecast is off by
{mape:.1f}% of the real value. Unlike RMSE/MAE (in pickups), MAPE is unitless,
which lets us compare across traffic regimes.
                """
            )

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


if __name__ == "__main__":
    main()
