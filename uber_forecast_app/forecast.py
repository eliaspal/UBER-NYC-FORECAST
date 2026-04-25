"""Hybrid LR + XGBoost forecaster for hourly Uber pickups.

Mirrors the modeling pipeline from the notebook:
  y = trend (LR) + residual_pattern (XGB on lag/calendar features)
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

LAGS = [1, 24, 48, 168]
ROLL_WINDOWS = [24, 168]
MAX_LAG = max(LAGS)

XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=1.0,
    random_state=42, verbosity=0,
)


def make_time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "hour":       idx.hour,
        "dayofweek":  idx.dayofweek,
        "month":      idx.month,
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "is_rush_am": ((idx.hour >= 7)  & (idx.hour <= 9)).astype(int),
        "is_rush_pm": ((idx.hour >= 17) & (idx.hour <= 19)).astype(int),
        "hour_sin":   np.sin(2 * np.pi * idx.hour / 24),
        "hour_cos":   np.cos(2 * np.pi * idx.hour / 24),
        "dow_sin":    np.sin(2 * np.pi * idx.dayofweek / 7),
        "dow_cos":    np.cos(2 * np.pi * idx.dayofweek / 7),
    }, index=idx)


def make_lag_features(ts: pd.Series, lags=LAGS) -> pd.DataFrame:
    return pd.concat({f"lag_{l}": ts.shift(l) for l in lags}, axis=1)


def make_rolling_features(ts: pd.Series, windows=ROLL_WINDOWS) -> pd.DataFrame:
    return pd.concat(
        {f"roll_{w}h": ts.shift(1).rolling(w).mean() for w in windows}, axis=1
    )


def build_xgb_features(ts: pd.Series) -> pd.DataFrame:
    return pd.concat([
        make_time_features(ts.index),
        make_lag_features(ts),
        make_rolling_features(ts),
    ], axis=1)


def build_lr_features(ts: pd.Series, t0: int = 0) -> pd.DataFrame:
    """t0 = the time-step index of ts.index[0] in the original training series."""
    return pd.DataFrame(
        {"const": 1.0, "trend": np.arange(t0, t0 + len(ts))},
        index=ts.index,
    )


@dataclass
class HybridForecaster:
    """Train-once / predict-many wrapper around LR + XGB."""
    lr: LinearRegression
    xgb: XGBRegressor
    train_start: pd.Timestamp
    feature_cols: list

    def fit(self, y: pd.Series) -> "HybridForecaster":
        raise NotImplementedError("Use HybridForecaster.train(y) classmethod")

    @classmethod
    def train(cls, y: pd.Series) -> "HybridForecaster":
        X_lr_full = pd.DataFrame(
            {"const": 1.0, "trend": np.arange(len(y))}, index=y.index
        )
        X_xgb_full = build_xgb_features(y)

        valid = X_xgb_full.dropna().index
        X_lr = X_lr_full.loc[valid]
        X_xgb = X_xgb_full.loc[valid]
        y_aligned = y.loc[valid]

        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_lr, y_aligned)
        resid = y_aligned - lr.predict(X_lr)

        xgb = XGBRegressor(**XGB_PARAMS)
        xgb.fit(X_xgb, resid)

        return cls(
            lr=lr, xgb=xgb,
            train_start=y.index[0],
            feature_cols=X_xgb.columns.tolist(),
        )

    def _trend_index(self, ts_index: pd.DatetimeIndex) -> np.ndarray:
        """Convert timestamps to integer offsets from train_start (1h step)."""
        delta = (ts_index - self.train_start) / pd.Timedelta("1h")
        return delta.to_numpy().astype(int)

    def predict_recursive(
        self,
        history: pd.Series,
        horizon: int,
    ) -> pd.Series:
        """Predict `horizon` future hours starting right after history.index[-1].

        Uses recursive forecasting: each predicted hour becomes input for the
        next hour's lag_1 / rolling features.
        """
        if len(history) < MAX_LAG:
            raise ValueError(
                f"Need at least {MAX_LAG} hours of history, got {len(history)}"
            )

        series = history.copy()
        future_idx = pd.date_range(
            start=history.index[-1] + pd.Timedelta("1h"),
            periods=horizon, freq="h",
        )
        preds = []

        for ts in future_idx:
            series.loc[ts] = np.nan
            X_xgb_row = build_xgb_features(series).loc[[ts], self.feature_cols]

            trend_step = self._trend_index(pd.DatetimeIndex([ts]))
            X_lr_row = pd.DataFrame(
                {"const": 1.0, "trend": trend_step}, index=[ts]
            )

            yhat = float(self.lr.predict(X_lr_row)[0] + self.xgb.predict(X_xgb_row)[0])
            yhat = max(0.0, yhat)
            series.loc[ts] = yhat
            preds.append(yhat)

        return pd.Series(preds, index=future_idx, name="Forecast")
