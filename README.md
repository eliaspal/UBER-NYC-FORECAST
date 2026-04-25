# Uber NYC — Hourly Demand Forecast

Hour-by-hour demand forecasting on the Uber NYC pickups dataset (Apr-Sep 2014,
~4.5M trips). Built as a portfolio piece for logistics & scheduling roles.

**Live demo:** _add Streamlit Community Cloud URL after deploy_

## Repository layout

| Folder | Purpose |
|---|---|
| [`uber_forecast_app/`](uber_forecast_app/) | Streamlit application — interactive same-day / next-day forecast with shift breakdown and peak alerts |
| [`notebooks/`](notebooks/) | Analysis notebook: EDA, periodogram, ACF/PACF, model selection, walk-forward cross-validation |

## Method (one-paragraph version)

A hybrid model splits the signal into trend (LinearRegression, extrapolatable)
and residual pattern (XGBoost on calendar + lag features, captures daily and
weekly seasonality). Validated with walk-forward cross-validation on 5 weekly
test windows: **mean RMSE 152.6, R² 0.953** — the linear baseline alone scores
RMSE 746.7, so the hybrid cuts error by ~80%.

For the full analysis (periodogram, seasonal plots, residual diagnostics) see
[notebooks/uber_nyc_forecast.ipynb](notebooks/uber_nyc_forecast.ipynb).

## Quick start

```bash
cd uber_forecast_app
pip install -r requirements.txt
streamlit run app.py
```

The pre-trained `model.pkl` is committed in the repo, so the app runs immediately.
To retrain from raw data, run `python train_model.py` first.

## Data source

[FiveThirtyEight Uber Pickups in New York City](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city)
