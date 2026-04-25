"""Load Uber NYC pickups (Apr-Sep 2014) and aggregate to hourly counts."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"C:/Users/Elias/.cache/kagglehub/datasets/fivethirtyeight/uber-pickups-in-new-york-city/versions/2"
)

FILES_2014 = [
    "uber-raw-data-apr14.csv", "uber-raw-data-may14.csv",
    "uber-raw-data-jun14.csv", "uber-raw-data-jul14.csv",
    "uber-raw-data-aug14.csv", "uber-raw-data-sep14.csv",
]


def load_hourly_pickups(data_dir: Path = DATA_DIR) -> pd.Series:
    """Return an hourly Series of NYC Uber pickups, cleaned and IQR-clipped."""
    dfs = [pd.read_csv(data_dir / f) for f in FILES_2014]
    raw = pd.concat(dfs, ignore_index=True)
    raw["Date/Time"] = pd.to_datetime(raw["Date/Time"])

    mask = (
        raw["Lat"].between(40.4, 41.0)
        & raw["Lon"].between(-74.3, -73.6)
        & raw["Date/Time"].notna()
    )
    raw = raw[mask].reset_index(drop=True)

    raw["Hour"] = raw["Date/Time"].dt.floor("h")
    uber = raw.groupby("Hour").size().rename("Pickups")

    full_idx = pd.date_range(uber.index.min(), uber.index.max(), freq="h")
    uber = uber.reindex(full_idx, fill_value=0)
    uber.index.name = "Hour"

    Q1, Q3 = uber.quantile(0.25), uber.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    uber = uber.clip(lower=lo, upper=hi)

    return uber


if __name__ == "__main__":
    y = load_hourly_pickups()
    print(f"Loaded {len(y):,} hourly observations")
    print(f"Range: {y.index[0]} -> {y.index[-1]}")
    print(y.head())
