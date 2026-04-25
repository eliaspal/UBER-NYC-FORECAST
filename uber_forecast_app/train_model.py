"""Train the hybrid forecaster on full history and pickle the result.

Run once:  python train_model.py
Produces:  model.pkl  (forecaster + hourly history series)
"""
import pickle
from pathlib import Path

from data_loader import load_hourly_pickups
from forecast import HybridForecaster

OUT_PATH = Path(__file__).parent / "model.pkl"


def main() -> None:
    print("Loading hourly pickups...")
    y = load_hourly_pickups()
    print(f"  {len(y):,} hours from {y.index[0]} to {y.index[-1]}")

    print("Training hybrid model (LR + XGBoost)...")
    forecaster = HybridForecaster.train(y)

    bundle = {"forecaster": forecaster, "history": y}
    with OUT_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
