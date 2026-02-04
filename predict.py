"""Load a saved model + scaler and predict a house price.

Example:
  python -m src.predict \
    --model models/house_price_model.keras \
    --scaler models/scaler.joblib \
    --features 2000,3,2,2005,1.2,1,7

Feature order must match the CSV (all columns except the target):
  Square_Footage,Num_Bedrooms,Num_Bathrooms,Year_Built,Lot_Size,Garage_Size,Neighborhood_Quality
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/house_price_model.keras")
    parser.add_argument("--scaler", type=str, default="models/scaler.joblib")
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Comma-separated numeric feature values in the correct order",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    x = np.array([[float(v) for v in args.features.split(",")]])

    scaler = joblib.load(scaler_path)
    x_s = scaler.transform(x)

    model = keras.models.load_model(model_path)
    pred = float(model.predict(x_s).ravel()[0])

    print(pred)


if __name__ == "__main__":
    main()
