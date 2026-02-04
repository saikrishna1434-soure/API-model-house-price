"""Train a neural network using ONLY scikit-learn (MLPRegressor).

Run:
  python -m src.train_sklearn_mlp --data data/house_price_regression_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/house_price_regression_dataset.csv")
    parser.add_argument("--target", type=str, default="House_Price")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--hidden", type=str, default="64,64")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="sklearn_mlp.joblib")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="adam",
                    learning_rate_init=args.lr,
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    print("\n=== Test Metrics ===")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R^2 : {r2:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / args.model_name
    joblib.dump(pipe, model_path)

    print("\nSaved:")
    print(f"- Model: {model_path}")


if __name__ == "__main__":
    main()
