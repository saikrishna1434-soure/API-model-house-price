"""Train a deep learning regression model (Keras) for house price prediction.

Workflow:
- Load CSV with pandas
- Split with scikit-learn
- Scale features with StandardScaler
- Train a small feed-forward neural network with TensorFlow/Keras
- Evaluate with MAE, RMSE, R^2
- Save model + scaler

Run:
  python -m src.train_tf --data data/house_price_regression_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(n_features: int, hidden: list[int], lr: float) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    for h in hidden:
        model.add(layers.Dense(h, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/house_price_regression_dataset.csv")
    parser.add_argument("--target", type=str, default="House_Price")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--hidden", type=str, default="64,64", help="Comma-separated hidden layer sizes")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="house_price_model.keras")
    parser.add_argument("--scaler_name", type=str, default="scaler.joblib")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target]).values
    y = df[args.target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]

    # Reproducibility (best effort)
    tf.random.set_seed(args.random_state)
    np.random.seed(args.random_state)

    model = build_model(n_features=X_train_s.shape[1], hidden=hidden, lr=args.lr)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )

    history = model.fit(
        X_train_s,
        y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    preds = model.predict(X_test_s).ravel()

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
    scaler_path = out_dir / args.scaler_name

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print("\nSaved:")
    print(f"- Model : {model_path}")
    print(f"- Scaler: {scaler_path}")


if __name__ == "__main__":
    main()
