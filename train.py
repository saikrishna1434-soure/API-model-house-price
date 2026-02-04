import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/house_price_regression_dataset.csv"
MODEL_PATH = "model/house_price_model.joblib"

FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality"
]

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

pipe.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(pipe, MODEL_PATH)

print("Model trained and saved successfully.")
