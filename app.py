import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="House Price Prediction API")

model = joblib.load("model/house_price_model.joblib")

class House(BaseModel):
    Square_Footage: int
    Num_Bedrooms: int
    Num_Bathrooms: int
    Year_Built: int
    Lot_Size: float
    Garage_Size: int
    Neighborhood_Quality: int

@app.post("/predict")
def predict(data: House):
    X = [[
        data.Square_Footage,
        data.Num_Bedrooms,
        data.Num_Bathrooms,
        data.Year_Built,
        data.Lot_Size,
        data.Garage_Size,
        data.Neighborhood_Quality
    ]]
    prediction = model.predict(X)[0]
    return {"predicted_house_price": prediction}
