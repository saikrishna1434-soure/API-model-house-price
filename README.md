# House Price Prediction API

This project trains a machine learning model to predict house prices and exposes it via a FastAPI REST API.

## End-to-End Flow
CSV data → train.py → trained model → FastAPI → /predict endpoint

## Run Instructions
```bash
pip install -r requirements.txt
python src/train.py
uvicorn src.app:app --reload
```
