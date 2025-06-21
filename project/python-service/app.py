# python-service/app.py
from fastapi import FastAPI, APIRouter, HTTPException
import pandas as pd
import boto3
import os
from pathlib import Path
from io import StringIO
import uvicorn

CACHE_BASE = Path(__file__).resolve().parent / "tmp"
CACHE_BASE.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# FROM  prediction.py
# ──────────────────────────────────────────────────────────────────────────
from prediction import (
    preprocess_data,  # Clean + Calculate DTW
    select_model,  #  Choose TFT or Fallback according to DTW
    predict_sales,  # Execute predictions
    validate_history,  # Verify that historical data meets the requirements
)

# ──────────────────────────────────────────────────────────────────────────
# FastAPI Init
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SME Forecasting API")
router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────
# S3 Config
# ──────────────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "web-forecast-data")

s3 = boto3.client("s3", region_name=AWS_REGION)


# ──────────────────────────────────────────────────────────────────────────
# read CSV → DataFrame from S3
# ──────────────────────────────────────────────────────────────────────────
def fetch_csv_cached(key: str) -> pd.DataFrame:
    local_path = CACHE_BASE / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"✅ Local cache hit: {local_path}")
        return pd.read_csv(local_path, parse_dates=["date"])

    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"S3 Key not found: {key}")

    # Write to local cache
    csv_str = obj["Body"].read().decode("utf-8")
    local_path.write_text(csv_str, encoding="utf-8")
    print(f"⬇️ Downloaded and cached from S3: {local_path}")

    return pd.read_csv(StringIO(csv_str), parse_dates=["date"])


# ──────────────────────────────────────────────────────────────────────────
# Routing: Check the file
# ──────────────────────────────────────────────────────────────────────────
# @router.get("/fetch-file")
# async def fetch_file(key: str):
#     """
#     Simple reading of CSV on S3, returning row count and column name for quick checking of front-end
#     """
#     df = fetch_csv_from_s3(key)
#     return {"s3_key": key, "rows": len(df), "columns": df.columns.tolist()}


# ──────────────────────────────────────────────────────────────────────────
# Routing: Core Prediction
# ──────────────────────────────────────────────────────────────────────────
@router.get("/predict")
def predict(key: str, horizon: int = 7):
    print(f"==[Predictive interface call]== key={key} horizon={horizon}")
    """
    Read user CSV → Preprocessing → DTW Similarity Judgment → Select Model → Predict the next 7 days。
    """

    # Step 1: Loading data
    df = fetch_csv_cached(key)
    print(f"[Step1] Loaded {len(df)} rows, columns: {df.columns.tolist()}")

    print("[Step1] Validating historical data...")
    validate_history(df)
    print("[Step1] Validation passed")

    # Step 2: Data preprocessing + DTW check
    print("[Step2] Preprocessing + DTW")
    preprocessed_df, can_use_tft = preprocess_data(df)
    print(
        f"[Step2] Preprocessing result: {preprocessed_df.shape}, can_use_tft={can_use_tft}"
    )

    # Step 3: Select model based on DTW
    print("[Step3] Selecting model")
    model, model_type = select_model(can_use_tft)
    print(f"[Step3] Model used: {model_type}")

    # Step 4: Run prediction
    print("[Step4] Running prediction")
    preds = predict_sales(model, preprocessed_df, steps=horizon, model_type=model_type)
    print(f"[Step4] model_type: {model_type}, prediction complete: {preds}")

    return {
        "model_used": model_type,  # "TFT" or "Fallback"
        "predicted_sales": preds,  # Predicted sales for the next 'horizon' days
    }


# Hang all routes under the "/py" prefix
app.include_router(router, prefix="/py")

# ──────────────────────────────────────────────────────────────────────────
# Docker / Local operation portal
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("🔗 Wait for VSCode to connect to the debugger…")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
