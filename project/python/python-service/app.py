# python-service/app.py
from fastapi import FastAPI, APIRouter, HTTPException
import boto3
import pandas as pd
import os
import joblib  # or tensorflow.keras.models.load_model
from io import StringIO
import uvicorn

app = FastAPI()
router = APIRouter()

# configure these via environment variables or application.yml
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "web-forecast-data")
MODEL_KEY = os.getenv(
    "MODEL_KEY", "models/retail_model/tft_retail.zip"
)  # your model artifact

# init S3 client
s3 = boto3.client("s3", region_name=AWS_REGION)


# load your model once at startup
def load_model():
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    with open("/tmp/model.zip", "wb") as f:
        f.write(obj["Body"].read())
    # assuming you zipped your model; unzip or load directly
    return joblib.load("/tmp/model.pkl")  # or load_model("/tmp/model.h5")


# model = load_model()


@router.get("/fetch-file")
async def fetch_file(key: str):
    """
    Fetches the CSV at S3://<bucket>/<key>, reads it,
    and returns basic info so Java can confirm.
    """
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Key not found: {key}")
    body = obj["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(body))
    return {"s3_key": key, "rows": len(df), "columns": df.columns.tolist()}


@router.get("/predict")
def predict(key: str):
    """
    key: the S3 object key of the uploaded CSV, e.g. "uploads/mydata.csv"
    """
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(404, f"No such key: {key}")

    # read CSV into DataFrame
    csv_str = obj["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(csv_str), parse_dates=["date"])

    # TODO: preprocessing exactly as in train_retail.py
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["unit_sales"] = df["unit_sales"].astype(float)
    df["sell_price"] = df["sell_price"].astype(float)
    # ... add your sliding-window, Dataset creation, etc.

    # run inference (example for TFT)
    # x = your prepared input
    # preds = model.predict(x)

    # for simplicity, echo back the first 5 rows
    return {"prediction": preds[:5].tolist()}


app.include_router(router, prefix="/py")

if __name__ == "__main__":
    # listen on all interfaces inside the container
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
