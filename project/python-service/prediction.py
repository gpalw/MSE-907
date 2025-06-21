# ───────────────── 依赖包 ─────────────────
import os
from pathlib import Path
from io import StringIO
from typing import List, Tuple, Dict

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import HTTPException
from dtaidistance import dtw

# Optional dependency: Must be installed only when using TFT
try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from torch.utils.data import DataLoader
except ImportError:
    TemporalFusionTransformer = None

# ───────────────── Environment/Constant ─────────────────
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "web-forecast-data")
MODEL_KEY = os.getenv("MODEL_KEY", "models/retail_model/tft_lightning_model.ckpt")

BASE_DIR = Path(__file__).resolve().parent
CACHE_BASE = BASE_DIR / "tmp"  # python-service/tmp/
CACHE_BASE.mkdir(exist_ok=True)

LOCAL_MODEL_PATH = CACHE_BASE / "tft_lightning_model.ckpt"
REFERENCE_CSV = (
    BASE_DIR / "retail_item_store.csv"
)  # Refer to large enterprise data (example)

s3 = boto3.client("s3", region_name=AWS_REGION)

ENCODER_LENGTH = 14  # Consistent with training


# ───────────────── 0. Local / S3 cache download tool ─────────────────
def fetch_csv_cached(key: str) -> pd.DataFrame:
    """Priority `/tmp/`; otherwise download and cache from S3"""
    local_path = CACHE_BASE / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"✅ Local cache hit: {local_path}")
        return pd.read_csv(local_path, parse_dates=["date"])

    print(f"⬇️ No files locally, try to get from S3 {key}…")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(404, f"S3 Key not found: {key}")

    csv_str = obj["Body"].read().decode("utf-8")
    local_path.write_text(csv_str, encoding="utf-8")
    print(f"✅ Cachedated {local_path}")
    return pd.read_csv(StringIO(csv_str), parse_dates=["date"])


# ───────────────── 1. DTW Matcher ─────────────────
class DTWMatcher:
    def __init__(self, reference_df: pd.DataFrame, window: int = ENCODER_LENGTH):
        self.window = window
        self.patterns = self._build(reference_df)

    def _build(self, df: pd.DataFrame) -> List[np.ndarray]:
        pats = []
        for _, g in df.groupby("store_id"):
            series = g.sort_values("date")["unit_sales"].to_numpy()
            for i in range(len(series) - self.window + 1):
                clip = series[i : i + self.window]
                std = clip.std() or 1
                pats.append((clip - clip.mean()) / std)
        return pats

    def distance(self, seq: np.ndarray) -> float:
        if len(seq) < self.window:
            return np.inf
        seq = seq[-self.window :]
        seq = (seq - seq.mean()) / (seq.std() or 1)
        return min(dtw.distance(seq, p) for p in self.patterns)


# Reference data is loaded only once
matcher = None


# ───────────────── 2. Pre-trained TFT adapter (optional) ─────────────────
class SmallBusinessAdapter(nn.Module):
    """Example: Freeze base_model, fine-tune with only one layer of adapter. Here only reasoning is not required for training."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        self.adapter = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.adapter(self.base(x))


# ───────────────── 3. Fallback Model ─────────────────
class MovingAverageFallback:
    def __init__(self, window: int = 7):
        self.window = window

    # def predict(self, series: np.ndarray, steps: int):
    #    mean = float(series[-self.window :].mean())
    #     return [mean] * steps

    def predict(self, series, steps, window=7):
        # Handle empty data situation
        if len(series) == 0:
            return [0] * steps

        # Adaptive window size: Take the minimum value of available data and default windows
        actual_window = min(len(series), window)

        # Calculate the moving average (consider the data for the last actual_window days)
        avg = np.mean(series[-actual_window:])

        # Ensure non-negative forecasts (sales cannot be negative)
        return [max(0, avg)] * steps


def load_fallback_model():
    return MovingAverageFallback(window=7)


# ───────────────── 4. TFT model loading (local priority) ─────────────────
def load_tft_model():
    if TemporalFusionTransformer is None:
        raise RuntimeError(
            "pytorch_forecasting is not installed, unable to load TFT model"
        )

    if not LOCAL_MODEL_PATH.exists():
        print("⬇️ No local ckpt found, downloading…")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
        LOCAL_MODEL_PATH.write_bytes(obj["Body"].read())
        print(f"✅ Model cached at {LOCAL_MODEL_PATH}")
    else:
        print(f"✅ Local model found: {LOCAL_MODEL_PATH}")

    model = TemporalFusionTransformer.load_from_checkpoint(str(LOCAL_MODEL_PATH))
    model.eval()
    return model


# ───────────────── 5. Business encapsulation function ─────────────────
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    df = df.dropna(subset=["unit_sales", "date"]).copy()

    # Make sure store_id is a string type
    df["store_id"] = df["store_id"].astype(str)

    df["date"] = pd.to_datetime(df["date"])
    df["unit_sales"] = df["unit_sales"].astype(float)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    can_use_tft = False
    if matcher and len(matcher.patterns) > 0:
        # Check each store separately
        store_distances = {}
        for store_id, store_df in df.groupby("store_id"):
            store_series = store_df.sort_values("date")["unit_sales"].to_numpy()
            dist = matcher.distance(store_series)
            store_distances[store_id] = dist
            print(f"Store {store_id}: DTW distance = {dist:.3f}")

        # Use the minimum distance to judge
        min_distance = min(store_distances.values()) if store_distances else np.inf
        can_use_tft = min_distance < 10.0  # This threshold can be adjusted

        print(f"Minimum DTW distance: {min_distance:.3f} → can use TFT: {can_use_tft}")
    else:
        print("⚠️ DTW matcher not available or no reference pattern")

    return df, can_use_tft


def select_model(can_use_tft: bool):
    if can_use_tft:
        try:
            return load_tft_model(), "TFT"
        except Exception as e:
            print(f"⚠️ TFT loading failed: {e}, fallback simple model")
    return load_fallback_model(), "Fallback"


def predict_sales(model, df: pd.DataFrame, steps: int, model_type: str):
    """
    return:
      {
        "model_used": "TFT",
        "predictions": [
          {"store_id": "101", "forecast": [...]},
          {"store_id": "102", "forecast": [...]},
          ...
        ]
      }
    """
    # ---------- Direct prediction of non-TFT models ----------
    if model_type != "TFT":
        results = []
        for store_id, sub_df in df.groupby("store_id"):
            y_pred = model.predict(sub_df["unit_sales"].to_numpy(), steps)
            results.append(
                {"store_id": str(store_id), "forecast": [float(v) for v in y_pred]}
            )
        return {"model_used": model_type, "predictions": results}

    # ---------- TFT model prediction process ----------
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    except ImportError:
        raise RuntimeError("pytorch_forecasting Not installed")

    check(df)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "unit_sales"])

    results = []
    for store_id, df_store in df.groupby("store_id"):
        pred_dict = _predict_single_store(model, df_store.copy(), steps, model_type)
        results.append(pred_dict)

    return {"model_used": model_type, "predictions": results}


def _predict_single_store(
    model,
    df: pd.DataFrame,
    steps: int,
    model_type: str,
    encoder_length: int = ENCODER_LENGTH,
) -> Dict:
    # Parameter configuration
    store_id = str(df["store_id"].iloc[-1])
    df["store_id"] = store_id
    df = df.sort_values("date")

    # History length check
    if len(df) < encoder_length:
        print(f"⚠️ {store_id} history less than {encoder_length} → using mean fallback")
        fallback_pred = MovingAverageFallback().predict(
            df["unit_sales"].to_numpy(), steps
        )
        return {"store_id": store_id, "forecast": fallback_pred}

    # Use the last encoder_length days as encoder
    hist = df.tail(encoder_length).copy()
    hist["is_prediction"] = 0

    # Prepare placeholder rows for future (decoder)
    last_date = hist["date"].max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=steps, freq="D"
    )
    future_df = pd.DataFrame(
        {
            "date": future_dates,
            "store_id": store_id,
            "unit_sales": hist["unit_sales"].iloc[-1],
            "is_prediction": 1,
        }
    )

    full_df = pd.concat([hist, future_df], ignore_index=True)
    full_df["time_idx"] = (full_df["date"] - full_df["date"].min()).dt.days.astype(
        "int64"
    )
    if "time" in full_df.columns:
        full_df = full_df.drop(columns=["time"])
    full_df["store_id"] = full_df["store_id"].astype(str)

    # Construct the dataset (parameters must be consistent with training)
    dataset = TimeSeriesDataSet(
        full_df,
        time_idx="time_idx",
        target="unit_sales",
        group_ids=["store_id"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=steps,
        max_prediction_length=steps,
        static_categoricals=["store_id"],
        time_varying_known_reals=["time_idx", "is_prediction"],
        time_varying_unknown_reals=["unit_sales"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    try:
        loader = dataset.to_dataloader(batch_size=1, train=False)
        with torch.no_grad():
            preds = model.predict(loader)  # shape: (1, steps)
        forecast = preds[0].cpu().numpy().flatten().tolist()
        return {"store_id": store_id, "forecast": forecast}

    except Exception as e:
        import traceback, sys

        print(f"⚠️ {store_id} TFT prediction failed: {e} → using mean fallback")
        traceback.print_exc(file=sys.stdout)
        fallback_pred = MovingAverageFallback().predict(
            df["unit_sales"].to_numpy(), steps
        )
        return {"store_id": store_id, "forecast": fallback_pred}


def check(df: pd.DataFrame) -> None:
    # Preview
    print(f"⚠️ Preview data (first 5 rows):")
    df = df.copy()
    # Force store_id to string type
    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].astype(str)
    print(df.head())

    # Ensure store_id exists
    if "store_id" not in df.columns:
        df["store_id"] = "user"


def validate_history(df: pd.DataFrame) -> None:
    """
    If any store has less than ENCODER_LENGTH days of history,
    raise HTTP 400 with a list of affected stores in the response body.
    """
    bad = (
        df.groupby("store_id")["date"]
        .nunique()
        .reset_index(name="days")
        .query("days < @ENCODER_LENGTH")
    )

    if not bad.empty:
        # Assemble a user-friendly message
        msg_lines = [
            f"- store_id {row.store_id}: {row.days} days"
            for row in bad.itertuples(index=False)
        ]
        message = (
            f"The following stores have less than {ENCODER_LENGTH} days of history and cannot use TFT:\n"
            + "\n".join(msg_lines)
        )
        # Raise HTTPException directly in FastAPI
        raise HTTPException(status_code=400, detail=message)


def predict_salesGPT(model, df: pd.DataFrame, steps: int, model_type: str):
    # ---------- Simple back ----------
    if model_type != "TFT":
        return model.predict(df["unit_sales"].to_numpy(), steps)

    if TemporalFusionTransformer is None:
        raise RuntimeError("pytorch_forecasting Not installed")

    store = str(df["store_id"].iloc[-1])
    last = df["date"].max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=steps)

    hist = (
        df[df["store_id"] == df["store_id"].iloc[-1]]
        .sort_values("date")
        .tail(ENCODER_LENGTH)
        .copy()
    )
    hist["is_prediction"] = 0
    default_val = hist["unit_sales"].iloc[-1]

    fut = pd.DataFrame(
        {
            "date": future_dates,
            "store_id": store,
            "unit_sales": default_val,  # Not NaN
            "is_prediction": 1,
        }
    )

    pred_df = pd.concat([hist, fut], ignore_index=True)
    pred_df["time_idx"] = (pred_df["date"] - pred_df["date"].min()).dt.days
    pred_df["store_id"] = pred_df["store_id"].astype(str)

    dataset = TimeSeriesDataSet(
        pred_df,
        time_idx="time_idx",
        target="unit_sales",
        group_ids=["store_id"],
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=steps,
        max_prediction_length=steps,
        time_varying_unknown_reals=["unit_sales"],
        time_varying_known_reals=["time_idx", "is_prediction"],
        static_categoricals=["store_id"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # -------- A. Convert to list first, filter out None --------
    samples = [dataset[i] for i in range(len(dataset)) if dataset[i] is not None]
    if len(samples) == 0:
        print("⚠️ No valid samples → using moving average fallback")
        return MovingAverageFallback().predict(df["unit_sales"].to_numpy(), steps)

    # -------- B. Use custom DataLoader --------
    from torch.utils.data import DataLoader, Dataset

    class _ListDataset(Dataset):
        def __init__(self, lst):
            self.lst = lst

        def __len__(self):
            return len(self.lst)

        def __getitem__(self, idx):
            return self.lst[idx]

    loader = DataLoader(_ListDataset(samples), batch_size=1, shuffle=False)

    with torch.no_grad():
        preds = model.predict(loader, return_index=False)

    return preds[0, :, 0].tolist()


# ───────────────── Global Initialization Section (Fixed) ─────────────────

# Do not initialize matcher at module level; use lazy initialization instead
_matcher_cache = None


def get_matcher():
    """Singleton pattern for obtaining the DTW matcher"""
    global _matcher_cache

    if _matcher_cache is not None:
        return _matcher_cache

    print("Initializing DTW matcher...")
    print(f"Reference file path: {REFERENCE_CSV}")
    print(f"File exists: {REFERENCE_CSV.exists()}")

    if not REFERENCE_CSV.exists():
        print(f"⚠️ Reference CSV file not found: {REFERENCE_CSV}")
        print(f"Current directory: {BASE_DIR}")
        print(f"CSV files in directory: {list(BASE_DIR.glob('*.csv'))}")
        _matcher_cache = None
        return None

    try:
        print("Reading reference data...")
        _reference_df = pd.read_csv(REFERENCE_CSV, parse_dates=["date"])
        print(f"Reference data shape: {_reference_df.shape}")
        print(f"Reference data columns: {_reference_df.columns.tolist()}")

        if _reference_df.empty:
            print("⚠️ Reference data is empty")
            _matcher_cache = None
            return None

        # Check for required columns
        required_cols = ["date", "store_id", "unit_sales"]
        missing_cols = [
            col for col in required_cols if col not in _reference_df.columns
        ]
        if missing_cols:
            print(f"❌ Reference data missing required columns: {missing_cols}")
            _matcher_cache = None
            return None

        # Check data quality
        valid_rows = _reference_df.dropna(subset=["unit_sales", "date"])
        print(f"Valid data rows: {len(valid_rows)} / {len(_reference_df)}")

        if len(valid_rows) == 0:
            print("⚠️ No valid data rows")
            _matcher_cache = None
            return None

        # Create DTW matcher
        _matcher_cache = DTWMatcher(valid_rows)
        print(
            f"✅ DTW matcher created successfully, number of patterns: {len(_matcher_cache.patterns)}"
        )
        return _matcher_cache

    except Exception as e:
        print(f"❌ Failed to initialize DTW matcher: {e}")
        import traceback

        traceback.print_exc()
        _matcher_cache = None
        return None


matcher = None  # This global variable is kept for compatibility, but is no longer used

# ───────────────── Fixed Preprocessing Function ─────────────────


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Fixed preprocessing function"""
    df = df.dropna(subset=["unit_sales", "date"]).copy()
    df["store_id"] = df["store_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["unit_sales"] = df["unit_sales"].astype(float)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    can_use_tft = False

    # Use new matcher retrieval
    current_matcher = get_matcher()

    if current_matcher is not None and len(current_matcher.patterns) > 0:
        print(
            f"Starting DTW similarity check, number of reference patterns: {len(current_matcher.patterns)}"
        )

        # Calculate DTW distance for each store
        store_results = []
        for store_id, store_df in df.groupby("store_id"):
            store_series = store_df.sort_values("date")["unit_sales"].to_numpy()

            if len(store_series) < current_matcher.window:
                print(
                    f"Store {store_id}: insufficient data ({len(store_series)} < {current_matcher.window})"
                )
                continue

            dist = current_matcher.distance(store_series)
            store_results.append((store_id, dist))
            print(f"Store {store_id}: DTW distance = {dist:.3f}")

        if store_results:
            min_distance = min(result[1] for result in store_results)
            can_use_tft = min_distance < 10.0
            print(
                f"Minimum DTW distance: {min_distance:.3f} → can use TFT: {can_use_tft}"
            )
        else:
            print("⚠️ No store has enough data for DTW comparison")
    else:
        print("⚠️ DTW matcher not available or no reference pattern")
        print("Possible reasons:")
        print("1. Reference CSV file does not exist")
        print("2. Reference CSV file is empty or formatted incorrectly")
        print("3. Reference data does not have enough valid records")

    print(f"preprocessed data store_id type: {df['store_id'].dtype}")
    return df, can_use_tft


# ───────────────── Add Debug Command ─────────────────


def debug_matcher_issue():
    """Function specifically for debugging matcher issues"""
    print("=" * 60)
    print("DTW MATCHER Debugging")
    print("=" * 60)

    # 1. Check base path
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"BASE_DIR exists: {BASE_DIR.exists()}")
    print(f"REFERENCE_CSV: {REFERENCE_CSV}")
    print(f"REFERENCE_CSV exists: {REFERENCE_CSV.exists()}")

    # 2. List directory contents
    if BASE_DIR.exists():
        print("\nBASE_DIR contents:")
        for item in BASE_DIR.iterdir():
            print(f"  {item.name} ({'file' if item.is_file() else 'directory'})")

    # 3. Try reading the file
    if REFERENCE_CSV.exists():
        try:
            df = pd.read_csv(REFERENCE_CSV)
            print("\n✅ Successfully read reference file")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First 3 rows:\n{df.head(3)}")

            # 4. Try creating matcher
            try:
                test_matcher = DTWMatcher(df)
                print("\n✅ DTW Matcher created successfully")
                print(f"Number of patterns: {len(test_matcher.patterns)}")
            except Exception as e:
                print(f"\n❌ Failed to create DTW Matcher: {e}")

        except Exception as e:
            print(f"\n❌ Failed to read reference file: {e}")
    else:
        print("\n❌ Reference file does not exist")

    print("=" * 60)


# Run debug immediately

debug_matcher_issue()
