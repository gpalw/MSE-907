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

# 可选依赖：仅在使用 TFT 时必须安装
try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from torch.utils.data import DataLoader
except ImportError:
    TemporalFusionTransformer = None  # 若未安装则仅能用 Fallback

# ───────────────── 环境/常量 ─────────────────
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "web-forecast-data")
MODEL_KEY = os.getenv("MODEL_KEY", "models/retail_model/tft_lightning_model.ckpt")

BASE_DIR = Path(__file__).resolve().parent
CACHE_BASE = BASE_DIR / "tmp"  # 本地缓存目录：python-service/tmp/
CACHE_BASE.mkdir(exist_ok=True)

LOCAL_MODEL_PATH = CACHE_BASE / "tft_lightning_model.ckpt"
REFERENCE_CSV = BASE_DIR / "retail_item_store.csv"  # 参考大企业数据(示例)

s3 = boto3.client("s3", region_name=AWS_REGION)


# ───────────────── 0. 本地 / S3 缓存下载工具 ─────────────────
def fetch_csv_cached(key: str) -> pd.DataFrame:
    """优先 `/tmp/<key>`；否则从 S3 下载并缓存"""
    local_path = CACHE_BASE / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"✅ 本地缓存命中: {local_path}")
        return pd.read_csv(local_path, parse_dates=["date"])

    print(f"⬇️ 本地无文件，尝试从 S3 获取 {key}…")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(404, f"S3 Key not found: {key}")

    csv_str = obj["Body"].read().decode("utf-8")
    local_path.write_text(csv_str, encoding="utf-8")
    print(f"✅ 已缓存到 {local_path}")
    return pd.read_csv(StringIO(csv_str), parse_dates=["date"])


# ───────────────── 1. DTW Matcher ─────────────────
class DTWMatcher:
    def __init__(self, reference_df: pd.DataFrame, window: int = 14):
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


# 参考数据只加载一次
if REFERENCE_CSV.exists():
    _reference_df = pd.read_csv(REFERENCE_CSV, parse_dates=["date"])
else:
    _reference_df = pd.DataFrame(columns=["date", "store_id", "unit_sales"])
matcher = DTWMatcher(_reference_df) if not _reference_df.empty else None
print("matcher loaded:", matcher is not None)


# ───────────────── 2. 预训练 TFT 适配器（可选） ─────────────────
class SmallBusinessAdapter(nn.Module):
    """示例：冻结 base_model，仅用一层 adapter 微调。此处仅推理无需训练。"""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        self.adapter = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.adapter(self.base(x))


# ───────────────── 3. Fallback 模型 ─────────────────
class MovingAverageFallback:
    def __init__(self, window: int = 7):
        self.window = window

    def predict(self, series: np.ndarray, steps: int):
        mean = float(series[-self.window :].mean())
        return [mean] * steps


def load_fallback_model():
    return MovingAverageFallback(window=7)


# ───────────────── 4. TFT 模型加载（本地优先） ─────────────────
def load_tft_model():
    if TemporalFusionTransformer is None:
        raise RuntimeError("pytorch_forecasting 未安装，无法加载 TFT 模型")

    if not LOCAL_MODEL_PATH.exists():
        print("⬇️ 本地无 ckpt，下载中…")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
        LOCAL_MODEL_PATH.write_bytes(obj["Body"].read())
        print(f"✅ 已缓存模型到 {LOCAL_MODEL_PATH}")
    else:
        print(f"✅ 本地模型命中: {LOCAL_MODEL_PATH}")

    model = TemporalFusionTransformer.load_from_checkpoint(str(LOCAL_MODEL_PATH))
    model.eval()
    return model


# ───────────────── 5. 业务封装函数 ─────────────────
ENCODER_LENGTH = 14  # 与训练一致


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    df = df.dropna(subset=["unit_sales", "date"]).copy()

    # 确保 store_id 是字符串类型
    df["store_id"] = df["store_id"].astype(str)

    df["date"] = pd.to_datetime(df["date"])
    df["unit_sales"] = df["unit_sales"].astype(float)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    can_use_tft = False
    if matcher:
        dist = matcher.distance(df["unit_sales"].to_numpy())
        can_use_tft = dist < 10
        print(f"DTW distance={dist:.2f} → can_use_tft={can_use_tft}")

    # 打印 store_id 类型信息，方便调试
    print(f"preprocessed data store_id type: {df['store_id'].dtype}")
    return df, can_use_tft


def select_model(can_use_tft: bool):
    if can_use_tft:
        try:
            return load_tft_model(), "TFT"
        except Exception as e:
            print(f"⚠️ TFT 加载失败: {e}，回退简单模型")
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
    # ---------- 非TFT模型直接预测 ----------
    if model_type != "TFT":
        results = []
        for store_id, sub_df in df.groupby("store_id"):
            y_pred = model.predict(sub_df["unit_sales"].to_numpy(), steps)
            results.append(
                {"store_id": str(store_id), "forecast": [float(v) for v in y_pred]}
            )
        return {"model_used": model_type, "predictions": results}

    # ---------- TFT模型预测流程 ----------
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    except ImportError:
        raise RuntimeError("pytorch_forecasting 未安装")

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
    encoder_length: int = 14,  # 编码器长度与训练时一致
) -> Dict:
    # 参数配置
    store_id = str(df["store_id"].iloc[-1])
    df["store_id"] = store_id  # 保证都是 str
    df = df.sort_values("date")

    # ① 历史长度检查
    if len(df) < encoder_length:
        print(f"⚠️ {store_id} 历史不足 {encoder_length} → 均值回退")
        fallback_pred = MovingAverageFallback().predict(
            df["unit_sales"].to_numpy(), steps
        )
        return {"store_id": store_id, "forecast": fallback_pred}

    # ② 取最后 encoder_length 天做 encoder
    hist = df.tail(encoder_length).copy()
    hist["is_prediction"] = 0

    # ③ 准备 future 占位行（decoder）
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
    full_df["time_idx"] = (full_df["date"] - full_df["date"].min()).dt.days
    full_df["store_id"] = full_df["store_id"].astype(str)

    # ④ 构造数据集（参数需与训练时一致）
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
        loader = dataset.to_dataloader(
            batch_size=1, train=False, batch_sampler="synchronized"
        )
        with torch.no_grad():
            preds = model.predict(loader)  # shape: (1, steps)
        forecast = preds[0].cpu().numpy().flatten().tolist()
        return {"store_id": store_id, "forecast": forecast}

    except Exception as e:
        print(f"⚠️ {store_id} TFT 预测失败: {e} → 均值回退")
        fallback_pred = MovingAverageFallback().predict(
            df["unit_sales"].to_numpy(), steps
        )
        return {"store_id": store_id, "forecast": fallback_pred}


# 回退策略实现
class MovingAverageFallback:
    def predict(self, series, steps, window=7):
        # 处理空数据情况
        if len(series) == 0:
            return [0] * steps

        # 自适应窗口大小：取可用数据和默认窗口的最小值
        actual_window = min(len(series), window)

        # 计算移动平均（考虑最后actual_window天的数据）
        avg = np.mean(series[-actual_window:])

        # 确保非负预测（销售不能为负）
        return [max(0, avg)] * steps


def check(df: pd.DataFrame) -> None:
    # 预览
    print(f"⚠️ 预览数据 (前5行)：")
    df = df.copy()
    # 强制转换 store_id 为字符串类型
    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].astype(str)
    print(df.head())

    # 保证有 store_id
    if "store_id" not in df.columns:
        df["store_id"] = "user"


def validate_history(df: pd.DataFrame) -> None:
    """
    若发现某些 store 历史记录少于 ENCODER_LENGTH 天，就抛出 HTTP 400，
    响应体里列出哪些门店不足多少天。
    """
    bad = (
        df.groupby("store_id")["date"]
        .nunique()
        .reset_index(name="days")
        .query("days < @ENCODER_LENGTH")
    )

    if not bad.empty:
        # 组装成友好的提示
        msg_lines = [
            f"- store_id {row.store_id}: {row.days} 天"
            for row in bad.itertuples(index=False)
        ]
        message = (
            f"下列门店历史记录不足 {ENCODER_LENGTH} 天，无法使用 TFT：\n"
            + "\n".join(msg_lines)
        )
        # FastAPI 内部直接抛 HTTPException
        raise HTTPException(status_code=400, detail=message)


def predict_salesGPT(model, df: pd.DataFrame, steps: int, model_type: str):
    # ---------- 简单回退 ----------
    if model_type != "TFT":
        return model.predict(df["unit_sales"].to_numpy(), steps)

    if TemporalFusionTransformer is None:
        raise RuntimeError("pytorch_forecasting 未安装")

    ENC = 14
    store = str(df["store_id"].iloc[-1])
    last = df["date"].max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=steps)

    hist = (
        df[df["store_id"] == df["store_id"].iloc[-1]]
        .sort_values("date")
        .tail(ENC)
        .copy()
    )
    hist["is_prediction"] = 0
    default_val = hist["unit_sales"].iloc[-1]

    fut = pd.DataFrame(
        {
            "date": future_dates,
            "store_id": store,
            "unit_sales": default_val,  # 绝不能 NaN
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
        min_encoder_length=ENC,
        max_encoder_length=ENC,
        min_prediction_length=steps,
        max_prediction_length=steps,
        time_varying_unknown_reals=["unit_sales"],
        time_varying_known_reals=["time_idx", "is_prediction"],
        static_categoricals=["store_id"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # -------- A. 先转成 list，过滤掉 None --------
    samples = [dataset[i] for i in range(len(dataset)) if dataset[i] is not None]
    if len(samples) == 0:
        print("⚠️ 没有有效样本 → 移动平均回退")
        return MovingAverageFallback().predict(df["unit_sales"].to_numpy(), steps)

    # -------- B. 用自定义 DataLoader --------
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
