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
# 业务逻辑函数全部来自 prediction.py
# ──────────────────────────────────────────────────────────────────────────
from prediction import (
    preprocess_data,  # 清洗 + 计算 DTW
    select_model,  # 根据 DTW 选择 TFT or Fallback
    predict_sales,  # 执行预测
    validate_history,  # 验证历史数据是否符合要求
)

# ──────────────────────────────────────────────────────────────────────────
# FastAPI 初始化
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SME Forecasting API")
router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────
# S3 配置（通过环境变量覆写更安全）
# ──────────────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "web-forecast-data")

s3 = boto3.client("s3", region_name=AWS_REGION)


# ──────────────────────────────────────────────────────────────────────────
# 工具：从 S3 读取 CSV → DataFrame
# ──────────────────────────────────────────────────────────────────────────
def fetch_csv_cached(key: str) -> pd.DataFrame:
    local_path = CACHE_BASE / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"✅ 本地缓存命中: {local_path}")
        return pd.read_csv(local_path, parse_dates=["date"])

    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"S3 Key not found: {key}")

    # 写入本地缓存
    csv_str = obj["Body"].read().decode("utf-8")
    local_path.write_text(csv_str, encoding="utf-8")
    print(f"⬇️ 已从 S3 下载并缓存到: {local_path}")

    return pd.read_csv(StringIO(csv_str), parse_dates=["date"])


# ──────────────────────────────────────────────────────────────────────────
# 路由：检查文件
# ──────────────────────────────────────────────────────────────────────────
# @router.get("/fetch-file")
# async def fetch_file(key: str):
#     """
#     简单读取 S3 上的 CSV，返回行数和列名，供前端快速检查。
#     """
#     df = fetch_csv_from_s3(key)
#     return {"s3_key": key, "rows": len(df), "columns": df.columns.tolist()}


# ──────────────────────────────────────────────────────────────────────────
# 路由：核心预测
# ──────────────────────────────────────────────────────────────────────────
@router.get("/predict")
def predict(key: str, horizon: int = 7):
    """
    读取用户 CSV → 预处理 → DTW 相似度判断 → 选择模型 → 预测未来 N 天。
    """
    # Step 1: 加载数据
    df = fetch_csv_cached(key)

    validate_history(df)

    # Step 2: 数据预处理 + DTW 判断
    preprocessed_df, can_use_tft = preprocess_data(df)

    # Step 3: 根据 DTW 选择模型
    model, model_type = select_model(can_use_tft)

    # Step 4: 执行预测
    preds = predict_sales(model, preprocessed_df, steps=horizon, model_type=model_type)

    return {
        "model_used": model_type,  # "TFT" or "Fallback"
        "predicted_sales": preds,  # 未来 horizon 天的销量预测
    }


# 将所有路由挂到 "/py" 前缀下
app.include_router(router, prefix="/py")

# ──────────────────────────────────────────────────────────────────────────
# Docker / 本地运行入口
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))  # 端口随意
    print("🔗 等待 VSCode 连接调试器…")
    # 监听所有网卡，端口 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
