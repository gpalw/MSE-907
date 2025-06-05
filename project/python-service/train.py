import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

# --------------------- 0. 超参数 ---------------------
# Sample
# CSV_PATH = r"E:\Yoobee\MSE907\Github\data\processed\retail\sample_retail_item_store.csv"

# Full
CSV_PATH = r"E:\Yoobee\MSE907\Github\data\processed\retail\retail_item_store.csv"
ENCODER_LENGTH = 5  # 历史窗口
DECODER_LENGTH = 2  # 预测窗口
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

print("🚀 启动训练脚本…")
print(f"CSV 路径: {CSV_PATH}")

# --------------------- 1. 读数据 ---------------------
df = pd.read_csv(
    CSV_PATH,
    parse_dates=["date"],
    dtype={"store_id": "str"},  # 避免 dtype 警告
    low_memory=False,  # 7 GB 文件一次读完
)
df["store_id"] = df["store_id"].astype(str)

# 1.1 按日聚合
df_agg = df.groupby(["store_id", "date"], as_index=False)["unit_sales"].sum()

# 1.2 补全日期
all_dates = pd.date_range(df_agg.date.min(), df_agg.date.max(), freq="D")
idx = pd.MultiIndex.from_product(
    [df_agg.store_id.unique(), all_dates], names=["store_id", "date"]
)
df_full = df_agg.set_index(["store_id", "date"]).reindex(idx).fillna(0).reset_index()
df_full["time_idx"] = (df_full["date"] - df_full["date"].min()).dt.days

# 1.3 取每店最后 ENCODER+DECODER 天 & 过滤全 0
WINDOW = ENCODER_LENGTH + DECODER_LENGTH
frames = []
for sid, g in df_full.groupby("store_id"):
    g = g.sort_values("date")
    # 找最长连续段
    block = (g.time_idx.diff() != 1).cumsum()
    longest = block.value_counts().idxmax()
    g = g[block == longest]
    if len(g) >= WINDOW and g.unit_sales.sum() > 0:
        frames.append(g.iloc[-WINDOW:].copy())

df_final = pd.concat(frames, ignore_index=True)
print(f"✅ 最终样本 shape: {df_final.shape}, 门店数: {df_final.store_id.nunique()}")

# --------------------- 2. 建 TimeSeriesDataSet ---------------------
training = TimeSeriesDataSet(
    df_final,
    time_idx="time_idx",
    target="unit_sales",
    group_ids=["store_id"],
    min_encoder_length=ENCODER_LENGTH,
    max_encoder_length=ENCODER_LENGTH,
    min_prediction_length=DECODER_LENGTH,
    max_prediction_length=DECODER_LENGTH,
    static_categoricals=["store_id"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["unit_sales"],
    target_normalizer=GroupNormalizer(groups=["store_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    randomize_length=False,  # 关键 1
    allow_missing_timesteps=False,  # 关键 2
)

# 2.1 检测 “坏窗口” (返回 None 的样本)
bad_idx = [i for i in range(len(training)) if training[i] is None]
if bad_idx:
    print(f"⚠️ 发现 {len(bad_idx)} 个窗口被丢弃 (索引示例: {bad_idx[:10]})")
else:
    print("👍 数据集中没有坏窗口")

# --------------------- 3. DataLoader ---------------------
train_loader: DataLoader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# quick sanity check
x, y = next(iter(train_loader))
print("🟢 第一个 batch 加载成功")
print("   ‣ x keys:", list(x.keys()))

target, weight = y  # y 是 (target, weight)
print("   ‣ target shape:", target.shape)
print("   ‣ weight:", "None" if weight is None else weight.shape)

# --------------------- 4. 定义 TFT ---------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=1,  # 关键 3
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# --------------------- 5. Trainer ---------------------
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    enable_checkpointing=False,  # demo 里不保存 ckpt
    log_every_n_steps=1,
)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"🔧 可训练参数量: {count_trainable(tft):,}")

# --------------------- 6. 开始训练 ---------------------
trainer.fit(tft, train_dataloaders=train_loader)

# --------------------- 7. 保存模型 ---------------------
MODEL_PATH = "tft_big_model.pt"
torch.save(tft.state_dict(), MODEL_PATH)
print(f"🎉 训练完成，模型已保存为 {MODEL_PATH}")
