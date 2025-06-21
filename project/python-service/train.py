import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

# --------------------- 0. Hyperparameters ---------------------
# Sample
# CSV_PATH = r"E:\Yoobee\MSE907\Github\data\processed\retail\sample_retail_item_store.csv"

# Full
CSV_PATH = r"E:\Yoobee\MSE907\Github\data\processed\retail\retail_item_store.csv"
ENCODER_LENGTH = 5  # History window
DECODER_LENGTH = 2  # Prediction window
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

print("🚀 Starting training script…")
print(f"CSV path: {CSV_PATH}")

# --------------------- 1. Load Data ---------------------
df = pd.read_csv(
    CSV_PATH,
    parse_dates=["date"],
    dtype={"store_id": "str"},  # Avoid dtype warning
    low_memory=False,  # Read 7GB file at once
)
df["store_id"] = df["store_id"].astype(str)

# 1.1 Aggregate by day
df_agg = df.groupby(["store_id", "date"], as_index=False)["unit_sales"].sum()

# 1.2 Fill missing dates
all_dates = pd.date_range(df_agg.date.min(), df_agg.date.max(), freq="D")
idx = pd.MultiIndex.from_product(
    [df_agg.store_id.unique(), all_dates], names=["store_id", "date"]
)
df_full = df_agg.set_index(["store_id", "date"]).reindex(idx).fillna(0).reset_index()
df_full["time_idx"] = (df_full["date"] - df_full["date"].min()).dt.days

# 1.3 For each store, take the last ENCODER+DECODER days & filter out all zeros
WINDOW = ENCODER_LENGTH + DECODER_LENGTH
frames = []
for sid, g in df_full.groupby("store_id"):
    g = g.sort_values("date")
    # Find the longest continuous block
    block = (g.time_idx.diff() != 1).cumsum()
    longest = block.value_counts().idxmax()
    g = g[block == longest]
    if len(g) >= WINDOW and g.unit_sales.sum() > 0:
        frames.append(g.iloc[-WINDOW:].copy())

df_final = pd.concat(frames, ignore_index=True)
print(
    f"✅ Final sample shape: {df_final.shape}, Number of stores: {df_final.store_id.nunique()}"
)

# --------------------- 2. Build TimeSeriesDataSet ---------------------
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
    randomize_length=False,  # Key 1
    allow_missing_timesteps=False,  # Key 2
)

# 2.1 Detect "bad windows" (samples that return None)
bad_idx = [i for i in range(len(training)) if training[i] is None]
if bad_idx:
    print(f"⚠️ Found {len(bad_idx)} windows discarded (example indices: {bad_idx[:10]})")
else:
    print("👍 No bad windows in the dataset")

# --------------------- 3. DataLoader ---------------------
train_loader: DataLoader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# quick sanity check
x, y = next(iter(train_loader))
print("🟢 First batch loaded successfully")
print("   ‣ x keys:", list(x.keys()))

target, weight = y  # y is (target, weight)
print("   ‣ target shape:", target.shape)
print("   ‣ weight:", "None" if weight is None else weight.shape)

# --------------------- 4. Define TFT ---------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=1,  # Key 3
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
    enable_checkpointing=False,  # Do not save ckpt in demo
    log_every_n_steps=1,
)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"🔧 Number of trainable parameters: {count_trainable(tft):,}")

# --------------------- 6. Start Training ---------------------
trainer.fit(tft, train_dataloaders=train_loader)

# --------------------- 7. Save Model ---------------------
MODEL_PATH = "tft_big_model.pt"
torch.save(tft.state_dict(), MODEL_PATH)
print(f"🎉 Training complete. Model saved as {MODEL_PATH}")
