import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
import os

# ------------------ Parameters ------------------
CSV_PATH = r"E:\Yoobee\MSE907\Github\data\processed\retail\retail_item_store.csv"
ENCODER_LENGTH = 14
DECODER_LENGTH = 7
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
N_STORES = 10  # Only select 10 stores for debugging

print("🚀 Starting memory-optimized training script…")

# ------------------ 1. Sample Store IDs ------------------
store_ids = set()
for chunk in pd.read_csv(
    CSV_PATH, usecols=["store_id"], dtype={"store_id": "str"}, chunksize=500000
):
    store_ids.update(chunk["store_id"].unique())
    if len(store_ids) >= N_STORES:
        break
store_ids = list(store_ids)[:N_STORES]
print("Sampled stores:", store_ids)

# ------------------ 2. Read main data in chunks ------------------
dfs = []
for chunk in pd.read_csv(
    CSV_PATH,
    parse_dates=["date"],
    dtype={"store_id": "str"},
    low_memory=False,
    chunksize=500000,
):
    dfs.append(chunk[chunk["store_id"].isin(store_ids)])
df = pd.concat(dfs, ignore_index=True)
print(f"Shape after sampling: {df.shape}")

# 1.1 Aggregate by day
df_agg = df.groupby(["store_id", "date"], as_index=False)["unit_sales"].sum()

# 1.2 Fill missing dates
all_dates = pd.date_range(df_agg.date.min(), df_agg.date.max(), freq="D")
idx = pd.MultiIndex.from_product(
    [df_agg.store_id.unique(), all_dates], names=["store_id", "date"]
)
df_full = df_agg.set_index(["store_id", "date"]).reindex(idx).fillna(0).reset_index()
df_full["time_idx"] = (df_full["date"] - df_full["date"].min()).dt.days

# 1.3 Data filtering
MIN_DAYS = ENCODER_LENGTH + DECODER_LENGTH + 10
MIN_SALES = 1

frames = []
for sid, g in df_full.groupby("store_id"):
    g = g.sort_values("date")
    if len(g) >= MIN_DAYS and g.unit_sales.sum() > MIN_SALES:
        frames.append(g.copy())

if frames:
    df_final = pd.concat(frames, ignore_index=True)
else:
    print("❌ No store data meets the requirements!")
    exit()

print(f"✅ Data statistics:")
print(f"   ‣ Final sample shape: {df_final.shape}")
print(f"   ‣ Number of stores: {df_final.store_id.nunique()}")
print(f"   ‣ Time span: {df_final.time_idx.max() - df_final.time_idx.min() + 1} days")
print(f"   ‣ Total sales: {df_final.unit_sales.sum():,.0f}")

# ------------------ 3. TimeSeriesDataSet ------------------
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
    randomize_length=True,
    allow_missing_timesteps=True,
)

print(f"📈 Training set size: {len(training):,} samples")

bad_count = sum(1 for i in range(min(1000, len(training))) if training[i] is None)
print(f"⚠️ Number of bad samples in the first 1000: {bad_count}")

train_loader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

print(f"🏋️ Number of training batches: {len(train_loader)}")

# ------------------ 4. Small TFT Model ------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=32,
    attention_head_size=2,
    dropout=0.2,
    hidden_continuous_size=16,
    output_size=1,
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=5,
)

total_params = sum(p.numel() for p in tft.parameters())
trainable_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
print(f"🔧 Model parameters:")
print(f"   ‣ Total parameters: {total_params:,}")
print(f"   ‣ Trainable parameters: {trainable_params:,}")
print(f"   ‣ Estimated model size: {trainable_params * 4 / 1024 / 1024:.1f} MB (float32)")

# ------------------ 5. Training ------------------
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    enable_checkpointing=True,
    log_every_n_steps=10,
    default_root_dir="./tft_checkpoints",
)

print("🚀 Starting training...")
trainer.fit(tft, train_dataloaders=train_loader)

# ------------------ 6. Save ------------------
torch.save(
    {
        "model_state_dict": tft.state_dict(),
        "model_config": tft.hparams,
        "training_dataset_params": training.get_parameters(),
    },
    "tft_complete_model.pt",
)
trainer.save_checkpoint("tft_lightning_model.ckpt")

for filename in ["tft_complete_model.pt", "tft_lightning_model.ckpt"]:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"💾 {filename}: {size:,} bytes ({size/1024/1024:.1f} MB)")

print("🎉 Training complete!")
