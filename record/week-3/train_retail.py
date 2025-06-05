import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, NHiTS
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

# 1) Load the data
df = pd.read_csv(
    "data/processed/retail/retail_item_store.csv",
    parse_dates=["date"],
    low_memory=False,
)

# 2) Create integer time index
df["time_idx"] = (df["date"] - df["date"].min()).dt.days

# 3) Cast sales and price to float (required for softplus normalizer)
df["unit_sales"] = df["unit_sales"].astype(float)
df["sell_price"] = df["sell_price"].astype(float)

# 4) Define train/validation cutoff
max_prediction_length = 7
max_encoder_length = 30
train_cutoff = df["time_idx"].max() - max_prediction_length

# 5) Split into train / validation DataFrames
train_df = df[df["time_idx"] <= train_cutoff].copy()
val_df = df[df["time_idx"] > train_cutoff].copy()

# 6) Common TimeSeriesDataSet arguments
common_args = dict(
    time_idx="time_idx",
    target="unit_sales",
    group_ids=["store_id", "item_id"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["store_id", "item_id"],
    time_varying_unknown_reals=["unit_sales", "sell_price"],
    target_normalizer=GroupNormalizer(
        groups=["store_id", "item_id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# 7) Create TimeSeriesDataSet objects
train_dataset = TimeSeriesDataSet(train_df, **common_args)
val_dataset = TimeSeriesDataSet(val_df, **common_args)

# 8) Create dataloaders
train_loader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=4)
val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=4)

# 9) Define and train the Temporal Fusion Transformer
tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(tft, train_loader, val_loader)
tft.save_model("models/retail_model/tft_retail")

# 10) Evaluate TFT
tft_metrics = trainer.test(tft, dataloaders=val_loader, verbose=False)
print("TFT validation metrics:", tft_metrics)

# 11) Define and train N-HiTS as a lightweight alternative
nhits = NHiTS.from_dataset(
    train_dataset, learning_rate=1e-3, log_interval=10, loss=MAE()
)

trainer.fit(nhits, train_loader, val_loader)
nhits.save_model("models/retail_model/nhits_retail")

# 12) Evaluate N-HiTS
nhits_metrics = trainer.test(nhits, dataloaders=val_loader, verbose=False)
print("N-HiTS validation metrics:", nhits_metrics)
