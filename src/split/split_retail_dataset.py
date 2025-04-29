import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed/retail"
FILE_NAME = "retail_sales_data.csv"

# create directory if not exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

# load data
raw_file = os.path.join(RAW_PATH, FILE_NAME)
data = pd.read_csv(raw_file)

# transform data
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

# confirm data types
print("Total number of stores:：", data["Store"].nunique())
print("Total record count：", len(data))

# Pre-training data（90%）
store_ids = data["Store"].unique()
train_stores = store_ids[: int(0.9 * len(store_ids))]
train_data = data[data["Store"].isin(train_stores)]

# SME data（With 10% of the stores, only the last 50-100 items are retained per store）
sme_stores = store_ids[int(0.9 * len(store_ids)) :]
sme_data = data[data["Store"].isin(sme_stores)].groupby("Store").tail(80)

# Save processed data
train_data.to_csv(os.path.join(PROCESSED_PATH, "large_scale_dataset.csv"), index=False)
sme_data.to_csv(os.path.join(PROCESSED_PATH, "SME_dataset.csv"), index=False)

print("Pre-training data length：", len(train_data))
print("SME length：", len(sme_data))
