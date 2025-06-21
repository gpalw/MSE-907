import os
import pandas as pd

RAW_FILE = "data/raw/coffee_sales_data.xlsx"
PROCESSED_DIR = "data/processed/coffee_shop"

# How many recent daily records are retained in each SME store
SME_RECORDS = 80

os.makedirs(PROCESSED_DIR, exist_ok=True)

# read Excel
df = pd.read_excel(RAW_FILE, engine="openpyxl")

# change data types
df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True)

# order by store_id + transaction_date
daily = (
    df.groupby(["store_id", "transaction_date"])
    .agg(
        Sales=("unit_price", "sum"),
        Customers=("transaction_id", "nunique"),
    )
    .reset_index()
)

print(f"After aggregation, there are {daily['store_id'].nunique()} stores and {len(daily)} daily records.")

# ─── 3. Split pretraining set & SME small sample ──────────────
store_ids = sorted(daily["store_id"].unique())
n_train = int(0.9 * len(store_ids))
train_ids = store_ids[:n_train]
sme_ids = store_ids[n_train:]

train_data = daily[daily["store_id"].isin(train_ids)]
sme_data = daily[daily["store_id"].isin(sme_ids)].groupby("store_id").tail(SME_RECORDS)

print(f"Pretraining store count: {len(train_ids)}, record count: {len(train_data)}")
print(f"SME store count: {len(sme_ids)}, record count: {len(sme_data)}")

# ─── 4. Save results ──────────────────────────────
train_data.to_csv(os.path.join(PROCESSED_DIR, "large_scale_dataset.csv"), index=False)
sme_data.to_csv(os.path.join(PROCESSED_DIR, "SME_dataset.csv"), index=False)

print("Processing complete, files saved in:", PROCESSED_DIR)
