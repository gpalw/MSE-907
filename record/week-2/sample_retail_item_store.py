import pandas as pd

# Path to your huge file
IN_PATH = "data/processed/retail/retail_item_store.csv"
OUT_PATH = "data/processed/retail/sample_retail_item_store.csv"

# Read in chunks of 100 000 rows, but stop after we've sampled 1 000
chunksize = 100_000
sample_rows = []
for chunk in pd.read_csv(IN_PATH, chunksize=chunksize):
    sample_rows.append(chunk.sample(n=200, random_state=42))  # 200×5 chunks ≈1 000 rows
    if len(sample_rows) * 200 >= 1_000:
        break

# Concatenate and save
sample_df = pd.concat(sample_rows, ignore_index=True)
sample_df.to_csv(OUT_PATH, index=False)
print(f"Sample written to {OUT_PATH} ({len(sample_df)} rows)")
