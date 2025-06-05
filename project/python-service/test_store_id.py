# test_store_id.py
import pandas as pd
import torch
from pathlib import Path

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from torch.utils.data import DataLoader
except ImportError:
    print("⚠️ PyTorch Forecasting not installed. Please install it first.")
    exit(1)

print("Testing store_id handling...")

# Create sample data
data = {
    'date': pd.date_range(start='2023-01-01', periods=20),
    'store_id': [123] * 20,  # Numeric store_id
    'unit_sales': [10, 12, 15, 14, 13, 17, 19, 21, 22, 20, 18, 15, 14, 16, 18, 19, 21, 20, 19, 17]
}

df = pd.DataFrame(data)
print("Original data:")
print(df.head())
print(f"Original store_id dtype: {df['store_id'].dtype}")

# Convert store_id to string
df["store_id"] = df["store_id"].astype(str)
print("\nAfter conversion:")
print(df.head())
print(f"Converted store_id dtype: {df['store_id'].dtype}")

# Add additional columns for TimeSeriesDataSet
df["time_idx"] = (df["date"] - df["date"].min()).dt.days
df["is_prediction"] = 0

print("\nPreparing dataset...")
try:
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="unit_sales",
        group_ids=["store_id"],
        min_encoder_length=10,
        max_encoder_length=10,
        min_prediction_length=5,
        max_prediction_length=5,
        static_categoricals=["store_id"],
        time_varying_known_reals=["time_idx", "is_prediction"],
        time_varying_unknown_reals=["unit_sales"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    print("✅ Dataset created successfully!")
    
    # Test creating a dataloader
    dataloader = dataset.to_dataloader(batch_size=1, train=False)
    print("✅ DataLoader created successfully!")
    
    # Check a sample
    x, y = next(iter(dataloader))
    print("\nDataset batch info:")
    print(f"x keys: {list(x.keys())}")
    print(f"store_id in batch: {x['static_categorical_features'][0].item()}")
    
except Exception as e:
    print(f"❌ Error creating dataset: {str(e)}")

print("\nTest completed.")
