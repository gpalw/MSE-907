import os
import pandas as pd

RAW = "data/raw/m5-forecasting-accuracy"
OUT = "data/processed/retail"
os.makedirs(OUT, exist_ok=True)

# 1. Load raw files
sales = pd.read_csv(f"{RAW}/sales_train_evaluation.csv")
cal = pd.read_csv(f"{RAW}/calendar.csv")
price = pd.read_csv(f"{RAW}/sell_prices.csv")

# 2. Wide→Long melt
d_cols = [c for c in sales.columns if c.startswith("d_")]
sales_long = sales.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    value_vars=d_cols,
    var_name="d",
    value_name="unit_sales",
)

# 3. Merge calendar for real date & time features
cal = cal[
    [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
]
cal["date"] = pd.to_datetime(cal["date"])
sales_long = sales_long.merge(cal, on="d", how="left")

# 4. Merge prices
sales_long = sales_long.merge(price, on=["store_id", "item_id", "wm_yr_wk"], how="left")
# Fill any missing price with the median
sales_long["sell_price"].fillna(sales_long["sell_price"].median(), inplace=True)

# 5. Basic cleaning
#   • Remove negative or absurd sales
sales_long = sales_long[sales_long["unit_sales"] >= 0]

# 6. Feature engineering
sales_long["is_weekend"] = sales_long["weekday"].isin([1, 7]).astype(int)
sales_long["is_holiday"] = (
    sales_long[["event_type_1", "event_type_2"]].notna().any(axis=1).astype(int)
)

# 7. Save the detailed “item–store–date” table
sales_long.to_csv(f"{OUT}/retail_item_store.csv", index=False)

# 8. Aggregate per store & date (for a simpler baseline)
daily = (
    sales_long.groupby(["store_id", "date"])
    .agg(
        Sales=("unit_sales", "sum"),
        AvgPrice=("sell_price", "mean"),
        Promo=("is_holiday", "max"),  # or pick a snap flag if you want
        IsWeekend=("is_weekend", "first"),
        Month=("month", "first"),
        Year=("year", "first"),
    )
    .reset_index()
)
daily.to_csv(f"{OUT}/retail_daily_store.csv", index=False)
