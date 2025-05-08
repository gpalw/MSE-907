# Week 2: Data Collection & Preprocessing

## 1. Raw Data Location  
All source files are under `data/raw/m5-forecasting-uncertainty/`:
- `calendar.csv`  
- `sell_prices.csv`  
- `sales_train_evaluation.csv`  

## 2. How to Run  
From the project root, execute:
```bash
python split_m5_uncertainty_dataset.py
# This will generate two cleaned CSVs in data/processed/retail/.

python sample_retail_item_store.py
# This will generate demo CSVs in data/processed/retail/.
```

## 3. Script
data_preprocessing_m5.py  
  - Wide→long melt of sales data  
  - Merge calendar and price features  
  - Impute missing prices (median), remove negative sales  
  - Derive is_weekend and is_holiday flags  
  - Save detailed (retail_item_store.csv) and aggregated (retail_daily_store.csv) outputs  

sample_retail_item_store.py  
  - Reads retail_item_store.csv in chunks  
  - Samples ~1,000 rows and writes sample_retail_item_store.csv  

## 4. Output Files  
All outputs are in data/processed/retail/:  
- retail_item_store.csv  
  - Long-form table: one row per (item_id, store_id, date)  
  - Used for pretraining and detailed feature analysis  
- retail_daily_store.csv  
  - Aggregated table: one row per (store_id, date)  
  - Used for store-level baseline models  
- sample_retail_item_store.csv  
  - 1,000-row sample of retail_item_store.csv  
  - For quick inspection in Excel or a text editor  