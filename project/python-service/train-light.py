import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

# --------------------- 0. 改进的超参数 ---------------------
CSV_PATH = r"E:\Yoobee\MSE907\Github\data\processed\retail\retail_item_store.csv"
ENCODER_LENGTH = 14  # 增加到2周历史
DECODER_LENGTH = 7  # 预测1周
BATCH_SIZE = 64
EPOCHS = 20  # 增加训练轮数
LR = 1e-3

print("🚀 启动改进的训练脚本…")

# --------------------- 1. 改进的数据处理 ---------------------
df = pd.read_csv(
    CSV_PATH,
    parse_dates=["date"],
    dtype={"store_id": "str"},
    low_memory=False,
)

# 1.1 按日聚合
df_agg = df.groupby(["store_id", "date"], as_index=False)["unit_sales"].sum()

# 1.2 补全日期
all_dates = pd.date_range(df_agg.date.min(), df_agg.date.max(), freq="D")
idx = pd.MultiIndex.from_product(
    [df_agg.store_id.unique(), all_dates], names=["store_id", "date"]
)
df_full = df_agg.set_index(["store_id", "date"]).reindex(idx).fillna(0).reset_index()
df_full["time_idx"] = (df_full["date"] - df_full["date"].min()).dt.days

# 1.3 改进的数据过滤策略
MIN_DAYS = ENCODER_LENGTH + DECODER_LENGTH + 10  # 至少需要31天数据
MIN_SALES = 1  # 至少有1次销售

frames = []
for sid, g in df_full.groupby("store_id"):
    g = g.sort_values("date")

    # 如果门店有足够天数且有销售记录
    if len(g) >= MIN_DAYS and g.unit_sales.sum() > MIN_SALES:
        # 不只取最后几天，而是取所有可用数据
        frames.append(g.copy())

if frames:
    df_final = pd.concat(frames, ignore_index=True)
else:
    print("❌ 没有符合条件的门店数据！")
    exit()

print(f"✅ 数据统计:")
print(f"   ‣ 最终样本 shape: {df_final.shape}")
print(f"   ‣ 门店数: {df_final.store_id.nunique()}")
print(f"   ‣ 时间跨度: {df_final.time_idx.max() - df_final.time_idx.min() + 1} 天")
print(f"   ‣ 总销售额: {df_final.unit_sales.sum():,.0f}")

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
    randomize_length=True,  # 允许随机长度增加样本多样性
    allow_missing_timesteps=True,  # 允许缺失时间步
)

print(f"📈 训练集大小: {len(training):,} 样本")

# 检查坏样本
bad_count = sum(1 for i in range(min(1000, len(training))) if training[i] is None)
print(f"⚠️ 前1000个样本中坏样本数: {bad_count}")

# --------------------- 3. DataLoader ---------------------
train_loader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

print(f"🏋️ 训练批次数: {len(train_loader)}")

# --------------------- 4. 更大的 TFT 模型 ---------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=128,  # 增加到128
    attention_head_size=8,  # 增加注意力头
    dropout=0.2,  # 增加dropout防止过拟合
    hidden_continuous_size=64,  # 增加连续特征隐藏层
    output_size=1,
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=5,
)

# 参数统计
total_params = sum(p.numel() for p in tft.parameters())
trainable_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
print(f"🔧 模型参数:")
print(f"   ‣ 总参数量: {total_params:,}")
print(f"   ‣ 可训练参数量: {trainable_params:,}")
print(f"   ‣ 预估模型大小: {trainable_params * 4 / 1024 / 1024:.1f} MB (float32)")

# --------------------- 5. 训练 ---------------------
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    enable_checkpointing=True,  # 启用检查点
    log_every_n_steps=10,
    default_root_dir="./tft_checkpoints",  # 检查点目录
)

print("🚀 开始训练...")
trainer.fit(tft, train_dataloaders=train_loader)

# --------------------- 6. 保存完整模型 ---------------------
# 方法1: 保存完整模型
torch.save(
    {
        "model_state_dict": tft.state_dict(),
        "model_config": tft.hparams,
        "training_dataset_params": training.get_parameters(),
    },
    "tft_complete_model.pt",
)

# 方法2: 使用 Lightning 的保存方式
trainer.save_checkpoint("tft_lightning_model.ckpt")

# 检查文件大小
import os

for filename in ["tft_complete_model.pt", "tft_lightning_model.ckpt"]:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"💾 {filename}: {size:,} bytes ({size/1024/1024:.1f} MB)")

print("🎉 训练完成！")
