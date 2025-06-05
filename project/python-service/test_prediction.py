import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pathlib import Path
import matplotlib.pyplot as plt
from prediction import predict_sales

# Path to your model checkpoint
MODEL_PATH = Path(__file__).resolve().parent / "tmp" / "tft_lightning_model.ckpt"


def create_test_data():
    """Create some test data for prediction"""
    # Sample store ID
    store_id = "1"

    # Create dates for historical data (14 days for ENCODER_LENGTH)
    dates = pd.date_range(start="2023-01-01", periods=14)

    # Create some sample sales data
    sales = [10, 12, 8, 15, 20, 18, 22, 25, 21, 19, 17, 15, 18, 20]

    # Create a dataframe
    df = pd.DataFrame({"date": dates, "store_id": store_id, "unit_sales": sales})

    return df


def test_prediction():
    """Test prediction functionality"""
    print("Loading TFT model...")
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    model.eval()

    print("Creating test data...")
    test_data = create_test_data()

    print("Input data:")
    print(test_data)

    # Predict 7 days ahead
    prediction_days = 7
    print(f"\nPredicting {prediction_days} days ahead...")

    predictions = predict_sales(model, test_data, prediction_days, "TFT")

    print("\nPrediction results:")
    print(predictions)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(test_data)), test_data["unit_sales"], "b-", label="Historical Data"
    )
    plt.plot(
        range(len(test_data), len(test_data) + prediction_days),
        predictions,
        "r-",
        label="Predictions",
    )
    plt.axvline(
        x=len(test_data) - 1, color="g", linestyle="--", label="Prediction Start"
    )
    plt.xlabel("Days")
    plt.ylabel("Unit Sales")
    plt.title("TFT Model Predictions")
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_test_results.png")
    print("\nPlot saved as 'prediction_test_results.png'")


if __name__ == "__main__":
    test_prediction()
