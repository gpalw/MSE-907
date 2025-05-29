# requirements:
# pandas, numpy, torch, pytorch_forecasting, dtaidistance
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dtaidistance import dtw


# DTWMatcher for pattern similarity
class DTWMatcher:
    def __init__(self, reference_data, window_size=14):
        self.window_size = window_size
        self.patterns = self.extract_patterns(reference_data)

    def extract_patterns(self, data):
        patterns = []
        for _, group in data.groupby("store_id"):
            sales = group["unit_sales"].values
            for i in range(len(sales) - self.window_size + 1):
                pattern = sales[i : i + self.window_size]
                if pattern.std() > 0:
                    pattern = (pattern - pattern.mean()) / pattern.std()
                patterns.append(pattern)
        return patterns

    def find_similar(self, target_sequence, top_k=5):
        target_sequence = target_sequence[-self.window_size :]
        target_norm = (target_sequence - np.mean(target_sequence)) / (
            np.std(target_sequence) or 1
        )
        distances = [(dtw.distance(target_norm, p), p) for p in self.patterns]
        distances.sort(key=lambda x: x[0])
        return distances[:top_k]


# SmallBusinessAdapter for adapting pretrained models
class SmallBusinessAdapter(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        for param in base_model.parameters():
            param.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x):
        base_out = self.base_model(x)
        adapted_out = self.adapter(base_out)
        return adapted_out


# SME Forecasting System
class SMEForecastingSystem:
    def __init__(self, pretrained_model_path, reference_data_path):
        self.reference_data = pd.read_csv(reference_data_path)
        self.base_model = torch.load(pretrained_model_path)
        self.matcher = DTWMatcher(self.reference_data)
        self.adapter_model = SmallBusinessAdapter(self.base_model)

    def preprocess_user_data(self, user_file):
        df = pd.read_csv(user_file)
        df = df.sort_values("date")
        sales = df["unit_sales"].values
        return sales

    def adapt_model(self, user_sales, epochs=50):
        similar_patterns = self.matcher.find_similar(user_sales)
        if not similar_patterns:
            print("No similar patterns found.")
            return
        X_train = np.array([p[1][:-1] for p in similar_patterns]).flatten()
        y_train = np.array(
            [user_sales[-self.matcher.window_size + 1 :]] * len(similar_patterns)
        ).flatten()

        X_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.adapter_model.adapter.parameters(), lr=0.01)

        self.adapter_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.adapter_model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, input_sequence, steps=7):
        self.adapter_model.eval()
        predictions = []
        input_seq = input_sequence.copy()
        with torch.no_grad():
            for _ in range(steps):
                next_pred = self.adapter_model(
                    torch.FloatTensor([input_seq[-1]]).unsqueeze(0)
                ).item()
                predictions.append(next_pred)
                input_seq = np.append(input_seq[1:], next_pred)
        return predictions


# Usage example
def main():
    forecasting_system = SMEForecastingSystem(
        pretrained_model_path="tft_complete_model.pt",
        reference_data_path="retail_item_store.csv",
    )

    user_sales = forecasting_system.preprocess_user_data("user_sales_data.csv")
    forecasting_system.adapt_model(user_sales)

    predictions = forecasting_system.predict(user_sales, steps=7)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
