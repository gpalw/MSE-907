import os
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
import hashlib
from typing import Dict, Any

# Will read from environment in Docker or use local default
API_URL = os.getenv("API_URL", "http://localhost:8080/api")

st.set_page_config("SME Forecast UI")

st.title("SME Sales Forecasting")

# Calculate absolute path for the assets directory (works from any working directory)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEMO_RETAIL_FILE_NAME = "demo_retail_item_store.csv"
TEMPLATE_PATH = ASSETS_DIR / DEMO_RETAIL_FILE_NAME


# ✨ Streamlit 1.25+ recommends st.cache_data
@st.cache_data(show_spinner="⌛ Calling prediction service…")
def call_backend(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Sends uploaded CSV to Java→Python service and returns pythonInfo.
    With @st.cache_data, the same bytes + name will only call once.
    """
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()["pythonInfo"]


# ================= Sidebar: Template Download =================
with st.sidebar:
    st.markdown(
        """
        **Field Description**  
        • `date`: YYYY-MM-DD  
        • `store_id`: Store number  
        • `unit_sales`: Daily sales
        """
    )
    with open(TEMPLATE_PATH, "rb") as f:
        st.download_button(
            label="Demo CSV",
            data=f,
            file_name=DEMO_RETAIL_FILE_NAME,
            mime="text/csv",
            help="Sample format",
        )

# ================= Main Area: File Upload =================


def render_forecast_view(py_info: dict) -> None:
    # Show model used
    model_name = py_info.get("model_used", "Unknown")
    st.info(f"**Model used:** {model_name}")

    # 1️⃣ Parse prediction blocks
    blocks = py_info["predicted_sales"]["predictions"]

    # 2️⃣ Build DataFrame: steps as rows, each store_id as a column
    df_pred = (
        pd.DataFrame({blk["store_id"]: blk["forecast"] for blk in blocks})
        .rename_axis("step")
        .reset_index()
    )
    df_pred["step"] += 1  # Start step from 1

    # 3️⃣ Store selection
    store_ids = df_pred.columns.drop("step").tolist()
    choice = st.selectbox("Select View Prediction Point", ["All Items"] + store_ids)

    # 4️⃣ Plot
    if choice == "All Items":
        st.line_chart(df_pred.set_index("step"), height=350)
    else:
        st.line_chart(df_pred.set_index("step")[[choice]], height=350)

    # 5️⃣ Download button (optional)
    st.download_button(
        "Download all predictions CSV",
        df_pred.to_csv(index=False).encode(),
        file_name="all_stores_forecast.csv",
        mime="text/csv",
    )


uploaded = st.file_uploader(
    "Upload your sales CSV", type=["csv"], help="Limit 20MB per file · CSV"
)

if uploaded:
    try:
        py_info = call_backend(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"Upload / predict failed: {e}")
        st.stop()

    st.success("✅ Please select to view the results")
    render_forecast_view(py_info)
