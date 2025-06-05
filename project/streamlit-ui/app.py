import os
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
import hashlib
from typing import Dict, Any

# will read from environment in Docker or locally default
API_URL = os.getenv("API_URL", "http://localhost:8080/api")

st.set_page_config("SME Forecast UI")

st.title("SME Sales Forecasting")

# 计算 assets 目录的绝对路径（保证在任何工作目录都能找到）
ASSETS_DIR = Path(__file__).parent / "assets"
DEMO_RETAIL_FILE_NAME = "demo_retail_item_store.csv"
TEMPLATE_PATH = ASSETS_DIR / DEMO_RETAIL_FILE_NAME


# ✨ Streamlit 1.25+ 推荐用 st.cache_data
@st.cache_data(show_spinner="⌛ Calling prediction service…")
def call_backend(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    把上传的 CSV 发给 Java→Python 服务，返回 pythonInfo。
    由于带 @st.cache_data，传入相同 bytes + name 时只调用一次。
    """
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()["pythonInfo"]


# ================= 侧边栏：模板下载 =================
with st.sidebar:
    col_exp, col_btn = st.columns([4, 2])
    with col_exp:
        st.markdown(
            """
            **Field Description**  
            • `date`：YYYY-MM-DD  
            • `store_id`：Store number  
            • `unit_sales`：Daily sales
            """
        )

    with col_btn:
        with open(TEMPLATE_PATH, "rb") as f:
            st.download_button(
                label="Demo CSV",
                data=f,
                file_name=DEMO_RETAIL_FILE_NAME,
                mime="text/csv",
                help="Sample format",
            )


# ================= 主区域：文件上传 =================


def render_forecast_view(py_info: dict) -> None:
    # 1️⃣ 解析预测块
    blocks = py_info["predicted_sales"]["predictions"]

    # 2️⃣ 拼 DataFrame：step 作为行，store_id 每列一条线
    df_pred = (
        pd.DataFrame({blk["store_id"]: blk["forecast"] for blk in blocks})
        .rename_axis("step")
        .reset_index()
    )
    df_pred["step"] += 1  # 让步长从 1 开始

    # 3️⃣ 选择门店
    store_ids = df_pred.columns.drop("step").tolist()
    choice = st.selectbox("Select View Prediction Point", ["All Items"] + store_ids)

    # 4️⃣ 画图
    if choice == "All Items":
        st.line_chart(df_pred.set_index("step"), height=350)
    else:
        st.line_chart(df_pred.set_index("step")[[choice]], height=350)

    # 5️⃣ 下载按钮（可选）
    st.download_button(
        "Download all predictions CSV",
        df_pred.to_csv(index=False).encode(),
        file_name="all_stores_forecast.csv",
        mime="text/csv",
    )


uploaded = st.file_uploader("Upload your sales CSV", type=["csv"])
# if uploaded:
# st.info("Uploading to server…")
# files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
# try:
#     resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
#     resp.raise_for_status()
# except Exception as e:
#     st.error(f"Upload failed: {e}")
# else:
#     data = resp.json()
#     st.success(f"File stored at S3 key: `{data['s3Key']}`")

#     st.write("Python service returned:")
#     # display the returned data
#     # st.json(data["pythonInfo"])

#     render_forecast_view(data["pythonInfo"])
#     # if you later add a /predict endpoint, you can call it here:
#     # pred = requests.get(f"{API_URL}/predict?key={data['s3Key']}").json()
#     # st.write(pred)

if uploaded:
    try:
        py_info = call_backend(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"Upload / predict failed: {e}")
        st.stop()

    st.success("✅ Please select to view the results")
    render_forecast_view(py_info)
