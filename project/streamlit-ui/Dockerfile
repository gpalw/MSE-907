FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY assets/ ./assets/
COPY .streamlit/ ./.streamlit/
ENV API_URL=http://nginx/api
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
