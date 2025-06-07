# 1. Base image with Python 3.12.6 -> 3.10-slim
FROM python:3.10-slim

# 2. Set a working directory
WORKDIR /app

# 3. Copy and install dependencies
COPY requirements.txt .
#RUN pip install --upgrade pip && \
#    pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt



# 4. Copy your FastAPI app,prediction code into the image
COPY app.py .
COPY prediction.py .

# 5. Tell Docker how to start the service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
