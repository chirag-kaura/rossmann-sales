# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies required for LightGBM (libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages directly
RUN pip install --no-cache-dir pandas numpy lightgbm scikit-learn joblib

# Copy project files
COPY src/ /app/src/
COPY data/raw/store.csv /app/data/raw/store.csv
COPY data/raw/test.csv /app/data/raw/test.csv
COPY models/best_model.pkl /app/models/best_model.pkl

# Run the inference script by default
CMD ["python", "src/models/predict.py"]