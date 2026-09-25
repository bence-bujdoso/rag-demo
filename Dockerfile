FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python environment
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir avoids caching pip packages
# --no-build-isolation builds packages in the current environment
RUN pip install --no-cache-dir --no-build-isolation \
    -r requirements.txt 2>&1 | tail -20 || \
    (pip install --no-cache-dir pip --upgrade && \
     pip install --no-cache-dir numpy scipy && \
     pip install --no-cache-dir -r requirements.txt 2>&1 | tail -20)

# Copy application code
COPY . .

# Create data directory if not exists
RUN mkdir -p /app/data && mkdir -p /app/logs

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
