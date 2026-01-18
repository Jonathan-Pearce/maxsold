# MaxSold Data Project - Dockerfile
# This Dockerfile creates an environment for running the MaxSold scrapers, ML pipeline, and utilities

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create data directories
RUN mkdir -p /app/data/raw/auction \
             /app/data/raw/item \
             /app/data/raw/item_enriched \
             /app/data/final_data \
             /app/data/models \
             /app/data/models/output

# Set Python path to include the project root
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command - show available commands
CMD ["python", "-c", "print('\\nMaxSold Docker Container\\n' + '='*50 + '\\n\\nAvailable commands:\\n' + '- Scrapers: python scrapers/<script_name>.py\\n' + '- ML Pipeline: python ml_pipeline/scripts/train_model_minimal.py\\n' + '- Test modules: python utils/test_modules.py\\n\\nFor interactive mode, run:\\n  docker run -it maxsold bash\\n')"]
