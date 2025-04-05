FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY . .

# Install package
RUN pip install --no-cache-dir -e ".[dev,docs]"

# Default command
CMD ["bash"]