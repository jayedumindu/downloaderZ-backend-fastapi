# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including ffmpeg and required libs for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libxss1 \
    libasound2 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libdrm2 \
    libcups2 \
    libdbus-1-3 \
    libx11-xcb1 \
    libxcb1 \
    libxrender1 \
    libxshmfence1 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.prod.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.prod.txt

# Install Playwright browsers (chromium)
RUN playwright install chromium

# Copy app files
COPY . .

# Expose port
EXPOSE 8000

# Run the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
