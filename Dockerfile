# Use an official lightweight Python image
FROM python:3.11-slim

# Environment settings to avoid pyc and enable unbuffered logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (ffmpeg + playwright libs)
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .

# Copy cookies during build (if you have them)
COPY instagram_cookies.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install latest yt-dlp to ensure support for new URL formats
RUN pip install --no-cache-dir --upgrade --pre "yt-dlp[default]" && yt-dlp --version

# Install Playwright Chromium browser
RUN playwright install chromium

# Copy application files
COPY . .

# Expose service port
EXPOSE 8000

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
