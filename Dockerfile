# Use a compatible base image such as python:3.11-bullseye or an Ubuntu image
FROM python:3.11-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget gnupg ffmpeg \
    libnss3 libatk-bridge2.0-0 libgtk-3-0 libx11-xcb1 libxcb-dri3-0 libxcomposite1 \
    libxdamage1 libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 libpango-1.0-0 \
    libatk1.0-0 libcups2 libxshmfence1 libxfixes3 libxrender1 libxi6 libxtst6 \
    libxss1 libxext6 libx11-6 fonts-unifont libgdk-pixbuf-xlib-2.0-0 \
    libenchant-2-2 libicu-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.prod.txt .
COPY . .

RUN pip install --no-cache-dir -r requirements.prod.txt

# Install Playwright system dependencies
RUN playwright install-deps

RUN python -m playwright install chromium

EXPOSE 8000

CMD ["uvicorn", "updated_backend:app", "--host", "0.0.0.0", "--port", "8000"]
