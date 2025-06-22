# GENESIS Dockerfile for PyQt5 GUI - ARCHITECT MODE v7.0.0
# Base image with GUI support
FROM python:3.11-slim

# Set working directory
WORKDIR /genesis

# Install system dependencies for PyQt5 and X11
RUN apt-get update && apt-get install -y \
    build-essential \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxtst-dev \
    libxi-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libssl-dev \
    libxcb1-dev \
    libgl1-mesa-glx \
    xvfb \
    x11-apps \
    qt5-qmake \
    qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Linux-compatible versions) 
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Create necessary directories
RUN mkdir -p /genesis/genesis_core /genesis/interface/dashboard /genesis/config /genesis/telemetry /genesis/logs /genesis/mt5_connector

# Copy application code
COPY . .

# Set environment variables
ENV DISPLAY=host.docker.internal:0
ENV QT_X11_NO_MITSHM=1
ENV QT_LOGGING_RULES="*=false"

# Create comprehensive entrypoint script with dashboard auto-detection
RUN echo '#!/bin/bash' > /genesis/entrypoint.sh && \
    echo 'cd /genesis' >> /genesis/entrypoint.sh && \
    echo 'export PYTHONPATH=/genesis:$PYTHONPATH' >> /genesis/entrypoint.sh && \
    echo 'export DOCKER_MODE=true' >> /genesis/entrypoint.sh && \
    echo 'echo "🚀 GENESIS ARCHITECT MODE v7.0.0 - Comprehensive Dashboard Starting..."' >> /genesis/entrypoint.sh && \
    echo 'echo "� Docker Mode: Enabled"' >> /genesis/entrypoint.sh && \
    echo 'echo "�🖥️ Display: $DISPLAY"' >> /genesis/entrypoint.sh && \
    echo 'echo "🔧 Starting Dashboard Engine..."' >> /genesis/entrypoint.sh && \# Copy and setup institutional dashboard launcher
COPY launch_institutional_dashboard.sh /genesis/
RUN chmod +x /genesis/launch_institutional_dashboard.sh

# Entrypoint
ENTRYPOINT ["/genesis/launch_institutional_dashboard.sh"]
