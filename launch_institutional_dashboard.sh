#!/bin/bash
# GENESIS INSTITUTIONAL DASHBOARD DOCKER LAUNCHER v7.0.0
# Properly launches the existing institutional dashboard with Docker/Xming support

set -e

echo "🏛️ GENESIS INSTITUTIONAL DASHBOARD - DOCKER MODE v7.0.0"
echo "=========================================================="

# Environment setup
export DOCKER_MODE=true
export PYTHONPATH=/genesis:$PYTHONPATH
export QT_X11_NO_MITSHM=1
export QT_LOGGING_RULES="*=false"

# Verify display
if [ -z "$DISPLAY" ]; then
    echo "⚠️  Warning: DISPLAY not set. Setting to host.docker.internal:0"
    export DISPLAY=host.docker.internal:0
fi

echo "🔧 Environment:"
echo "   DISPLAY: $DISPLAY"
echo "   DOCKER_MODE: $DOCKER_MODE"
echo "   PYTHONPATH: $PYTHONPATH"

# Check for required files
echo "🔍 Checking system files..."
if [ ! -f "/genesis/genesis_institutional_dashboard.py" ]; then
    echo "❌ CRITICAL: genesis_institutional_dashboard.py not found"
    exit 1
fi

if [ ! -f "/genesis/core/genesis_real_mt5_connection.py" ]; then
    echo "❌ CRITICAL: MT5 connection module not found"
    exit 1
fi

# Create required directories
mkdir -p /genesis/logs/dashboard
mkdir -p /genesis/telemetry
mkdir -p /genesis/config

# Initialize system
echo "🚀 Starting GENESIS system initialization..."

# Start MT5 connection manager first
echo "📡 Initializing MT5 connection..."
python3 /genesis/core/genesis_real_mt5_connection.py &
MT5_PID=$!

# Wait a moment for MT5 to initialize
sleep 3

# Start the institutional dashboard
echo "🏛️ Launching GENESIS Institutional Dashboard..."
cd /genesis

# Launch with Streamlit mode for Docker compatibility
python3 genesis_institutional_dashboard.py

# Clean up background processes
echo "🛑 Shutting down background processes..."
kill $MT5_PID 2>/dev/null || true

echo "✅ GENESIS Institutional Dashboard session completed"
