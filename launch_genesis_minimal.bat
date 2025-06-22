@echo off
title GENESIS Minimal Launcher v7.0.0

echo.
echo ====================================================================
echo    🏛️ GENESIS INSTITUTIONAL TRADING SYSTEM v7.0.0
echo    ARCHITECT MODE v7.0.0 - MINIMAL FUNCTIONAL EDITION
echo ====================================================================
echo.
echo 🚀 Launching GENESIS Minimal Application...
echo 📡 Using built-in Python libraries for maximum compatibility...
echo 🛡️ Real MT5 integration available when MetaTrader5 is installed...
echo.

REM Use system Python to avoid virtual environment issues
python genesis_minimal_launcher.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ GENESIS failed to start
    echo Trying with Python 3...
    python3 genesis_minimal_launcher.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Both Python attempts failed
    echo Please ensure Python is installed and in your PATH
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ✅ GENESIS shut down normally
pause
