@echo off
title GENESIS Institutional Trading System v7.0.0

echo.
echo ====================================================================
echo    🏛️ GENESIS INSTITUTIONAL TRADING SYSTEM v7.0.0
echo    ARCHITECT MODE v7.0.0 - ZERO TOLERANCE EDITION
echo ====================================================================
echo.
echo 🚀 Launching GENESIS with REAL MT5 Integration...
echo 📡 Initializing institutional-grade trading interface...
echo 🛡️ FTMO compliance enforcement active...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Error: Virtual environment not found
    echo Please run the installation process first
    pause
    exit /b 1
)

REM Launch GENESIS Desktop
echo 🔧 Starting GENESIS Desktop Application...
".venv\Scripts\python.exe" genesis_desktop.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ GENESIS failed to start properly
    echo Check the error messages above
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ✅ GENESIS shut down normally
pause
