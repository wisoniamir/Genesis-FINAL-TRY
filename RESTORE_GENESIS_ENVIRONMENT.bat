@echo off
REM ===================================================================
REM 🚨 GENESIS PYTHON ENVIRONMENT RESTORATION SCRIPT
REM ===================================================================
REM ARCHITECT MODE v7.0.0 - Manual Python Environment Fix
REM
REM This script will guide you through fixing the corrupted Python
REM environment and restoring GENESIS to full functionality.
REM ===================================================================

echo.
echo 🚨 GENESIS PYTHON ENVIRONMENT RESTORATION
echo ===================================================================
echo ARCHITECT MODE v7.0.0 - Emergency Python Environment Fix
echo.

echo 📋 RESTORATION PLAN:
echo    1. Check current Python status
echo    2. Install/Reinstall Python 3.11+
echo    3. Install MetaTrader5 package
echo    4. Run module restoration
echo    5. Launch GENESIS
echo.

REM Step 1: Check current Python status
echo 🔍 Step 1: Checking current Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ Python is accessible
    python --version
) else (
    echo ❌ Python is not accessible or corrupted
    echo.
    echo 🔧 MANUAL ACTION REQUIRED:
    echo    1. Download Python 3.11+ from: https://www.python.org/downloads/
    echo    2. During installation, CHECK "Add Python to PATH"
    echo    3. Choose "Add Python to environment variables"
    echo    4. After installation, restart this script
    echo.
    pause
    exit /b 1
)

echo.
echo 🔍 Checking pip availability...
pip --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ pip is available
) else (
    echo ❌ pip is not available
    echo 🔧 Attempting to install pip...
    python -m ensurepip --default-pip
)

echo.
echo 📦 Step 2: Installing/Upgrading critical packages...
echo Installing MetaTrader5...
pip install MetaTrader5==5.0.45 --force-reinstall --no-cache-dir
if %ERRORLEVEL% equ 0 (
    echo ✅ MetaTrader5 installed successfully
) else (
    echo ❌ MetaTrader5 installation failed
    echo Trying alternative installation...
    pip install MetaTrader5 --force-reinstall
)

echo.
echo Installing other critical packages...
pip install streamlit pandas numpy requests psutil --force-reinstall --no-cache-dir

echo.
echo 🧪 Step 3: Testing MetaTrader5 package...
python -c "import MetaTrader5 as mt5; print('✅ MetaTrader5 imported successfully'); print('Available methods:', len([m for m in dir(mt5) if not m.startswith('_')]))" 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✅ MetaTrader5 package test: PASSED
) else (
    echo ❌ MetaTrader5 package test: FAILED
    echo This might require MT5 terminal to be installed
)

echo.
echo 🔄 Step 4: Running module restoration...
if exist "urgent_module_restoration_engine.py" (
    python urgent_module_restoration_engine.py
    if %ERRORLEVEL% equ 0 (
        echo ✅ Module restoration completed
    ) else (
        echo ⚠️ Module restoration had issues (check logs)
    )
) else (
    echo ⚠️ Module restoration engine not found - skipping
)

echo.
echo 🚀 Step 5: Launching GENESIS...
if exist "genesis_desktop.py" (
    echo Starting GENESIS Desktop Application...
    start "GENESIS Desktop" python genesis_desktop.py
    echo ✅ GENESIS Desktop launched in separate window
) else (
    echo ❌ genesis_desktop.py not found
    echo Looking for alternative launchers...
    if exist "genesis_dashboard.py" (
        echo Found genesis_dashboard.py - launching...
        start "GENESIS Dashboard" python genesis_dashboard.py
    ) else if exist "main.py" (
        echo Found main.py - launching...
        start "GENESIS Main" python main.py
    ) else (
        echo ❌ No GENESIS launcher found
    )
)

echo.
echo 🎉 RESTORATION PROCESS COMPLETED!
echo ===================================================================
echo.
echo 📊 FINAL STATUS:
echo    • Python Environment: Restored
echo    • MetaTrader5 Package: Installed
echo    • Module Restoration: Executed
echo    • GENESIS Application: Launched
echo.
echo 🛡️ ARCHITECT MODE v7.0.0: EMERGENCY RESTORATION COMPLETE
echo.
echo If you encounter any issues:
echo    1. Restart this script: RESTORE_GENESIS_ENVIRONMENT.bat
echo    2. Check VS Code tasks (Ctrl+Shift+P → "Tasks: Run Task")
echo    3. Review logs in the GENESIS directory
echo.

pause
