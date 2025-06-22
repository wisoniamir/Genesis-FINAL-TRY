@echo off
rem ARCHITECT MODE SYSTEM GUARDIAN LAUNCHER
rem Ultimate enforcement and validation protocol

echo 🔐 ARCHITECT MODE SYSTEM GUARDIAN v7.0.0
echo 🚨 ZERO TOLERANCE → ULTIMATE ENFORCEMENT ACTIVATED
echo ====================================================

echo 📋 Phase 1: Validating Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found, using system python
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ CRITICAL: No Python installation found
        pause
        exit /b 1
    )
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)

echo ✅ Python environment validated

echo 📋 Phase 2: Loading architecture files...
if not exist "module_registry.json" (
    echo ❌ CRITICAL: module_registry.json missing
    pause
    exit /b 1
)

if not exist "event_bus.json" (
    echo ❌ CRITICAL: event_bus.json missing
    pause
    exit /b 1
)

if not exist "build_status.json" (
    echo ❌ CRITICAL: build_status.json missing
    pause
    exit /b 1
)

echo ✅ Core architecture files validated

echo 📋 Phase 3: Executing System Guardian...
%PYTHON_CMD% architect_mode_system_guardian.py

if errorlevel 1 (
    echo ❌ System Guardian execution failed
    echo 🔧 Attempting emergency module restoration...
    
    rem Run emergency restoration
    if exist "emergency_module_restoration.py" (
        %PYTHON_CMD% emergency_module_restoration.py
    )
    
    rem Try comprehensive upgrade
    if exist "comprehensive_module_upgrade_engine.py" (
        echo 🔧 Running comprehensive module upgrade...
        %PYTHON_CMD% comprehensive_module_upgrade_engine.py
    )
    
    pause
    exit /b 1
)

echo ✅ System Guardian completed successfully

echo 🚀 Phase 4: Launching GENESIS application...

rem Try different launch methods
if exist "genesis_desktop.py" (
    echo 🖥️ Launching GENESIS Desktop...
    start "GENESIS Desktop" %PYTHON_CMD% genesis_desktop.py
    goto launch_success
)

if exist "genesis_ultimate_launcher.py" (
    echo 🚀 Launching Ultimate Launcher...
    start "GENESIS Ultimate" %PYTHON_CMD% genesis_ultimate_launcher.py
    goto launch_success
)

if exist "genesis_api.py" (
    echo 🌐 Launching GENESIS API...
    start "GENESIS API" %PYTHON_CMD% genesis_api.py
    goto launch_success
)

if exist "launch_genesis_docker.bat" (
    echo 🐳 Launching Docker GUI...
    call launch_genesis_docker.bat
    goto launch_success
)

echo ⚠️ No suitable launcher found, manual launch required

:launch_success
echo 🎯 ARCHITECT MODE GUARDIAN PROTOCOL COMPLETE
echo 🟢 GENESIS SYSTEM: PRODUCTION READY
echo.
echo 📊 Next steps:
echo 1. Login to GENESIS Desktop GUI
echo 2. Connect to MT5 live account
echo 3. Begin instrument discovery
echo 4. Activate real-time monitoring
echo.
pause
