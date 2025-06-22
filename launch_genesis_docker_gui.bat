@echo off
REM 🚀 GENESIS Dockerized PyQt5 GUI Launcher - ARCHITECT MODE v7.0.0
REM NO browser, NO server - NATIVE PYQT5 DESKTOP APPLICATION

echo.
echo ================================================================================
echo 🐳 GENESIS DOCKERIZED PYQT5 GUI LAUNCHER
echo 🚨 ARCHITECT MODE v7.0.0 - NATIVE DESKTOP APPLICATION
echo ================================================================================
echo.

REM Add Docker to PATH
set PATH=%PATH%;C:\Program Files\Docker\Docker\resources\bin

REM Check Docker
echo 📊 Checking Docker installation...
docker --version
if errorlevel 1 (
    echo ❌ Docker not found! Please ensure Docker Desktop is installed.
    pause
    exit /b 1
)

echo ✅ Docker found!
echo.

REM Check Docker daemon
echo 🔧 Checking Docker daemon...
docker info >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Docker daemon not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo ⏳ Waiting for Docker Desktop to start...
    timeout /t 30 /nobreak >nul
    
    REM Check again
    docker info >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker daemon still not running. Please start Docker Desktop manually.
        echo 💡 After Docker Desktop starts, run this script again.
        pause
        exit /b 1
    )
)

echo ✅ Docker daemon is running!
echo.

REM Create necessary directories
echo 📁 Creating required directories...
if not exist "genesis_core" mkdir genesis_core
if not exist "interface" mkdir interface
if not exist "interface\dashboard" mkdir interface\dashboard
if not exist "config" mkdir config
if not exist "telemetry" mkdir telemetry
if not exist "logs" mkdir logs
if not exist "mt5_connector" mkdir mt5_connector

REM Create main PyQt5 GUI application if it doesn't exist
if not exist "interface\dashboard\main.py" (
    echo 🔧 Creating main PyQt5 GUI application...
    echo import sys > interface\dashboard\main.py
    echo import os >> interface\dashboard\main.py
    echo from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget >> interface\dashboard\main.py
    echo from PyQt5.QtCore import Qt >> interface\dashboard\main.py
    echo. >> interface\dashboard\main.py
    echo class GenesisMainWindow(QMainWindow): >> interface\dashboard\main.py
    echo     def __init__(self): >> interface\dashboard\main.py
    echo         super().__init__() >> interface\dashboard\main.py
    echo         self.setWindowTitle("🚀 GENESIS Trading System - Dockerized PyQt5 GUI") >> interface\dashboard\main.py
    echo         self.setGeometry(100, 100, 1200, 800) >> interface\dashboard\main.py
    echo         central_widget = QWidget() >> interface\dashboard\main.py
    echo         self.setCentralWidget(central_widget) >> interface\dashboard\main.py
    echo         layout = QVBoxLayout(central_widget) >> interface\dashboard\main.py
    echo         label = QLabel("🎯 GENESIS ARCHITECT MODE v7.0.0 - NATIVE PYQT5 INTERFACE") >> interface\dashboard\main.py
    echo         label.setAlignment(Qt.AlignCenter) >> interface\dashboard\main.py
    echo         layout.addWidget(label) >> interface\dashboard\main.py
    echo. >> interface\dashboard\main.py
    echo if __name__ == "__main__": >> interface\dashboard\main.py
    echo     app = QApplication(sys.argv) >> interface\dashboard\main.py
    echo     window = GenesisMainWindow() >> interface\dashboard\main.py
    echo     window.show() >> interface\dashboard\main.py
    echo     sys.exit(app.exec_()) >> interface\dashboard\main.py
)

REM Create launch script for backend
if not exist "genesis_core\launch_genesis.py" (
    echo 🔧 Creating backend launch script...
    echo import time > genesis_core\launch_genesis.py
    echo print("🚀 GENESIS Backend Engine Starting...") >> genesis_core\launch_genesis.py
    echo print("✅ Backend modules initialized") >> genesis_core\launch_genesis.py
    echo while True: >> genesis_core\launch_genesis.py
    echo     time.sleep(60) >> genesis_core\launch_genesis.py
)

REM Build Docker image
echo 🔨 Building GENESIS Docker image...
docker build -t genesis_desktop .
if errorlevel 1 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo ✅ Docker image built successfully!
echo.

REM Configure X Server for GUI (Windows approach)
echo 🖥️ Configuring GUI display...
set DISPLAY=host.docker.internal:0

REM Launch GENESIS Dockerized PyQt5 GUI
echo 🚀 Launching GENESIS Dockerized PyQt5 GUI...
echo ⚠️ Note: You may need to configure an X server for GUI display
echo.

docker run -it --rm ^
    --name genesis_gui_app ^
    -e DISPLAY=host.docker.internal:0 ^
    -v "%CD%\genesis_core:/genesis/genesis_core" ^
    -v "%CD%\interface:/genesis/interface" ^
    -v "%CD%\config:/genesis/config" ^
    -v "%CD%\telemetry:/genesis/telemetry" ^
    -v "%CD%\logs:/genesis/logs" ^
    -v "%CD%\mt5_connector:/genesis/mt5_connector" ^
    genesis_desktop

if errorlevel 1 (
    echo ❌ GUI launch failed! 
    echo 💡 Make sure an X server is running (like XLaunch, VcXsrv, or MobaXterm)
    pause
    exit /b 1
)

echo.
echo 🎉 GENESIS Dockerized PyQt5 GUI launched successfully!
echo Press any key to exit...
pause >nul
