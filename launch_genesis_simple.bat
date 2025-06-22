@echo off
REM GENESIS Dockerized PyQt5 GUI Launcher - ARCHITECT MODE v7.0.0
REM NO browser, NO server - NATIVE PYQT5 DESKTOP APPLICATION

echo.
echo ================================================================================
echo GENESIS DOCKERIZED PYQT5 GUI LAUNCHER
echo ARCHITECT MODE v7.0.0 - NATIVE DESKTOP APPLICATION
echo ================================================================================
echo.

REM Check Docker
echo Checking Docker installation...
docker --version
if errorlevel 1 (
    echo ERROR: Docker not found! Please ensure Docker Desktop is installed.
    pause
    exit /b 1
)

echo SUCCESS: Docker found!
echo.

REM Check Docker daemon
echo Checking Docker daemon...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker daemon not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo SUCCESS: Docker daemon is running!
echo.

REM Build Docker image
echo Building GENESIS Docker image...
docker build -t genesis_desktop .
if errorlevel 1 (
    echo ERROR: Docker build failed!
    pause
    exit /b 1
)

echo SUCCESS: Docker image built!
echo.

REM Configure X Server for GUI
echo Configuring GUI display...
set DISPLAY=host.docker.internal:0

REM Launch GENESIS Dockerized PyQt5 GUI
echo Launching GENESIS Dockerized PyQt5 GUI...
echo NOTE: Make sure X server is running (XLaunch, VcXsrv, or Xming)
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
    echo ERROR: GUI launch failed! 
    echo Make sure an X server is running (XLaunch, VcXsrv, or Xming)
    pause
    exit /b 1
)

echo.
echo SUCCESS: GENESIS Dockerized PyQt5 GUI launched!
echo Press any key to exit...
pause >nul
