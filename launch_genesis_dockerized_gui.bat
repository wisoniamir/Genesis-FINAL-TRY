@echo off
echo.
echo **************************************************
echo ARCHITECT MODE v7.0.2: GENESIS Docker GUI Launch  
echo **************************************************
echo.

REM Check if XLaunch is running
tasklist /FI "IMAGENAME eq vcxsrv.exe" 2>NUL | find /I /N "vcxsrv.exe">NUL
if %ERRORLEVEL% neq 0 (
    echo Starting XLaunch X11 Server...
    start /B "XLaunch" "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -ac -terminate -lesspointer -multiwindow -clipboard -wgl -dpi auto
    timeout /t 3
)

REM Get host IP for X11 forwarding
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do set HOST_IP=%%a
set HOST_IP=%HOST_IP: =%

echo Starting GENESIS Trading Bot in Docker...
echo Host IP: %HOST_IP%
echo.

docker run -it --rm ^
    --name genesis-trading-bot ^
    -e DISPLAY=%HOST_IP%:0.0 ^
    -e QT_X11_NO_MITSHM=1 ^
    -v "%cd%\data":/genesis/data ^
    -v "%cd%\logs":/genesis/logs ^
    -v "%cd%\config":/genesis/config ^
    -p 8501:8501 ^
    -p 5000:5000 ^
    genesis-trading-bot:v7.0.0

echo.
echo GENESIS Trading Bot has stopped.
pause
