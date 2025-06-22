@echo off
:: GENESIS AI TRADING BOT SYSTEM - Safe Script Runner
:: ARCHITECT MODE: v2.7
:: This script safely runs Python scripts with proper termination

setlocal enabledelayedexpansion

:: Check if a script path was provided
if "%1"=="" (
    echo ERROR: Please provide a Python script path
    echo Usage: run_safe.bat script.py [timeout_seconds] [script_args...]
    exit /b 1
)

:: Get the script path
set SCRIPT_PATH=%1
shift

:: Get the timeout (default: 60 seconds)
set TIMEOUT=60
if not "%1"=="" (
    set TIMEOUT=%1
    shift
)

:: Collect any remaining arguments for the script
set SCRIPT_ARGS=
:collect_args
if not "%1"=="" (
    set SCRIPT_ARGS=!SCRIPT_ARGS! %1
    shift
    goto collect_args
)

echo [%date% %time%] [INFO] GENESIS Safe Script Runner - Running %SCRIPT_PATH%
echo [%date% %time%] [INFO] Timeout set to %TIMEOUT% seconds

:: Check if the script exists
if not exist %SCRIPT_PATH% (
    echo [%date% %time%] [ERROR] Script not found: %SCRIPT_PATH%
    exit /b 1
)

:: Run the script using PowerShell for better process control
powershell -ExecutionPolicy Bypass -File "run_safe.ps1" -ScriptPath "%SCRIPT_PATH%" -TimeoutSeconds %TIMEOUT% %SCRIPT_ARGS%

:: Get the exit code
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% EQU 0 (
    echo [%date% %time%] [INFO] Script completed successfully with exit code %EXIT_CODE%
) else (
    echo [%date% %time%] [WARNING] Script exited with code %EXIT_CODE%
)

exit /b %EXIT_CODE%
