@echo off
REM Fast VS Code Launcher - Guardian Free
echo 🚀 Starting VS Code with optimized settings...
echo Guardian services: DISABLED
echo Performance mode: ENABLED
echo.

REM Clear any VS Code caches that might slow startup
if exist "%APPDATA%\Code\User\workspaceStorage" (
    echo Clearing workspace storage cache...
    rmdir /s /q "%APPDATA%\Code\User\workspaceStorage" 2>nul
)

REM Start VS Code with performance flags
code --disable-extensions --max-memory=4096 "Genesis FINAL TRY.code-workspace"

echo.
echo ✅ VS Code launched with performance optimizations
echo ⚡ Guardian-free environment active
pause
