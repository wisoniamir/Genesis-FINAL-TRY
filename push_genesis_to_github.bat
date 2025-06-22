@echo off
:: Genesis GitHub Push Script
:: This batch file helps you push the entire Genesis project to GitHub

echo.
echo ╔═════════════════════════════════════════════════════════════╗
echo ║          GENESIS GitHub Project Push Utility                ║
echo ╚═════════════════════════════════════════════════════════════╝
echo.

echo This utility will help you push your entire Genesis project to GitHub.
echo It will:
echo  1. Initialize a Git repository (if needed)
echo  2. Create a proper .gitignore file for your project
echo  3. Commit all files in the project
echo  4. Push to your GitHub repository
echo.

echo Do you want to continue?
set /p choice=Enter Y to continue or any other key to exit: 

if /i not "%choice%"=="Y" (
    echo Operation cancelled.
    goto :end
)

echo.
echo Starting PowerShell script to push your Genesis project...
echo.

powershell -ExecutionPolicy Bypass -File push_genesis_to_github.ps1

:end
echo.
pause
