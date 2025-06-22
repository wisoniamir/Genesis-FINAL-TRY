@echo off
:: GitHub Integration Setup Script for GENESIS Trading Bot
:: This script helps set up and manage GitHub integration for the GENESIS trading bot

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         GENESIS TRADING BOT - GITHUB INTEGRATION SETUP         ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

:menu
echo Please select an option:
echo.
echo 1. Initialize GitHub repository
echo 2. Show current status
echo 3. Synchronize with remote repository
echo 4. Commit local changes
echo 5. Push changes to remote
echo 6. Start CI/CD workflow monitoring
echo 7. Run full system audit
echo 8. Exit
echo.

set /p choice=Enter your choice (1-8): 

if "%choice%"=="1" goto init_repo
if "%choice%"=="2" goto show_status
if "%choice%"=="3" goto sync_repo
if "%choice%"=="4" goto commit_changes
if "%choice%"=="5" goto push_changes
if "%choice%"=="6" goto start_workflow
if "%choice%"=="7" goto run_audit
if "%choice%"=="8" goto exit
echo Invalid choice, please try again.
goto menu

:init_repo
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║               INITIALIZE GITHUB REPOSITORY                     ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
set /p repo_url=Enter GitHub repository URL (leave empty to skip): 
set /p branch=Enter main branch name [main]: 
if "%branch%"=="" set branch=main

if "%repo_url%"=="" (
    powershell -ExecutionPolicy Bypass -File github_integration_setup.ps1 -Action setup -Branch "%branch%"
) else (
    powershell -ExecutionPolicy Bypass -File github_integration_setup.ps1 -Action setup -RepoUrl "%repo_url%" -Branch "%branch%"
)

echo.
pause
goto menu

:show_status
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║               GITHUB REPOSITORY STATUS                         ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
powershell -ExecutionPolicy Bypass -File github_integration_setup.ps1 -Action status

echo.
echo Would you also like to check CI/CD workflow status?
set /p check_cicd=Enter Y for Yes, any other key for No: 
if /i "%check_cicd%"=="Y" (
    python github_ci_cd_workflow.py status
)

echo.
pause
goto menu

:sync_repo
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║               SYNCHRONIZE WITH REMOTE                          ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
powershell -ExecutionPolicy Bypass -File github_integration_setup.ps1 -Action sync

echo.
pause
goto menu

:commit_changes
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║               COMMIT LOCAL CHANGES                             ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
set /p commit_msg=Enter commit message: 
if "%commit_msg%"=="" (
    echo Commit message is required.
) else (
    powershell -ExecutionPolicy Bypass -File github_integration_setup.ps1 -Action commit -CommitMessage "%commit_msg%"
)

echo.
pause
goto menu

:push_changes
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║               PUSH CHANGES TO REMOTE                           ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
powershell -ExecutionPolicy Bypass -File github_integration_setup.ps1 -Action push

echo.
pause
goto menu

:start_workflow
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║               START CI/CD WORKFLOW MONITORING                  ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
python github_ci_cd_workflow.py start

echo.
pause
goto menu

:run_audit
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║               RUN FULL SYSTEM AUDIT                            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
python github_sync_cli.py audit

echo.
pause
goto menu

:exit
echo.
echo Thank you for using GENESIS GitHub Integration Setup.
echo.
exit /b 0
