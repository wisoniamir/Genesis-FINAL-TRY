@echo off
rem GENESIS Comprehensive Module Upgrade Script
rem Scans and upgrades all modules to institutional-grade compliance

echo 🚀 GENESIS COMPREHENSIVE MODULE UPGRADE v8.0.0
echo =============================================

echo 📊 Scanning Python modules...
for /f %%i in ('dir /s /b *.py ^| find /c /v ""') do set TOTAL_MODULES=%%i
echo Found %TOTAL_MODULES% Python modules

echo 🔍 Checking compliance status...

rem Count modules needing EventBus integration
set NEEDS_EVENTBUS=0
for /f %%f in ('dir /s /b *.py') do (
    findstr /i /c:"from event_bus import" /c:"from core.hardened_event_bus import" "%%f" >nul 2>&1
    if errorlevel 1 (
        set /a NEEDS_EVENTBUS+=1
    )
)

rem Count modules needing telemetry
set NEEDS_TELEMETRY=0
for /f %%f in ('dir /s /b *.py') do (
    findstr /i /c:"emit_telemetry" "%%f" >nul 2>&1
    if errorlevel 1 (
        set /a NEEDS_TELEMETRY+=1
    )
)

rem Count modules needing FTMO compliance
set NEEDS_FTMO=0
for /f %%f in ('dir /s /b *.py') do (
    findstr /i /c:"ftmo" /c:"drawdown" /c:"daily_loss" "%%f" >nul 2>&1
    if errorlevel 1 (
        set /a NEEDS_FTMO+=1
    )
)

rem Count modules needing kill switch
set NEEDS_KILL_SWITCH=0
for /f %%f in ('dir /s /b *.py') do (
    findstr /i /c:"kill_switch" /c:"emergency_stop" "%%f" >nul 2>&1
    if errorlevel 1 (
        set /a NEEDS_KILL_SWITCH+=1
    )
)

echo.
echo 📋 COMPLIANCE ANALYSIS RESULTS:
echo ================================
echo Total Python Modules: %TOTAL_MODULES%
echo Need EventBus Integration: %NEEDS_EVENTBUS%
echo Need Telemetry Hooks: %NEEDS_TELEMETRY%
echo Need FTMO Compliance: %NEEDS_FTMO%
echo Need Kill Switch Logic: %NEEDS_KILL_SWITCH%

rem Calculate compliance rate
set /a COMPLIANT_MODULES=%TOTAL_MODULES%-%NEEDS_EVENTBUS%-%NEEDS_TELEMETRY%-%NEEDS_FTMO%-%NEEDS_KILL_SWITCH%
if %COMPLIANT_MODULES% lss 0 set COMPLIANT_MODULES=0

echo Fully Compliant Modules: %COMPLIANT_MODULES%

echo.
echo 🎯 UPGRADE RECOMMENDATIONS:
echo ===========================
if %NEEDS_EVENTBUS% gtr 0 echo - Priority: Integrate EventBus in %NEEDS_EVENTBUS% modules
if %NEEDS_TELEMETRY% gtr 0 echo - Important: Add telemetry to %NEEDS_TELEMETRY% modules
if %NEEDS_FTMO% gtr 0 echo - Critical: Implement FTMO compliance in %NEEDS_FTMO% modules
if %NEEDS_KILL_SWITCH% gtr 0 echo - Critical: Add kill switch to %NEEDS_KILL_SWITCH% modules

echo.
echo 📊 SYSTEM STATUS: 
if %COMPLIANT_MODULES% gtr %NEEDS_EVENTBUS% (
    echo ✅ SYSTEM MOSTLY COMPLIANT
) else (
    echo ⚠️ SYSTEM NEEDS SIGNIFICANT UPGRADES
)

echo.
echo 🏁 Scan complete. Run comprehensive_module_upgrade_engine.py for detailed upgrades.
pause
