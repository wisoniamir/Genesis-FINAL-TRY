# 🚀 GENESIS AUTO-STARTUP SCRIPT (PowerShell)
# ARCHITECT MODE v7.0.0 - Windows Integration

Write-Host ""
Write-Host "🔐 GENESIS SYSTEM AUTO-STARTUP" -ForegroundColor Green
Write-Host "🛡️ ARCHITECT MODE v7.0.0 - ULTIMATE ENFORCEMENT" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Gray -NoNewline; for ($i=1; $i -le 60; $i++) { Write-Host "=" -ForegroundColor Gray -NoNewline }; Write-Host ""

# Check if we're in the correct directory
$currentPath = Get-Location
$expectedPath = "C:\Users\patra\Genesis FINAL TRY"

if ($currentPath.Path -ne $expectedPath) {
    Write-Host "⚠️  Warning: Not in GENESIS workspace directory" -ForegroundColor Yellow
    Write-Host "   Current: $($currentPath.Path)" -ForegroundColor Gray
    Write-Host "   Expected: $expectedPath" -ForegroundColor Gray
    Write-Host ""
    
    # Ask user if they want to change directory
    $response = Read-Host "Change to GENESIS workspace? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        try {
            Set-Location $expectedPath
            Write-Host "✅ Changed to GENESIS workspace" -ForegroundColor Green
        } catch {
            Write-Host "❌ Failed to change directory: $_" -ForegroundColor Red
            exit 1
        }
    }
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "🐍 Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python and add to PATH." -ForegroundColor Red
    exit 1
}

# Check for auto-startup script
if (Test-Path "genesis_auto_startup.py") {
    Write-Host "🚀 Starting GENESIS Auto-Startup System..." -ForegroundColor Cyan
    Write-Host ""
    
    # Run the auto-startup script
    python genesis_auto_startup.py
    
    $exitCode = $LASTEXITCODE
    Write-Host ""
    
    if ($exitCode -eq 0) {
        Write-Host "✅ GENESIS Auto-Startup completed successfully!" -ForegroundColor Green
        
        # Show status information
        if (Test-Path "genesis_startup_status.json") {
            Write-Host ""
            Write-Host "📊 Startup Status Information:" -ForegroundColor Cyan
            $status = Get-Content "genesis_startup_status.json" | ConvertFrom-Json
            Write-Host "   Timestamp: $($status.startup_timestamp)" -ForegroundColor Gray
            Write-Host "   Services Started: $($status.services_started -join ', ')" -ForegroundColor Gray
            Write-Host "   ARCHITECT MODE: $(if($status.architect_mode_active){'ACTIVE'}else{'INACTIVE'})" -ForegroundColor $(if($status.architect_mode_active){'Green'}else{'Red'})
        }
        
        Write-Host ""
        Write-Host "🎯 Your GENESIS development environment is ready!" -ForegroundColor Green
        Write-Host "   • Watchdog monitoring active"
        Write-Host "   • Git pre-commit hooks installed"
        Write-Host "   • Zero tolerance enforcement enabled"
        Write-Host ""
        
    } else {
        Write-Host "⚠️  GENESIS Auto-Startup completed with warnings" -ForegroundColor Yellow
        Write-Host "   Check the startup log for details" -ForegroundColor Gray
    }
    
} else {
    Write-Host "❌ GENESIS auto-startup script not found!" -ForegroundColor Red
    Write-Host "   Expected: genesis_auto_startup.py" -ForegroundColor Gray
    Write-Host "   Make sure you're in the correct GENESIS workspace directory" -ForegroundColor Gray
    exit 1
}

# Keep PowerShell window open if run directly
if ($Host.Name -eq "ConsoleHost") {
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
