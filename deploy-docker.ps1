# GENESIS INSTITUTIONAL DASHBOARD - Docker Deployment Script
# PowerShell script for Windows with Xming support

param(
    [Parameter()]
    [ValidateSet("build", "start", "stop", "restart", "logs", "clean")]
    [string]$Action = "start"
)

Write-Host "🏛️ GENESIS INSTITUTIONAL DASHBOARD - Docker Deployment v7.0.0" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if Xming is available (for GUI support)
$xmingProcess = Get-Process -Name "Xming" -ErrorAction SilentlyContinue
if ($xmingProcess) {
    Write-Host "✅ Xming X Server is running" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: Xming X Server not detected. GUI features may not work." -ForegroundColor Yellow
    Write-Host "   Download Xming from: https://sourceforge.net/projects/xming/" -ForegroundColor Yellow
}

switch ($Action) {
    "build" {
        Write-Host "🔨 Building GENESIS Institutional Dashboard container..." -ForegroundColor Yellow
        docker-compose -f docker-compose-institutional.yml build --no-cache
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Build completed successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Build failed" -ForegroundColor Red
            exit 1
        }
    }
    
    "start" {
        Write-Host "🚀 Starting GENESIS Institutional Dashboard..." -ForegroundColor Yellow
        
        # Set DISPLAY environment variable for Windows
        $env:DISPLAY = "host.docker.internal:0"
        
        docker-compose -f docker-compose-institutional.yml up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ GENESIS Institutional Dashboard started successfully" -ForegroundColor Green
            Write-Host ""
            Write-Host "📊 Access your dashboard at:" -ForegroundColor Cyan
            Write-Host "   Streamlit UI: http://localhost:8501" -ForegroundColor White
            Write-Host ""
            Write-Host "🔍 Monitor logs with: ./deploy-docker.ps1 logs" -ForegroundColor Cyan
        } else {
            Write-Host "❌ Failed to start dashboard" -ForegroundColor Red
            exit 1
        }
    }
    
    "stop" {
        Write-Host "🛑 Stopping GENESIS Institutional Dashboard..." -ForegroundColor Yellow
        docker-compose -f docker-compose-institutional.yml down
        Write-Host "✅ Dashboard stopped" -ForegroundColor Green
    }
    
    "restart" {
        Write-Host "🔄 Restarting GENESIS Institutional Dashboard..." -ForegroundColor Yellow
        docker-compose -f docker-compose-institutional.yml restart
        Write-Host "✅ Dashboard restarted" -ForegroundColor Green
    }
    
    "logs" {
        Write-Host "📋 Showing GENESIS Dashboard logs..." -ForegroundColor Yellow
        docker-compose -f docker-compose-institutional.yml logs -f
    }
    
    "clean" {
        Write-Host "🧹 Cleaning up GENESIS containers and images..." -ForegroundColor Yellow
        docker-compose -f docker-compose-institutional.yml down --rmi all --volumes
        docker system prune -f
        Write-Host "✅ Cleanup completed" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🏛️ GENESIS Institutional Dashboard deployment complete!" -ForegroundColor Green
