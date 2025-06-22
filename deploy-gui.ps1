# GENESIS Docker GUI Deployment Script - ARCHITECT MODE v7.0.0
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("build", "start", "stop", "restart", "logs", "clean", "status")]
    [string]$Command = "start"
)

$ErrorActionPreference = "Stop"

# Configuration
$ImageName = "genesis-desktop-gui"
$ContainerName = "genesis_desktop_app"
$ComposeFile = "docker-compose-desktop-gui.yml"
$DockerFile = "Dockerfile.desktop-gui"

function Write-GenesisHeader {
    Write-Host ""
    Write-Host "==================================================================================" -ForegroundColor Blue
    Write-Host "🚀 GENESIS DOCKER GUI DASHBOARD MANAGER - ARCHITECT MODE v7.0.0" -ForegroundColor Cyan
    Write-Host "🖥️ Native PyQt5 Desktop Interface with Docker/Xming Support" -ForegroundColor Green
    Write-Host "==================================================================================" -ForegroundColor Blue
    Write-Host ""
}

function Test-Prerequisites {
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker not found! Please install Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    # Check Docker daemon
    try {
        docker info | Out-Null
        Write-Host "✅ Docker daemon is running" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker daemon not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    # Check Xming (for GUI)
    $xmingRunning = Get-Process | Where-Object { $_.ProcessName -like "*Xming*" -or $_.ProcessName -like "*VcXsrv*" }
    if ($xmingRunning) {
        Write-Host "✅ X Server found: GUI support available" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ X Server not detected. Install Xming or VcXsrv for GUI support." -ForegroundColor Yellow
        Write-Host "   Download: https://sourceforge.net/projects/xming/" -ForegroundColor Cyan
    }
      # Check required files
    $requiredFiles = @("launch_desktop_app.py", $DockerFile, $ComposeFile, "interface/dashboard/main.py")
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✅ Found: $file" -ForegroundColor Green
        }
        else {
            Write-Host "❌ Missing: $file" -ForegroundColor Red
            exit 1
        }
    }
    
    # Check for required directories
    $requiredDirs = @(
        "logs", "config", "telemetry", "interface/dashboard", "mt5_connector"
    )
    foreach ($dir in $requiredDirs) {
        if (Test-Path $dir) {
            Write-Host "✅ Found directory: $dir" -ForegroundColor Green
        }
        else {
            Write-Host "📁 Creating directory: $dir" -ForegroundColor Yellow
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Host ""
}

function Build-GenesisImage {
    Write-Host "🔨 Building GENESIS GUI Docker image..." -ForegroundColor Yellow
    
    try {
        docker-compose -f $ComposeFile build
        Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker build failed!" -ForegroundColor Red
        throw
    }
    
    Write-Host ""
}

function Start-GenesisDashboard {
    Write-Host "🚀 Starting GENESIS GUI Dashboard..." -ForegroundColor Yellow
    
    # Stop existing container if running
    try {
        docker stop $ContainerName 2>$null
        docker rm $ContainerName 2>$null
    }
    catch {
        # Container might not exist, that's OK
    }
    
    # Start new container
    try {
        docker-compose -f $ComposeFile up -d
        Write-Host "✅ GENESIS GUI Dashboard started successfully!" -ForegroundColor Green
        
        # Wait a moment for startup
        Start-Sleep -Seconds 3
        
        # Check status
        $containerStatus = docker ps --filter "name=$ContainerName" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        Write-Host ""
        Write-Host "📊 Container Status:" -ForegroundColor Cyan
        Write-Host $containerStatus
        
        Write-Host ""
        Write-Host "🖥️ GUI should appear via X Server (Xming/VcXsrv)" -ForegroundColor Green
        Write-Host "🔧 If GUI doesn't appear, check X Server configuration" -ForegroundColor Yellow
    }
    catch {
        Write-Host "❌ Failed to start dashboard!" -ForegroundColor Red
        throw
    }
    
    Write-Host ""
}

function Stop-GenesisDashboard {
    Write-Host "🛑 Stopping GENESIS GUI Dashboard..." -ForegroundColor Yellow
    
    try {
        docker-compose -f $ComposeFile down
        Write-Host "✅ Dashboard stopped successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to stop dashboard!" -ForegroundColor Red
        throw
    }
    
    Write-Host ""
}

function Restart-GenesisDashboard {
    Write-Host "🔄 Restarting GENESIS GUI Dashboard..." -ForegroundColor Yellow
    
    Stop-GenesisDashboard
    Start-GenesisDashboard
}

function Show-GenesisLogs {
    Write-Host "📋 Showing GENESIS Dashboard logs..." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to exit log view" -ForegroundColor Cyan
    Write-Host ""
    
    try {
        docker-compose -f $ComposeFile logs -f
    }
    catch {
        Write-Host "❌ Failed to retrieve logs!" -ForegroundColor Red
        throw
    }
}

function Clean-GenesisContainers {
    Write-Host "🧹 Cleaning up GENESIS containers and images..." -ForegroundColor Yellow
    
    try {
        # Stop and remove containers
        docker-compose -f $ComposeFile down -v
        
        # Remove images
        docker rmi $ImageName 2>$null
        
        # Clean up unused resources
        docker system prune -f
        
        Write-Host "✅ Cleanup completed!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Cleanup failed!" -ForegroundColor Red
        throw
    }
    
    Write-Host ""
}

function Show-GenesisStatus {
    Write-Host "📊 GENESIS System Status:" -ForegroundColor Cyan
    Write-Host ""
    
    # Container status
    Write-Host "🐳 Container Status:" -ForegroundColor Yellow
    $containers = docker ps -a --filter "name=$ContainerName" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
    if ($containers) {
        Write-Host $containers
    } else {
        Write-Host "No GENESIS containers found" -ForegroundColor Gray
    }
    
    Write-Host ""
    
    # Image status
    Write-Host "📦 Image Status:" -ForegroundColor Yellow
    $images = docker images --filter "reference=$ImageName" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    if ($images) {
        Write-Host $images
    } else {
        Write-Host "No GENESIS images found" -ForegroundColor Gray
    }
    
    Write-Host ""
    
    # System resources
    Write-Host "💻 System Resources:" -ForegroundColor Yellow
    try {
        $dockerStats = docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $ContainerName 2>$null
        if ($dockerStats) {
            Write-Host $dockerStats
        } else {
            Write-Host "Container not running" -ForegroundColor Gray
        }
    }
    catch {
        Write-Host "Unable to retrieve resource usage" -ForegroundColor Gray
    }
    
    Write-Host ""
}

# Main execution
Write-GenesisHeader

switch ($Command) {
    "build" {
        Test-Prerequisites
        Build-GenesisImage
    }
    "start" {
        Test-Prerequisites
        Start-GenesisDashboard
    }
    "stop" {
        Stop-GenesisDashboard
    }
    "restart" {
        Test-Prerequisites
        Restart-GenesisDashboard
    }
    "logs" {
        Show-GenesisLogs
    }
    "clean" {
        Clean-GenesisContainers
    }
    "status" {
        Show-GenesisStatus
    }
    default {
        Write-Host "❌ Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Write-Host "📖 Available commands:" -ForegroundColor Cyan
        Write-Host "   build   - Build Docker image" -ForegroundColor White
        Write-Host "   start   - Start dashboard (default)" -ForegroundColor White
        Write-Host "   stop    - Stop dashboard" -ForegroundColor White
        Write-Host "   restart - Restart dashboard" -ForegroundColor White
        Write-Host "   logs    - View logs" -ForegroundColor White
        Write-Host "   clean   - Clean up containers and images" -ForegroundColor White
        Write-Host "   status  - Show system status" -ForegroundColor White
        Write-Host ""
        Write-Host "📝 Usage examples:" -ForegroundColor Cyan
        Write-Host "   .\deploy-gui.ps1 build" -ForegroundColor Gray
        Write-Host "   .\deploy-gui.ps1 start" -ForegroundColor Gray
        Write-Host "   .\deploy-gui.ps1 logs" -ForegroundColor Gray
        exit 1
    }
}

Write-Host "🎯 GENESIS Docker GUI deployment command completed!" -ForegroundColor Green
Write-Host ""
