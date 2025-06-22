# GENESIS Discovery GUI Docker Deployment Script
# Production-grade deployment with full validation

param(
    [string]$Action = "deploy",
    [switch]$Validate = $false,
    [switch]$Cleanup = $false,
    [switch]$Development = $false
)

Write-Host "🔍 GENESIS Discovery GUI Deployment System v7.0.0" -ForegroundColor Cyan
Write-Host "🚨 ARCHITECT MODE - ZERO TOLERANCE ENFORCEMENT" -ForegroundColor Red
Write-Host ""

# Configuration
$COMPOSE_FILE = "docker-compose-discovery-gui.yml"
$DOCKERFILE = "Dockerfile.discovery-gui"
$PROJECT_NAME = "genesis-discovery"
$CONTAINER_NAME = "genesis_discovery_dashboard"

function Test-Prerequisites {
    Write-Host "🔍 Checking Prerequisites..." -ForegroundColor Yellow
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker not found or not running" -ForegroundColor Red
        return $false
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version
        Write-Host "✅ Docker Compose: $composeVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker Compose not found" -ForegroundColor Red
        return $false
    }
    
    # Check Xming (for Windows GUI)
    $xmingProcess = Get-Process -Name "Xming" -ErrorAction SilentlyContinue
    if ($xmingProcess) {
        Write-Host "✅ Xming X11 Server: Running" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ Xming X11 Server: Not running (GUI may not display)" -ForegroundColor Yellow
        Write-Host "   Please start Xming with display :0.0" -ForegroundColor Yellow
    }
    
    # Check required files
    $requiredFiles = @($COMPOSE_FILE, $DOCKERFILE, "launch_desktop_app.py", "interface/dashboard/discovery_main.py")
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✅ Required file: $file" -ForegroundColor Green
        }
        else {
            Write-Host "❌ Missing required file: $file" -ForegroundColor Red
            return $false
        }
    }
    
    return $true
}

function Start-XmingServer {
    Write-Host "🖥️ Starting Xming X11 Server..." -ForegroundColor Yellow
    
    # Check if Xming is already running
    $xmingProcess = Get-Process -Name "Xming" -ErrorAction SilentlyContinue
    if ($xmingProcess) {
        Write-Host "✅ Xming already running" -ForegroundColor Green
        return
    }
    
    # Try to start Xming
    $xmingPath = Get-Command "Xming" -ErrorAction SilentlyContinue
    if ($xmingPath) {
        Start-Process "Xming" -ArgumentList ":0", "-clipboard", "-wgl" -WindowStyle Hidden
        Start-Sleep 3
        Write-Host "✅ Xming started" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ Xming not found in PATH. Please install and start manually." -ForegroundColor Yellow
        Write-Host "   Download from: https://sourceforge.net/projects/xming/" -ForegroundColor Yellow
    }
}

function Build-DiscoveryImage {
    Write-Host "🔨 Building GENESIS Discovery Docker Image..." -ForegroundColor Yellow
    
    # Set display environment for Docker
    $env:DISPLAY = "host.docker.internal:0.0"
    
    try {
        docker build -f $DOCKERFILE -t "genesis-discovery-gui:latest" .
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Docker image built successfully" -ForegroundColor Green
        }
        else {
            Write-Host "❌ Docker image build failed" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Docker build error: $_" -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Deploy-DiscoverySystem {
    Write-Host "🚀 Deploying GENESIS Discovery System..." -ForegroundColor Yellow
    
    # Create necessary directories
    $directories = @("data", "logs", "backup", "config")
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "📁 Created directory: $dir" -ForegroundColor Green
        }
    }
    
    # Set environment variables for deployment
    $env:DISPLAY = "host.docker.internal:0.0"
    $env:GENESIS_MODE = "production"
    $env:DISCOVERY_ENABLED = "true"
    
    try {
        # Deploy with Docker Compose
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ GENESIS Discovery System deployed successfully" -ForegroundColor Green
            
            # Wait for container to be ready
            Write-Host "⏳ Waiting for system to initialize..." -ForegroundColor Yellow
            Start-Sleep 10
            
            # Check container status
            $containerStatus = docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            Write-Host "📊 Container Status:" -ForegroundColor Cyan
            Write-Host $containerStatus
            
            return $true
        }
        else {
            Write-Host "❌ Deployment failed" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Deployment error: $_" -ForegroundColor Red
        return $false
    }
}

function Test-DiscoverySystem {
    Write-Host "🧪 Testing GENESIS Discovery System..." -ForegroundColor Yellow
    
    # Check if container is running
    $containerCheck = docker ps --filter "name=$CONTAINER_NAME" --quiet
    if (-not $containerCheck) {
        Write-Host "❌ Discovery container not running" -ForegroundColor Red
        return $false
    }
    
    # Run discovery validation
    try {
        Write-Host "🔍 Running discovery validation inside container..." -ForegroundColor Yellow
        $validationResult = docker exec $CONTAINER_NAME python /genesis/validate_discovery.py
        Write-Host $validationResult -ForegroundColor Green
        
        # Check health endpoint
        Write-Host "🩺 Checking container health..." -ForegroundColor Yellow
        $healthCheck = docker inspect $CONTAINER_NAME --format='{{.State.Health.Status}}'
        Write-Host "Health Status: $healthCheck" -ForegroundColor $(if ($healthCheck -eq "healthy") { "Green" } else { "Yellow" })
        
        return $true
    }
    catch {
        Write-Host "❌ Discovery validation failed: $_" -ForegroundColor Red
        return $false
    }
}

function Show-SystemStatus {
    Write-Host "📊 GENESIS Discovery System Status" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    
    # Container status
    docker ps --filter "name=genesis" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # Resource usage
    Write-Host ""
    Write-Host "💾 Resource Usage:" -ForegroundColor Yellow
    docker stats --no-stream --filter "name=genesis"
    
    # Logs preview
    Write-Host ""
    Write-Host "📝 Recent Logs:" -ForegroundColor Yellow
    docker logs $CONTAINER_NAME --tail 10
}

function Stop-DiscoverySystem {
    Write-Host "🛑 Stopping GENESIS Discovery System..." -ForegroundColor Yellow
    
    try {
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
        Write-Host "✅ System stopped successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error stopping system: $_" -ForegroundColor Red
    }
}

function Remove-DiscoverySystem {
    Write-Host "🗑️ Removing GENESIS Discovery System..." -ForegroundColor Yellow
    
    try {
        # Stop and remove containers
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down --volumes --remove-orphans
        
        # Remove images if requested
        if ($Cleanup) {
            docker rmi "genesis-discovery-gui:latest" -f
            Write-Host "✅ Images removed" -ForegroundColor Green
        }
        
        Write-Host "✅ System removed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error removing system: $_" -ForegroundColor Red
    }
}

# Main execution logic
try {
    switch ($Action.ToLower()) {
        "deploy" {
            if (!(Test-Prerequisites)) {
                Write-Host "❌ Prerequisites check failed. Aborting deployment." -ForegroundColor Red
                exit 1
            }
            
            Start-XmingServer
            
            if (Build-DiscoveryImage) {
                if (Deploy-DiscoverySystem) {
                    if ($Validate) {
                        Test-DiscoverySystem
                    }
                    Show-SystemStatus
                    
                    Write-Host ""
                    Write-Host "🎉 GENESIS Discovery GUI Deployment Complete!" -ForegroundColor Green
                    Write-Host "🖥️ Dashboard should be visible via Xming" -ForegroundColor Cyan
                    Write-Host "📊 Access URLs:" -ForegroundColor Cyan
                    Write-Host "   - Main GUI: Via Xming X11 display" -ForegroundColor White
                    Write-Host "   - Fallback Web: http://localhost:8501" -ForegroundColor White
                    Write-Host "   - Telemetry: http://localhost:8503" -ForegroundColor White
                }
                else {
                    exit 1
                }
            }
            else {
                exit 1
            }
        }
        
        "status" {
            Show-SystemStatus
        }
        
        "test" {
            Test-DiscoverySystem
        }
        
        "stop" {
            Stop-DiscoverySystem
        }
        
        "remove" {
            Remove-DiscoverySystem
        }
        
        "restart" {
            Stop-DiscoverySystem
            Start-Sleep 3
            Deploy-DiscoverySystem
        }
        
        default {
            Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
            Write-Host "Available actions: deploy, status, test, stop, remove, restart" -ForegroundColor Yellow
            exit 1
        }
    }
}
catch {
    Write-Host "❌ Unexpected error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🔍 GENESIS Discovery GUI Deployment System - Complete" -ForegroundColor Cyan
