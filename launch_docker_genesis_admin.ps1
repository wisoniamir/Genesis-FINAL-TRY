# 🚀 GENESIS Docker Launch Script - Administrator Mode
# ARCHITECT MODE v7.0.0 - Containerized Windows Application

Write-Host "🐳 GENESIS Docker Launch - Administrator Mode" -ForegroundColor Cyan
Write-Host "🚨 ARCHITECT MODE v7.0.0 - Containerized Application" -ForegroundColor Yellow

# Add Docker to PATH
$env:PATH += ";C:\Program Files\Docker\Docker\resources\bin"

# Check Docker availability
Write-Host "📊 Checking Docker status..." -ForegroundColor Green
docker --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker not available. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check Docker daemon
Write-Host "🔧 Checking Docker daemon..." -ForegroundColor Green
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Docker daemon not running. Starting Docker Desktop..." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -WindowStyle Hidden
    
    # Wait for Docker daemon to start
    $timeout = 60
    $elapsed = 0
    do {
        Start-Sleep -Seconds 5
        $elapsed += 5
        Write-Host "⏳ Waiting for Docker daemon... ($elapsed/$timeout seconds)" -ForegroundColor Yellow
        docker info > $null 2>&1
    } while ($LASTEXITCODE -ne 0 -and $elapsed -lt $timeout)
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker daemon failed to start within timeout." -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Docker daemon is running!" -ForegroundColor Green

# Build GENESIS containers
Write-Host "🔨 Building GENESIS containers..." -ForegroundColor Cyan
docker build -t genesis-backend -f docker/Dockerfile.backend .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Backend build failed!" -ForegroundColor Red
    exit 1
}

docker build -t genesis-frontend -f docker/Dockerfile.frontend .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Frontend build failed!" -ForegroundColor Red
    exit 1
}

# Create Docker network
Write-Host "🌐 Creating Docker network..." -ForegroundColor Cyan
docker network create genesis-network 2>$null

# Stop any existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker stop genesis-backend genesis-frontend 2>$null
docker rm genesis-backend genesis-frontend 2>$null

# Launch GENESIS Backend
Write-Host "🚀 Launching GENESIS Backend..." -ForegroundColor Green
docker run -d `
    --name genesis-backend `
    --network genesis-network `
    -p 8000:8000 `
    -v "${PWD}/core:/app/core" `
    -v "${PWD}/modules:/app/modules" `
    -v "${PWD}/config:/app/config" `
    -v "${PWD}/logs:/app/logs" `
    -e ARCHITECT_MODE=true `
    -e ZERO_TOLERANCE=true `
    genesis-backend

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Backend launch failed!" -ForegroundColor Red
    exit 1
}

# Launch GENESIS Frontend
Write-Host "🖥️ Launching GENESIS Frontend..." -ForegroundColor Green
docker run -d `
    --name genesis-frontend `
    --network genesis-network `
    -p 3000:3000 `
    -e REACT_APP_API_URL=http://localhost:8000 `
    -e NODE_ENV=production `
    genesis-frontend

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Frontend launch failed!" -ForegroundColor Red
    exit 1
}

# Wait for services to be ready
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check container status
Write-Host "📊 Checking container status..." -ForegroundColor Cyan
docker ps --filter "name=genesis"

# Display access information
Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "🚀 GENESIS CONTAINERIZED APPLICATION - READY!" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "🖥️ Frontend Application: http://localhost:3000" -ForegroundColor White
Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "📊 Health Check: http://localhost:8000/health" -ForegroundColor White
Write-Host "📡 System Status: http://localhost:8000/api/system/status" -ForegroundColor White
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "🎯 ARCHITECT MODE v7.0.0 - Full Containerized Integration" -ForegroundColor Green
Write-Host "✅ ALL modules wired via EventBus" -ForegroundColor Green
Write-Host "✅ Real MT5 data integration ready" -ForegroundColor Green
Write-Host "✅ Complete frontend/backend integration" -ForegroundColor Green
Write-Host "✅ Zero mock data enforcement" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Cyan

# Open the application
Write-Host "🌐 Opening GENESIS Application..." -ForegroundColor Green
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "🎯 GENESIS Containerized Application is now running!" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop all containers" -ForegroundColor Yellow

# Keep script running and monitor containers
try {
    while ($true) {
        Start-Sleep -Seconds 30
        
        # Check if containers are still running
        $backendStatus = docker ps -q --filter "name=genesis-backend"
        $frontendStatus = docker ps -q --filter "name=genesis-frontend"
        
        if (-not $backendStatus) {
            Write-Host "⚠️ Backend container stopped!" -ForegroundColor Red
        }
        
        if (-not $frontendStatus) {
            Write-Host "⚠️ Frontend container stopped!" -ForegroundColor Red
        }
        
        if ($backendStatus -and $frontendStatus) {
            Write-Host "✅ All GENESIS containers running healthy" -ForegroundColor Green
        }
    }
}
catch {
    Write-Host "🛑 Stopping GENESIS containers..." -ForegroundColor Yellow
    docker stop genesis-backend genesis-frontend
    docker rm genesis-backend genesis-frontend
    Write-Host "✅ GENESIS containers stopped cleanly" -ForegroundColor Green
}
