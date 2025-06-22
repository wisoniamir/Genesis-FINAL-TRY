@echo off
REM 🐳 GENESIS Docker Startup Script - Windows
REM Architect Mode v7.0 - Full System Launch

echo 🚀 GENESIS TRADING PLATFORM - DOCKER DEPLOYMENT
echo 📊 Architect Mode v7.0 - Zero Tolerance Compliance  
echo 🔐 Full Stack Dashboard with Real-time Integration
echo ============================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed.
    pause
    exit /b 1
)

echo ✅ Docker environment ready

REM Build and start services
echo 🔨 Building GENESIS services...
docker-compose build --no-cache

echo 🚀 Starting GENESIS platform...
docker-compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 15 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check API health
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Backend API is not responding
) else (
    echo ✅ Backend API is healthy
)

REM Check frontend
curl -f http://localhost:3000 >nul 2>&1
if errorlevel 1 (
    echo ❌ Frontend dashboard is not responding
) else (
    echo ✅ Frontend dashboard is accessible
)

echo.
echo 🎉 GENESIS PLATFORM DEPLOYMENT COMPLETE!
echo.
echo 📊 Access Points:
echo    Frontend Dashboard: http://localhost:3000
echo    Backend API:       http://localhost:8000
echo    API Health:        http://localhost:8000/health
echo.
echo 🔧 Management Commands:
echo    View logs:         docker-compose logs -f
echo    Stop platform:     docker-compose down
echo    Restart services:  docker-compose restart
echo    View status:       docker-compose ps
echo.
echo 🚨 Monitor the logs for any startup issues:
echo    docker-compose logs -f genesis-api
echo    docker-compose logs -f genesis-dashboard
echo.
echo ✅ GENESIS is now running in containerized mode!
echo 🔐 Architect Mode v7.0 - Full Compliance Active
echo.
echo Press any key to open the dashboard in your browser...
pause >nul

REM Open dashboard in default browser
start http://localhost:3000
