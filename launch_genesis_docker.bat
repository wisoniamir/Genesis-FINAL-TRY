@echo off
REM 🚀 GENESIS Docker Launch - Containerized Windows Application
REM ARCHITECT MODE v7.0.0 - NO LOCAL SERVERS, FULL CONTAINER INTEGRATION

echo.
echo ================================================================================
echo 🐳 GENESIS CONTAINERIZED WINDOWS APPLICATION LAUNCHER
echo 🚨 ARCHITECT MODE v7.0.0 - FULL DOCKER INTEGRATION
echo ================================================================================
echo.

REM Add Docker to PATH
set PATH=%PATH%;C:\Program Files\Docker\Docker\resources\bin

REM Check Docker
echo 📊 Checking Docker installation...
docker --version
if errorlevel 1 (
    echo ❌ Docker not found! Please install Docker Desktop.
    pause
    exit /b 1
)

echo ✅ Docker found!
echo.

REM Check Docker daemon
echo 🔧 Checking Docker daemon...
docker info >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Docker daemon not running. Please start Docker Desktop manually.
    echo 💡 After Docker Desktop starts, run this script again.
    pause
    exit /b 1
)

echo ✅ Docker daemon is running!
echo.

REM Create network
echo 🌐 Creating Docker network...
docker network create genesis-network 2>nul

REM Stop existing containers
echo 🛑 Cleaning up existing containers...
docker stop genesis-backend genesis-frontend 2>nul
docker rm genesis-backend genesis-frontend 2>nul

REM Build containers
echo 🔨 Building GENESIS Backend container...
docker build -t genesis-backend -f docker/Dockerfile.backend .
if errorlevel 1 (
    echo ❌ Backend build failed!
    pause
    exit /b 1
)

echo 🔨 Building GENESIS Frontend container...
docker build -t genesis-frontend -f docker/Dockerfile.frontend .
if errorlevel 1 (
    echo ❌ Frontend build failed!
    pause
    exit /b 1
)

REM Launch Backend
echo 🚀 Launching GENESIS Backend...
docker run -d --name genesis-backend --network genesis-network -p 8000:8000 -v "%CD%/core:/app/core" -v "%CD%/modules:/app/modules" -v "%CD%/config:/app/config" -v "%CD%/logs:/app/logs" -e ARCHITECT_MODE=true -e ZERO_TOLERANCE=true genesis-backend
if errorlevel 1 (
    echo ❌ Backend launch failed!
    pause
    exit /b 1
)

REM Launch Frontend
echo 🖥️ Launching GENESIS Frontend...
docker run -d --name genesis-frontend --network genesis-network -p 3000:3000 -e REACT_APP_API_URL=http://localhost:8000 -e NODE_ENV=production genesis-frontend
if errorlevel 1 (
    echo ❌ Frontend launch failed!
    pause
    exit /b 1
)

echo.
echo ⏳ Waiting for services to initialize...
timeout /t 10 /nobreak >nul

echo.
echo ================================================================================
echo 🎉 GENESIS CONTAINERIZED APPLICATION - OPERATIONAL!
echo ================================================================================
echo 🖥️ GENESIS Application: http://localhost:3000
echo 🔧 Backend API: http://localhost:8000  
echo 📊 Health Check: http://localhost:8000/health
echo ================================================================================
echo 🎯 ARCHITECT MODE v7.0.0 FEATURES:
echo ✅ Containerized frontend + backend integration
echo ✅ ALL modules wired via EventBus
echo ✅ Real MT5 data integration ready
echo ✅ Zero mock data enforcement
echo ✅ Complete Docker isolation
echo ================================================================================
echo.

REM Open application
echo 🌐 Opening GENESIS Application...
start http://localhost:3000

echo.
echo 🎯 GENESIS is now running in containers!
echo 📊 Container status:
docker ps --filter "name=genesis"

echo.
echo Press any key to stop all containers...
pause >nul

echo.
echo 🛑 Stopping GENESIS containers...
docker stop genesis-backend genesis-frontend
docker rm genesis-backend genesis-frontend

echo.
echo ✅ All containers stopped successfully!
echo Press any key to exit...
pause >nul
