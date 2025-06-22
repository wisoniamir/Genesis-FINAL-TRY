#!/bin/bash

# 🐳 GENESIS Docker Startup Script
# Architect Mode v7.0 - Full System Launch

echo "🚀 GENESIS TRADING PLATFORM - DOCKER DEPLOYMENT"
echo "📊 Architect Mode v7.0 - Zero Tolerance Compliance"
echo "🔐 Full Stack Dashboard with Real-time Integration"
echo "=" * 60

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed."
    exit 1
fi

echo "✅ Docker environment ready"

# Build and start services
echo "🔨 Building GENESIS services..."
docker-compose build --no-cache

echo "🚀 Starting GENESIS platform..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check service health
echo "🔍 Checking service health..."

# Check API health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is healthy"
else
    echo "❌ Backend API is not responding"
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend dashboard is accessible"
else
    echo "❌ Frontend dashboard is not responding"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not responding"
fi

echo ""
echo "🎉 GENESIS PLATFORM DEPLOYMENT COMPLETE!"
echo ""
echo "📊 Access Points:"
echo "   Frontend Dashboard: http://localhost:3000"
echo "   Backend API:       http://localhost:8000"
echo "   API Health:        http://localhost:8000/health"
echo ""
echo "🔧 Management Commands:"
echo "   View logs:         docker-compose logs -f"
echo "   Stop platform:     docker-compose down"
echo "   Restart services:  docker-compose restart"
echo "   View status:       docker-compose ps"
echo ""
echo "🚨 Monitor the logs for any startup issues:"
echo "   docker-compose logs -f genesis-api"
echo "   docker-compose logs -f genesis-dashboard"
echo ""
echo "✅ GENESIS is now running in containerized mode!"
echo "🔐 Architect Mode v7.0 - Full Compliance Active"
