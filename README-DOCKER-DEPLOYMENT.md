# 🏛️ GENESIS Institutional Trading Dashboard - Docker Deployment Guide

## Overview

The GENESIS Institutional Trading Dashboard is an enterprise-grade real-time trading platform that runs as a native GUI application inside Docker containers with X11 support for Windows via Xming.

## ✅ Features

- **Real-time MT5 Integration**: Live trading data from MetaTrader 5 terminals
- **Institutional-Grade UI**: Professional Streamlit + PyQt5 hybrid interface
- **Docker Compatibility**: Full containerization with X11 GUI support
- **EventBus Architecture**: All modules connected via event-driven architecture
- **Comprehensive Telemetry**: Real-time monitoring and compliance tracking
- **Zero Mock Data**: 100% live data feeds only

## 🛠️ Prerequisites

### Windows Requirements
1. **Docker Desktop for Windows** (latest version)
2. **Xming X Server** - Download from [Xming SourceForge](https://sourceforge.net/projects/xming/)
3. **MetaTrader 5** terminal installed (for live data)
4. **PowerShell 5.0+** (usually pre-installed)

### Xming Setup
1. Install Xming with default settings
2. Launch Xming from Start Menu
3. Ensure Xming is running (check system tray for X icon)

## 🚀 Quick Start

### Option 1: PowerShell Script (Recommended)
```powershell
# Build the container
./deploy-docker.ps1 build

# Start the dashboard
./deploy-docker.ps1 start

# Access dashboard at: http://localhost:8501
```

### Option 2: Docker Compose
```bash
# Build and start
docker-compose -f docker-compose-institutional.yml up --build

# Run in background
docker-compose -f docker-compose-institutional.yml up -d
```

### Option 3: Direct Docker
```bash
# Build image
docker build -t genesis-institutional-dashboard .

# Run container
docker run -d \
  --name genesis-dashboard \
  -p 8501:8501 \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  genesis-institutional-dashboard
```

## 📊 Dashboard Access

- **Streamlit Interface**: http://localhost:8501
- **Health Check**: http://localhost:8501/_stcore/health

## 🔧 Management Commands

### PowerShell Script Commands
```powershell
./deploy-docker.ps1 build    # Build container
./deploy-docker.ps1 start    # Start dashboard
./deploy-docker.ps1 stop     # Stop dashboard  
./deploy-docker.ps1 restart  # Restart dashboard
./deploy-docker.ps1 logs     # View logs
./deploy-docker.ps1 clean    # Clean up containers
```

### Docker Compose Commands
```bash
docker-compose -f docker-compose-institutional.yml up -d     # Start
docker-compose -f docker-compose-institutional.yml down     # Stop
docker-compose -f docker-compose-institutional.yml logs -f  # Logs
docker-compose -f docker-compose-institutional.yml restart  # Restart
```

## 📁 Directory Structure

```
Genesis FINAL TRY/
├── genesis_institutional_dashboard.py    # Main dashboard application
├── core/
│   └── genesis_real_mt5_connection.py    # MT5 connection module
├── Dockerfile                            # Container definition
├── docker-compose-institutional.yml     # Compose configuration
├── launch_institutional_dashboard.sh    # Container entrypoint
├── deploy-docker.ps1                    # Windows deployment script
├── requirements_docker.txt              # Python dependencies
└── logs/                                # Application logs
```

## 🔍 Troubleshooting

### Container Won't Start
1. Check Docker Desktop is running
2. Verify no port conflicts on 8501
3. Check container logs: `./deploy-docker.ps1 logs`

### GUI Not Displaying
1. Ensure Xming is running (check system tray)
2. Verify DISPLAY environment variable is set
3. Check Windows Firewall settings for Xming

### MT5 Connection Issues
1. Verify MetaTrader 5 is installed
2. Check MT5 terminal is not running in demo mode
3. Ensure MT5 Expert Advisors are enabled

### Performance Issues
1. Allocate more memory to Docker Desktop
2. Close unnecessary applications
3. Check system resources in Task Manager

## 🔐 Security Notes

- The dashboard requires real MT5 terminal access
- No demo/mock data is permitted (ARCHITECT MODE compliance)
- All modules are EventBus connected for security
- Telemetry monitoring is always active

## 📈 Monitoring

### Health Checks
The container includes automatic health monitoring:
- HTTP health endpoint at `/_stcore/health`
- 30-second interval checks
- Automatic restart on failure

### Logs
- Container logs: `./deploy-docker.ps1 logs`
- Application logs: `./logs/dashboard/`
- Telemetry logs: `./telemetry/`

## 🆘 Support

### Common Issues
1. **Port 8501 already in use**: Stop other Streamlit apps or change port mapping
2. **Xming connection refused**: Restart Xming and ensure firewall allows connections
3. **MT5 connection failed**: Verify MT5 installation and account setup

### Log Analysis
```bash
# View real-time logs
docker logs -f genesis-institutional-dashboard

# Check specific errors
docker logs genesis-institutional-dashboard 2>&1 | grep "ERROR"
```

## 🔄 Updates

To update the dashboard:
```powershell
./deploy-docker.ps1 stop
./deploy-docker.ps1 build
./deploy-docker.ps1 start
```

## 📋 System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **CPU**: Dual-core 2.5GHz+, Recommended Quad-core 3.0GHz+
- **Storage**: 5GB free space
- **Network**: Stable internet connection for MT5 data feeds
- **OS**: Windows 10/11 with Docker Desktop support

---

## 🏛️ Enterprise Features

This institutional dashboard includes:
- Multi-asset portfolio monitoring
- Advanced risk management visualization  
- Real-time execution quality tracking
- Compliance monitoring and reporting
- Performance analytics and backtesting
- Pattern recognition and signal detection
- Trade journal and audit trails

**Status**: ✅ Production Ready - ARCHITECT MODE v7.0.0 Compliant
