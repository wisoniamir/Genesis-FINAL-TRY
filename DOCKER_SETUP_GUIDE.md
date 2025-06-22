# 🚀 GENESIS PyQt5 GUI - Docker and X Server Setup Guide
# ARCHITECT MODE v7.0.0

## Docker Desktop Installation

### Step 1: Install Docker Desktop
1. Download Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Run the installer and follow the on-screen instructions
3. When prompted, enable required Hyper-V or WSL2 features
4. Complete the installation and restart your computer if required

### Step 2: Start Docker Desktop
1. After restarting, Docker Desktop should start automatically
2. If not, find Docker Desktop in the Start Menu and launch it
3. Wait for the Docker Engine to start (shows "Engine running" in status bar)
4. The Docker icon in the system tray should be steady (not animated)

## X Server Setup for GUI Display

### Option 1: VcXsrv (Recommended)
1. Download and install VcXsrv from [https://sourceforge.net/projects/vcxsrv/](https://sourceforge.net/projects/vcxsrv/)
2. Run XLaunch from the Start Menu
3. Select "Multiple windows" and click Next
4. Select "Start no client" and click Next
5. Check "Disable access control" and click Next
6. Click Finish to start the X Server

### Option 2: Xming
1. Download and install Xming from [https://sourceforge.net/projects/xming/](https://sourceforge.net/projects/xming/)
2. Run Xming from the Start Menu
3. Use the default settings and ensure "No Access Control" is checked

## Testing Your Configuration

### Test the X Server
1. Run the X Server test script:
   ```
   python test_x_server.py
   ```
2. If a window appears, your X Server is configured correctly

### Test Docker
1. Open a Command Prompt or PowerShell and run:
   ```
   docker --version
   ```
2. You should see the Docker version output

## Launching the GENESIS PyQt5 GUI

### Using the Batch File
1. Open a Command Prompt or PowerShell
2. Navigate to your GENESIS directory:
   ```
   cd C:\Users\patra\Genesis FINAL TRY
   ```
3. Run the launcher:
   ```
   .\launch_genesis_docker_gui.bat
   ```
4. The GENESIS PyQt5 GUI should appear as a native desktop application

### Troubleshooting

#### Docker Issues
- Ensure Docker Desktop is running (check the system tray icon)
- If Docker commands fail, try restarting Docker Desktop
- Ensure your user has permissions to run Docker

#### X Server Issues
- Ensure the X Server is running (check the system tray for XLaunch or Xming icons)
- Try restarting the X Server with "Disable access control" checked
- Set DISPLAY environment variable manually:
  ```
  set DISPLAY=localhost:0.0
  ```

#### PyQt5 GUI Issues
- If the GUI doesn't appear, check Docker logs:
  ```
  docker logs genesis_gui_app
  ```
- Ensure all required directories exist
- Check the X Server configuration
