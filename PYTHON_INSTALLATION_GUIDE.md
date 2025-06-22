# 🐍 PYTHON INSTALLATION GUIDE FOR GENESIS
===========================================

**CRITICAL:** The GENESIS system requires a working Python installation to function. Follow this guide to restore your Python environment.

## 🚨 **CURRENT ISSUE**
Your Python environment is corrupted with the error:
```
failed to locate pyvenv.cfg: The system cannot find the file specified.
```

This indicates a broken virtual environment or corrupted Python installation.

## 🔧 **SOLUTION 1: Fresh Python Installation (RECOMMENDED)**

### Step 1: Download Python
1. Go to **https://www.python.org/downloads/**
2. Download **Python 3.11.x** or **Python 3.12.x** (latest stable)
3. Choose the **Windows installer (64-bit)**

### Step 2: Install Python
1. **RUN AS ADMINISTRATOR** - Right-click installer → "Run as administrator"
2. **✅ CHECK "Add Python to PATH"** (CRITICAL!)
3. **✅ CHECK "Add Python to environment variables"**
4. Choose **"Customize installation"**
5. **✅ CHECK ALL optional features**
6. **✅ CHECK "Add Python to environment variables"** (again)
7. **✅ CHECK "Precompile standard library"**
8. Click **"Install"**

### Step 3: Verify Installation
Open **Command Prompt** (Win+R → cmd) and test:
```bash
python --version
pip --version
```

You should see version numbers, not errors.

## 🔧 **SOLUTION 2: Anaconda/Miniconda (ALTERNATIVE)**

### Download and Install
1. Go to **https://www.anaconda.com/download**
2. Download **Anaconda Individual Edition**
3. Install with default settings

### Create GENESIS Environment
```bash
conda create -n genesis python=3.11
conda activate genesis
pip install MetaTrader5==5.0.45 streamlit pandas numpy
```

## 🔧 **SOLUTION 3: Portable Python (EMERGENCY)**

### Download WinPython
1. Go to **https://winpython.github.io/**
2. Download **WinPython 3.11.x**
3. Extract to **C:\WinPython**
4. Add **C:\WinPython\python-3.11.x.amd64** to PATH

## 🚀 **AFTER PYTHON IS FIXED**

### Automatic Restoration
Run the restoration script:
```bash
RESTORE_GENESIS_ENVIRONMENT.bat
```

### Manual Steps
```bash
# 1. Install MetaTrader5
pip install MetaTrader5==5.0.45 --force-reinstall

# 2. Test installation
python -c "import MetaTrader5 as mt5; print('MT5 OK')"

# 3. Run module restoration
python urgent_module_restoration_engine.py

# 4. Launch GENESIS
python genesis_desktop.py
```

### VS Code Tasks
Use **Ctrl+Shift+P** → **"Tasks: Run Task"** → **"🛠️ Complete GENESIS Restoration"**

## 🛡️ **VERIFICATION CHECKLIST**

After installation, verify these work:
- [ ] `python --version` shows Python 3.11+
- [ ] `pip --version` shows pip version
- [ ] `python -c "import MetaTrader5"` works without errors
- [ ] `python genesis_desktop.py` launches GENESIS

## ⚠️ **TROUBLESHOOTING**

### "Python not found"
- Restart Command Prompt after Python installation
- Restart VS Code after Python installation
- Check Windows PATH includes Python directories

### "MetaTrader5 import error"
```bash
pip uninstall MetaTrader5
pip install MetaTrader5==5.0.45 --no-cache-dir
```

### "Permission denied"
- Run Command Prompt as Administrator
- Install Python as Administrator

## 🎯 **EXPECTED OUTCOME**

After successful restoration:
✅ Python environment working  
✅ MetaTrader5 package functional  
✅ GENESIS modules restored  
✅ Desktop application launches  
✅ Full MT5 integration available  

---

**ARCHITECT MODE v7.0.0:** Ready to resume full operation after Python restoration.
