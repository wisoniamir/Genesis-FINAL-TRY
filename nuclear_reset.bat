@echo off
echo NUCLEAR RESET: Clearing all Python cache and processes...

REM Kill all Python processes
taskkill /F /IM python.exe 2>NUL
taskkill /F /IM pythonw.exe 2>NUL

REM Clear Python cache files
del /S /Q __pycache__ 2>NUL
del /S /Q *.pyc 2>NUL
del /S /Q *.pyo 2>NUL

REM Clear corrupted JSON files that might be causing blocking
del event_bus.json.bak 2>NUL
del telemetry.json.bak 2>NUL
copy NUL event_bus.json 2>NUL
copy NUL telemetry.json 2>NUL

echo RESET COMPLETE
echo Testing basic Python...
python -c "print('Python works!')"
echo READY FOR TESTING
pause
