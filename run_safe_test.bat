@echo off
echo Running Safe MetaSignalHarmonizer Test
cd /d "c:\Users\patra\Genesis FINAL TRY"
python test_meta_signal_harmonizer_safe.py
if %errorlevel% neq 0 (
    echo Test completed with exit code %errorlevel%
)
echo Test execution finished.
pause
