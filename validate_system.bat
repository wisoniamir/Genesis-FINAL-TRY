@echo off
REM Quick system validation check
echo 🔍 Running GENESIS system validation...

"C:\Users\patra\Genesis FINAL TRY\.venv\Scripts\python.exe" -c "from lightweight_validation_hook import validate_and_patch, get_system_status; import json; result = validate_and_patch(); status = get_system_status(); print('=== VALIDATION RESULT ==='); print('Passed:', result.get('validation_passed', False)); print('Issues:', len(result.get('issues_found', []))); print('=== SYSTEM STATUS ==='); print(json.dumps(status, indent=2))"

echo.
echo ✅ Validation complete!
pause
