@echo off
echo ===== GitHub Desktop.ini Error Prevention Utility =====
echo This tool helps prevent desktop.ini errors in your Git repository

:: Navigate to repository root
cd /d "C:\Users\patra\Genesis FINAL TRY"

:: Add .gitignore entries if needed
echo Updating .gitignore to exclude desktop.ini files...
findstr /c:"desktop.ini" .gitignore >nul || echo desktop.ini>> .gitignore
findstr /c:"Thumbs.db" .gitignore >nul || echo Thumbs.db>> .gitignore
findstr /c:"**/desktop.ini" .gitignore >nul || echo **/desktop.ini>> .gitignore

:: Remove any desktop.ini files from git tracking
echo Removing any desktop.ini files from git tracking...
git rm --cached -f "desktop.ini" 2>nul
git rm --cached -f "**/desktop.ini" 2>nul
git rm --cached -f "**/*/desktop.ini" 2>nul

:: Fix broken refs
echo Fixing any broken references...
git update-ref -d refs/tags/desktop.ini 2>nul
git gc --prune=now

:: Commit changes
echo Committing changes to .gitignore...
git add .gitignore
git commit -m "Prevent desktop.ini tracking in Git"

echo ===== Cleanup Complete =====
echo Your repository should now be protected against desktop.ini errors.
echo You can now safely use GitHub Desktop.
pause
