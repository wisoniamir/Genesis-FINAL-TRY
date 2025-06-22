@echo off
SETLOCAL EnableDelayedExpansion

echo.
echo ========================================================
echo    DIRECT FORCED FIX FOR DESKTOP.INI REFERENCE ERRORS
echo ========================================================
echo.

echo This tool will FORCEFULLY remove all desktop.ini references
echo from your Git repository with administrator privileges.
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo NOTICE: This script works best with administrator privileges.
    echo Please run as administrator if this fix doesn't resolve the issue.
    echo.
    pause
    echo.
)

echo Working directory: %CD%
echo.

:: Force add desktop.ini to gitignore
echo STEP 1: Updating .gitignore
echo desktop.ini>.gitignore.temp
type .gitignore>>.gitignore.temp
move /Y .gitignore.temp .gitignore
echo Updated .gitignore successfully.
echo.

:: Direct filesystem operations to remove desktop.ini references
echo STEP 2: Direct filesystem operations to remove desktop.ini files...
echo.

echo Fixing Git database...
if exist .git\refs\tags\desktop.ini (
    echo Found desktop.ini reference, removing...
    del /f /q .git\refs\tags\desktop.ini 2>nul
)

if exist .git\refs\tags\desktop.ini (
    echo Stubborn file detected! Using alternate removal...
    attrib -r -h -s .git\refs\tags\desktop.ini
    del /f /q .git\refs\tags\desktop.ini
)

:: Remove any potential directory with that name
if exist .git\refs\tags\desktop.ini\ (
    echo Removing desktop.ini directory...
    rd /s /q .git\refs\tags\desktop.ini\
)

echo Cleaning Git packed references...
if exist .git\packed-refs (
    echo Processing packed-refs file...
    type .git\packed-refs | findstr /v "desktop.ini" > .git\packed-refs.new
    move /Y .git\packed-refs.new .git\packed-refs
)

echo.
echo STEP 3: Direct Git operations...

echo Removing from Git index...
git rm -f --cached **/desktop.ini >nul 2>&1
git rm -f --cached desktop.ini >nul 2>&1

echo Aggressive Git cleanup...
git gc --aggressive --prune=now >nul 2>&1
git reflog expire --expire=now --all >nul 2>&1

echo Committing changes...
git add .gitignore >nul 2>&1
git commit -m "Final fix for desktop.ini reference issues" >nul 2>&1

echo.
echo STEP 4: Radical actions if needed...

set /p radical=Would you like to try radical repair actions? Only use if still having issues (Y/N): 

if /i "%radical%"=="Y" (
    echo.
    echo Performing radical repair operations...
    
    echo Backing up important Git data...
    if not exist .git_backup mkdir .git_backup
    
    if exist .git\config (
        copy .git\config .git_backup\config
        echo Config file backed up
    )
    
    :: Get remote URL
    git remote -v > .git_backup\remotes.txt
    
    echo Getting branch information...
    git branch -a > .git_backup\branches.txt
    
    echo Saving current origin URL...
    for /f "tokens=2" %%u in ('git remote get-url origin 2^>nul') do (
        set origin_url=%%u
        echo !origin_url! > .git_backup\origin_url.txt
    )
    
    echo.
    echo =====================================
    echo WARNING: ABOUT TO RECONSTRUCT REPOSITORY
    echo =====================================
    echo.
    echo This will completely remove .git directory
    echo and recreate the Git repository from scratch.
    echo.
    
    set /p confirm=Are you ABSOLUTELY SURE? This cannot be undone (Y/N): 
    
    if /i "!confirm!"=="Y" (
        echo.
        echo Removing Git repository...
        rd /s /q .git
        
        echo Initializing new repository...
        git init
        
        echo Restoring remote origin...
        if exist .git_backup\origin_url.txt (
            for /f "delims=" %%u in (.git_backup\origin_url.txt) do (
                git remote add origin %%u
                echo Restored origin: %%u
            )
        )
        
        echo Adding all files...
        git add .
        
        echo Creating initial commit...
        git commit -m "Complete repository reconstruction to fix reference errors"
        
        echo Repository reconstruction complete.
        echo.
        echo If you had a remote repository, you'll need to force push:
        echo git push -f origin main
        echo.
    ) else (
        echo Repository reconstruction canceled.
    )
)

echo.
echo ========================================================
echo    FIX COMPLETED
echo ========================================================
echo.
echo The desktop.ini error should now be fixed.
echo Please close and reopen GitHub Desktop.
echo.

pause
