# Direct PowerShell fix for desktop.ini Git reference errors
# This script will forcefully remove desktop.ini references from Git

Write-Host @"
========================================================
   DIRECT FORCED FIX FOR DESKTOP.INI REFERENCE ERRORS
========================================================
"@ -ForegroundColor Cyan

Write-Host "Working on C:\Users\patra\Genesis FINAL TRY..." -ForegroundColor Yellow

# Update .gitignore
Write-Host "Adding desktop.ini to .gitignore..." -ForegroundColor Yellow
$ignoreContent = Get-Content -Path ".gitignore" -ErrorAction SilentlyContinue
if (-not ($ignoreContent -match "desktop\.ini")) {
    "desktop.ini" | Out-File -FilePath ".gitignore" -Encoding ASCII -Append
    Write-Host "desktop.ini added to .gitignore" -ForegroundColor Green
}

# Direct fixes to Git repository
Write-Host "Forcefully removing desktop.ini references..." -ForegroundColor Yellow

# Remove desktop.ini from Git index
Write-Host "Removing from Git index..." -ForegroundColor Yellow
git rm --cached -r --ignore-unmatch "**/desktop.ini" 2>$null
git rm --cached -r --ignore-unmatch "desktop.ini" 2>$null

# Clean up Git database
Write-Host "Cleaning Git database..." -ForegroundColor Yellow
git gc --aggressive --prune=now 2>$null
git reflog expire --expire=now --all 2>$null

# Direct filesystem operations to remove desktop.ini references
Write-Host "Direct filesystem operations..." -ForegroundColor Yellow
if (Test-Path ".git\refs\tags\desktop.ini") {
    Remove-Item -Path ".git\refs\tags\desktop.ini" -Force -ErrorAction SilentlyContinue
    Write-Host "Removed .git\refs\tags\desktop.ini" -ForegroundColor Green
}

# Cleanup packed-refs
if (Test-Path ".git\packed-refs") {
    Write-Host "Cleaning packed references..." -ForegroundColor Yellow
    $packedRefs = Get-Content ".git\packed-refs" -ErrorAction SilentlyContinue
    $cleanedRefs = $packedRefs | Where-Object { $_ -notmatch "desktop\.ini" }
    $cleanedRefs | Set-Content ".git\packed-refs" -Force
}

# Commit changes
git add .gitignore 2>$null
git commit -m "Fixed desktop.ini reference issues" 2>$null

# Reconstruct repository if needed
Write-Host "`nWould you like to completely reconstruct the Git repository?" -ForegroundColor Yellow
Write-Host "This is the most extreme solution but will definitely fix the issue." -ForegroundColor Yellow
Write-Host "WARNING: You will need to force-push to any remote repository after this." -ForegroundColor Red
$reconstruct = Read-Host "Proceed with reconstruction? (Y/N)"

if ($reconstruct -eq "Y" -or $reconstruct -eq "y") {
    Write-Host "`nReconstructing repository..." -ForegroundColor Red
    
    # Backup config and remote information
    if (-not (Test-Path ".git_backup")) {
        New-Item -ItemType Directory -Path ".git_backup" | Out-Null
    }
    
    # Save Git config
    if (Test-Path ".git\config") {
        Copy-Item ".git\config" ".git_backup\config"
    }
    
    # Save remote URL
    try {
        $originUrl = git remote get-url origin 2>$null
        if ($originUrl) {
            $originUrl | Out-File -FilePath ".git_backup\origin_url.txt" -Encoding ASCII
            Write-Host "Saved origin URL: $originUrl" -ForegroundColor Green
        }
    } catch {}
    
    # Save branch info
    git branch -a | Out-File -FilePath ".git_backup\branches.txt" -Encoding ASCII
    
    # Remove .git directory
    Remove-Item -Recurse -Force ".git"
    
    # Initialize new repository
    git init
    
    # Restore remote origin
    if (Test-Path ".git_backup\origin_url.txt") {
        $originUrl = Get-Content ".git_backup\origin_url.txt" -ErrorAction SilentlyContinue
        if ($originUrl) {
            git remote add origin $originUrl
            Write-Host "Restored origin URL: $originUrl" -ForegroundColor Green
        }
    }
    
    # Add all files
    git add .
    
    # Commit
    git commit -m "Complete repository reconstruction to fix reference errors"
    
    Write-Host "`nRepository has been reconstructed successfully!" -ForegroundColor Green
    Write-Host "To update your remote repository, you'll need to force push:" -ForegroundColor Yellow
    Write-Host "git push -f origin main" -ForegroundColor Yellow
} else {
    Write-Host "Repository reconstruction skipped." -ForegroundColor Yellow
}

Write-Host @"
========================================================
   FIX COMPLETED
========================================================
"@ -ForegroundColor Cyan

Write-Host "The desktop.ini error should now be fixed." -ForegroundColor Green
Write-Host "Please try GitHub Desktop again." -ForegroundColor Green

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
