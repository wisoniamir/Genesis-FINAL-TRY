# Fix GitHub desktop.ini error and set up repository
# This script fixes the "bad ref refs/tags/desktop.ini" error and sets up GitHub repository properly

# Set error action preference
$ErrorActionPreference = "SilentlyContinue"

# Function to display formatted messages
function Write-FormattedMessage {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$false)]
        [System.ConsoleColor]$ForegroundColor = [System.ConsoleColor]::White
    )
    
    Write-Host $Message -ForegroundColor $ForegroundColor
}

# Function to display a header
function Write-Header {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Title
    )
    
    $border = "=" * ($Title.Length + 10)
    Write-FormattedMessage "`n$border" -ForegroundColor Cyan
    Write-FormattedMessage "    $Title    " -ForegroundColor Cyan
    Write-FormattedMessage "$border`n" -ForegroundColor Cyan
}

Write-Header "GENESIS GitHub Repository Fix"

Write-FormattedMessage "This script will fix the 'bad ref refs/tags/desktop.ini' error and set up your repository." -ForegroundColor Yellow

# Step 1: Update .gitignore
Write-FormattedMessage "`nStep 1: Updating .gitignore to exclude desktop.ini files..." -ForegroundColor Green
$gitignorePath = ".gitignore"
$gitignoreContent = Get-Content $gitignorePath -ErrorAction SilentlyContinue
$desktopIniLine = "desktop.ini"

if ($gitignoreContent -notcontains $desktopIniLine) {
    Add-Content -Path $gitignorePath -Value "`n# Windows system files`n$desktopIniLine"
    Write-FormattedMessage "Added desktop.ini to .gitignore" -ForegroundColor Green
} else {
    Write-FormattedMessage "desktop.ini is already in .gitignore" -ForegroundColor Green
}

# Step 2: Remove desktop.ini files from git index
Write-FormattedMessage "`nStep 2: Removing desktop.ini files from Git tracking..." -ForegroundColor Green
git rm --cached -r --ignore-unmatch **/desktop.ini 2>$null
git rm --cached -r --ignore-unmatch desktop.ini 2>$null
Write-FormattedMessage "Removed desktop.ini files from Git tracking" -ForegroundColor Green

# Step 3: Clean the git repository
Write-FormattedMessage "`nStep 3: Cleaning Git repository..." -ForegroundColor Green
git gc --prune=now
Write-FormattedMessage "Git repository cleaned" -ForegroundColor Green

# Step 4: Reset bad references
Write-FormattedMessage "`nStep 4: Resetting bad references..." -ForegroundColor Green
$refs = git for-each-ref --format="%(refname)" refs/tags
foreach ($ref in $refs) {
    if ($ref -like "*desktop.ini*") {
        git update-ref -d $ref
        Write-FormattedMessage "Deleted bad reference: $ref" -ForegroundColor Yellow
    }
}
Write-FormattedMessage "Bad references have been reset" -ForegroundColor Green

# Step 5: Commit changes to .gitignore
Write-FormattedMessage "`nStep 5: Committing changes to .gitignore..." -ForegroundColor Green
git add .gitignore
git commit -m "Update .gitignore to exclude desktop.ini files" | Out-Null
Write-FormattedMessage "Committed changes to .gitignore" -ForegroundColor Green

# Step 6: Ask if user wants to push to GitHub
Write-FormattedMessage "`nStep 6: Setting up GitHub repository..." -ForegroundColor Green
$repoExists = git remote -v | Where-Object { $_ -like "*origin*" }

if (-not $repoExists) {
    $repoUrl = Read-Host "Enter your GitHub repository URL (or leave empty to skip)"
    
    if (-not [string]::IsNullOrWhiteSpace($repoUrl)) {
        git remote add origin $repoUrl
        Write-FormattedMessage "Added remote repository: $repoUrl" -ForegroundColor Green
        
        $pushNow = Read-Host "Do you want to push to GitHub now? (Y/N)"
        if ($pushNow -eq "Y" -or $pushNow -eq "y") {
            Write-FormattedMessage "Pushing to GitHub..." -ForegroundColor Yellow
            git push -u origin main
            
            if ($LASTEXITCODE -eq 0) {
                Write-FormattedMessage "Successfully pushed to GitHub!" -ForegroundColor Green
            } else {
                Write-FormattedMessage "Failed to push to GitHub. You may need to:" -ForegroundColor Red
                Write-FormattedMessage "1. Create the repository on GitHub first" -ForegroundColor Yellow
                Write-FormattedMessage "2. Ensure you have proper access rights" -ForegroundColor Yellow
                Write-FormattedMessage "3. Try pushing manually: git push -u origin main" -ForegroundColor Yellow
            }
        }
    } else {
        Write-FormattedMessage "Skipping GitHub setup" -ForegroundColor Yellow
    }
} else {
    Write-FormattedMessage "Remote repository already exists" -ForegroundColor Green
    $remoteUrl = git remote get-url origin
    Write-FormattedMessage "Current remote URL: $remoteUrl" -ForegroundColor Yellow
    
    $updateRemote = Read-Host "Do you want to update the remote URL? (Y/N)"
    if ($updateRemote -eq "Y" -or $updateRemote -eq "y") {
        $newUrl = Read-Host "Enter new GitHub repository URL"
        git remote set-url origin $newUrl
        Write-FormattedMessage "Updated remote URL to: $newUrl" -ForegroundColor Green
    }
    
    $pushNow = Read-Host "Do you want to push to GitHub now? (Y/N)"
    if ($pushNow -eq "Y" -or $pushNow -eq "y") {
        Write-FormattedMessage "Pushing to GitHub..." -ForegroundColor Yellow
        git push -u origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-FormattedMessage "Successfully pushed to GitHub!" -ForegroundColor Green
        } else {
            Write-FormattedMessage "Failed to push to GitHub. You may need to:" -ForegroundColor Red
            Write-FormattedMessage "1. Create the repository on GitHub first" -ForegroundColor Yellow
            Write-FormattedMessage "2. Ensure you have proper access rights" -ForegroundColor Yellow
            Write-FormattedMessage "3. Try force push if needed: git push -u origin main --force" -ForegroundColor Yellow
            
            $forcePush = Read-Host "Do you want to try force pushing? This will override remote changes (Y/N)"
            if ($forcePush -eq "Y" -or $forcePush -eq "y") {
                Write-FormattedMessage "Force pushing to GitHub..." -ForegroundColor Yellow
                git push -u origin main --force
                
                if ($LASTEXITCODE -eq 0) {
                    Write-FormattedMessage "Force push successful!" -ForegroundColor Green
                } else {
                    Write-FormattedMessage "Force push failed. Please check your repository settings and credentials." -ForegroundColor Red
                }
            }
        }
    }
}

Write-FormattedMessage "`nGitHub repository setup complete!" -ForegroundColor Green
Write-FormattedMessage "You can now use GitHub Desktop or the 'Push GENESIS to GitHub' task." -ForegroundColor Green
Write-FormattedMessage "The desktop.ini error should be fixed."  -ForegroundColor Green

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
