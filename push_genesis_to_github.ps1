# Push GENESIS Project to GitHub
# This PowerShell script commits and pushes the entire Genesis project to GitHub
# It creates a new repository if needed and pushes ALL files from C:\Users\patra\Genesis FINAL TRY

# Set error action preference
$ErrorActionPreference = "Stop"

# Define colors for output
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Red = [System.ConsoleColor]::Red
$Cyan = [System.ConsoleColor]::Cyan

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
    
    $Border = "═" * ($Title.Length + 10)
    Write-FormattedMessage "╔$Border╗" -ForegroundColor $Cyan
    Write-FormattedMessage "║     $Title     ║" -ForegroundColor $Cyan
    Write-FormattedMessage "╚$Border╝" -ForegroundColor $Cyan
    Write-Host
}

# Function to check if Git is installed
function Test-GitInstalled {    try {
        git --version | Out-Null
        return $true
    }
    catch {
        Write-FormattedMessage "Git is not installed or not in PATH. Please install Git and try again." -ForegroundColor $Red
        return $false
    }
}

# Function to load configuration
function Get-GitHubConfig {
    $ConfigPath = "github_integration_config.json"
    if (Test-Path $ConfigPath) {
        try {
            $config = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
            return $config
        }
        catch {
            Write-FormattedMessage "Failed to load configuration from $ConfigPath. Error: $_" -ForegroundColor $Red
            return $null
        }
    }
    else {
        Write-FormattedMessage "Configuration file not found at $ConfigPath" -ForegroundColor $Yellow
        return $null
    }
}

# Function to save configuration
function Save-GitHubConfig {
    param(
        [Parameter(Mandatory=$true)]
        [PSCustomObject]$Config
    )
    
    $ConfigPath = "github_integration_config.json"
    try {
        $Config | ConvertTo-Json -Depth 10 | Out-File -FilePath $ConfigPath -Encoding utf8
        Write-FormattedMessage "Configuration saved to $ConfigPath" -ForegroundColor $Green
    }
    catch {
        Write-FormattedMessage "Failed to save configuration to $ConfigPath. Error: $_" -ForegroundColor $Red
    }
}

# Function to create .gitignore file with common Python and development exclusions
function New-GitIgnoreFile {
    Write-FormattedMessage "Creating .gitignore file..." -ForegroundColor $Yellow
    
    $gitIgnoreContent = @"
# GENESIS Project .gitignore

# Python cache files
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
ENV/
env/

# IDE files
.idea/
.vscode/*
!.vscode/tasks.json
!.vscode/launch.json
.vs/
*.swp
*.swo

# Logs and databases
*.log
*.sqlite3
*.db

# Local configurations that should not be committed
local_settings.py
config.local.json

# OS specific files
.DS_Store
Thumbs.db
desktop.ini

# Backup files
*.bak
*~
*.backup
*.old

# Jupyter Notebook
.ipynb_checkpoints

# MetaTrader5 specific
*.ex4
*.ex5
*.mqproj
*.mq4.swb
*.mq5.swb
*.mqh.swb

# Trading sensitive data that should not be committed
# Uncomment if needed
# credentials.json
# api_keys.json
# secrets.json
"@
    
    try {
        $gitIgnoreContent | Out-File -FilePath ".gitignore" -Encoding utf8
        Write-FormattedMessage ".gitignore file created successfully" -ForegroundColor $Green
        return $true
    }
    catch {
        Write-FormattedMessage "Failed to create .gitignore file: $_" -ForegroundColor $Red
        return $false
    }
}

# Function to create GitHub repository and push all files
function Push-GenesisFolderToGitHub {
    param(
        [string]$RepoUrl,
        [string]$Branch = "main"
    )
    
    Write-Header "PUSHING GENESIS PROJECT TO GITHUB"
    
    # Check if Git is installed
    if (-not (Test-GitInstalled)) {
        exit 1
    }
    
    # Get current path
    $currentPath = Get-Location
    Write-FormattedMessage "Current directory: $currentPath" -ForegroundColor $Yellow
    
    # Ask for repository URL if not provided
    if ([string]::IsNullOrWhiteSpace($RepoUrl)) {
        $RepoUrl = Read-Host "Enter GitHub repository URL (leave empty to create only local repository)"
    }
    
    # Initialize Git repository if it doesn't exist
    if (-not (Test-Path ".git")) {
        Write-FormattedMessage "Initializing new Git repository..." -ForegroundColor $Yellow
        git init
        if ($LASTEXITCODE -ne 0) {
            Write-FormattedMessage "Failed to initialize Git repository" -ForegroundColor $Red
            exit 1
        }
    } else {
        Write-FormattedMessage "Git repository already initialized" -ForegroundColor $Green
    }
      # Create .gitignore file
    New-GitIgnoreFile
    
    # Check status to see what will be committed
    Write-FormattedMessage "`nChecking Git status..." -ForegroundColor $Yellow
    $status = git status --porcelain
    
    if (-not $status) {
        Write-FormattedMessage "No changes to commit" -ForegroundColor $Yellow
        
        # Check if we already have commits
        $commits = git log -1 --oneline 2>$null
        if (-not $commits) {
            Write-FormattedMessage "No commits yet. You need to add files before pushing to GitHub." -ForegroundColor $Red
            exit 1
        }
    } else {
        # Show files that will be committed
        Write-FormattedMessage "`nThe following files will be committed:" -ForegroundColor $Cyan
        $totalFiles = 0
        foreach ($line in $status) {
            $statusCode = $line.Substring(0, 2)
            $file = $line.Substring(3)
            
            $statusText = switch ($statusCode.Trim()) {
                "M" { "Modified:"; break }
                "A" { "Added:   "; break }
                "D" { "Deleted: "; break }
                "R" { "Renamed: "; break }
                "C" { "Copied:  "; break }
                "U" { "Updated: "; break }
                "??" { "Untracked:"; break }
                default { "Changed: "; break }
            }
            
            if ($totalFiles -lt 10) {
                Write-FormattedMessage "  $statusText $file" -ForegroundColor $Yellow
                $totalFiles++
            } elseif ($totalFiles -eq 10) {
                Write-FormattedMessage "  ... and more files" -ForegroundColor $Yellow
                $totalFiles++
            }
        }
        
        # Prompt for confirmation
        $confirmation = Read-Host "`nDo you want to commit ALL files in the Genesis project? (Y/N)"
        if ($confirmation -ne "Y" -and $confirmation -ne "y") {
            Write-FormattedMessage "Operation cancelled by user" -ForegroundColor $Yellow
            exit 0
        }
        
        # Add all files
        Write-FormattedMessage "Adding all files to Git..." -ForegroundColor $Yellow
        git add -A
        if ($LASTEXITCODE -ne 0) {
            Write-FormattedMessage "Failed to add files to Git" -ForegroundColor $Red
            exit 1
        }
        
        # Commit changes
        $commitMessage = "GENESIS Project Initial Commit - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-FormattedMessage "Committing files with message: $commitMessage" -ForegroundColor $Yellow
        git commit -m $commitMessage
        if ($LASTEXITCODE -ne 0) {
            Write-FormattedMessage "Failed to commit files" -ForegroundColor $Red
            exit 1
        }
        
        Write-FormattedMessage "Files committed successfully" -ForegroundColor $Green
    }
    
    # Push to GitHub if URL provided
    if ($RepoUrl) {
        # Set remote if not set
        $remoteExists = git remote -v | Where-Object { $_ -like "*origin*" }
        if (-not $remoteExists) {
            Write-FormattedMessage "Setting remote URL to $RepoUrl..." -ForegroundColor $Yellow
            git remote add origin $RepoUrl
            if ($LASTEXITCODE -ne 0) {
                Write-FormattedMessage "Failed to add remote" -ForegroundColor $Red
                exit 1
            }
        } else {
            # Check if the remote URL matches
            $currentUrl = git remote get-url origin
            if ($currentUrl -ne $RepoUrl) {
                Write-FormattedMessage "Updating remote URL from $currentUrl to $RepoUrl..." -ForegroundColor $Yellow
                git remote set-url origin $RepoUrl
                if ($LASTEXITCODE -ne 0) {
                    Write-FormattedMessage "Failed to update remote URL" -ForegroundColor $Red
                    exit 1
                }
            }
        }
        
        # Push to GitHub
        Write-FormattedMessage "Pushing to GitHub..." -ForegroundColor $Yellow
        git push -u origin $Branch
        
        if ($LASTEXITCODE -ne 0) {
            Write-FormattedMessage "Failed to push to GitHub. You might need to:" -ForegroundColor $Red
            Write-FormattedMessage "1. Check if the repository exists on GitHub" -ForegroundColor $Yellow
            Write-FormattedMessage "2. Ensure you have proper access rights" -ForegroundColor $Yellow
            Write-FormattedMessage "3. Try creating the repository on GitHub first" -ForegroundColor $Yellow
            
            # Ask if user wants to try force push
            $forcePush = Read-Host "`nDo you want to try force pushing? This will override remote changes (Y/N)"
            if ($forcePush -eq "Y" -or $forcePush -eq "y") {
                Write-FormattedMessage "Force pushing to GitHub..." -ForegroundColor $Yellow
                git push -u origin $Branch --force
                
                if ($LASTEXITCODE -ne 0) {
                    Write-FormattedMessage "Force push failed. Please check your repository settings and credentials." -ForegroundColor $Red
                    exit 1
                } else {
                    Write-FormattedMessage "Force push successful!" -ForegroundColor $Green
                }
            } else {
                exit 1
            }
        } else {
            Write-FormattedMessage "Successfully pushed to GitHub!" -ForegroundColor $Green
        }
        
        # Update config
        $config = Get-GitHubConfig
        if ($null -eq $config) {
            $config = [PSCustomObject]@{
                repository_url = $RepoUrl
                main_branch = $Branch
                local_path = (Get-Location).Path
                polling_interval_minutes = 5
                enable_webhooks = $false
                webhook_port = 9000
                webhook_secret = ""
                auto_pull = $true
                auto_push = $false
                push_requires_approval = $true
                auto_audit = $true
                auto_patch = $false
                notify_on_changes = $true
                protected_files = @(
                    "build_status.json",
                    "system_tree.json",
                    "module_registry.json",
                    "event_bus.json",
                    "signal_manager.json",
                    "compliance.json"
                )
                github_api_token = ""
            }
        } else {
            $config.repository_url = $RepoUrl
            $config.main_branch = $Branch
        }
        
        Save-GitHubConfig -Config $config
    } else {
        Write-FormattedMessage "No GitHub URL provided, skipping push." -ForegroundColor $Yellow
        Write-FormattedMessage "Repository is initialized locally only." -ForegroundColor $Yellow
    }
    
    Write-FormattedMessage "`nGENESIS project is now under version control!" -ForegroundColor $Green
    if ($RepoUrl) {
        Write-FormattedMessage "GitHub Repository: $RepoUrl" -ForegroundColor $Green
    }
    Write-FormattedMessage "Local Repository: $(Get-Location)" -ForegroundColor $Green
}

# Main script execution
try {
    $config = Get-GitHubConfig
    $repoUrl = ""
    $branch = "main"
    
    if ($config -and $config.repository_url) {
        $repoUrl = $config.repository_url
        $branch = $config.main_branch
        
        Write-FormattedMessage "Found existing GitHub configuration:" -ForegroundColor $Yellow
        Write-FormattedMessage "Repository URL: $repoUrl" -ForegroundColor $Yellow
        Write-FormattedMessage "Branch: $branch" -ForegroundColor $Yellow
        
        $useExisting = Read-Host "Do you want to use this configuration? (Y/N)"
        if ($useExisting -ne "Y" -and $useExisting -ne "y") {
            $repoUrl = Read-Host "Enter GitHub repository URL"
            $customBranch = Read-Host "Enter branch name (default: main)"
            if (-not [string]::IsNullOrWhiteSpace($customBranch)) {
                $branch = $customBranch
            }
        }
    } else {
        Write-FormattedMessage "No existing GitHub configuration found." -ForegroundColor $Yellow
        $repoUrl = Read-Host "Enter GitHub repository URL (leave empty to initialize local repository only)"
        $customBranch = Read-Host "Enter branch name (default: main)"
        if (-not [string]::IsNullOrWhiteSpace($customBranch)) {
            $branch = $customBranch
        }
    }
    
    Push-GenesisFolderToGitHub -RepoUrl $repoUrl -Branch $branch
}
catch {
    Write-FormattedMessage "An error occurred: $_" -ForegroundColor $Red
    exit 1
}

# Pause before exit
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
