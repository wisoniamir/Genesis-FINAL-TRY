# GitHub Integration Setup and Management Script for GENESIS
# This PowerShell script helps set up and manage GitHub integration for the GENESIS trading bot
# It ensures architect mode compliance while providing useful utility functions for Git operations

# Parameters
param(
    [Parameter(HelpMessage="Action to perform: setup, status, sync, pull, push, commit")]
    [string]$Action = "status",
    
    [Parameter(HelpMessage="GitHub repository URL")]
    [string]$RepoUrl,
    
    [Parameter(HelpMessage="GitHub branch to use")]
    [string]$Branch = "main",
    
    [Parameter(HelpMessage="Commit message for commit action")]
    [string]$CommitMessage = "Update from GENESIS system",
    
    [Parameter(HelpMessage="Path to GitHub configuration file")]
    [string]$ConfigPath = "github_integration_config.json",
    
    [Parameter(HelpMessage="GitHub personal access token")]
    [string]$Token
)

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
function Test-GitInstalled {
    try {
        $gitVersion = git --version
        return $true
    }
    catch {
        Write-FormattedMessage "Git is not installed or not in PATH. Please install Git and try again." -ForegroundColor $Red
        return $false
    }
}

# Function to load configuration
function Get-GitHubConfig {
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
    
    try {
        $Config | ConvertTo-Json -Depth 10 | Out-File -FilePath $ConfigPath -Encoding utf8
        Write-FormattedMessage "Configuration saved to $ConfigPath" -ForegroundColor $Green
    }
    catch {
        Write-FormattedMessage "Failed to save configuration to $ConfigPath. Error: $_" -ForegroundColor $Red
    }
}

# Function to set up GitHub repository
function Initialize-GitHubRepo {
    param(
        [Parameter(Mandatory=$true)]
        [string]$RepoUrl,
        
        [Parameter(Mandatory=$true)]
        [string]$Branch
    )
    
    Write-Header "SETTING UP GITHUB REPOSITORY"
    
    # Check if .git directory exists
    if (Test-Path ".git") {
        Write-FormattedMessage "Git repository already initialized" -ForegroundColor $Green
        
        # Check if remote URL matches configuration
        $currentUrl = git config --get remote.origin.url
        if ($currentUrl -ne $RepoUrl) {
            Write-FormattedMessage "Updating remote URL from $currentUrl to $RepoUrl" -ForegroundColor $Yellow
            git remote set-url origin $RepoUrl
        }
    }
    else {
        # Initialize new repository
        if ([string]::IsNullOrWhiteSpace($RepoUrl)) {
            Write-FormattedMessage "Initializing new local Git repository" -ForegroundColor $Green
            git init
        }
        else {
            Write-FormattedMessage "Cloning repository from $RepoUrl" -ForegroundColor $Green
            git clone $RepoUrl .
        }
    }
    
    # Update configuration
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
            github_api_token = $Token
        }
    }
    else {
        $config.repository_url = $RepoUrl
        $config.main_branch = $Branch
        $config.github_api_token = $Token
    }
    
    Save-GitHubConfig -Config $config
    
    Write-FormattedMessage "GitHub repository setup complete!" -ForegroundColor $Green
}

# Function to get repository status
function Get-GitHubStatus {
    Write-Header "GITHUB REPOSITORY STATUS"
    
    if (-not (Test-Path ".git")) {
        Write-FormattedMessage "No Git repository initialized in current directory" -ForegroundColor $Red
        return
    }
    
    try {
        # Get current branch
        $currentBranch = git rev-parse --abbrev-ref HEAD
        Write-FormattedMessage "Current branch: $currentBranch" -ForegroundColor $Green
        
        # Get current commit
        $currentCommit = git rev-parse HEAD
        $commitDate = git log -1 --format=%cd
        Write-FormattedMessage "Current commit: $currentCommit" -ForegroundColor $Green
        Write-FormattedMessage "Commit date: $commitDate" -ForegroundColor $Green
        
        # Get commit count
        $commitCount = git rev-list --count HEAD
        Write-FormattedMessage "Total commits: $commitCount" -ForegroundColor $Green
        
        # Get status
        Write-FormattedMessage "`nLocal changes:" -ForegroundColor $Cyan
        $status = git status --porcelain
        if ($status) {
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
                
                Write-FormattedMessage "  $statusText $file" -ForegroundColor $Yellow
            }
        }
        else {
            Write-FormattedMessage "  No local changes" -ForegroundColor $Green
        }
        
        # Check if remote exists and if we're behind/ahead
        $remoteExists = git remote -v
        if ($remoteExists) {
            $remoteBranch = git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>$null
            if ($remoteBranch) {
                git fetch origin --quiet
                
                $behind = git rev-list --count HEAD..$remoteBranch
                $ahead = git rev-list --count $remoteBranch..HEAD
                
                Write-FormattedMessage "`nRemote status:" -ForegroundColor $Cyan
                if ($behind -gt 0) {
                    Write-FormattedMessage "  Behind by $behind commit(s)" -ForegroundColor $Yellow
                }
                else {
                    Write-FormattedMessage "  Up to date with remote" -ForegroundColor $Green
                }
                
                if ($ahead -gt 0) {
                    Write-FormattedMessage "  Ahead by $ahead commit(s)" -ForegroundColor $Yellow
                }
            }
            else {
                Write-FormattedMessage "`nRemote status: No tracking branch set" -ForegroundColor $Yellow
            }
        }
        else {
            Write-FormattedMessage "`nRemote status: No remote configured" -ForegroundColor $Yellow
        }
    }
    catch {
        Write-FormattedMessage "Failed to get repository status: $_" -ForegroundColor $Red
    }
}

# Function to synchronize with remote repository
function Sync-GitHubRepo {
    Write-Header "SYNCHRONIZING WITH GITHUB"
    
    if (-not (Test-Path ".git")) {
        Write-FormattedMessage "No Git repository initialized in current directory" -ForegroundColor $Red
        return
    }
    
    try {
        # Get configuration
        $config = Get-GitHubConfig
        if ($null -eq $config -or [string]::IsNullOrWhiteSpace($config.repository_url)) {
            Write-FormattedMessage "No repository URL configured" -ForegroundColor $Red
            return
        }
        
        $branch = $config.main_branch
        
        # Fetch updates
        Write-FormattedMessage "Fetching updates from remote..." -ForegroundColor $Cyan
        git fetch origin
        
        # Compare local and remote
        $localCommit = git rev-parse HEAD
        $remoteCommit = git rev-parse origin/$branch
        
        if ($localCommit -eq $remoteCommit) {
            Write-FormattedMessage "Already up to date with remote" -ForegroundColor $Green
            return
        }
        
        Write-FormattedMessage "Updates found! Local: $localCommit, Remote: $remoteCommit" -ForegroundColor $Yellow
        
        # Pull changes
        Write-FormattedMessage "Pulling changes from remote..." -ForegroundColor $Cyan
        git pull origin $branch
        
        # Get commit details
        $commitDetails = git log -1 --pretty=format:"%an|%ae|%s|%b"
        $parts = $commitDetails -split '\|', 4
        $author = $parts[0]
        $email = $parts[1]
        $subject = $parts[2]
        $body = $parts[3]
        
        Write-FormattedMessage "Successfully pulled changes from remote" -ForegroundColor $Green
        Write-FormattedMessage "Latest commit: $subject by $author" -ForegroundColor $Green
        
        # Run Python script to initiate audit if configured
        if ($config.auto_audit) {
            Write-FormattedMessage "`nInitiating system audit..." -ForegroundColor $Cyan
            $pythonCmd = "python github_integration_sync.py --audit"
            
            try {
                Invoke-Expression $pythonCmd
                Write-FormattedMessage "Audit initiated successfully" -ForegroundColor $Green
            }
            catch {
                Write-FormattedMessage "Failed to initiate audit: $_" -ForegroundColor $Red
            }
        }
    }
    catch {
        Write-FormattedMessage "Failed to synchronize with remote: $_" -ForegroundColor $Red
    }
}

# Function to commit changes
function Commit-Changes {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message
    )
    
    Write-Header "COMMITTING CHANGES"
    
    if (-not (Test-Path ".git")) {
        Write-FormattedMessage "No Git repository initialized in current directory" -ForegroundColor $Red
        return
    }
    
    try {
        # Check if there are changes to commit
        $status = git status --porcelain
        if (-not $status) {
            Write-FormattedMessage "No changes to commit" -ForegroundColor $Yellow
            return
        }
        
        # Show changes that will be committed
        Write-FormattedMessage "The following changes will be committed:" -ForegroundColor $Cyan
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
            
            Write-FormattedMessage "  $statusText $file" -ForegroundColor $Yellow
        }
        
        # Prompt for confirmation
        $confirmation = Read-Host "Do you want to commit these changes? (Y/N)"
        if ($confirmation -ne "Y" -and $confirmation -ne "y") {
            Write-FormattedMessage "Commit cancelled" -ForegroundColor $Yellow
            return
        }
        
        # Add all changes
        Write-FormattedMessage "Adding all changes..." -ForegroundColor $Cyan
        git add -A
        
        # Commit
        Write-FormattedMessage "Committing with message: $Message" -ForegroundColor $Cyan
        git commit -m $Message
        
        # Get commit hash
        $commitHash = git rev-parse HEAD
        Write-FormattedMessage "Changes committed successfully. Commit hash: $commitHash" -ForegroundColor $Green
        
        # Check if auto-push is enabled
        $config = Get-GitHubConfig
        if ($null -ne $config -and $config.auto_push) {
            Push-Changes
        }
    }
    catch {
        Write-FormattedMessage "Failed to commit changes: $_" -ForegroundColor $Red
    }
}

# Function to push changes
function Push-Changes {
    Write-Header "PUSHING CHANGES TO GITHUB"
    
    if (-not (Test-Path ".git")) {
        Write-FormattedMessage "No Git repository initialized in current directory" -ForegroundColor $Red
        return
    }
    
    try {
        # Get configuration
        $config = Get-GitHubConfig
        if ($null -eq $config -or [string]::IsNullOrWhiteSpace($config.repository_url)) {
            Write-FormattedMessage "No repository URL configured" -ForegroundColor $Red
            return
        }
        
        $branch = $config.main_branch
        
        # Check if there are commits to push
        $ahead = git rev-list --count @{u}..HEAD 2>$null
        if ($ahead -eq 0) {
            Write-FormattedMessage "No commits to push" -ForegroundColor $Yellow
            return
        }
        
        # Show commits that will be pushed
        Write-FormattedMessage "The following commits will be pushed:" -ForegroundColor $Cyan
        git log --oneline @{u}..HEAD
        
        # Push
        if ($config.push_requires_approval) {
            $confirmation = Read-Host "Do you want to push these commits? (Y/N)"
            if ($confirmation -ne "Y" -and $confirmation -ne "y") {
                Write-FormattedMessage "Push cancelled" -ForegroundColor $Yellow
                return
            }
        }
        
        Write-FormattedMessage "Pushing commits to remote..." -ForegroundColor $Cyan
        git push origin $branch
        
        Write-FormattedMessage "Changes pushed successfully" -ForegroundColor $Green
    }
    catch {
        Write-FormattedMessage "Failed to push changes: $_" -ForegroundColor $Red
    }
}

# Main script logic
if (-not (Test-GitInstalled)) {
    exit 1
}

# Execute action based on parameter
switch ($Action.ToLower()) {
    "setup" {
        Initialize-GitHubRepo -RepoUrl $RepoUrl -Branch $Branch
    }
    "status" {
        Get-GitHubStatus
    }
    "sync" {
        Sync-GitHubRepo
    }
    "pull" {
        Sync-GitHubRepo
    }
    "commit" {
        Commit-Changes -Message $CommitMessage
    }
    "push" {
        Push-Changes
    }
    default {
        Write-FormattedMessage "Unknown action: $Action" -ForegroundColor $Red
        Write-FormattedMessage "Available actions: setup, status, sync, pull, commit, push" -ForegroundColor $Yellow
    }
}
