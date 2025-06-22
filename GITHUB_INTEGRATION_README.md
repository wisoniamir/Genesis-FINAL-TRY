# GENESIS GitHub Integration & Continuous Integration Module

This module establishes robust continuous integration workflow between the local Genesis folder and a GitHub repository, enabling real-time monitoring, auditing, patching, and building directly based on the GitHub commit history and latest repository state.

## Features

- **Real-time GitHub Synchronization**: Automatically sync your local Genesis system with a GitHub repository.
- **Continuous Integration**: Run automated tests, architecture validation, and compliance checks with each commit.
- **EventBus Integration**: Fully integrated with the GENESIS EventBus for event-driven workflows.
- **Telemetry**: Complete monitoring of all GitHub operations with detailed logging.
- **Compliance Enforcement**: Ensures all architect mode rules are enforced during synchronization.
- **Automated Auditing**: Run comprehensive system audits after each pull from GitHub.
- **Patch Generation**: Automatically generate patches to fix issues detected during audits.

## Setup

1. Run the setup script to configure your GitHub repository:
   ```
   setup_github_integration.bat
   ```
   
2. Follow the on-screen prompts to connect your GitHub repository.

3. The system will automatically initialize and configure the integration.

## Configuration Files

- `github_integration_config.json`: Main configuration for GitHub synchronization
- `github_ci_cd_config.json`: Configuration for CI/CD workflows
- `.github/workflows/genesis_ci.yml`: GitHub Actions workflow template

## Usage

### Command Line Interface

Use the CLI tool for manual operations:

```
# Show current status
python github_sync_cli.py status

# Synchronize with remote repository
python github_sync_cli.py sync

# Initialize GitHub integration
python github_sync_cli.py init --url https://github.com/your/repo.git --branch main

# Run a system audit
python github_sync_cli.py audit

# Commit local changes
python github_sync_cli.py commit --message "Update files"

# Push committed changes to remote
python github_sync_cli.py push
```

### CI/CD Workflow Management

Manage the CI/CD workflows:

```
# Start CI/CD workflow monitoring
python github_ci_cd_workflow.py start

# Check workflow status
python github_ci_cd_workflow.py status

# Run a workflow manually
python github_ci_cd_workflow.py run --trigger manual

# Stop workflow monitoring
python github_ci_cd_workflow.py stop
```

### PowerShell Script

Use the PowerShell script for more detailed operations:

```powershell
# Show current status
.\github_integration_setup.ps1 -Action status

# Set up GitHub repository
.\github_integration_setup.ps1 -Action setup -RepoUrl "https://github.com/your/repo.git" -Branch main

# Sync with remote repository
.\github_integration_setup.ps1 -Action sync

# Commit changes
.\github_integration_setup.ps1 -Action commit -CommitMessage "Update files"

# Push changes to remote
.\github_integration_setup.ps1 -Action push
```

## Architect Mode Compliance

This module strictly enforces all architect mode compliance rules:

- **No Simplified Data**: All operations use real, valid data.
- **No Isolated Functions**: All functions emit/consume via EventBus.
- **No Orphans**: Every component is connected to the system tree.
- **No Duplicates**: Detects and prevents duplicate code or modules.
- **No Gaps**: Identifies and reports gaps in logic or connections.
- **No Mock Data**: All tests use real data from the system.
- **No Bypassing**: Respects the build architecture strictness.

## Continuous Integration Flow

The CI/CD workflow automatically:

1. Detects new commits on GitHub
2. Pulls the latest changes
3. Runs architecture validation
4. Checks compliance with architect mode rules
5. Validates all module connections
6. Runs tests with real data
7. Generates reports and notifications
8. Creates patches for any issues

## Logs and Reports

- All operations are logged to `github_integration_sync.log`
- CI/CD workflows are logged to `github_ci_cd_workflow.log`
- Workflow results are stored in the `workflow_results/` directory
- Audit reports are generated after each synchronization

## Required Environment

- Python 3.8 or higher
- Git installed and available in PATH
- PowerShell 5.0 or higher (for Windows)
- MetaTrader5 Python package (for trading data validation)
