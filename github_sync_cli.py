#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GENESIS GitHub Integration Command Line Interface
------------------------------------------------
This script provides a command-line interface to the GitHub Integration System
for the GENESIS trading bot, allowing manual control of synchronization operations
while maintaining architect mode compliance rules.

Usage:
    python github_sync_cli.py [command] [options]

Commands:
    status       - Show current sync status
    sync         - Synchronize with remote repository
    init         - Initialize GitHub integration
    audit        - Run system audit
    set-url      - Set repository URL
    set-branch   - Set main branch
    enable-auto  - Enable auto-pull/push
    disable-auto - Disable auto-pull/push
"""

import os
import sys
import json
import argparse
import datetime
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GITHUB_CLI")

# Try to import the GitHubIntegrationSync class
try:
    from github_integration_sync import GitHubIntegrationSync
    has_module = True
    GitHubSyncClass = GitHubIntegrationSync  # Store the class in a global variable
except ImportError:
    logger.warning("Could not import GitHubIntegrationSync. Running in standalone mode.")
    has_module = False
    GitHubSyncClass = None  # Set to None if not available

# Define colors for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    
    @staticmethod
    def colored(text: str, color: str) -> str:
        """Return colored text if terminal supports colors."""
        if sys.platform == "win32":
            # Windows doesn't support ANSI color codes by default
            return text
        return f"{color}{text}{Colors.RESET}"

def load_config() -> Dict[str, Any]:
    """Load GitHub integration configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'github_integration_config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {config_path}")
            return {}
    else:
        logger.warning(f"Config file not found: {config_path}")
        return {}

def save_config(config: Dict[str, Any]) -> bool:
    """Save GitHub integration configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'github_integration_config.json')
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {str(e)}")
        return False

def get_status() -> None:
    """Get and display the current status of GitHub integration."""
    config = load_config()
    
    print(Colors.colored("\n╔═════════════════════════════════════════════════╗", Colors.CYAN))
    print(Colors.colored("║     GENESIS GITHUB INTEGRATION STATUS           ║", Colors.CYAN))
    print(Colors.colored("╚═════════════════════════════════════════════════╝", Colors.CYAN))
    
    print(f"\nRepository URL: {Colors.colored(config.get('repository_url', 'Not configured'), Colors.YELLOW)}")
    print(f"Branch: {Colors.colored(config.get('main_branch', 'main'), Colors.YELLOW)}")
    print(f"Local path: {Colors.colored(config.get('local_path', os.path.dirname(os.path.abspath(__file__))), Colors.YELLOW)}")
    print(f"Auto-pull: {Colors.colored('Enabled' if config.get('auto_pull', False) else 'Disabled', Colors.GREEN if config.get('auto_pull', False) else Colors.RED)}")
    print(f"Auto-push: {Colors.colored('Enabled' if config.get('auto_push', False) else 'Disabled', Colors.GREEN if config.get('auto_push', False) else Colors.RED)}")
    print(f"Auto-audit: {Colors.colored('Enabled' if config.get('auto_audit', False) else 'Disabled', Colors.GREEN if config.get('auto_audit', False) else Colors.RED)}")
    print(f"Auto-patch: {Colors.colored('Enabled' if config.get('auto_patch', False) else 'Disabled', Colors.GREEN if config.get('auto_patch', False) else Colors.RED)}")
    print(f"Push requires approval: {Colors.colored('Yes' if config.get('push_requires_approval', True) else 'No', Colors.GREEN if config.get('push_requires_approval', True) else Colors.YELLOW)}")
    print(f"Webhooks: {Colors.colored('Enabled' if config.get('enable_webhooks', False) else 'Disabled', Colors.GREEN if config.get('enable_webhooks', False) else Colors.YELLOW)}")
      # If the module is available, try to get more detailed status
    if has_module and GitHubSyncClass:
        try:
            github_sync = GitHubSyncClass()
            sync_status = github_sync.get_sync_status()
            
            print(f"\nSync status: {Colors.colored(sync_status['sync_status'], Colors.GREEN)}")
            print(f"Monitoring active: {Colors.colored('Yes' if sync_status['sync_active'] else 'No', Colors.GREEN if sync_status['sync_active'] else Colors.RED)}")
            print(f"Last commit: {Colors.colored(sync_status['last_commit_hash'], Colors.YELLOW)}")
            print(f"Last sync: {Colors.colored(sync_status['last_sync_time'] or 'Never', Colors.YELLOW)}")
        except Exception as e:
            logger.error(f"Failed to get detailed status: {str(e)}")
    
    print("\n")

def init_github(args: argparse.Namespace) -> None:
    """Initialize GitHub integration with provided URL and branch."""
    config = load_config()
    
    if args.url:
        config['repository_url'] = args.url
    
    if args.branch:
        config['main_branch'] = args.branch
    
    if args.token:
        config['github_api_token'] = args.token
        
    # Save the updated config
    if save_config(config):
        print(Colors.colored("Configuration updated successfully.", Colors.GREEN))
      # If the module is available, initialize the repository
    if has_module and GitHubSyncClass:
        try:
            github_sync = GitHubSyncClass()
            if github_sync.initialize_git_repository():
                print(Colors.colored("Git repository initialized successfully.", Colors.GREEN))
                if args.start_monitoring:
                    github_sync.start_monitoring()
                    print(Colors.colored("GitHub monitoring started.", Colors.GREEN))
            else:
                print(Colors.colored("Failed to initialize Git repository.", Colors.RED))
        except Exception as e:
            logger.error(f"Failed to initialize GitHub integration: {str(e)}")
            print(Colors.colored(f"Error: {str(e)}", Colors.RED))
    else:
        # If the module is not available, just print a message
        print(Colors.colored("Configuration saved. Please restart GENESIS to apply changes.", Colors.YELLOW))

def set_config_value(args: argparse.Namespace) -> None:
    """Set a value in the configuration."""
    config = load_config()
    
    if args.key and args.value is not None:
        # Convert string value to appropriate type
        value = args.value
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        
        config[args.key] = value
        
        if save_config(config):
            print(Colors.colored(f"Configuration updated: {args.key} = {value}", Colors.GREEN))
        else:
            print(Colors.colored("Failed to update configuration.", Colors.RED))
    else:
        print(Colors.colored("Error: Both key and value must be provided.", Colors.RED))

def sync_repository(args: argparse.Namespace) -> None:
    """Synchronize with the remote repository."""
    if not has_module or not GitHubSyncClass:
        print(Colors.colored("GitHub integration module not available.", Colors.RED))
        return
    
    try:
        github_sync = GitHubSyncClass()
        
        print(Colors.colored("Checking for updates...", Colors.CYAN))
        updates_found = github_sync.check_for_updates()
        
        if updates_found:
            print(Colors.colored("Updates found! Pulling changes...", Colors.YELLOW))
            if github_sync.pull_changes():
                print(Colors.colored("Changes pulled successfully.", Colors.GREEN))
            else:
                print(Colors.colored("Failed to pull changes.", Colors.RED))
        else:
            print(Colors.colored("Already up to date with remote.", Colors.GREEN))
    except Exception as e:
        logger.error(f"Failed to synchronize: {str(e)}")
        print(Colors.colored(f"Error: {str(e)}", Colors.RED))

def run_audit(args: argparse.Namespace) -> None:
    """Run a system audit."""
    if not has_module or not GitHubSyncClass:
        print(Colors.colored("GitHub integration module not available.", Colors.RED))
        return
    
    try:
        github_sync = GitHubSyncClass()
        
        print(Colors.colored("Running system audit...", Colors.CYAN))
        github_sync.run_system_audit("manual_audit")
        print(Colors.colored("Audit initiated. Check logs for results.", Colors.GREEN))
    except Exception as e:
        logger.error(f"Failed to run audit: {str(e)}")
        print(Colors.colored(f"Error: {str(e)}", Colors.RED))

def commit_changes(args: argparse.Namespace) -> None:
    """Commit local changes."""
    if not has_module or not GitHubSyncClass:
        print(Colors.colored("GitHub integration module not available.", Colors.RED))
        return
    
    if not args.message:
        print(Colors.colored("Error: Commit message is required.", Colors.RED))
        return
    
    try:
        github_sync = GitHubSyncClass()
        
        print(Colors.colored(f"Committing changes with message: {args.message}", Colors.CYAN))
        # Convert file string to list or use empty list instead of None
        files_list = args.files.split(",") if args.files else []
        if github_sync.commit_changes(args.message, files_list):
            print(Colors.colored("Changes committed successfully.", Colors.GREEN))
        else:
            print(Colors.colored("Failed to commit changes.", Colors.RED))
    except Exception as e:
        logger.error(f"Failed to commit changes: {str(e)}")
        print(Colors.colored(f"Error: {str(e)}", Colors.RED))

def push_changes(args: argparse.Namespace) -> None:
    """Push committed changes to remote."""
    if not has_module or not GitHubSyncClass:
        print(Colors.colored("GitHub integration module not available.", Colors.RED))
        return
    
    try:
        github_sync = GitHubSyncClass()
        
        print(Colors.colored("Pushing changes to remote...", Colors.CYAN))
        if github_sync.push_changes():
            print(Colors.colored("Changes pushed successfully.", Colors.GREEN))
        else:
            print(Colors.colored("Failed to push changes.", Colors.RED))
    except Exception as e:
        logger.error(f"Failed to push changes: {str(e)}")
        print(Colors.colored(f"Error: {str(e)}", Colors.RED))

def display_help() -> None:
    """Display help information."""
    print(Colors.colored("\n╔═════════════════════════════════════════════════╗", Colors.CYAN))
    print(Colors.colored("║     GENESIS GITHUB INTEGRATION CLI HELP         ║", Colors.CYAN))
    print(Colors.colored("╚═════════════════════════════════════════════════╝", Colors.CYAN))
    
    print("\nCommands:")
    print(f"  {Colors.colored('status', Colors.YELLOW)}       - Show current sync status")
    print(f"  {Colors.colored('sync', Colors.YELLOW)}         - Synchronize with remote repository")
    print(f"  {Colors.colored('init', Colors.YELLOW)}         - Initialize GitHub integration")
    print(f"  {Colors.colored('audit', Colors.YELLOW)}        - Run system audit")
    print(f"  {Colors.colored('commit', Colors.YELLOW)}       - Commit local changes")
    print(f"  {Colors.colored('push', Colors.YELLOW)}         - Push committed changes to remote")
    print(f"  {Colors.colored('set', Colors.YELLOW)}          - Set a configuration value")
    
    print("\nExamples:")
    print(f"  python github_sync_cli.py status")
    print(f"  python github_sync_cli.py init --url https://github.com/user/repo.git --branch main")
    print(f"  python github_sync_cli.py sync")
    print(f"  python github_sync_cli.py commit --message 'Update files' --files file1.py,file2.py")
    print(f"  python github_sync_cli.py set --key auto_pull --value true")
    
    print("\n")

def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="GENESIS GitHub Integration CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show current sync status")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize GitHub integration")
    init_parser.add_argument("--url", help="GitHub repository URL")
    init_parser.add_argument("--branch", help="Main branch name")
    init_parser.add_argument("--token", help="GitHub API token")
    init_parser.add_argument("--start-monitoring", action="store_true", help="Start monitoring after initialization")
    
    # Set command
    set_parser = subparsers.add_parser("set", help="Set a configuration value")
    set_parser.add_argument("--key", required=True, help="Configuration key")
    set_parser.add_argument("--value", required=True, help="Configuration value")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Synchronize with remote repository")
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Run system audit")
    
    # Commit command
    commit_parser = subparsers.add_parser("commit", help="Commit local changes")
    commit_parser.add_argument("--message", required=True, help="Commit message")
    commit_parser.add_argument("--files", help="Comma-separated list of files to commit (optional)")
    
    # Push command
    push_parser = subparsers.add_parser("push", help="Push committed changes to remote")
    
    # Help command
    help_parser = subparsers.add_parser("help", help="Show help information")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process commands
    if args.command == "status":
        get_status()
    elif args.command == "init":
        init_github(args)
    elif args.command == "set":
        set_config_value(args)
    elif args.command == "sync":
        sync_repository(args)
    elif args.command == "audit":
        run_audit(args)
    elif args.command == "commit":
        commit_changes(args)
    elif args.command == "push":
        push_changes(args)
    elif args.command == "help" or not args.command:
        display_help()
    else:
        print(Colors.colored(f"Unknown command: {args.command}", Colors.RED))
        display_help()

if __name__ == "__main__":
    main()
