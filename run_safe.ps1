# PowerShell script to safely run Python scripts in GENESIS with proper timeouts
# GENESIS AI TRADING BOT SYSTEM - Safe Script Runner
# ARCHITECT MODE: v2.7

param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ScriptPath,
    
    [Parameter(Mandatory=$false)]
    [int]$TimeoutSeconds = 60,
    
    [Parameter(Mandatory=$false, ValueFromRemainingArguments=$true)]
    [string[]]$ScriptArgs
)

# Current timestamp
function Get-Timestamp {
    return Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

# Log message with timestamp
function Write-LogMessage {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Timestamp
    Write-Host "[$timestamp] [$Level] $Message"
}

# Main execution function
function Start-PythonScriptSafely {
    param(
        [string]$ScriptPath,
        [int]$TimeoutSeconds,
        [string[]]$ScriptArgs
    )
    
    # Validate script exists
    if (-Not (Test-Path $ScriptPath)) {
        Write-LogMessage "Script not found: $ScriptPath" "ERROR"
        return 1
    }
    
    # Build the command
    $pythonPath = "python"  # Use system default Python
    $commandArgs = @($ScriptPath) + $ScriptArgs
    
    Write-LogMessage "Starting script: $pythonPath $($commandArgs -join ' ')"
    Write-LogMessage "Timeout set to $TimeoutSeconds seconds"
    
    try {
        # Start the process
        $process = Start-Process -FilePath $pythonPath -ArgumentList $commandArgs -PassThru -NoNewWindow
        
        # Wait for the process to complete or timeout
        $completed = $process.WaitForExit($TimeoutSeconds * 1000)  # Convert to milliseconds
        
        if (-Not $completed) {
            Write-LogMessage "Timeout ($TimeoutSeconds seconds) exceeded, terminating process" "WARNING"
            
            # Try graceful termination first
            $process.CloseMainWindow() | Out-Null
            
            # Give it a moment to terminate gracefully
            if (-Not $process.WaitForExit(5000)) {
                Write-LogMessage "Process did not terminate gracefully, force killing" "WARNING"
                $process.Kill()
            }
            
            return -1
        }
        
        $exitCode = $process.ExitCode
        
        if ($exitCode -eq 0) {
            Write-LogMessage "Script completed successfully with exit code $exitCode"
        }
        else {
            Write-LogMessage "Script exited with code $exitCode" "WARNING"
        }
        
        return $exitCode
    }
    catch {
        Write-LogMessage "Error running script: $_" "ERROR"
        return -3
    }
}

# Main script execution
Write-LogMessage "GENESIS Safe Script Runner - Running $ScriptPath"
$result = Start-PythonScriptSafely -ScriptPath $ScriptPath -TimeoutSeconds $TimeoutSeconds -ScriptArgs $ScriptArgs
exit $result
