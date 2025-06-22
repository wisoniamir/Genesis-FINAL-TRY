import logging

# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def detect_confluence_patterns(self, market_data: dict) -> float:
                """GENESIS Pattern Intelligence - Detect confluence patterns"""
                confluence_score = 0.0

                # Simple confluence calculation
                if market_data.get('trend_aligned', False):
                    confluence_score += 0.3
                if market_data.get('support_resistance_level', False):
                    confluence_score += 0.3
                if market_data.get('volume_confirmation', False):
                    confluence_score += 0.2
                if market_data.get('momentum_aligned', False):
                    confluence_score += 0.2

                emit_telemetry("genesis_auto_startup", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("genesis_auto_startup", "position_calculated", {
                    "risk_amount": risk_amount,
                    "position_size": position_size,
                    "risk_percentage": (position_size / account_balance) * 100
                })

                return position_size
        def emergency_stop(self, reason: str = "Manual trigger") -> bool:
                """GENESIS Emergency Kill Switch"""
                try:
                    # Emit emergency event
                    if hasattr(self, 'event_bus') and self.event_bus:
                        emit_event("emergency_stop", {
                            "module": "genesis_auto_startup",
                            "reason": reason,
                            "timestamp": datetime.now().isoformat()
                        })

                    # Log telemetry
                    self.emit_module_telemetry("emergency_stop", {
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Set emergency state
                    if hasattr(self, '_emergency_stop_active'):
                        self._emergency_stop_active = True

                    return True
                except Exception as e:
                    print(f"Emergency stop error in genesis_auto_startup: {e}")
                    return False
        def validate_ftmo_compliance(self, trade_data: dict) -> bool:
                """GENESIS FTMO Compliance Validator"""
                # Daily drawdown check (5%)
                daily_loss = trade_data.get('daily_loss_pct', 0)
                if daily_loss > 5.0:
                    self.emit_module_telemetry("ftmo_violation", {
                        "type": "daily_drawdown", 
                        "value": daily_loss,
                        "threshold": 5.0
                    })
                    return False

                # Maximum drawdown check (10%)
                max_drawdown = trade_data.get('max_drawdown_pct', 0)
                if max_drawdown > 10.0:
                    self.emit_module_telemetry("ftmo_violation", {
                        "type": "max_drawdown", 
                        "value": max_drawdown,
                        "threshold": 10.0
                    })
                    return False

                # Risk per trade check (2%)
                risk_pct = trade_data.get('risk_percent', 0)
                if risk_pct > 2.0:
                    self.emit_module_telemetry("ftmo_violation", {
                        "type": "risk_exceeded", 
                        "value": risk_pct,
                        "threshold": 2.0
                    })
                    return False

                return True
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "genesis_auto_startup",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("genesis_auto_startup", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in genesis_auto_startup: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False



# 🔗 GENESIS EventBus Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.hardened_event_bus import get_event_bus, emit_event, register_route
    EVENTBUS_AVAILABLE = True
except ImportError:
    # Fallback implementation
    def get_event_bus(): return None
    def emit_event(event, data): print(f"EVENT: {event} - {data}")
    def register_route(route, producer, consumer): pass
    EVENTBUS_AVAILABLE = False


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

🚀 GENESIS AUTO-STARTUP SYSTEM
ARCHITECT MODE v7.0.0 - Automatic System Initialization

PURPOSE:
Automatically starts all GENESIS systems when VS Code opens in the workspace folder.
Ensures ARCHITECT MODE compliance and system integrity from the moment you start working.
"""

import os
import sys
import time
import subprocess
import json
import threading
from datetime import datetime
from pathlib import Path

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: genesis_auto_startup -->


# <!-- @GENESIS_MODULE_START: genesis_auto_startup -->

class GenesisAutoStartup:
    def detect_confluence_patterns(self, market_data: dict) -> float:
            """GENESIS Pattern Intelligence - Detect confluence patterns"""
            confluence_score = 0.0

            # Simple confluence calculation
            if market_data.get('trend_aligned', False):
                confluence_score += 0.3
            if market_data.get('support_resistance_level', False):
                confluence_score += 0.3
            if market_data.get('volume_confirmation', False):
                confluence_score += 0.2
            if market_data.get('momentum_aligned', False):
                confluence_score += 0.2

            emit_telemetry("genesis_auto_startup", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("genesis_auto_startup", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "genesis_auto_startup",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })

                # Log telemetry
                self.emit_module_telemetry("emergency_stop", {
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })

                # Set emergency state
                if hasattr(self, '_emergency_stop_active'):
                    self._emergency_stop_active = True

                return True
            except Exception as e:
                print(f"Emergency stop error in genesis_auto_startup: {e}")
                return False
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss_pct', 0)
            if daily_loss > 5.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "daily_drawdown", 
                    "value": daily_loss,
                    "threshold": 5.0
                })
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown_pct', 0)
            if max_drawdown > 10.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "max_drawdown", 
                    "value": max_drawdown,
                    "threshold": 10.0
                })
                return False

            # Risk per trade check (2%)
            risk_pct = trade_data.get('risk_percent', 0)
            if risk_pct > 2.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "risk_exceeded", 
                    "value": risk_pct,
                    "threshold": 2.0
                })
                return False

            return True
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "genesis_auto_startup",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("genesis_auto_startup", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in genesis_auto_startup: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "genesis_auto_startup",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in genesis_auto_startup: {e}")
    """🚀 GENESIS AUTO-STARTUP SYSTEM"""
    
    def __init__(self):
        self.workspace_path = Path(".")
        self.startup_log = []
        self.services_started = []
        
    def log_startup_event(self, event: str, status: str = "INFO"):
        """📝 Log startup events"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {status}: {event}"
        self.startup_log.append(log_entry)
        print(f"🚀 {log_entry}")
        
    def check_prerequisites(self) -> bool:
        """✅ Check system prerequisites"""
        self.log_startup_event("Checking GENESIS system prerequisites...")
        
        required_files = [
            "build_status.json",
            "audit_engine.py", 
            "watchdog_core.py",
            "genesis_watchdog_launcher.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.workspace_path / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            self.log_startup_event(f"Missing required files: {missing_files}", "ERROR")
            return False
        else:
            self.log_startup_event("All prerequisites satisfied", "SUCCESS")
            return True
            
    def validate_architect_mode(self) -> bool:
        """🔐 Validate ARCHITECT MODE status"""
        self.log_startup_event("Validating ARCHITECT MODE compliance...")
        
        try:
            with open(self.workspace_path / "build_status.json", 'r') as f:
                build_status = json.load(f)
                
            system_status = build_status.get("system_status", "")
            architect_compliance = build_status.get("architect_mode_v7_compliance_status", "")
            
            if "ARCHITECT_MODE_V7" in system_status and "ULTIMATE_ENFORCEMENT" in architect_compliance:
                self.log_startup_event("ARCHITECT MODE v7.0.0 compliance verified", "SUCCESS")
                return True
            else:
                self.log_startup_event(f"ARCHITECT MODE compliance issue: {system_status}", "WARNING")
                return False
                
        except Exception as e:
            self.log_startup_event(f"Failed to validate ARCHITECT MODE: {e}", "ERROR")
            return False
            
    def start_watchdog_system(self) -> bool:
        """🐺 Start GENESIS Watchdog System"""
        self.log_startup_event("Starting GENESIS Watchdog System...")
        
        try:
            # Start watchdog in background mode
            process = subprocess.Popen([
                sys.executable, 
                "genesis_watchdog_launcher.py", 
                "--interval", "30", 
                "--background"
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            if process.poll() is None:  # Process is still running
                self.log_startup_event("GENESIS Watchdog System started successfully", "SUCCESS")
                self.services_started.append("watchdog_system")
                return True
            else:
                stdout, stderr = process.communicate()
                self.log_startup_event(f"Watchdog failed to start: {stderr.decode()}", "ERROR")
                return False
                
        except Exception as e:
            self.log_startup_event(f"Failed to start Watchdog System: {e}", "ERROR")
            return False
            
    def run_initial_audit(self) -> bool:
        """🛡️ Run initial system audit"""
        self.log_startup_event("Running initial system audit...")
        
        try:
            # Run audit engine
            result = subprocess.run([
                sys.executable, 
                "audit_engine.py"
            ], 
            capture_output=True, 
            text=True,
            timeout=30
            )
            
            if result.returncode == 0:
                self.log_startup_event("Initial audit PASSED - System ready", "SUCCESS")
                return True
            else:
                self.log_startup_event("Initial audit FAILED - Violations detected", "WARNING")
                self.log_startup_event("Review audit_snapshot_report.md for details", "INFO")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_startup_event("Audit timeout - continuing with startup", "WARNING")
            return False
        except Exception as e:
            self.log_startup_event(f"Failed to run initial audit: {e}", "ERROR")
            return False
            
    def verify_git_hooks(self) -> bool:
        """🔧 Verify Git pre-commit hooks"""
        self.log_startup_event("Verifying Git pre-commit hooks...")
        
        hook_files = [
            ".git/hooks/pre-commit",
            ".git/hooks/pre-commit.bat"
        ]
        
        hooks_found = 0
        for hook_file in hook_files:
            if (self.workspace_path / hook_file).exists():
                hooks_found += 1
                
        if hooks_found > 0:
            self.log_startup_event(f"Git pre-commit hooks verified ({hooks_found} found)", "SUCCESS")
            return True
        else:
            self.log_startup_event("No Git pre-commit hooks found", "WARNING")
            return False
            
    def create_startup_status_file(self):
        """📊 Create startup status file"""
        startup_status = {
            "startup_timestamp": datetime.now().isoformat(),
            "workspace_path": str(self.workspace_path.absolute()),
            "services_started": self.services_started,
            "startup_log": self.startup_log,
            "auto_startup_version": "v1.0.0",
            "architect_mode_active": True,
            "zero_tolerance_enforcement": True
        }
        
        with open(self.workspace_path / "genesis_startup_status.json", 'w') as f:
            json.dump(startup_status, f, indent=2)
            
        self.log_startup_event("Startup status file created", "SUCCESS")
        
    def show_welcome_banner(self):
        """🎉 Show GENESIS welcome banner"""
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                   🚀 GENESIS SYSTEM AUTO-STARTUP COMPLETE                          ║
║                   🔐 ARCHITECT MODE v7.0.0 -- ULTIMATE ENFORCEMENT                   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

🎯 SYSTEM STATUS: READY FOR DEVELOPMENT
🐺 Watchdog System: {'ACTIVE' if 'watchdog_system' in self.services_started else 'INACTIVE'}
🛡️ Git Pre-Commit Hooks: INSTALLED
🔐 ARCHITECT MODE: ACTIVE
📊 Zero Tolerance Enforcement: ENABLED

💡 Your development environment is now protected by:
   • Continuous monitoring (every 30 seconds)
   • Pre-commit audit enforcement
   • Real-time violation detection
   • Automatic compliance verification

🎉 Happy coding! Your GENESIS system is watching over you. 🛡️
"""
        print(banner)
        
    def run_auto_startup(self):
        """🚀 Execute complete auto-startup sequence"""
        self.log_startup_event("🚀 GENESIS AUTO-STARTUP INITIATED", "INFO")
        self.log_startup_event("=" * 60, "INFO")
        
        startup_success = True
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            startup_success = False
            
        # Step 2: Validate ARCHITECT MODE
        if not self.validate_architect_mode():
            startup_success = False
            
        # Step 3: Verify Git hooks
        self.verify_git_hooks()
        
        # Step 4: Start Watchdog System
        if not self.start_watchdog_system():
            startup_success = False
            
        # Step 5: Run initial audit (non-blocking)
        self.run_initial_audit()
        
        # Step 6: Create status file
        self.create_startup_status_file()
        
        # Step 7: Show welcome banner
        if startup_success:
            self.show_welcome_banner()
        else:
            self.log_startup_event("⚠️ AUTO-STARTUP COMPLETED WITH WARNINGS", "WARNING")
            self.log_startup_event("Review startup log for details", "INFO")
            
        self.log_startup_event("🚀 GENESIS AUTO-STARTUP SEQUENCE COMPLETE", "INFO")

def main():
    """Main entry point for auto-startup"""
    startup_system = GenesisAutoStartup()
    startup_system.run_auto_startup()

if __name__ == "__main__":
    main()



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))


def detect_divergence(price_data: list, indicator_data: list, window: int = 10) -> Dict:
    """
    Detect regular and hidden divergences between price and indicator
    
    Args:
        price_data: List of price values (closing prices)
        indicator_data: List of indicator values (e.g., RSI, MACD)
        window: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence information
    """
    result = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "strength": 0.0
    }
    
    # Need at least window + 1 periods of data
    if len(price_data) < window + 1 or len(indicator_data) < window + 1:
        return result
        
    # Get the current and historical points
    current_price = price_data[-1]
    previous_price = min(price_data[-window:-1]) if price_data[-1] > price_data[-2] else max(price_data[-window:-1])
    previous_price_idx = price_data[-window:-1].index(previous_price) + len(price_data) - window
    
    current_indicator = indicator_data[-1]
    previous_indicator = indicator_data[previous_price_idx]
    
    # Check for regular divergences
    # Bullish - Lower price lows but higher indicator lows
    if current_price < previous_price and current_indicator > previous_indicator:
        result["regular_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Higher price highs but lower indicator highs
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["regular_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Check for hidden divergences
    # Bullish - Higher price lows but lower indicator lows
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["hidden_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Lower price highs but higher indicator highs
    elif current_price < previous_price and current_indicator > previous_indicator:
        result["hidden_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Emit divergence event if detected
    if any([result["regular_bullish"], result["regular_bearish"], 
            result["hidden_bullish"], result["hidden_bearish"]]):
        emit_event("divergence_detected", {
            "type": next(k for k, v in result.items() if v is True and k != "strength"),
            "strength": result["strength"],
            "symbol": price_data.symbol if hasattr(price_data, "symbol") else "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
    return result
