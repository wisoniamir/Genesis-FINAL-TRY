
# Real Data Access Integration
import MetaTrader5 as mt5
from datetime import datetime

class RealDataAccess:
    """Provides real market data access"""
    
    def __init__(self):
        self.mt5_connected = False
        self.data_source = "live"
    
    def get_live_data(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, count=100):
        """Get live market data"""
        try:
            if not self.mt5_connected:
                mt5.initialize()
                self.mt5_connected = True
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            return rates
        except Exception as e:
            logger.error(f"Live data access failed: {e}")
            return None
    
    def get_account_info(self):
        """Get live account information"""
        try:
            return mt5.account_info()
        except Exception as e:
            logger.error(f"Account info access failed: {e}")
            return None

# Initialize real data access
_real_data = RealDataAccess()



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

                emit_telemetry("watchdog_core", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("watchdog_core", "position_calculated", {
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
                            "module": "watchdog_core",
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
                    print(f"Emergency stop error in watchdog_core: {e}")
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
                    "module": "watchdog_core",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("watchdog_core", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in watchdog_core: {e}")
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
"""
🐺 GENESIS WATCHDOG CORE — SYSTEM SENTINEL v1.0.0
CONTINUOUS MONITORING ENGINE FOR ARCHITECT MODE v7.0.0

PURPOSE:
Implements the core watchdog functions for continuous monitoring of GENESIS system
compliance, violations detection, and automatic remediation.
"""

import json
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set
import threading
import logging


# <!-- @GENESIS_MODULE_END: watchdog_core -->


# <!-- @GENESIS_MODULE_START: watchdog_core -->

# Setup logging for watchdog operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - WATCHDOG - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watchdog_sentinel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GenesisWatchdogCore:
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

            emit_telemetry("watchdog_core", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("watchdog_core", "position_calculated", {
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
                        "module": "watchdog_core",
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
                print(f"Emergency stop error in watchdog_core: {e}")
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
                "module": "watchdog_core",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("watchdog_core", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in watchdog_core: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "watchdog_core",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in watchdog_core: {e}")
    """🐺 GENESIS WATCHDOG CORE — Continuous System Monitoring"""
    
    def __init__(self, workspace_path="c:\\Users\\patra\\Genesis FINAL TRY"):
        self.workspace_path = Path(workspace_path)
        self.violations_detected = []
        self.quarantine_count = 0
        self.last_scan_time = None
        self.watchdog_active = True
        
        # Critical files to monitor
        self.core_files = [
            "build_status.json", "build_tracker.md", "system_tree.json", 
            "event_bus.json", "telemetry.json", "module_registry.json", 
            "test_runners.json", "compliance.json", "live_data.json", "real_data.json"
        ]
        
        # Violation patterns to detect
        self.violation_patterns = {
            "live_data": [
                r"mock[\s_]", r"simulate[\s_]", r"dummy[\s_]", r"test[\s_]data",
                r"fallback[\s_]", r"stub[\s_]", r"placeholder", r"sample[\s_]data"
            ],
            "eventbus_bypass": [
                r"direct[\s_]call", r"bypass[\s_]eventbus", r"skip[\s_]emit",
                r"local[\s_]function", r"isolated[\s_]logic"
            ],
            "telemetry_evasion": [
                r"skip[\s_]telemetry", r"disable[\s_]logging", r"bypass[\s_]metrics",
                r"no[\s_]tracking", r"silent[\s_]mode"
            ]
        }
        
        logger.info("🐺 GENESIS WATCHDOG CORE initialized")
        
    def system_alive(self) -> bool:
        """Check if the system is alive and watchdog should continue"""
        return self.watchdog_active and os.path.exists(self.workspace_path / "build_status.json")
        
    def load_and_validate_core_files(self, core_files: List[str]) -> Dict[str, Any]:
        """📁 Step 1: Validate Core System Files"""
        logger.info("📁 Validating core system files...")
        
        validation_results = {
            "valid_files": [],
            "missing_files": [],
            "corrupted_files": [],
            "validation_timestamp": datetime.now().isoformat()
        }
        
        for file_name in core_files:
            file_path = self.workspace_path / file_name
            
            if not file_path.exists():
                validation_results["missing_files"].append(file_name)
                logger.warning(f"⚠️ Missing core file: {file_name}")
                continue
                
            try:
                if file_name.endswith('.json'):
                    with open(file_path, 'r') as f:
                        json.load(f)  # Validate JSON structure
                        
                validation_results["valid_files"].append(file_name)
                logger.info(f"✅ Validated: {file_name}")
                
            except (json.JSONDecodeError, Exception) as e:
                validation_results["corrupted_files"].append({
                    "file": file_name,
                    "error": str(e)
                })
                logger.error(f"❌ Corrupted file: {file_name} - {e}")
                
        return validation_results
        
    def validate_eventbus_routes(self, event_bus_file: str) -> Dict[str, Any]:
        """📡 Step 2: Validate EventBus Routes"""
        logger.info("📡 Validating EventBus routes...")
        
        try:
            event_bus_path = self.workspace_path / event_bus_file
            with open(event_bus_path, 'r') as f:
                event_bus_data = json.load(f)
                
            validation_results = {
                "total_routes": 0,
                "active_routes": 0,
                "orphaned_routes": [],
                "empty_function_arrays": [],
                "validation_status": "PASS"
            }
            
            # Validate active routes
            active_routes = event_bus_data.get("active_routes", {})
            validation_results["total_routes"] = len(active_routes)
            
            for route_name, route_data in active_routes.items():
                if isinstance(route_data, dict):
                    functions = route_data.get("functions", [])
                    
                    if not functions:  # Empty functions array
                        validation_results["empty_function_arrays"].append(route_name)
                        logger.warning(f"⚠️ Empty functions array in route: {route_name}")
                    else:
                        validation_results["active_routes"] += 1
                        
            if validation_results["empty_function_arrays"]:
                validation_results["validation_status"] = "VIOLATIONS_DETECTED"
                
            logger.info(f"✅ EventBus validation: {validation_results['active_routes']}/{validation_results['total_routes']} routes active")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ EventBus validation failed: {e}")
            return {"validation_status": "FAILED", "error": str(e)}
            
    def verify_system_tree_connections(self, system_tree_file: str, registry_file: str) -> List[str]:
        """📊 Step 3: Scan for orphan or disconnected modules"""
        logger.info("📊 Scanning for orphan modules...")
        
        orphan_modules = []
        
        try:
            # Load system tree
            system_tree_path = self.workspace_path / system_tree_file
            with open(system_tree_path, 'r') as f:
                system_tree = json.load(f)
                
            # Scan for modules with compliance violations
            connected_modules = system_tree.get("connected_modules", {})
            
            for category, modules in connected_modules.items():
                if isinstance(modules, list):
                    for module in modules:
                        if isinstance(module, dict):
                            # Check for orphan indicators
                            if (module.get("eventbus_integrated") is False or 
                                module.get("telemetry_enabled") is False or
                                module.get("compliance_status") == "NEEDS_ATTENTION"):
                                
                                orphan_modules.append(module.get("name", "unknown"))
                                
            if orphan_modules:
                logger.warning(f"⚠️ Found {len(orphan_modules)} orphan modules")
            else:
                logger.info("✅ No orphan modules detected")
                
            return orphan_modules
            
        except Exception as e:
            logger.error(f"❌ System tree validation failed: {e}")
            return []
            
    def scan_for_live_data(self, violation_keywords: List[str]) -> List[Dict[str, Any]]:
        """🚫 Step 4: Check for Mock/Stub/Simulated Logic"""
        logger.info("🚫 Scanning for mock data violations...")
        
        mock_hits = []
        
        # Scan all Python files in the workspace
        for py_file in self.workspace_path.glob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Check for violation patterns
                for keyword in violation_keywords:
                    pattern = re.compile(keyword, re.IGNORECASE)
                    matches = pattern.findall(content)
                    
                    if matches:
                        mock_hits.append({
                            "file": py_file.name,
                            "keyword": keyword,
                            "matches": len(matches),
                            "severity": "HIGH" if keyword in ["mock", "simulate", "dummy"] else "MEDIUM"
                        })
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not scan {py_file.name}: {e}")
                
        if mock_hits:
            logger.warning(f"⚠️ Found {len(mock_hits)} mock data violations")
        else:
            logger.info("✅ No mock data violations detected")
            
        return mock_hits
        
    def enforce_module_wiring(self, system_tree_file: str, event_bus_file: str) -> List[str]:
        """🔗 Step 5: Enforce EventBus Wiring"""
        logger.info("🔗 Enforcing EventBus wiring...")
        
        unwired_modules = []
        
        try:
            # Load system tree
            system_tree_path = self.workspace_path / system_tree_file
            with open(system_tree_path, 'r') as f:
                system_tree = json.load(f)
                
            # Load event bus
            event_bus_path = self.workspace_path / event_bus_file
            with open(event_bus_path, 'r') as f:
                event_bus = json.load(f)
                
            # Check wiring
            connected_modules = system_tree.get("connected_modules", {})
            active_routes = event_bus.get("active_routes", {})
            
            for category, modules in connected_modules.items():
                if isinstance(modules, list):
                    for module in modules:
                        if isinstance(module, dict):
                            module_name = module.get("name", "")
                            
                            # Check if module has EventBus route
                            if module_name not in active_routes:
                                unwired_modules.append(module_name)
                                
            if unwired_modules:
                logger.warning(f"⚠️ Found {len(unwired_modules)} unwired modules")
            else:
                logger.info("✅ All modules properly wired")
                
            return unwired_modules
            
        except Exception as e:
            logger.error(f"❌ Module wiring enforcement failed: {e}")
            return []
            
    def check_telemetry_integrity(self, telemetry_file: str) -> Dict[str, Any]:
        """📊 Check Telemetry Integrity"""
        logger.info("📊 Checking telemetry integrity...")
        
        try:
            telemetry_path = self.workspace_path / telemetry_file
            with open(telemetry_path, 'r') as f:
                telemetry_data = json.load(f)
                
            integrity_results = {
                "telemetry_active": telemetry_data.get("active", False),
                "collection_endpoints": len(telemetry_data.get("collection_endpoints", [])),
                "metric_definitions": len(telemetry_data.get("metric_definitions", [])),
                "status": "HEALTHY" if telemetry_data.get("active", False) else "DISABLED"
            }
            
            logger.info(f"✅ Telemetry integrity: {integrity_results['status']}")
            return integrity_results
            
        except Exception as e:
            logger.error(f"❌ Telemetry integrity check failed: {e}")
            return {"status": "FAILED", "error": str(e)}
            
    def quarantine_violations(self, violations: List[Any], reason: str) -> None:
        """🔒 Quarantine Violations"""
        logger.warning(f"🔒 Quarantining {len(violations)} violations: {reason}")
        
        # Log violations
        violation_entry = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "violations": violations,
            "action": "QUARANTINED",
            "quarantine_count": len(violations)
        }
        
        self.violations_detected.append(violation_entry)
        self.quarantine_count += len(violations)
        
        # Log to build tracker
        self.log_watchdog_alert(violation_entry)
        
    def log_watchdog_alerts(self) -> None:
        """🛠️ Step 6: Log Watchdog Alerts"""
        if self.violations_detected:
            logger.info(f"📝 Logging {len(self.violations_detected)} watchdog alerts")
            
            # Update build tracker
            tracker_path = self.workspace_path / "build_tracker.md"
            
            alert_summary = f"""

## 🐺 WATCHDOG SENTINEL ALERT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATUS **⚠️ VIOLATIONS DETECTED AND QUARANTINED**

STATS **Watchdog Alert Summary:**
- Total Violations: {len(self.violations_detected)}
- Items Quarantined: {self.quarantine_count}
- Last Scan: {datetime.now().isoformat()}
- Watchdog Status: ACTIVE

VIOLATIONS **Recent Violations:**"""

            for violation in self.violations_detected[-5:]:  # Show last 5
                alert_summary += f"""
- {violation['reason']}: {len(violation['violations'])} items ({violation['timestamp']})"""
                
            alert_summary += f"""

NEXT **Action Required:** Review quarantined items and repair violations

---

"""
            
            try:
                with open(tracker_path, 'a', encoding='utf-8') as f:
                    f.write(alert_summary)
                    
                logger.info("✅ Watchdog alerts logged to build tracker")
                
            except Exception as e:
                logger.error(f"❌ Failed to log watchdog alerts: {e}")
                
        # Clear processed violations
        self.violations_detected = []
        self.quarantine_count = 0
        
    def log_watchdog_alert(self, violation_entry: Dict[str, Any]) -> None:
        """Log individual watchdog alert"""
        logger.warning(f"🚨 WATCHDOG ALERT: {violation_entry['reason']} - {violation_entry['quarantine_count']} items")
        
    def start_watchdog_daemon(self, scan_interval: int = 30) -> None:
        """🔁 Start the watchdog daemon"""
        logger.info(f"🐺 Starting GENESIS Watchdog Daemon (scan interval: {scan_interval}s)")
        
        def watchdog_loop():
            while self.system_alive():
                try:
                    # Execute watchdog scan cycle
                    self.execute_full_scan()
                    time.sleep(scan_interval)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Watchdog daemon stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Watchdog daemon error: {e}")
                    time.sleep(scan_interval)  # Continue after error
                    
        # Start daemon in background thread
        daemon_thread = threading.Thread(target=watchdog_loop, daemon=True)
        daemon_thread.start()
        logger.info("✅ Watchdog daemon started successfully")
        
    def execute_full_scan(self) -> Dict[str, Any]:
        """Execute a full watchdog scan cycle"""
        scan_start = datetime.now()
        logger.info("🔍 Executing full watchdog scan...")
        
        # Step 1: Validate core files
        core_validation = self.load_and_validate_core_files(self.core_files)
        
        # Step 2: Validate EventBus
        eventbus_validation = self.validate_eventbus_routes("event_bus.json")
        
        # Step 3: Check for orphans
        orphan_modules = self.verify_system_tree_connections("system_tree.json", "module_registry.json")
        if orphan_modules:
            self.quarantine_violations(orphan_modules, "ORPHAN_MODULES")
            
        # Step 4: Scan for mock data
        mock_hits = self.scan_for_live_data(["mock", "stub", "simulate", "fallback", "dummy"])
        if mock_hits:
            self.quarantine_violations(mock_hits, "MOCK_DATA_DETECTED")
            
        # Step 5: Check module wiring
        unwired = self.enforce_module_wiring("system_tree.json", "event_bus.json")
        if unwired:
            self.quarantine_violations(unwired, "UNWIRED_MODULES")
            
        # Step 6: Check telemetry
        telemetry_status = self.check_telemetry_integrity("telemetry.json")
        
        # Log alerts if any violations found
        if self.violations_detected:
            self.log_watchdog_alerts()
            
        scan_duration = (datetime.now() - scan_start).total_seconds()
        self.last_scan_time = datetime.now()
        
        scan_results = {
            "scan_timestamp": scan_start.isoformat(),
            "scan_duration_seconds": scan_duration,
            "core_files_status": core_validation,
            "eventbus_status": eventbus_validation,
            "orphan_modules_count": len(orphan_modules),
            "mock_violations_count": len(mock_hits),
            "unwired_modules_count": len(unwired),
            "telemetry_status": telemetry_status,
            "violations_detected": len(self.violations_detected) > 0
        }
        
        logger.info(f"✅ Watchdog scan completed in {scan_duration:.2f}s")
        return scan_results
        
    def stop_watchdog(self) -> None:
        """Stop the watchdog system"""
        self.watchdog_active = False
        logger.info("🛑 GENESIS Watchdog stopped")

# Global watchdog instance
_watchdog_instance = None

def get_watchdog() -> GenesisWatchdogCore:
    """Get the global watchdog instance"""
    global _watchdog_instance
    if _watchdog_instance is None:
        _watchdog_instance = GenesisWatchdogCore()
    return _watchdog_instance

# Convenience functions for external use
def validate_eventbus_routes(event_bus_file: str) -> Dict[str, Any]:
    return get_watchdog().validate_eventbus_routes(event_bus_file)

def verify_system_tree_connections(system_tree_file: str, registry_file: str) -> List[str]:
    return get_watchdog().verify_system_tree_connections(system_tree_file, registry_file)

def scan_for_live_data(violation_keywords: List[str]) -> List[Dict[str, Any]]:
    return get_watchdog().scan_for_live_data(violation_keywords)

def enforce_module_wiring(system_tree_file: str, event_bus_file: str) -> List[str]:
    return get_watchdog().enforce_module_wiring(system_tree_file, event_bus_file)

def check_telemetry_integrity(telemetry_file: str) -> Dict[str, Any]:
    return get_watchdog().check_telemetry_integrity(telemetry_file)

def quarantine_violations(violations: List[Any], reason: str) -> None:
    get_watchdog().quarantine_violations(violations, reason)

def log_watchdog_alerts() -> None:
    get_watchdog().log_watchdog_alerts()

def system_alive() -> bool:
    return get_watchdog().system_alive()

def load_and_validate_core_files(core_files: List[str]) -> Dict[str, Any]:
    return get_watchdog().load_and_validate_core_files(core_files)

if __name__ == "__main__":
    # Start watchdog in standalone mode
    watchdog = GenesisWatchdogCore()
    print("🐺 GENESIS WATCHDOG CORE — Starting standalone mode")
    
    try:
        watchdog.start_watchdog_daemon(scan_interval=10)  # 10 second interval for demo
        
        # Keep the main thread alive
        while watchdog.system_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping GENESIS Watchdog...")
        watchdog.stop_watchdog()


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
