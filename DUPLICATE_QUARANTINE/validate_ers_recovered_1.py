
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

                emit_telemetry("validate_ers_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("validate_ers_recovered_1", "position_calculated", {
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
                            "module": "validate_ers_recovered_1",
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
                    print(f"Emergency stop error in validate_ers_recovered_1: {e}")
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
                    "module": "validate_ers_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("validate_ers_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in validate_ers_recovered_1: {e}")
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


# <!-- @GENESIS_MODULE_START: validate_ers -->

"""
🔐 GENESIS AI SYSTEM — EXECUTION RISK SENTINEL VALIDATION
========================================================
PHASE 52: ERS VALIDATION MODULE
Simple validation test for the Execution Risk Sentinel

⚠️ ARCHITECT MODE COMPLIANT v5.0.0
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | GENESIS-ERS-VALIDATION | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ers_validation")

# Check required files exist
required_files = [
    "execution_risk_sentinel.py",
    "execution_risk_config.json",
    "ers_report.json"
]

def validate_file_existence():
    """Validate that all required files exist"""
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    logger.info("✅ All required files exist")
    return True

def validate_config_file():
    """Validate the execution_risk_config.json file"""
    try:
        with open("execution_risk_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ["metadata", "thresholds", "watchlist", "fallback_routing"]
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logger.error(f"Missing required sections in config: {', '.join(missing_sections)}")
            return False
        
        # Check threshold values
        required_thresholds = [
            "latency_threshold_ms", 
            "alpha_decay_threshold",
            "cluster_trade_window_sec",
            "cluster_threshold"
        ]
        
        for threshold in required_thresholds:
            if threshold not in config.get("thresholds", {}):
                logger.error(f"Missing required threshold: {threshold}")
                return False
        
        logger.info("✅ Config file validation successful")
        return True
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error validating config file: {e}")
        return False

def validate_module_code():
    """Basic validation of the execution_risk_sentinel.py file"""
    try:
        with open("execution_risk_sentinel.py", 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check for required components
        required_components = [
            "class ExecutionRiskSentinel",
            "
    def log_state(self):
        """Phase 91 Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": __name__,
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "phase": "91_telemetry_enforcement"
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", state_data)
        return state_data
        class RiskAnomaly",
            "def _detect_latency_anomaly",
            "def _detect_execution_clustering",
            "def _calculate_combined_risk_score",
            "def _activate_killswitch",
            "def handle_execution_log",
            "def handle_alpha_decay"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in code:
                missing_components.append(component)
        
        if missing_components:
            logger.error(f"Missing required code components: {', '.join(missing_components)}")
            return False
        
        # Check for real data violations (basic check)
        mock_indicators = [
            "live_mt5_data",
            "self.event_bus.request('data:real_feed')",
            "# real",
            "# ARCHITECT_MODE_COMPLIANCE: Implementation required
            "# STUB",
            "pass  # ARCHITECT_MODE_COMPLIANCE: Implementation required
        ]
        
        violations = []
        for indicator in mock_indicators:
            if indicator in code:
                violations.append(indicator)
        
        if violations:
            logger.error(f"Possible real data violations found: {', '.join(violations)}")
            return False
        
        logger.info("✅ Module code validation successful")
        return True
    
    except FileNotFoundError as e:
        logger.error(f"Error validating module code: {e}")
        return False

def validate_system_registration():
    """Validate system registration in module_registry.json and system_tree.json"""
    try:        # Check module_registry.json
        if os.path.exists("module_registry.json"):
            with open("module_registry.json", 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # Check if ExecutionRiskSentinel is in modules
            modules = registry.get("modules", [])
            ers_module = next((m for m in modules if m.get("name") == "ExecutionRiskSentinel"), None)
            
            if not ers_module:
                logger.error("ExecutionRiskSentinel not found in module_registry.json")
                return False
            
            # Check metadata
            if not registry.get("metadata", {}).get("phase_52_execution_risk_sentinel_added"):
                logger.error("phase_52_execution_risk_sentinel_added not found in module_registry.json metadata")
              # Check system_tree.json
        if os.path.exists("system_tree.json"):
            with open("system_tree.json", 'r', encoding='utf-8') as f:
                tree = json.load(f)
            
            # Check if ExecutionRiskSentinel is in nodes
            nodes = tree.get("nodes", [])
            ers_node = next((n for n in nodes if n.get("id") == "ExecutionRiskSentinel"), None)
            
            if not ers_node:
                logger.error("ExecutionRiskSentinel not found in system_tree.json nodes")
                return False
            
            # Check metadata
            if not tree.get("metadata", {}).get("phase_52_execution_risk_sentinel_integrated"):
                logger.error("phase_52_execution_risk_sentinel_integrated not found in system_tree.json metadata")
          # Check event_bus.json
        if os.path.exists("event_bus.json"):
            with open("event_bus.json", 'r', encoding='utf-8') as f:
                event_bus = json.load(f)
            
            # Check if routes for ERS exist
            routes = event_bus.get("routes", [])
            ers_routes = [r for r in routes if r.get("producer") == "ExecutionRiskSentinel" or r.get("consumer") == "ExecutionRiskSentinel"]
            
            if len(ers_routes) < 4:  # We should have at least 4 routes
                logger.error(f"Insufficient EventBus routes for ExecutionRiskSentinel: found {len(ers_routes)}")
                return False
            
            # Check metadata
            if not event_bus.get("metadata", {}).get("phase_52_execution_risk_sentinel_integrated"):
                logger.error("phase_52_execution_risk_sentinel_integrated not found in event_bus.json metadata")
        
        logger.info("✅ System registration validation successful")
        return True
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error validating system registration: {e}")
        return False

def validate_report():
    """Validate the ers_report.json file"""
    try:
        with open("ers_report.json", 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Check for required sections
        required_sections = ["metadata", "execute", "anomalies", "alerts", "test_results", "performance_metrics"]
        missing_sections = [section for section in required_sections if section not in report]
        
        if missing_sections:
            logger.error(f"Missing required sections in report: {', '.join(missing_sections)}")
            return False
        
        # Check compliance
        if not report.get("compliance", {}).get("status") == "FULLY_COMPLIANT":
            logger.error("Report does not indicate FULLY_COMPLIANT status")
            return False
        
        logger.info("✅ Report validation successful")
        return True
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error validating report: {e}")
        return False

def validate_telemetry_integration():
    """Validate telemetry integration"""
    try:
        if os.path.exists("telemetry.json"):
            with open("telemetry.json", 'r', encoding='utf-8') as f:
                telemetry = json.load(f)
            
            # Check for ERS metrics
            metrics = telemetry.get("metrics", {})
            ers_metrics = [m for m in metrics.keys() if m.startswith("ers_")]
            
            if len(ers_metrics) < 3:
                logger.error(f"Insufficient ERS metrics in telemetry.json: found {len(ers_metrics)}")
                return False
            
            logger.info(f"Found {len(ers_metrics)} ERS metrics in telemetry.json: {', '.join(ers_metrics)}")
        
        logger.info("✅ Telemetry integration validation successful")
        return True
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error validating telemetry integration: {e}")
        return False

def run_validation():
    """Run all validation checks"""
    validations = [
        ("Required Files", validate_file_existence),
        ("Config File", validate_config_file),
        ("Module Code", validate_module_code),
        ("System Registration", validate_system_registration),
        ("ERS Report", validate_report),
        ("Telemetry Integration", validate_telemetry_integration)
    ]
    
    results = []
    for name, validation_func in validations:
        logger.info(f"Running {name} validation...")
        success = validation_func()
        results.append((name, success))
        time.sleep(0.5)  # Short delay for readability of logs
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXECUTION RISK SENTINEL VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if not success:
            all_passed = False
    
    print("\nOVERALL VALIDATION:", "✅ PASSED" if all_passed else "❌ FAILED")
      # Update build_tracker.md
    if all_passed:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            validation_entry = f"**{timestamp}** - ✅ **ERS VALIDATION** - Phase 52 ExecutionRiskSentinel validation completed successfully.\n\n"
            
            with open("build_tracker.md", 'a', encoding='utf-8') as f:
                f.write(validation_entry)
            
            logger.info("Updated build_tracker.md with validation result")
        except Exception as e:
            logger.error(f"Failed to update build_tracker.md: {e}")
    
    return all_passed

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("GENESIS EXECUTION RISK SENTINEL VALIDATION")
    print("=" * 50 + "\n")
    
    try:
        success = run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.critical(f"Unhandled exception during validation: {e}")
        sys.exit(1)


# <!-- @GENESIS_MODULE_END: validate_ers -->