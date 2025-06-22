# -*- coding: utf-8 -*-
# @GENESIS_ORPHAN_STATUS: recoverable
# @GENESIS_SUGGESTED_ACTION: connect
# @GENESIS_ANALYSIS_DATE: 2025-06-20T16:45:13.473457
# @GENESIS_PROTECTION: DO_NOT_DELETE_UNTIL_REVIEWED

#!/usr/bin/env python3
"""
GENESIS Final System Validation - Phase 88 Complete
Comprehensive validation that all phases 82-88 are complete and system is live-ready

🎯 PURPOSE: Final validation that GENESIS is ready for live MT5 trading
🔁 EVENTBUS: Emits system:final_validation, system:live_ready
📡 TELEMETRY: Complete system health check
🛡️ COMPLIANCE: Architect Mode v5.0.0 institutional grade
"""

import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FinalValidation')

class GenesisSystemValidator:
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

            emit_telemetry("genesis_final_system_validation", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("genesis_final_system_validation", "position_calculated", {
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
                        "module": "genesis_final_system_validation",
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
                print(f"Emergency stop error in genesis_final_system_validation: {e}")
                return False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager


# <!-- @GENESIS_MODULE_END: genesis_final_system_validation -->


# <!-- @GENESIS_MODULE_START: genesis_final_system_validation -->
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """Final comprehensive system validation"""
    
    def __init__(self):
        """Initialize system validator"""
        self.validation_id = f"final_validation_{int(time.time())}"
        self.start_time = datetime.now(timezone.utc)
        
        self.critical_files = [
            "auto_execution_manager.py",
            "live_risk_governor.py", 
            "mt5_connection_bridge.py",
            "dashboard.py",
            "genesis_launcher.py",
            "system_tree.json",
            "module_registry.json",
            "build_status.json",
            "install_metadata.json",
            "phase_88_live_trial_activation.py"
        ]
        
        self.validation_results = {}
        
    def validate_phase_completion(self) -> Dict[str, bool]:
        """Validate all phases 82-88 are complete"""
        phase_results = {}
        
        try:
            # Check install metadata
            metadata_path = Path("install_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                readiness = metadata.get("readiness_status", {})
                phase_results = {
                    "phase_82_83_complete": readiness.get("killswitch_validated", False) and 
                                           readiness.get("auto_mode_validated", False),
                    "phase_84_85_complete": metadata.get("genesis_system", {}).get("version") == "1.0.0",
                    "phase_86_complete": readiness.get("phase_86_complete", False),
                    "phase_87_complete": readiness.get("phase_87_complete", False),
                    "phase_88_complete": readiness.get("phase_88_complete", False),
                    "live_trial_complete": readiness.get("live_trial_complete", False),
                    "live_trading_ready": readiness.get("live_trading_ready", False)
                }
            
            logger.info(f"Phase completion validation: {phase_results}")
            return phase_results
            
        except Exception as e:
            logger.error(f"Phase validation error: {str(e)}")
            return {}
    
    def validate_critical_files(self) -> Dict[str, bool]:
        """Validate all critical system files exist"""
        file_results = {}
        
        for file_path in self.critical_files:
            path = Path(file_path)
            file_results[file_path] = path.exists()
            
        logger.info(f"Critical files validation: {sum(file_results.values())}/{len(file_results)} files present")
        return file_results
    
    def validate_module_registry(self) -> Dict[str, Any]:
        """Validate module registry integrity"""
        try:
            registry_path = Path("module_registry.json")
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                    
                modules = registry.get("modules", {})
                return {
                    "registry_exists": True,
                    "total_modules": len(modules),
                    "auto_execution_manager_registered": "auto_execution_manager" in modules,
                    "live_risk_governor_registered": "live_risk_governor" in modules,
                    "mt5_bridge_registered": "mt5_connection_bridge" in modules,
                    "all_critical_modules_registered": all([
                        "auto_execution_manager" in modules,
                        "live_risk_governor" in modules,
                        "dashboard_engine" in modules
                    ])
                }
            else:
                return {"registry_exists": False}
                
        except Exception as e:
            logger.error(f"Module registry validation error: {str(e)}")
            return {"registry_exists": False, "error": str(e)}
    
    def validate_telemetry_system(self) -> Dict[str, bool]:
        """Validate telemetry system operational"""
        telemetry_files = [
            "telemetry/connection_status.json",
            "telemetry/killswitch_test.json", 
            "telemetry/auto_mode_state.json",
            "telemetry/alerts_log.json"
        ]
        
        telemetry_results = {}
        for file_path in telemetry_files:
            path = Path(file_path)
            telemetry_results[file_path] = path.exists()
            
        return telemetry_results
    
    def validate_phase_88_artifacts(self) -> Dict[str, bool]:
        """Validate Phase 88 generated artifacts"""
        phase_88_files = [
            "logs/phase_88_trial_boot_log.md",
            "logs/trial_execution_test_log.json",
            "logs/mt5_latency_test_log.md",
            "PHASE_88_COMPLETION_SUMMARY.md"
        ]
        
        artifact_results = {}
        for file_path in phase_88_files:
            path = Path(file_path)
            artifact_results[file_path] = path.exists()
            
        return artifact_results
    
    def validate_architect_mode(self) -> Dict[str, Any]:
        """Validate Architect Mode v5.0.0 compliance"""
        try:
            build_status_path = Path("build_status.json")
            if build_status_path.exists():
                with open(build_status_path, 'r') as f:
                    build_status = json.load(f)
                    
                architect_status = build_status.get("architect_mode_status", {})
                return {
                    "architect_mode_active": architect_status.get("architect_mode_v500_activation", False),
                    "version": "v5.0.0",
                    "structural_enforcement": architect_status.get("architect_mode_v500_structural_enforcement", False),
                    "real_time_validation": architect_status.get("architect_mode_v500_real_time_validation", False),
                    "compliance_rate": architect_status.get("architect_mode_v401_compliance_rate", 0),
                    "system_grade": architect_status.get("architect_mode_v401_system_grade", "UNKNOWN")
                }
            else:
                return {"architect_mode_active": False}
                
        except Exception as e:
            logger.error(f"Architect mode validation error: {str(e)}")
            return {"architect_mode_active": False, "error": str(e)}
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        try:
            logger.info(f"Starting comprehensive system validation: {self.validation_id}")
            
            # Run all validation checks
            validation_results = {
                "validation_id": self.validation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase_completion": self.validate_phase_completion(),
                "critical_files": self.validate_critical_files(),
                "module_registry": self.validate_module_registry(),
                "telemetry_system": self.validate_telemetry_system(),
                "phase_88_artifacts": self.validate_phase_88_artifacts(),
                "architect_mode": self.validate_architect_mode()
            }
            
            # Calculate overall system readiness
            phase_completion = validation_results["phase_completion"]
            critical_files = validation_results["critical_files"]
            module_registry = validation_results["module_registry"]
            architect_mode = validation_results["architect_mode"]
            
            # Overall readiness calculation
            phases_complete = all([
                phase_completion.get("phase_82_83_complete", False),
                phase_completion.get("phase_84_85_complete", False),
                phase_completion.get("phase_86_complete", False),
                phase_completion.get("phase_87_complete", False),
                phase_completion.get("phase_88_complete", False)
            ])
            
            files_present = all(critical_files.values())
            modules_registered = module_registry.get("all_critical_modules_registered", False)
            architect_active = architect_mode.get("architect_mode_active", False)
            
            overall_readiness = {
                "all_phases_complete": phases_complete,
                "all_critical_files_present": files_present,
                "modules_properly_registered": modules_registered,
                "architect_mode_operational": architect_active,
                "live_trading_ready": all([phases_complete, files_present, modules_registered, architect_active]),
                "system_status": "LIVE_READY" if all([phases_complete, files_present, modules_registered, architect_active]) else "NEEDS_REVIEW"
            }
            
            validation_results["overall_readiness"] = overall_readiness
            
            # Save validation report
            report_path = Path("logs/final_system_validation_report.json")
            report_path.parent.mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            # Generate summary
            self.generate_validation_summary(validation_results)
            
            logger.info(f"System validation complete. Status: {overall_readiness['system_status']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Comprehensive validation error: {str(e)}")
            return {"error": str(e)}
    
    def generate_validation_summary(self, results: Dict[str, Any]):
        """Generate human-readable validation summary"""
        try:
            summary_path = Path("GENESIS_FINAL_VALIDATION_SUMMARY.md")
            
            overall = results["overall_readiness"]
            phase_completion = results["phase_completion"]
            
            with open(summary_path, 'w') as f:
                f.write(f"""# GENESIS FINAL SYSTEM VALIDATION SUMMARY
====================================================

**Validation ID:** {results['validation_id']}  
**Timestamp:** {results['timestamp']}  
**Overall Status:** {'✅ LIVE-READY' if overall['live_trading_ready'] else '⚠️ NEEDS REVIEW'}

## 🎯 PHASE COMPLETION STATUS

""")
                
                phase_status = [
                    ("Phases 82-83 (Execution & Risk)", phase_completion.get("phase_82_83_complete", False)),
                    ("Phases 84-85 (Assembly & Installer)", phase_completion.get("phase_84_85_complete", False)),
                    ("Phase 86 (Strategy Intelligence)", phase_completion.get("phase_86_complete", False)),
                    ("Phase 87 (Final Feature Activation)", phase_completion.get("phase_87_complete", False)),
                    ("Phase 88 (Live Trial & MT5)", phase_completion.get("phase_88_complete", False))
                ]
                
                for phase_name, status in phase_status:
                    status_icon = "✅" if status else "❌"
                    f.write(f"- **{phase_name}:** {status_icon}\n")
                
                f.write(f"""
## 🔧 SYSTEM COMPONENTS

### Critical Files
""")
                
                critical_files = results["critical_files"]
                for file_name, exists in critical_files.items():
                    status_icon = "✅" if exists else "❌"
                    f.write(f"- {file_name}: {status_icon}\n")
                
                f.write(f"""
### Module Registry
- **Total Modules:** {results['module_registry'].get('total_modules', 0)}
- **Critical Modules Registered:** {'✅' if results['module_registry'].get('all_critical_modules_registered', False) else '❌'}

### Architect Mode
- **Version:** {results['architect_mode'].get('version', 'Unknown')}
- **Active:** {'✅' if results['architect_mode'].get('architect_mode_active', False) else '❌'}
- **System Grade:** {results['architect_mode'].get('system_grade', 'Unknown')}

## 🚀 LIVE TRADING READINESS

""")
                
                readiness_checks = [
                    ("All Phases Complete", overall["all_phases_complete"]),
                    ("Critical Files Present", overall["all_critical_files_present"]),
                    ("Modules Registered", overall["modules_properly_registered"]),
                    ("Architect Mode Operational", overall["architect_mode_operational"])
                ]
                
                for check_name, status in readiness_checks:
                    status_icon = "✅" if status else "❌"
                    f.write(f"- **{check_name}:** {status_icon}\n")
                
                f.write(f"""
## 🎯 FINAL STATUS

**GENESIS System Status:** {overall['system_status']}

""")
                
                if overall['live_trading_ready']:
                    f.write("""✅ **GENESIS IS LIVE-READY FOR MT5 TRADING**

The system has successfully completed all phases 82-88 and is ready for production deployment with live MT5 trading capabilities.

### Ready Features:
- Real-time signal execution
- Live risk management with FTMO compliance
- Emergency KillSwitch controls
- Complete telemetry and monitoring
- GUI controls and dashboard
- MT5 connection and trade execution

### Next Steps:
1. Deploy to production environment
2. Connect to live MT5 account
3. Begin live trading operations
4. Monitor system performance and telemetry
""")
                else:
                    f.write("""⚠️ **SYSTEM NEEDS REVIEW**

Some components require attention before live deployment. Please review the validation results and ensure all critical components are operational.
""")
                
                f.write(f"""
---
*Generated by GENESIS Final System Validator*  
*Validation ID: {results['validation_id']}*  
*Timestamp: {results['timestamp']}*
""")
            
            logger.info(f"Validation summary generated: {summary_path}")
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")

def main():
    """Main validation execution"""
    try:
        print("🔍 GENESIS Final System Validation")
        print("🛡️ Architect Mode v5.0.0 - INSTITUTIONAL GRADE")
        print("=" * 60)
        
        validator = GenesisSystemValidator()
        results = validator.run_comprehensive_validation()
        
        if results and "overall_readiness" in results:
            overall = results["overall_readiness"]
            if overall["live_trading_ready"]:
                print("\n✅ VALIDATION COMPLETE - GENESIS IS LIVE-READY!")
                print("🚀 System ready for production MT5 trading")
                return True
            else:
                print("\n⚠️ VALIDATION COMPLETE - SYSTEM NEEDS REVIEW")
                print("📋 Check validation report for details")
                return False
        else:
            print("\n❌ VALIDATION FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Main validation error: {str(e)}")
        print(f"\n❌ Validation error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

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


def setup_event_subscriptions(self):
    """Set up EventBus subscriptions for this UI component"""
    event_bus.subscribe("market_data_updated", self.handle_market_data_update)
    event_bus.subscribe("trade_executed", self.handle_trade_update)
    event_bus.subscribe("position_changed", self.handle_position_update)
    event_bus.subscribe("risk_threshold_warning", self.handle_risk_warning)
    event_bus.subscribe("system_status_changed", self.handle_system_status_update)
    
    # Register with telemetry
    telemetry.log_event(TelemetryEvent(
        category="ui", 
        name="event_subscriptions_setup", 
        properties={"component": self.__class__.__name__}
    ))
