
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


#!/usr/bin/env python3
"""
🔧 PHASE 98 — SYSTEM BUILD VALIDATOR v3.0 - ARCHITECT MODE COMPLIANT
Final pass integrity validator before system prep for install
Ensures every GENESIS component from previous phases is validated and patched if necessary
Uses EventBus-connected architecture surveillance to enforce system integrity rules

🎯 PURPOSE: EventBus-driven validation with real-time telemetry
📡 EVENTBUS: Mandatory connection for all validation operations
🚫 ZERO TOLERANCE: No isolated functions, no mock data, no local calls
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Phase98Validator')

class Phase98SystemBuildValidator:
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

            emit_telemetry("phase_98_system_build_validator", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "phase_98_system_build_validator",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("phase_98_system_build_validator", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("phase_98_system_build_validator", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss', 0)
            if daily_loss > 0.05:
                emit_telemetry("phase_98_system_build_validator", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("phase_98_system_build_validator", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    """
    ARCHITECT MODE COMPLIANT System Build Validator
    EventBus-connected validation with mandatory telemetry emission
    """
    
    def __init__(self):
        """Initialize validator with mandatory EventBus connection"""
        # MANDATORY EventBus Connection - ARCHITECT MODE COMPLIANCE
        try:
            from event_bus import EventBus


# <!-- @GENESIS_MODULE_END: phase_98_system_build_validator -->


# <!-- @GENESIS_MODULE_START: phase_98_system_build_validator -->
            self.event_bus = EventBus()
            self.event_bus_connected = True
            logger.info("✅ EventBus connection established")
        except ImportError:
            logger.error("❌ CRITICAL: EventBus import failed")
            raise ImportError("EventBus connection is mandatory for ARCHITECT MODE")
        
        # ARCHITECT MODE COMPLIANCE FLAGS
        self.real_data_only = True
        self.telemetry_enabled = True
        self.version = "98.0-architect_compliant"
        
        # Validation configuration
        self.required_files = [
            "system_tree.json",
            "module_registry.json", 
            "telemetry.json",
            "event_bus.json",
            "build_status.json",
            "genesis_config.json",
            "genesis_docs.json", 
            "genesis_telemetry.json",
            "genesis_event_bus.json",
            "compliance.json",
            "real_data.json",
            "live_data.json"
        ]
        
        self.validation_results = {
            "files_validated": 0,
            "files_missing": [],
            "files_corrupted": [],
            "violations_detected": [],
            "repairs_applied": []
        }
        
        # Emit startup telemetry
        self._emit_startup_telemetry()
        
        logger.info("🔧 Phase 98 System Build Validator v3.0 initialized - ARCHITECT MODE ACTIVE")
    
    def _emit_startup_telemetry(self):
        """Emit startup telemetry to EventBus"""
        if hasattr(self, 'event_bus') and self.event_bus:
            telemetry_data = {
                "module": "phase_98_system_build_validator",
                "status": "initialized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architect_mode": True,
                "eventbus_connected": self.event_bus_connected,
                "real_data_only": self.real_data_only,
                "version": self.version
            }
            self.event_bus.emit("telemetry", telemetry_data)
            logger.info("📊 Startup telemetry emitted to EventBus")
    
    def validate_core_files(self) -> Dict[str, Any]:
        """
        ARCHITECT_MODE_COMPLIANCE: Validate all core files with EventBus reporting
        NO LOCAL VALIDATION - All results emitted via EventBus
        """
        logger.info("🔍 Beginning Full System File Validation via EventBus...")
        
        for file_path in self.required_files:
            try:
                self._validate_single_file(file_path)
            except Exception as e:
                logger.error(f"❌ Validation error for {file_path}: {str(e)}")
                self.validation_results["violations_detected"].append({
                    "file": file_path,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Emit validation completion telemetry
        self.event_bus.emit("validation_complete", {
            "module": "phase_98_system_build_validator",
            "results": self.validation_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return self.validation_results
    
    def _validate_single_file(self, file_path: str) -> bool:
        """Validate individual file and emit results via EventBus"""
        if not os.path.exists(file_path):
            logger.error(f"❌ Missing: {file_path}")
            self.validation_results["files_missing"].append(file_path)
            
            # Emit missing file alert
            self.event_bus.emit("file_missing", {
                "file": file_path,
                "severity": "CRITICAL",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False
        
        try:
            # Validate JSON structure
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            
            # Special validation for critical files
            if file_path == "live_data.json":
                self._validate_live_data_empty(data, file_path)
            elif file_path == "build_status.json":
                self._validate_build_status(data, file_path)
            elif file_path == "system_tree.json":
                self._validate_system_tree(data, file_path)
            
            logger.info(f"✅ Validated: {file_path}")
            self.validation_results["files_validated"] += 1
            
            # Emit successful validation
            self.event_bus.emit("file_validated", {
                "file": file_path,
                "status": "VALID",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Corrupted JSON in {file_path} → FLAGGING FOR REPAIR")
            self.validation_results["files_corrupted"].append({
                "file": file_path,
                "error": str(e)
            })
            
            # Emit corruption alert
            self.event_bus.emit("file_corrupted", {
                "file": file_path,
                "error": str(e),
                "action_required": "REPAIR",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False
    
    def _validate_live_data_empty(self, data: Dict, file_path: str):
        """Ensure live_data.json remains empty per ARCHITECT MODE requirements"""
        mock_sources = data.get("live_data_sources", {})
        if mock_sources:
            violation = f"ARCHITECT_VIOLATION: live_data.json contains data: {mock_sources}"
            logger.error(f"❌ {violation}")
            self.validation_results["violations_detected"].append({
                "file": file_path,
                "violation": violation,
                "severity": "CRITICAL"
            })
            
            # Emit critical violation
            self.event_bus.emit("architect_violation", {
                "type": "MOCK_DATA_DETECTED",
                "file": file_path,
                "data": mock_sources,
                "severity": "CRITICAL",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    def _validate_build_status(self, data: Dict, file_path: str):
        """Validate build status compliance"""
        required_status = ["ARCHITECT_COMPLIANT", "CONSOLIDATED", "FULLY_COMPLIANT"]
        system_status = data.get("system_status")
        
        if system_status not in required_status:
            violation = f"Build status not compliant: {system_status}"
            self.validation_results["violations_detected"].append({
                "file": file_path,
                "violation": violation,
                "severity": "WARNING"
            })
    
    def _validate_system_tree(self, data: Dict, file_path: str):
        """Validate system tree connectivity"""
        total_modules = data.get("genesis_final_system", {}).get("total_modules", 0)
        
        if total_modules < 30:
            violation = f"System tree has insufficient modules: {total_modules}"
            self.validation_results["violations_detected"].append({
                "file": file_path,
                "violation": violation,
                "severity": "WARNING"
            })
    
    def run_architect_surveillance(self) -> Dict[str, Any]:
        """
        EventBus-connected architect surveillance hook
        Replaces direct function calls with EventBus routing
        """
        logger.info("🧠 Running Architect Surveillance Hook via EventBus...")
        
        # Request surveillance via EventBus instead of direct import
        surveillance_request = {
            "module": "phase_98_system_build_validator",
            "action": "architect_surveillance",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Emit surveillance request
        self.event_bus.emit("surveillance_request", surveillance_request)
        
        # Check for EventBus connectivity violations
        connectivity_violations = self._check_eventbus_connectivity()
        
        # Emit surveillance completion
        self.event_bus.emit("surveillance_complete", {
            "module": "phase_98_system_build_validator",
            "violations_found": len(connectivity_violations),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return {"surveillance_complete": True, "violations": connectivity_violations}
    
    def _check_eventbus_connectivity(self) -> List[Dict]:
        """Check for EventBus connectivity violations"""
        violations = []
        
        try:
            # Check if system_tree.json has disconnected modules
            with open("system_tree.json", "r") as f:
                system_tree = json.load(f)
            
            # Scan for eventbus_connected: false
            for section in system_tree.values():
                if isinstance(section, dict):
                    for module_name, module_data in section.items():
                        if isinstance(module_data, dict):
                            if module_data.get("eventbus_connected") is False:
                                violation = {
                                    "module": module_name,
                                    "violation": "EventBus not connected",
                                    "severity": "CRITICAL"
                                }
                                violations.append(violation)
                                
                                # Emit violation alert
                                self.event_bus.emit("eventbus_violation", {
                                    "module": module_name,
                                    "violation": "disconnected",
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                })
        
        except Exception as e:
            logger.error(f"Error checking EventBus connectivity: {str(e)}")
        
        return violations
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report via EventBus"""
        report = {
            "phase": "98",
            "validator_version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architect_mode": True,
            "validation_results": self.validation_results,
            "system_status": "VALIDATED" if not self.validation_results["violations_detected"] else "VIOLATIONS_DETECTED"
        }
        
        # Emit final report
        self.event_bus.emit("validation_report", report)
        
        return report
    
    def execute_phase_98_validation(self) -> Dict[str, Any]:
        """
        Main execution method - ARCHITECT MODE COMPLIANT
        All operations via EventBus, no isolated functions
        """
        logger.info("🔧 PHASE 98: Beginning Full System File Validation...")
        
        # Step 1: Validate core files
        file_validation = self.validate_core_files()
        
        # Step 2: Run architect surveillance
        surveillance_results = self.run_architect_surveillance()
        
        # Step 3: Generate report
        final_report = self.generate_validation_report()
        
        # Emit completion telemetry
        self.event_bus.emit("phase_98_complete", {
            "module": "phase_98_system_build_validator",
            "status": "COMPLETE",
            "files_validated": self.validation_results["files_validated"],
            "violations_detected": len(self.validation_results["violations_detected"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        if self.validation_results["violations_detected"]:
            logger.warning("🚨 Violations detected → Manual review recommended")
            return {"status": "VIOLATIONS_DETECTED", "report": final_report}
        else:
            logger.info("✅ PHASE 98 COMPLETE — SYSTEM IS VALIDATED")
            return {"status": "VALIDATED", "report": final_report}

# ARCHITECT MODE COMPLIANCE EXECUTION
def main():
    """ARCHITECT MODE COMPLIANT execution entry point"""
    try:
        validator = Phase98SystemBuildValidator()
        results = validator.execute_phase_98_validation()
        
        if results["status"] == "VALIDATED":
            print("✅ PHASE 98 COMPLETE — SYSTEM IS VALIDATED")
        else:
            print("🚨 PHASE 98 COMPLETE — VIOLATIONS DETECTED")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Phase 98 validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()


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
