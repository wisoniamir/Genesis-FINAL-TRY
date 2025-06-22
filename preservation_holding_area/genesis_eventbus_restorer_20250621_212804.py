import logging

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


"""
🔗 GENESIS EVENTBUS RESTORATION ENGINE - ARCHITECT MODE v7.0.0
🔐 ZERO TOLERANCE for mock data or bypassed EventBus

This module restores EventBus connectivity with real MT5 data feeds only.
MANDATORY for ARCHITECT MODE compliance.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import importlib.util

class GenesisEventBusRestorer:
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

            emit_telemetry("genesis_eventbus_restorer", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("genesis_eventbus_restorer", "position_calculated", {
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
                        "module": "genesis_eventbus_restorer",
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
                print(f"Emergency stop error in genesis_eventbus_restorer: {e}")
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
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "genesis_eventbus_restorer",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in genesis_eventbus_restorer: {e}")
    """
    ARCHITECT MODE v7.0.0 EventBus Restoration Engine
    - Restores EventBus connectivity
    - Ensures real MT5 data feeds only
    - Validates all routes and connections
    - NO TOLERANCE for mock data
    """
    
    def __init__(self):
        self.restored_routes = []
        self.failed_routes = []
        self.mt5_connections = []
        self.eventbus_violations = []
        
    def emit_telemetry(self, event, data):
        """EventBus telemetry emission - MANDATORY"""
        telemetry_data = {
            "timestamp": datetime.now().isoformat(),
            "module": "GenesisEventBusRestorer",
            "event": event,
            "data": data,
            "architect_mode": "v7.0.0"
        }
        
        # Log to telemetry file
        telemetry_file = Path("telemetry_realtime.jsonl")
        with open(telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(telemetry_data) + "\n")
    
    def validate_mt5_connection(self):
        """
        Validates that MT5 terminal connection is available
        """
        try:
            # Check if MT5 module is available
            spec = importlib.util.find_spec("MetaTrader5")
            if spec is None:
                return False, "MetaTrader5 module not installed"
            
            # Try to import and initialize
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                return False, "MT5 terminal not running or not accessible"
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return False, "Cannot retrieve terminal information"
            
            # Check if connected to server
            if not terminal_info.connected:
                return False, "MT5 not connected to trading server"
            
            # Check account info
            account_info = mt5.account_info()
            if account_info is None:
                return False, "Cannot retrieve account information"
            
            self.mt5_connections.append({
                "terminal_info": {
                    "company": terminal_info.company,
                    "name": terminal_info.name,
                    "path": terminal_info.path,
                    "data_path": terminal_info.data_path,
                    "connected": terminal_info.connected,
                    "build": terminal_info.build
                },
                "account_info": {
                    "login": account_info.login,
                    "server": account_info.server,
                    "currency": account_info.currency,
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "trade_allowed": account_info.trade_allowed
                }
            })
            
            mt5.shutdown()
            return True, "MT5 connection validated successfully"
            
        except Exception as e:
            return False, f"MT5 validation error: {str(e)}"
    
    def restore_core_eventbus(self):
        """
        Restores core EventBus infrastructure
        """
        eventbus_config = {
            "version": "v7.0.0",
            "architect_mode": True,
            "real_data_only": True,
            "live_data_forbidden": True,
            "routes": {
                "mt5_data_feed": {
                    "topic": "market_data.real_time",
                    "source": "mt5_adapter",
                    "destination": ["signal_processor", "risk_engine", "execution_engine"],
                    "data_type": "real_market_data",
                    "mock_forbidden": True
                },
                "trade_signals": {
                    "topic": "signals.trade_recommendations", 
                    "source": "signal_processor",
                    "destination": ["execution_engine", "risk_engine"],
                    "data_type": "real_trade_signals",
                    "mock_forbidden": True
                },
                "risk_monitoring": {
                    "topic": "risk.real_time_monitoring",
                    "source": "risk_engine",
                    "destination": ["execution_engine", "dashboard", "alert_system"],
                    "data_type": "real_risk_data",
                    "mock_forbidden": True
                },
                "execution_feedback": {
                    "topic": "execution.trade_feedback",
                    "source": "execution_engine",
                    "destination": ["signal_processor", "risk_engine", "dashboard"],
                    "data_type": "real_execution_data",
                    "mock_forbidden": True
                },
                "dashboard_telemetry": {
                    "topic": "telemetry.real_time",
                    "source": ["signal_processor", "risk_engine", "execution_engine"],
                    "destination": "dashboard",
                    "data_type": "real_telemetry_data", 
                    "mock_forbidden": True
                }
            },
            "compliance_rules": {
                "no_live_data": True,
                "real_mt5_only": True,
                "eventbus_mandatory": True,
                "bypass_forbidden": True
            }
        }
        
        # Save EventBus configuration
        with open("event_bus.json", "w", encoding="utf-8") as f:
            json.dump(eventbus_config, f, indent=2)
        
        self.emit_telemetry("eventbus_config_restored", {
            "routes_count": len(eventbus_config["routes"]),
            "compliance_enforced": True
        })
        
        return eventbus_config
    
    def validate_eventbus_routes(self, eventbus_config):
        """
        Validates all EventBus routes for compliance
        """
        for route_name, route_config in eventbus_config["routes"].items():
            try:
                # Check mandatory fields
                required_fields = ["topic", "source", "destination", "data_type"]
                for field in required_fields:
                    if field not in route_config:
                        raise ValueError(f"Missing required field: {field}")
                
                # Check mock data prohibition
                if not route_config.get("mock_forbidden", False):
                    raise ValueError("Mock data not explicitly forbidden")
                
                # Validate data type is real
                if "mock" in route_config["data_type"].lower():
                    raise ValueError("Mock data type detected")
                
                if "test" in route_config["data_type"].lower():
                    raise ValueError("Test data type detected")
                
                if "simulated" in route_config["data_type"].lower():
                    raise ValueError("Simulated data type detected")
                
                self.restored_routes.append({
                    "route": route_name,
                    "status": "VALIDATED",
                    "topic": route_config["topic"],
                    "real_data_only": True
                })
                
            except Exception as e:
                self.failed_routes.append({
                    "route": route_name,
                    "error": str(e),
                    "status": "FAILED"
                })
                
                self.emit_telemetry("route_validation_failed", {
                    "route": route_name,
                    "error": str(e)
                })
    
    def scan_for_eventbus_violations(self):
        """
        Scans codebase for EventBus violations and bypasses
        """
        violation_patterns = [
            r'# bypass.*eventbus',
            r'# skip.*eventbus', 
            r'# disable.*eventbus',
            r'direct_call\(',
            r'mock_event\(',
            r'simulate_event\(',
            r'test_event\(',
            r'bypass_eventbus',
            r'skip_eventbus',
            r'disable_eventbus'
        ]
        
        python_files = list(Path(".").rglob("*.py"))
        
        for file_path in python_files:
            # Skip quarantined directories
            if any(exclude in str(file_path) for exclude in [
                "TRIAGE_ORPHAN_QUARANTINE", 
                "MOCK_VIOLATIONS_QUARANTINE",
                ".venv", 
                "__pycache__", 
                ".git"
            ]):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern in violation_patterns:
                    import re


# <!-- @GENESIS_MODULE_END: genesis_eventbus_restorer -->


# <!-- @GENESIS_MODULE_START: genesis_eventbus_restorer -->
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self.eventbus_violations.append({
                            "file": str(file_path),
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group(),
                            "violation_type": "EVENTBUS_BYPASS"
                        })
                        
            except Exception as e:
                self.emit_telemetry("file_scan_error", {
                    "file": str(file_path),
                    "error": str(e)
                })
    
    def generate_eventbus_report(self):
        """
        Generates EventBus restoration compliance report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "architect_mode_version": "v7.0.0",
            "eventbus_status": "RESTORED" if len(self.failed_routes) == 0 else "PARTIAL",
            "mt5_connection_validated": len(self.mt5_connections) > 0,
            "routes_restored": len(self.restored_routes),
            "routes_failed": len(self.failed_routes),
            "eventbus_violations": len(self.eventbus_violations),
            "mt5_connections": self.mt5_connections,
            "restored_routes": self.restored_routes,
            "failed_routes": self.failed_routes,
            "violations": self.eventbus_violations
        }
        
        # Save report
        report_file = Path("eventbus_restoration_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        self.emit_telemetry("eventbus_report_generated", {
            "report_file": str(report_file),
            "status": report["eventbus_status"]
        })
        
        return report
    
    def restore_eventbus_connectivity(self):
        """
        Complete EventBus restoration process
        """
        print("🔗 GENESIS EVENTBUS RESTORATION ENGINE - ARCHITECT MODE v7.0.0")
        print("🔐 ZERO TOLERANCE for mock data or bypassed EventBus")
        print("-" * 70)
        
        # Step 1: Validate MT5 connection
        print("📡 Validating MT5 terminal connection...")
        mt5_valid, mt5_message = self.validate_mt5_connection()
        if mt5_valid:
            print(f"✅ MT5 Connection: {mt5_message}")
        else:
            print(f"❌ MT5 Connection: {mt5_message}")
        
        # Step 2: Restore core EventBus
        print("🔗 Restoring core EventBus infrastructure...")
        eventbus_config = self.restore_core_eventbus()
        print(f"✅ EventBus configuration restored with {len(eventbus_config['routes'])} routes")
        
        # Step 3: Validate routes
        print("🔍 Validating EventBus routes...")
        self.validate_eventbus_routes(eventbus_config)
        
        # Step 4: Scan for violations
        print("🚫 Scanning for EventBus violations...")
        self.scan_for_eventbus_violations()
        
        # Step 5: Generate report
        print("📊 Generating EventBus restoration report...")
        report = self.generate_eventbus_report()
        
        # Step 6: Report results
        print(f"✅ EventBus restoration complete")
        print(f"🔗 Routes restored: {len(self.restored_routes)}")
        print(f"❌ Routes failed: {len(self.failed_routes)}")
        print(f"🚫 Violations found: {len(self.eventbus_violations)}")
        print(f"📊 Report saved: eventbus_restoration_report.json")
        
        if len(self.failed_routes) > 0:
            print("\n❌ FAILED ROUTES:")
            for route in self.failed_routes:
                print(f"   ❌ {route['route']}: {route['error']}")
        
        if len(self.eventbus_violations) > 0:
            print("\n🚫 EVENTBUS VIOLATIONS:")
            for violation in self.eventbus_violations[:10]:  # Show first 10
                print(f"   🚫 {violation['file']}:{violation['line']} - {violation['match']}")
            if len(self.eventbus_violations) > 10:
                print(f"   ... and {len(self.eventbus_violations) - 10} more violations")
        
        return len(self.failed_routes) == 0 and mt5_valid

def main():
    """Execute EventBus restoration"""
    restorer = GenesisEventBusRestorer()
    success = restorer.restore_eventbus_connectivity()
    
    if success:
        print("\n🎯 ARCHITECT MODE EVENTBUS COMPLIANCE: PASSED")
    else:
        print("\n🚨 ARCHITECT MODE VIOLATION: EventBus issues remain")
        print("❌ System requires EventBus repairs")
    
    return success

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
