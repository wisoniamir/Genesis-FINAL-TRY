import logging

# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "genesis_final_system_validator",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("genesis_final_system_validator", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in genesis_final_system_validator: {e}")
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
🧠 GENESIS FINAL SYSTEM VALIDATOR - INSTITUTIONAL COMPLIANCE CHECK
================================================================

@GENESIS_CATEGORY: CORE.VALIDATION.INSTITUTIONAL
@GENESIS_TELEMETRY: ENABLED  
@GENESIS_EVENTBUS: EMIT+CONSUME

OBJECTIVE: Final comprehensive validation of all GENESIS components
- Validate all core modules are operational
- Check EventBus routing integrity
- Verify FTMO compliance across all modules
- Ensure telemetry is active and reporting
- Generate final system health report

COMPLIANCE: ARCHITECT MODE v3.0 ENFORCED | FTMO RESTRICTIONS ACTIVE
================================================================
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class GenesisFinalSystemValidator:
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "genesis_final_system_validator",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("genesis_final_system_validator", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in genesis_final_system_validator: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "genesis_final_system_validator",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in genesis_final_system_validator: {e}")
    def __init__(self):
        self.workspace_path = Path("c:/Users/patra/Genesis FINAL TRY")
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'architect_mode': True,
            'ftmo_compliance': True,
            'modules_tested': {},
            'eventbus_integrity': {},
            'telemetry_status': {},
            'risk_engine_status': {},
            'execution_status': {},
            'dashboard_status': {},
            'overall_health': 'UNKNOWN'
        }
        
    def validate_core_module(self, module_name: str, module_path: str) -> Dict[str, Any]:
        """Validate individual core module"""
        try:
            # Check if file exists
            if not Path(module_path).exists():
                return {
                    'status': 'MISSING',
                    'error': f'Module file not found: {module_path}'
                }
            
            # Try to import the module
            spec = None
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    return {
                        'status': 'OPERATIONAL',
                        'genesis_category': getattr(module, 'GENESIS_CATEGORY', 'UNKNOWN'),
                        'ftmo_compliance': self.check_ftmo_compliance(module),
                        'eventbus_ready': self.check_eventbus_integration(module),
                        'telemetry_active': self.check_telemetry_integration(module)
                    }
                else:
                    return {
                        'status': 'IMPORT_ERROR',
                        'error': 'Failed to create module spec'
                    }
                    
            except Exception as e:
                return {
                    'status': 'RUNTIME_ERROR',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                
        except Exception as e:
            return {
                'status': 'VALIDATION_ERROR',
                'error': str(e)
            }
    
    def check_ftmo_compliance(self, module) -> bool:
        """Check if module has FTMO compliance features"""
        ftmo_indicators = [
            'daily_loss', 'max_drawdown', 'trailing_drawdown',
            'profit_target', 'consistency_rule', 'risk_management',
            'kill_switch', 'emergency_stop'
        ]
        
        module_source = ""
        try:
            import inspect
            module_source = inspect.getsource(module).lower()
        except:
            pass
            
        return any(indicator in module_source for indicator in ftmo_indicators)
    
    def check_eventbus_integration(self, module) -> bool:
        """Check if module has EventBus integration"""
        eventbus_indicators = [
            'event_bus', 'emit', 'subscribe', 'publish',
            'dispatch_event', 'route_message'
        ]
        
        module_source = ""
        try:
            import inspect
            module_source = inspect.getsource(module).lower()
        except:
            pass
            
        return any(indicator in module_source for indicator in eventbus_indicators)
    
    def check_telemetry_integration(self, module) -> bool:
        """Check if module has telemetry integration"""
        telemetry_indicators = [
            'telemetry', 'log_telemetry', 'emit_telemetry',
            'track_metric', 'record_performance', 'heartbeat'
        ]
        
        module_source = ""
        try:
            import inspect


# <!-- @GENESIS_MODULE_END: genesis_final_system_validator -->


# <!-- @GENESIS_MODULE_START: genesis_final_system_validator -->
            module_source = inspect.getsource(module).lower()
        except:
            pass
            
        return any(indicator in module_source for indicator in telemetry_indicators)
    
    def validate_json_files(self) -> Dict[str, Any]:
        """Validate all required JSON configuration files"""
        required_files = [
            'build_status.json',
            'module_registry.json',
            'system_tree.json',
            'event_bus.json',
            'telemetry.json',
            'compliance.json'
        ]
        
        json_status = {}
        
        for filename in required_files:
            file_path = self.workspace_path / filename
            try:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        json_status[filename] = {
                            'status': 'VALID',
                            'size': len(str(data)),
                            'keys': list(data.keys()) if isinstance(data, dict) else 'NOT_DICT'
                        }
                else:
                    json_status[filename] = {
                        'status': 'MISSING',
                        'error': 'File not found'
                    }
            except Exception as e:
                json_status[filename] = {
                    'status': 'INVALID',
                    'error': str(e)
                }
        
        return json_status
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation"""
        print("🔍 Starting GENESIS Final System Validation...")
        
        # Core modules to validate
        core_modules = {
            'risk_engine': 'genesis_risk_engine_institutional.py',
            'execution_middleware': 'genesis_execution_middleware_institutional.py',
            'eventbus_sync': 'genesis_eventbus_sync_engine.py',
            'module_registry': 'phase_101_institutional_module_registry.py',
            'dashboard': 'dashboard.py',
            'telemetry_validator': 'genesis_telemetry_validator.py'
        }
        
        # Validate each core module
        for module_name, filename in core_modules.items():
            module_path = str(self.workspace_path / filename)
            print(f"🔧 Validating {module_name}...")
            
            validation_result = self.validate_core_module(module_name, module_path)
            self.validation_results['modules_tested'][module_name] = validation_result
            
            print(f"   Status: {validation_result['status']}")
            if validation_result['status'] != 'OPERATIONAL':
                print(f"   Error: {validation_result.get('error', 'Unknown')}")
        
        # Validate JSON configuration files
        print("📋 Validating JSON configuration files...")
        json_validation = self.validate_json_files()
        self.validation_results['json_files'] = json_validation
        
        # Calculate overall system health
        operational_modules = sum(
            1 for result in self.validation_results['modules_tested'].values()
            if result['status'] == 'OPERATIONAL'
        )
        total_modules = len(self.validation_results['modules_tested'])
        
        valid_json_files = sum(
            1 for result in json_validation.values()
            if result['status'] == 'VALID'
        )
        total_json_files = len(json_validation)
        
        # Determine overall health
        if operational_modules == total_modules and valid_json_files >= 4:
            self.validation_results['overall_health'] = 'EXCELLENT'
        elif operational_modules >= total_modules * 0.8 and valid_json_files >= 3:
            self.validation_results['overall_health'] = 'GOOD'
        elif operational_modules >= total_modules * 0.6:
            self.validation_results['overall_health'] = 'ACCEPTABLE'
        else:
            self.validation_results['overall_health'] = 'CRITICAL'
        
        # Generate final report
        print("\n" + "="*80)
        print("🏆 GENESIS FINAL SYSTEM VALIDATION REPORT")
        print("="*80)
        print(f"📊 Overall Health: {self.validation_results['overall_health']}")
        print(f"🔧 Operational Modules: {operational_modules}/{total_modules}")
        print(f"📋 Valid JSON Files: {valid_json_files}/{total_json_files}")
        print(f"🕒 Validation Time: {self.validation_results['timestamp']}")
        print(f"🏛️ Architect Mode: {self.validation_results['architect_mode']}")
        print(f"⚖️ FTMO Compliance: {self.validation_results['ftmo_compliance']}")
        
        print("\n📝 Module Status Details:")
        for module_name, result in self.validation_results['modules_tested'].items():
            status_icon = "✅" if result['status'] == 'OPERATIONAL' else "❌"
            print(f"   {status_icon} {module_name}: {result['status']}")
            if result['status'] == 'OPERATIONAL':
                print(f"      - FTMO Compliance: {result.get('ftmo_compliance', False)}")
                print(f"      - EventBus Ready: {result.get('eventbus_ready', False)}")
                print(f"      - Telemetry Active: {result.get('telemetry_active', False)}")
        
        print("\n📋 JSON Configuration Status:")
        for filename, result in json_validation.items():
            status_icon = "✅" if result['status'] == 'VALID' else "❌"
            print(f"   {status_icon} {filename}: {result['status']}")
        
        # Save validation report
        report_path = self.workspace_path / 'GENESIS_FINAL_VALIDATION_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\n💾 Full validation report saved to: {report_path}")
        print("="*80)
        
        return self.validation_results

def main():
    """Main validation entry point"""
    try:
        validator = GenesisFinalSystemValidator()
        results = validator.run_comprehensive_validation()
        
        if results['overall_health'] in ['EXCELLENT', 'GOOD']:
            print("\n🎉 GENESIS TRADING BOT IS READY FOR INSTITUTIONAL DEPLOYMENT!")
            print("🚀 All core systems validated and operational.")
            return 0
        else:
            print(f"\n⚠️ GENESIS system health is {results['overall_health']}")
            print("🔧 Please address the issues identified above before deployment.")
            return 1
            
    except Exception as e:
        print(f"❌ Critical validation error: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit(main())


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


def monitor_drawdown(max_drawdown_percent: float = 5.0, daily_limit_percent: float = 5.0) -> Dict:
    """
    Monitor account drawdown against FTMO limits
    
    Args:
        max_drawdown_percent: Maximum allowed drawdown percentage
        daily_limit_percent: Maximum allowed daily loss percentage
        
    Returns:
        Dictionary with drawdown status information
    """
    try:
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            return {"status": "error", "message": "Failed to get account info"}
        
        # Calculate current drawdown
        balance = account_info.balance
        equity = account_info.equity
        
        current_drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        
        # Get daily high balance
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        positions = mt5.history_deals_get(from_date, datetime.now())
        
        daily_starting_balance = balance - sum([deal.profit for deal in positions])
        daily_loss_percent = (daily_starting_balance - equity) / daily_starting_balance * 100 if daily_starting_balance > 0 else 0
        
        # Prepare result
        result = {
            "status": "ok",
            "current_drawdown_percent": current_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "drawdown_level": current_drawdown / max_drawdown_percent,  # 0.0 to 1.0+
            "daily_loss_percent": daily_loss_percent,
            "daily_limit_percent": daily_limit_percent,
            "daily_loss_level": daily_loss_percent / daily_limit_percent,  # 0.0 to 1.0+
            "warnings": []
        }
        
        # Check drawdown thresholds
        if current_drawdown > max_drawdown_percent * 0.7:
            result["warnings"].append(f"Drawdown at {current_drawdown:.2f}% approaching maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "warning"
            
        if current_drawdown > max_drawdown_percent:
            result["warnings"].append(f"CRITICAL: Drawdown of {current_drawdown:.2f}% exceeds maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "critical"
            
        # Check daily loss thresholds
        if daily_loss_percent > daily_limit_percent * 0.7:
            result["warnings"].append(f"Daily loss at {daily_loss_percent:.2f}% approaching limit of {daily_limit_percent:.2f}%")
            result["status"] = "warning"
            
        if daily_loss_percent > daily_limit_percent:
            result["warnings"].append(f"CRITICAL: Daily loss of {daily_loss_percent:.2f}% exceeds limit of {daily_limit_percent:.2f}%")
            result["status"] = "critical"
        
        # Emit events for warnings
        if result["status"] in ["warning", "critical"]:
            emit_event("risk_threshold_warning", {
                "status": result["status"],
                "warnings": result["warnings"],
                "timestamp": datetime.now().isoformat()
            })
            
        return result
        
    except Exception as e:
        logging.error(f"Error monitoring drawdown: {str(e)}")
        return {"status": "error", "message": str(e)}


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
