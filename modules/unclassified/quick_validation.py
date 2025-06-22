import logging
# <!-- @GENESIS_MODULE_START: quick_validation -->

"""


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

GENESIS Quick System Validation - BULLETPROOF VERSION
====================================================
This bypasses all blocking operations and just validates system integrity.
NO EVENT LOOPS, NO WAITING, NO SUBSCRIPTIONS - JUST VALIDATION
"""

import json
import os
import sys
from datetime import datetime

from hardened_event_bus import EventBus, Event

def validate_core_files():
    """Validate core system files exist and are readable"""
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "validation_status": "RUNNING",
        "files_checked": {},
        "system_status": "UNKNOWN"
    }
    
    core_files = [
        "build_status.json",
        "build_tracker.md", 
        "system_tree.json",
        "event_bus.json",
        "module_registry.json",
        "telemetry.json"
    ]
    
    print("🔍 GENESIS SYSTEM VALIDATION - QUICK CHECK")
    print("=" * 50)
    
    all_files_ok = True
    
    for file_name in core_files:
        try:
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    if file_name.endswith('.json'):
                        data = json.load(f)
                        results["files_checked"][file_name] = {
                            "exists": True,
                            "readable": True,
                            "valid_json": True,
                            "size_bytes": os.path.getsize(file_name)
                        }
                        print(f"✓ {file_name}: OK (JSON valid, {os.path.getsize(file_name)} bytes)")
                    else:
                        content = f.read()
                        results["files_checked"][file_name] = {
                            "exists": True,
                            "readable": True,
                            "size_bytes": len(content)
                        }
                        print(f"✓ {file_name}: OK ({len(content)} chars)")
            else:
                results["files_checked"][file_name] = {
                    "exists": False,
                    "readable": False
                }
                print(f"✗ {file_name}: MISSING")
                all_files_ok = False
                
        except Exception as e:
            results["files_checked"][file_name] = {
                "exists": True,
                "readable": False,
                "error": str(e)
            }
            print(f"✗ {file_name}: ERROR - {str(e)}")
            all_files_ok = False
    
    return results, all_files_ok

def validate_module_files():
    """Validate key module files exist"""
    modules = [
        "event_bus.py",
        "smart_execution_monitor.py", 
        "signal_engine.py",
        "signal_validator.py",
        "risk_engine.py",
        "execution_engine.py"
    ]
    
    print("\n🔍 MODULE FILES VALIDATION")
    print("=" * 30)
    
    module_status = {}
    all_modules_ok = True
    
    for module in modules:
        try:
            if os.path.exists(module):
                size = os.path.getsize(module)
                module_status[module] = {
                    "exists": True,
                    "size_bytes": size
                }
                print(f"✓ {module}: OK ({size} bytes)")
            else:
                module_status[module] = {"exists": False}
                print(f"✗ {module}: MISSING")
                all_modules_ok = False
        except Exception as e:
            module_status[module] = {"exists": True, "error": str(e)}
            print(f"✗ {module}: ERROR - {str(e)}")
            all_modules_ok = False
    
    return module_status, all_modules_ok

def check_build_status():
    """Check build status without importing anything"""
    try:
        with open('build_status.json', 'r') as f:
            build_data = json.load(f)
        
        print("\n🔍 BUILD STATUS CHECK")
        print("=" * 25)
        
        key_indicators = [
            ("real_data_passed", "Real Data"),
            ("compliance_ok", "Compliance"),
            ("architect_mode", "Architect Mode"),
            ("system_health", "System Health"),
            ("permanent_directive_status", "Directive Status")
        ]
        
        status_ok = True
        for key, label in key_indicators:
            value = build_data.get(key, "UNKNOWN")
            if value in [True, "ACTIVE", "ENABLED", "FULLY_COMPLIANT"]:
                print(f"✓ {label}: {value}")
            else:
                print(f"⚠ {label}: {value}")
                if key in ["real_data_passed", "compliance_ok"]:
                    status_ok = False
        
        total_modules = build_data.get("total_modules", 0)
        modules_connected = len(build_data.get("modules_connected", []))
        
        print(f"\n📊 MODULES: {modules_connected}/{total_modules} connected")
        
        return build_data, status_ok
        
    except Exception as e:
        print(f"✗ Build Status Check Failed: {str(e)}")
        raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed"), False

def main():
    """Main validation function - NO BLOCKING OPERATIONS"""
    print("GENESIS QUICK VALIDATION STARTING...")
    print("This validation bypasses all blocking operations")
    print("=" * 60)
    
    try:
        # Validate core files
        file_results, files_ok = validate_core_files()
        
        # Validate modules
        module_results, modules_ok = validate_module_files()
        
        # Check build status
        build_data, build_ok = check_build_status()
        
        # Final assessment
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 20)
        
        overall_status = "PASS" if (files_ok and modules_ok and build_ok) else "ISSUES_DETECTED"
        
        print(f"Core Files: {'✓ PASS' if files_ok else '✗ ISSUES'}")
        print(f"Module Files: {'✓ PASS' if modules_ok else '✗ ISSUES'}")
        print(f"Build Status: {'✓ PASS' if build_ok else '✗ ISSUES'}")
        print(f"\nOVERALL STATUS: {overall_status}")
        
        if overall_status == "PASS":
            print("\n🎉 GENESIS SYSTEM VALIDATION SUCCESSFUL!")
            print("System files are intact and build status is compliant.")
            return 0
        else:
            print("\n⚠️ SYSTEM ISSUES DETECTED")
            print("Some components need attention.")
            return 1
            
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nValidation completed with exit code: {exit_code}")
    sys.exit(exit_code)


# <!-- @GENESIS_MODULE_END: quick_validation -->


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
