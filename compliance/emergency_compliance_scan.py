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

                emit_telemetry("emergency_compliance_scan", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("emergency_compliance_scan", "position_calculated", {
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
                            "module": "emergency_compliance_scan",
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
                    print(f"Emergency stop error in emergency_compliance_scan: {e}")
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
                    "module": "emergency_compliance_scan",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("emergency_compliance_scan", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in emergency_compliance_scan: {e}")
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
🚨 ARCHITECT COMPLIANCE EMERGENCY SCAN - IMMEDIATE EXECUTION
"""

import json
import os
from datetime import datetime
from pathlib import Path


# <!-- @GENESIS_MODULE_END: emergency_compliance_scan -->


# <!-- @GENESIS_MODULE_START: emergency_compliance_scan -->

def emergency_compliance_scan():
    """Execute emergency compliance scan"""
    print("🚨 ARCHITECT COMPLIANCE EMERGENCY SCAN INITIATED")
    print("="*70)
    
    # Load system_tree.json
    system_tree_path = Path("system_tree.json")
    if system_tree_path.exists():
        with open(system_tree_path, 'r', encoding='utf-8') as f:
            system_tree = json.load(f)
        
        registered_modules = set(system_tree.get("modules", {}).keys())
        print(f"📊 Registered modules in system_tree.json: {len(registered_modules)}")
    else:
        print("❌ CRITICAL: system_tree.json NOT FOUND")
        registered_modules = set()
    
    # Scan Python files in workspace root only
    python_files = list(Path(".").glob("*.py"))
    orphaned_modules = []
    
    for py_file in python_files:
        module_name = py_file.stem
        if module_name not in registered_modules and module_name != "__init__":
            orphaned_modules.append(module_name)
    
    print(f"⚠️ ORPHANED MODULES DETECTED (workspace root): {len(orphaned_modules)}")
    
    # Show first 15 orphaned modules
    if orphaned_modules:
        print("🔧 ORPHANED MODULES (first 15):")
        for module in orphaned_modules[:15]:
            print(f"   - ❌ {module}: Not registered in system_tree.json")
        
        if len(orphaned_modules) > 15:
            print(f"   ... and {len(orphaned_modules) - 15} more orphaned modules")
    
    # Check EventBus routes
    event_bus_path = Path("event_bus.json")
    if event_bus_path.exists():
        with open(event_bus_path, 'r', encoding='utf-8') as f:
            event_bus = json.load(f)
        
        routes = event_bus.get("routes", {})
        print(f"📊 EventBus routes registered: {len(routes)}")
    else:
        print("❌ CRITICAL: event_bus.json NOT FOUND")
        routes = {}
    
    # Check for isolated modules (no EventBus routes)
    isolated_modules = []
    for py_file in python_files:
        module_name = py_file.stem
        expected_route = f"{module_name}_events"
        if expected_route not in routes:
            isolated_modules.append(module_name)
    
    print(f"⚠️ ISOLATED MODULES (NO EVENTBUS): {len(isolated_modules)}")
    
    # Update build_status.json
    build_status_path = Path("build_status.json")
    if build_status_path.exists():
        with open(build_status_path, 'r', encoding='utf-8') as f:
            build_status = json.load(f)
    else:
        build_status = {}
    
    total_violations = len(orphaned_modules) + len(isolated_modules)
    
    build_status.update({
        "architect_compliance_emergency_scan": {
            "timestamp": datetime.now().isoformat(),
            "total_violations": total_violations,
            "orphaned_modules_count": len(orphaned_modules),
            "isolated_modules_count": len(isolated_modules),
            "compliance_status": "VIOLATIONS_DETECTED" if total_violations > 0 else "COMPLIANT",
            "emergency_repair_required": total_violations > 0,
            "architect_lock_status": "BROKEN" if total_violations > 50 else "MONITORING"
        }
    })
    
    with open(build_status_path, 'w', encoding='utf-8') as f:
        json.dump(build_status, f, indent=2)
    
    # Update build_tracker.md
    build_tracker_path = Path("build_tracker.md")
    
    progress_entry = f"""

## 🚨 ARCHITECT COMPLIANCE EMERGENCY SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 🔧 MODULE REPAIR REQUIRED - IMMEDIATE ACTION

**TOTAL VIOLATIONS DETECTED: {total_violations}**

#### 🗂 ORPHANED MODULES ({len(orphaned_modules)}):
"""
    
    for module in orphaned_modules[:10]:  # Show first 10
        progress_entry += f"- ❌ **{module}**: Not registered in system_tree.json → register_in_system_tree\n"
    
    if len(orphaned_modules) > 10:
        progress_entry += f"- ... and {len(orphaned_modules) - 10} more orphaned modules\n"
    
    progress_entry += f"""
#### 🔗 DISCONNECTED MODULES ({len(isolated_modules)}):
"""
    
    for module in isolated_modules[:10]:  # Show first 10
        progress_entry += f"- ❌ **{module}**: No EventBus route → create_eventbus_route\n"
    
    progress_entry += f"""
### 🔧 REPAIR STATUS:
- **Emergency Repair Required**: {'YES' if total_violations > 0 else 'NO'}
- **System Compliance**: {'FAILED' if total_violations > 0 else 'PASSED'}
- **Guardian Action**: {'IMMEDIATE_INTERVENTION_REQUIRED' if total_violations > 50 else 'MONITORING'}
- **Architect Lock Status**: {'BROKEN' if total_violations > 50 else 'MONITORING'}

"""
    
    # Append to build tracker
    with open(build_tracker_path, 'a', encoding='utf-8') as f:
        f.write(progress_entry)
    
    print("="*70)
    print("🚨 ARCHITECT COMPLIANCE EMERGENCY SCAN COMPLETE")
    print("="*70)
    print(f"📊 VIOLATION SUMMARY:")
    print(f"   Total Violations: {total_violations}")
    print(f"   Orphaned Modules: {len(orphaned_modules)}")
    print(f"   Isolated Modules: {len(isolated_modules)}")
    print(f"   Compliance Status: {'FAILED' if total_violations > 0 else 'PASSED'}")
    print(f"   Emergency Repair: {'REQUIRED' if total_violations > 0 else 'NOT_NEEDED'}")
    
    if total_violations > 0:
        print(f"🚨 ARCHITECT_LOCK_BROKEN - IMMEDIATE REPAIR REQUIRED")
        return False
    else:
        print(f"✅ ARCHITECT_COMPLIANT - All systems operational")
        return True
    
    print("="*70)

if __name__ == "__main__":
    emergency_compliance_scan()
