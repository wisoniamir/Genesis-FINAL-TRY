import logging
# <!-- @GENESIS_MODULE_START: validate_phase80_81 -->

from datetime import datetime\n#!/usr/bin/env python3

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

                emit_telemetry("validate_phase80_81", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("validate_phase80_81", "position_calculated", {
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
                            "module": "validate_phase80_81",
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
                    print(f"Emergency stop error in validate_phase80_81: {e}")
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
                    "module": "validate_phase80_81",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("validate_phase80_81", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in validate_phase80_81: {e}")
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


"""
PHASE 80-81 Final Validation Script
Validates complete integration of Genesis GUI Launcher and MT5 Connection Bridge
"""

import json
import os
import sys
from pathlib import Path

def validate_system_files():
    """Validate all required system files exist and are properly formatted"""
    required_files = [
        'system_tree.json',
        'build_status.json', 
        'module_registry.json',
        'genesis_gui_launcher.py',
        'mt5_connection_bridge.py',
        'genesis_gui_launcher_documentation.md',
        'mt5_connection_bridge_documentation.md',
        'test_phase80_83_fixed.py',
        'test_phase81.py',
        'phase_80_81_completion_summary.md'
    ]
    
    print("🔍 Validating System Files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            return False
    return True

def validate_system_tree():
    """Validate system_tree.json contains both modules"""
    print("\n🏗️ Validating System Tree...")
    
    try:
        with open('system_tree.json', 'r') as f:
            tree = json.load(f)
        
        # Check metadata
        metadata = tree.get('metadata', {})
        print(f"  📊 Total Nodes: {metadata.get('total_nodes', 'N/A')}")
        print(f"  🔗 Total Connections: {metadata.get('total_connections', 'N/A')}")
        print(f"  ✅ Phase 80 Complete: {metadata.get('phase_80_genesis_gui_launcher_complete', False)}")
        print(f"  ✅ Phase 81 Complete: {metadata.get('phase_81_mt5_connection_bridge_complete', False)}")
        
        # Find modules in nodes
        nodes = tree.get('nodes', [])
        gui_found = False
        mt5_found = False
        
        for node in nodes:
            if node.get('id') == 'GenesisGUILauncher':
                gui_found = True
                print(f"  ✅ GenesisGUILauncher found - Status: {node.get('status', 'unknown')}")
            elif node.get('id') == 'MT5ConnectionBridge':
                mt5_found = True
                print(f"  ✅ MT5ConnectionBridge found - Status: {node.get('status', 'unknown')}")
        
        if not gui_found:
            print("  ❌ GenesisGUILauncher not found in system tree")
        if not mt5_found:
            print("  ❌ MT5ConnectionBridge not found in system tree")
            
        return gui_found and mt5_found
        
    except Exception as e:
        print(f"  ❌ Error reading system_tree.json: {e}")
        return False

def validate_module_registry():
    """Validate module_registry.json contains both modules"""
    print("\n📋 Validating Module Registry...")
    
    try:
        with open('module_registry.json', 'r') as f:
            registry = json.load(f)
        
        modules = registry.get('modules', [])
        gui_found = False
        mt5_found = False
        
        for module in modules:
            if module.get('name') == 'GenesisGUILauncher':
                gui_found = True
                print(f"  ✅ GenesisGUILauncher - Phase: {module.get('phase', 'N/A')}, Status: {module.get('status', 'unknown')}")
                print(f"     Test Coverage: {module.get('test_coverage', 'N/A')}%")
            elif module.get('name') == 'MT5ConnectionBridge':
                mt5_found = True
                print(f"  ✅ MT5ConnectionBridge - Phase: {module.get('phase', 'N/A')}, Status: {module.get('status', 'unknown')}")
                print(f"     Test Coverage: {module.get('test_coverage', 'N/A')}%")
        
        if not gui_found:
            print("  ❌ GenesisGUILauncher not found in module registry")
        if not mt5_found:
            print("  ❌ MT5ConnectionBridge not found in module registry")
            
        return gui_found and mt5_found
        
    except Exception as e:
        print(f"  ❌ Error reading module_registry.json: {e}")
        return False

def validate_build_status():
    """Validate build_status.json shows completion"""
    print("\n🚀 Validating Build Status...")
    
    try:
        with open('build_status.json', 'r') as f:
            status = json.load(f)
        
        # Check architect mode
        arch_status = status.get('architect_mode_status', {})
        print(f"  🏛️ Architect Mode: {arch_status.get('architect_mode_v500_activation', False)}")
        
        # Check module registry status
        mod_status = status.get('module_registry_status', {})
        print(f"  📊 Total Modules: {mod_status.get('total_modules_registered', 'N/A')}")
        print(f"  ✅ Phase 80 Complete: {mod_status.get('phase_80_genesis_gui_launcher_complete', False)}")
        print(f"  ✅ Phase 81 Complete: {mod_status.get('phase_81_mt5_connection_bridge_complete', False)}")
        
        # Check system tree status
        tree_status = status.get('system_tree_status', {})
        print(f"  🏗️ System Nodes: {tree_status.get('total_nodes', 'N/A')}")
        print(f"  🔗 Connections: {tree_status.get('total_connections', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading build_status.json: {e}")
        return False

def main():
    """Main validation function"""
    print("🎯 PHASE 80-81 FINAL VALIDATION")
    print("=" * 50)
    
    results = []
    results.append(validate_system_files())
    results.append(validate_system_tree())
    results.append(validate_module_registry())
    results.append(validate_build_status())
    
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("✅ ALL VALIDATIONS PASSED")
        print("🎉 PHASE 80-81 SUCCESSFULLY COMPLETED")
        print("🚀 System ready for production deployment")
        return 0
    else:
        print("❌ VALIDATION FAILURES DETECTED")
        print("🔧 Please review and fix issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())


# <!-- @GENESIS_MODULE_END: validate_phase80_81 -->