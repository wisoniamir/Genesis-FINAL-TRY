import logging
# <!-- @GENESIS_MODULE_START: validate_signal_fusion_matrix -->

from datetime import datetime\nfrom event_bus import EventBus

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

                emit_telemetry("validate_signal_fusion_matrix", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("validate_signal_fusion_matrix", "position_calculated", {
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
                            "module": "validate_signal_fusion_matrix",
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
                    print(f"Emergency stop error in validate_signal_fusion_matrix: {e}")
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
                    "module": "validate_signal_fusion_matrix",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("validate_signal_fusion_matrix", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in validate_signal_fusion_matrix: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
"""
🧪 GENESIS Phase 34: Signal Fusion Matrix Validation Test v1.0.0
ARCHITECT MODE COMPLIANT | SYSTEM INTEGRATION TEST

This test validates the SignalFusionMatrix module registration and basic functionality.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_signal_fusion_matrix_registration():
    """Test that SignalFusionMatrix is properly registered in all system files"""
    print("🔍 Testing SignalFusionMatrix system registration...")
    
    results = {
        "module_exists": False,
        "build_status_registered": False,
        "module_registry_registered": False,
        "system_tree_registered": False,
        "event_bus_registered": False,
        "telemetry_registered": False
    }
    
    # Test 1: Module file exists
    if os.path.exists("signal_fusion_matrix.py"):
        results["module_exists"] = True
        print("✅ signal_fusion_matrix.py exists")
    else:
        print("❌ signal_fusion_matrix.py not found")
        return results
    
    # Test 2: Check build_status.json
    try:
        with open("build_status.json", "r") as f:
            build_status = json.load(f)
            if "phase_34" in build_status and "SignalFusionMatrix" in str(build_status):
                results["build_status_registered"] = True
                print("✅ SignalFusionMatrix registered in build_status.json")
            else:
                print("❌ SignalFusionMatrix not found in build_status.json")
    except Exception as e:
        print(f"❌ Error reading build_status.json: {e}")
    
    # Test 3: Check module_registry.json  
    try:
        with open("module_registry.json", "r") as f:
            module_registry = json.load(f)
            module_names = [m.get("name", "") for m in module_registry.get("modules", [])]
            if "SignalFusionMatrix" in module_names:
                results["module_registry_registered"] = True
                print("✅ SignalFusionMatrix registered in module_registry.json")
            else:
                print("❌ SignalFusionMatrix not found in module_registry.json")
    except Exception as e:
        print(f"❌ Error reading module_registry.json: {e}")
    
    # Test 4: Check system_tree.json
    try:
        with open("system_tree.json", "r") as f:
            system_tree = json.load(f)
            node_ids = [n.get("id", "") for n in system_tree.get("nodes", [])]
            if "SignalFusionMatrix" in node_ids:
                results["system_tree_registered"] = True
                print("✅ SignalFusionMatrix registered in system_tree.json")
            else:
                print("❌ SignalFusionMatrix not found in system_tree.json")
    except Exception as e:
        print(f"❌ Error reading system_tree.json: {e}")
    
    # Test 5: Check event_bus.json
    try:
        with open("event_bus.json", "r") as f:
            event_bus = json.load(f)
            if "phase_34_signal_fusion_matrix_routes_added" in event_bus.get("metadata", {}):
                results["event_bus_registered"] = True
                print("✅ SignalFusionMatrix routes registered in event_bus.json")
            else:
                print("❌ SignalFusionMatrix routes not found in event_bus.json")
    except Exception as e:
        print(f"❌ Error reading event_bus.json: {e}")
    
    # Test 6: Check telemetry.json
    try:
        with open("telemetry.json", "r") as f:
            telemetry = json.load(f)
            telemetry_str = str(telemetry)
            if "fusion_matrix" in telemetry_str:
                results["telemetry_registered"] = True
                print("✅ SignalFusionMatrix telemetry registered in telemetry.json")
            else:
                print("❌ SignalFusionMatrix telemetry not found in telemetry.json")
    except Exception as e:
        print(f"❌ Error reading telemetry.json: {e}")
    
    return results

def test_signal_fusion_matrix_import():
    """Test that SignalFusionMatrix can be imported successfully"""
    print("\n🔍 Testing SignalFusionMatrix import...")
    
    try:
        from signal_fusion_matrix import SignalFusionMatrix, MultiStrategySignal, WeightVector, FusedSignal
        print("✅ Successfully imported SignalFusionMatrix classes")
        
        # Test instantiation
        fusion_matrix = SignalFusionMatrix()
        print("✅ Successfully instantiated SignalFusionMatrix")
        
        # Test basic attributes
        if hasattr(fusion_matrix, 'event_bus'):
            print("✅ SignalFusionMatrix has event_bus attribute")
        if hasattr(fusion_matrix, 'signal_buffer'):
            print("✅ SignalFusionMatrix has signal_buffer attribute")
        if hasattr(fusion_matrix, 'weight_vectors'):
            print("✅ SignalFusionMatrix has weight_vectors attribute")
            
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_signal_fusion_matrix_basic_functionality():
    """Test basic SignalFusionMatrix functionality"""
    print("\n🔍 Testing SignalFusionMatrix basic functionality...")
    
    try:
        from signal_fusion_matrix import SignalFusionMatrix, MultiStrategySignal
        
        # Create instance
        fusion_matrix = SignalFusionMatrix()
        
        # Test start/stop
        fusion_matrix.start()
        print("✅ SignalFusionMatrix started successfully")
        
        time.sleep(0.1)  # Allow initialization
        
        fusion_matrix.stop()
        print("✅ SignalFusionMatrix stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def run_validation_suite():
    """Run the complete validation suite"""
    print("🚀 GENESIS PHASE 34: SIGNAL FUSION MATRIX VALIDATION")
    print("=" * 60)
    print("🔐 ARCHITECT MODE COMPLIANT | SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Registration
    print("\n📋 TEST 1: SYSTEM REGISTRATION")
    print("-" * 40)
    registration_results = test_signal_fusion_matrix_registration()
    registration_passed = all(registration_results.values())
    
    # Test 2: Import
    print("\n📦 TEST 2: MODULE IMPORT")
    print("-" * 40)
    import_passed = test_signal_fusion_matrix_import()
    
    # Test 3: Basic functionality
    print("\n⚙️ TEST 3: BASIC FUNCTIONALITY")
    print("-" * 40)
    functionality_passed = test_signal_fusion_matrix_basic_functionality()
    
    # Overall results
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = 3
    passed_tests = sum([registration_passed, import_passed, functionality_passed])
    
    print(f"✅ Registration Test: {'PASSED' if registration_passed else 'FAILED'}")
    print(f"✅ Import Test: {'PASSED' if import_passed else 'FAILED'}")
    print(f"✅ Functionality Test: {'PASSED' if functionality_passed else 'FAILED'}")
    print(f"\n📈 Overall Score: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATION TESTS PASSED")
        print("✅ Phase 34 SignalFusionMatrix is ready for production")
        return True
    else:
        print("\n⚠️ SOME VALIDATION TESTS FAILED")
        print("🔧 Phase 34 SignalFusionMatrix requires fixes")
        return False

if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)


# <!-- @GENESIS_MODULE_END: validate_signal_fusion_matrix -->