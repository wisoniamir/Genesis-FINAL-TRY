
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

                emit_telemetry("test_phase_92_93_validation_recovered_2", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_phase_92_93_validation_recovered_2", "position_calculated", {
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
                            "module": "test_phase_92_93_validation_recovered_2",
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
                    print(f"Emergency stop error in test_phase_92_93_validation_recovered_2: {e}")
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
                    "module": "test_phase_92_93_validation_recovered_2",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_phase_92_93_validation_recovered_2", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_phase_92_93_validation_recovered_2: {e}")
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


# <!-- @GENESIS_MODULE_START: test_phase_92_93_validation -->

#!/usr/bin/env python3
"""
🔐 GENESIS PHASE 92-93 VALIDATION TEST
Integration test for enhanced dashboard with backtesting and comparison modules

🎯 PURPOSE:
Validate that the Phase 92/93 integration is working correctly
- Test dashboard loading with new tabs
- Verify module imports and EventBus integration
- Check error handling and graceful fallbacks
"""

import sys
import os
import logging
import traceback
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phase_92_93_integration():
    """Test the complete Phase 92-93 integration"""
    
    test_results = {
        "dashboard_import": False,
        "backtest_module_import": False,
        "comparison_engine_import": False,
        "phase_92_93_available": False,
        "dashboard_creation": False,
        "error_details": []
    }
    
    print("🔍 GENESIS Phase 92-93 Integration Validation")
    print("=" * 50)
    
    # Test 1: Dashboard UI Import
    try:
        print("📊 Testing dashboard UI import...")
        import genesis_dashboard_ui
        test_results["dashboard_import"] = True
        print("✅ Dashboard UI imported successfully")
        
        # Check if Phase 92/93 modules are available
        if hasattr(genesis_dashboard_ui, 'PHASES_92_93_AVAILABLE'):
            test_results["phase_92_93_available"] = genesis_dashboard_ui.PHASES_92_93_AVAILABLE
            print(f"📡 Phase 92/93 availability: {test_results['phase_92_93_available']}")
        
    except Exception as e:
        test_results["error_details"].append(f"Dashboard import failed: {str(e)}")
        print(f"❌ Dashboard UI import failed: {str(e)}")
    
    # Test 2: Backtest Module Import
    try:
        print("📈 Testing backtest module import...")
        import backtest_dashboard_module
        test_results["backtest_module_import"] = True
        print("✅ Backtest module imported successfully")
        
        # Test create_backtest_panel function
        if hasattr(backtest_dashboard_module, 'create_backtest_panel'):
            print("✅ create_backtest_panel function available")
        else:
            print("⚠️ create_backtest_panel function not found")
            
    except Exception as e:
        test_results["error_details"].append(f"Backtest module import failed: {str(e)}")
        print(f"❌ Backtest module import failed: {str(e)}")
    
    # Test 3: Comparison Engine Import
    try:
        print("📊 Testing comparison engine import...")
        import live_backtest_comparison_engine
        test_results["comparison_engine_import"] = True
        print("✅ Comparison engine imported successfully")
        
        # Test create_comparison_engine function
        if hasattr(live_backtest_comparison_engine, 'create_comparison_engine'):
            print("✅ create_comparison_engine function available")
        else:
            print("⚠️ create_comparison_engine function not found")
            
    except Exception as e:
        test_results["error_details"].append(f"Comparison engine import failed: {str(e)}")
        print(f"❌ Comparison engine import failed: {str(e)}")
    
    # Test 4: Dashboard Creation (without GUI)
    try:
        print("🖥️ Testing dashboard creation (headless)...")
        if test_results["dashboard_import"]:
            # This would normally create the GUI, but we'll just check the class exists
            dashboard_class = genesis_dashboard_ui.GenesisInstitutionalDashboard
            print("✅ Dashboard 
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
        class available for creation")
            test_results["dashboard_creation"] = True
        else:
            print("⚠️ Skipping dashboard creation due to import failure")
            
    except Exception as e:
        test_results["error_details"].append(f"Dashboard creation test failed: {str(e)}")
        print(f"❌ Dashboard creation test failed: {str(e)}")
    
    # Test Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY:")
    print("=" * 50)
    
    total_tests = 4
    passed_tests = sum([
        test_results["dashboard_import"],
        test_results["backtest_module_import"], 
        test_results["comparison_engine_import"],
        test_results["dashboard_creation"]
    ])
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"📡 Phase 92/93 Available: {test_results['phase_92_93_available']}")
    
    if test_results["error_details"]:
        print("\n⚠️ ERRORS DETECTED:")
        for error in test_results["error_details"]:
            print(f"  - {error}")
    
    if passed_tests == total_tests:
        print("\n🎉 PHASE 92-93 INTEGRATION: FULLY OPERATIONAL")
        print("✅ Ready for production use with enhanced dashboard")
        print("🎯 Launch command: python genesis_dashboard_ui.py")
        return True
    else:
        print(f"\n⚠️ PHASE 92-93 INTEGRATION: PARTIAL SUCCESS ({success_rate:.1f}%)")
        print("⚠️ Some components may not be fully functional")
        return False

def test_file_existence():
    """Test that all required files exist"""
    print("\n📁 FILE EXISTENCE CHECK:")
    print("-" * 30)
    
    required_files = [
        "genesis_dashboard_ui.py",
        "backtest_dashboard_module.py", 
        "live_backtest_comparison_engine.py",
        "telemetry.json",
        "module_registry.json",
        "build_status.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print(f"🕐 Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run file existence check
    files_ok = test_file_existence()
    
    # Run integration tests
    integration_ok = test_phase_92_93_integration()
    
    print(f"\n🕐 Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if files_ok and integration_ok:
        print("\n🚀 VALIDATION RESULT: SUCCESS - READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n⚠️ VALIDATION RESULT: ISSUES DETECTED - REVIEW REQUIRED")
        sys.exit(1)


# <!-- @GENESIS_MODULE_END: test_phase_92_93_validation -->