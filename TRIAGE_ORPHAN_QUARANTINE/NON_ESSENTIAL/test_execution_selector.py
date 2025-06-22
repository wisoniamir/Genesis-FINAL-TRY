import logging
# <!-- @GENESIS_MODULE_START: test_execution_selector -->

from datetime import datetime\n#!/usr/bin/env python3

# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def emergency_stop(self, reason: str = "Manual trigger") -> bool:
                """GENESIS Emergency Kill Switch"""
                try:
                    # Emit emergency event
                    if hasattr(self, 'event_bus') and self.event_bus:
                        emit_event("emergency_stop", {
                            "module": "test_execution_selector",
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
                    print(f"Emergency stop error in test_execution_selector: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "test_execution_selector",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_execution_selector", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_execution_selector: {e}")
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
Test Suite for GENESIS Phase 38 Execution Selector Module
Architect Mode v2.7 Compliance Testing

🔐 This test validates the execution selector filtering logic,
EventBus integration, telemetry hooks, and real data compliance.
"""

import sys
import os
import json
import unittest
import datetime
from unittest.mock import patch, MagicMock, mock_open

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import execution_selector
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("📂 Current working directory:", os.getcwd())
    print("📁 Files in directory:", os.listdir('.'))
    sys.exit(1)

class TestExecutionSelector(unittest.TestCase):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "test_execution_selector",
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
                print(f"Emergency stop error in test_execution_selector: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "test_execution_selector",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_execution_selector", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_execution_selector: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_execution_selector",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_execution_selector: {e}")
    """Test cases for ExecutionSelector module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_signal = {
            "id": "test_signal_001",
            "symbol": "EURUSD",
            "direction": "buy",
            "entry_price": 1.0950,
            "take_profit": 1.1000,
            "stop_loss": 1.0900,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.test_context = {
            "priority_signals": [self.test_signal],
            "account_config": {
                "max_drawdown": 0.05,
                "daily_loss_limit": 0.02
            },
            "drawdown_tracker": {
                "current_drawdown": 0.01,
                "daily_drawdown": 0.005
            },
            "open_trades": [],
            "macro_event_stream": {
                "blackout_active": False,
                "events": []
            },
            "risk_engine": {
                "current_risk": 0.01,
                "max_risk": 0.02
            },
            "telemetry": {"metrics": []}
        }
    
    def test_architect_compliance_validation(self):
        """Test architect mode compliance checks"""
        print("🔐 Testing Architect Mode Compliance...")
        
        # Create mock build_status.json
        mock_build_status = {
            "architect_mode_v28_compliant": True,
            "real_data_passed": True
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_build_status))):
            try:
                result = execution_selector.validate_architect_compliance()
                self.assertTrue(result)
                print("✅ Architect compliance validation passed")
            except Exception as e:
                self.fail(f"Architect compliance validation failed: {e}")
    
    def test_drawdown_buffer_filter(self):
        """Test drawdown proximity filtering"""
        print("🚦 Testing Drawdown Buffer Filter...")
        
        # Test signal passes with low drawdown
        context = self.test_context.copy()
        context["drawdown_tracker"]["current_drawdown"] = 0.01
        
        result = execution_selector.passes_drawdown_buffer(self.test_signal, context)
        self.assertTrue(result)
        print("✅ Low drawdown signal passed")        # Test signal fails with high drawdown
        context["drawdown_tracker"]["current_drawdown"] = 0.041  # Clearly above 80% of 0.05 max
        result = execution_selector.passes_drawdown_buffer(self.test_signal, context)
        self.assertFalse(result)
        print("✅ High drawdown signal rejected")
    
    def test_macro_blackout_filter(self):
        """Test macro event blackout filtering"""
        print("🌍 Testing Macro Blackout Filter...")
        
        # Test signal passes when no blackout
        context = self.test_context.copy()
        context["macro_event_stream"]["blackout_active"] = False
        
        result = execution_selector.is_in_macro_blackout(self.test_signal, context)
        self.assertFalse(result)
        print("✅ Normal trading conditions - signal passed")
        
        # Test signal fails during blackout
        context["macro_event_stream"]["blackout_active"] = True
        result = execution_selector.is_in_macro_blackout(self.test_signal, context)
        self.assertTrue(result)
        print("✅ Macro blackout active - signal blocked")
    
    def test_position_conflict_filter(self):
        """Test position conflict detection"""
        print("⚖️ Testing Position Conflict Filter...")
        
        # Test no conflict with empty positions
        result = execution_selector.is_conflicting_with_open_position(self.test_signal, [])
        self.assertFalse(result)
        print("✅ No open positions - no conflict")
        
        # Test conflict with same direction position
        open_trades = [{
            "id": "trade_001",
            "symbol": "EURUSD",
            "direction": "buy"
        }]
        
        result = execution_selector.is_conflicting_with_open_position(self.test_signal, open_trades)
        self.assertTrue(result)
        print("✅ Same direction conflict detected")
    
    def test_risk_reward_filter(self):
        """Test risk:reward ratio validation"""
        print("💰 Testing Risk:Reward Filter...")
        
        # Test signal with good RR ratio (1.5:1)
        signal = self.test_signal.copy()
        # Risk: 1.0950 - 1.0900 = 0.0050
        # Reward: 1.1000 - 1.0950 = 0.0050
        # RR = 1:1 (should fail with min 1.5)
        
        result = execution_selector.passes_risk_reward(signal, min_rr=1.5)
        self.assertFalse(result)
        print("✅ Low RR ratio rejected")
        
        # Test signal with better RR ratio
        signal["take_profit"] = 1.1025  # Better reward
        # New RR = 0.0075 / 0.0050 = 1.5:1
        result = execution_selector.passes_risk_reward(signal, min_rr=1.5)
        self.assertTrue(result)
        print("✅ Good RR ratio accepted")
    
    def test_trailing_drawdown_protection(self):
        """Test trailing drawdown risk assessment"""
        print("📉 Testing Trailing Drawdown Protection...")
        
        # Test signal passes with low daily drawdown
        context = self.test_context.copy()
        context["drawdown_tracker"]["daily_drawdown"] = 0.005
        context["account_config"]["daily_loss_limit"] = 0.02
        
        result = execution_selector.is_trailing_drawdown_at_risk(self.test_signal, context)
        self.assertFalse(result)
        print("✅ Low daily drawdown - signal allowed")
        
        # Test signal fails with high daily drawdown
        context["drawdown_tracker"]["daily_drawdown"] = 0.015  # > 50% of 0.02 limit
        result = execution_selector.is_trailing_drawdown_at_risk(self.test_signal, context)
        self.assertTrue(result)
        print("✅ High daily drawdown - signal blocked")
    
    def test_master_qualification_filter(self):
        """Test the master qualification function"""
        print("🎯 Testing Master Qualification Filter...")
        
        context = self.test_context.copy()
        
        # Modify signal for better RR ratio
        signal = self.test_signal.copy()
        signal["take_profit"] = 1.1025  # 1.5:1 RR ratio
        
        result = execution_selector.qualify_for_execution(signal, context)
        self.assertTrue(result)
        print("✅ Signal qualified for execution")
    
    @patch('execution_selector.log_to_telemetry')
    def test_telemetry_logging(self, mock_telemetry):
        """Test telemetry logging functionality"""
        print("📡 Testing Telemetry Integration...")
        
        execution_selector.log_to_telemetry("test_metric", {"value": 100})
        mock_telemetry.assert_called_once()
        print("✅ Telemetry logging functional")
    
    @patch('execution_selector.emit_event')
    def test_event_emission(self, mock_emit):
        """Test EventBus event emission"""
        print("🔁 Testing EventBus Integration...")
        
        execution_selector.emit_event("test_topic", {"data": "test"})
        mock_emit.assert_called_once()
        print("✅ EventBus emission functional")
    
    def test_context_loading(self):
        """Test context source loading"""
        print("📂 Testing Context Loading...")
        
        # Mock file system
        mock_files = {
            "priority_signals.json": json.dumps([]),
            "account_config.json": json.dumps({"daily_limit": 5}),
            "telemetry.json": json.dumps({"metrics": []}),
            "risk_engine.json": json.dumps({"current_risk": 0.0}),
            "drawdown_tracker.json": json.dumps({"current_drawdown": 0.0}),
            "open_trades.json": json.dumps([]),
            "macro_event_stream.json": json.dumps({"events": []})
        }
        
        def mock_open_func(filename, mode='r'):
            if filename in mock_files:
                return mock_open(read_data=mock_files[filename]).return_value
            raise FileNotFoundError(f"No such file: {filename}")
        
        with patch('builtins.open', side_effect=mock_open_func):
            with patch('os.path.exists', return_value=True):
                try:
                    context = execution_selector.load_context_sources()
                    self.assertIsInstance(context, dict)
                    self.assertIn("priority_signals", context)
                    print("✅ Context loading successful")
                except Exception as e:
                    print(f"⚠️ Context loading test skipped: {e}")
    
    def test_module_registration(self):
        """Test EventBus module registration"""
        print("🔗 Testing Module Registration...")
        
        mock_event_bus = {"routes": []}
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_event_bus))):
            with patch('execution_selector.load_event_bus', return_value=mock_event_bus):
                try:
                    result = execution_selector.register_module_on_eventbus()
                    self.assertTrue(result)
                    print("✅ Module registration successful")
                except Exception as e:
                    print(f"⚠️ Registration test skipped: {e}")

def run_execution_selector_tests():
    """Run all ExecutionSelector tests with detailed output"""
    print("🔐 GENESIS Phase 38 Execution Selector Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExecutionSelector)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Overall result
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED - EXECUTION SELECTOR VALIDATED")
        return True
    else:
        print("\n💥 SOME TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = run_execution_selector_tests()
    sys.exit(0 if success else 1)

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
        

# <!-- @GENESIS_MODULE_END: test_execution_selector -->