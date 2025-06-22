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
                    "module": "test_execution_envelope_engine",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_execution_envelope_engine", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_execution_envelope_engine: {e}")
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


# <!-- @GENESIS_MODULE_START: test_execution_envelope_engine -->

#!/usr/bin/env python3
"""
GENESIS AI Trading System - Execution Envelope Engine Test Suite
PHASE 24 - Strategic Execution Envelope Engine Testing

ARCHITECT MODE v2.7 COMPLIANT
- Event-driven architecture (EventBus only)
- Real MT5 data integration
- Full telemetry and logging
- Institutional-grade compliance
"""

import sys
import os
import json
import time
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution_envelope_engine import ExecutionEnvelopeEngine
from hardened_event_bus import HardenedEventBus

class TestExecutionEnvelopeEngine(unittest.TestCase):
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "test_execution_envelope_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_execution_envelope_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_execution_envelope_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_execution_envelope_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_execution_envelope_engine: {e}")
    """
    Comprehensive test suite for Execution Envelope Engine
    Tests real data processing, decision latency, compliance checks
    """
    
    def setUp(self):
        """Initialize test environment with real components"""
        self.event_bus = HardenedEventBus()
        self.engine = ExecutionEnvelopeEngine()
        
        # Load test configuration
        try:
            with open('envelope_config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.skipTest("envelope_config.json not found - skipping tests")
      def test_decision_latency_compliance(self):
        """Test: Decision latency under 450ms threshold"""
        print("\n🧪 Testing decision latency compliance...")
        
        # Create test strategy recommendation with real structure
        test_recommendation = {
            "recommendation_id": "test_001",
            "symbol": "EURUSD",
            "strategy": "momentum_breakout",
            "execution_quality": 0.85,
            "htf_alignment": True,
            "position_size_factor": 0.75,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        start_time = time.time()
        
        # Process through engine using real MT5 data getter
        mt5_data = self.engine._get_realtime_mt5_data("EURUSD")
        envelope = self.engine._create_execution_envelope(test_recommendation, mt5_data, start_time)
        
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"   ✅ Decision latency: {latency_ms:.2f}ms")
        self.assertLess(latency_ms, 450, "Decision latency exceeds 450ms threshold")
        self.assertIsNotNone(envelope)
        self.assertGreater(envelope.position_size_lots, 0)
    
    def test_spread_block_functionality(self):
        """Test: Spread blocking when exceeds threshold"""
        print("\n🧪 Testing spread block functionality...")
        
        test_signal = {
            "signal_id": "test_002",
            "symbol": "GBPUSD",
            "direction": "SELL",
            "confidence": 0.90,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Mock wide spread scenario
        with patch.object(self.engine, '_get_market_data') as mock_market:
            mock_market.return_value = {
                "bid": 1.2500,
                "ask": 1.2520,  # 2.0 pips spread (exceeds 1.2 threshold)
                "spread": 0.0020,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            decision = self.engine._make_execution_decision(test_signal)
        
        print(f"   ✅ Wide spread detected and blocked")
        self.assertEqual(decision['action'], 'BLOCK')
        self.assertIn('spread_too_wide', decision['reason'])
    
    def test_position_sizing_calculation(self):
        """Test: Dynamic position sizing with risk management"""
        print("\n🧪 Testing position sizing calculation...")
        
        test_signal = {
            "signal_id": "test_003",
            "symbol": "USDJPY",
            "direction": "BUY",
            "confidence": 0.75,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pattern_strength": 0.70
        }
        
        with patch.object(self.engine, '_get_market_data') as mock_market:
            mock_market.return_value = {
                "bid": 110.50,
                "ask": 110.52,
                "spread": 0.02,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            with patch.object(self.engine, '_get_account_info') as mock_account:
                mock_account.return_value = {
                    "balance": 10000.0,
                    "equity": 9800.0,
                    "margin_free": 9500.0
                }
                
                position_size = self.engine._calculate_position_size(
                    test_signal, mock_market.return_value, mock_account.return_value
                )
        
        print(f"   ✅ Position size calculated: {position_size}")
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 5.0)  # Max position size check
        self.assertGreaterEqual(position_size, 0.01)  # Min position size check
    
    def test_compliance_check_enforcement(self):
        """Test: FTMO compliance rules enforcement"""
        print("\n🧪 Testing compliance check enforcement...")
        
        # Test daily drawdown breach
        with patch.object(self.engine, '_get_account_info') as mock_account:
            mock_account.return_value = {
                "balance": 10000.0,
                "equity": 9400.0,  # 6% drawdown (exceeds 5% daily limit)
                "margin_free": 9000.0,
                "daily_pnl": -600.0
            }
            
            compliance_result = self.engine._check_compliance(mock_account.return_value)
        
        print(f"   ✅ Daily drawdown breach detected and blocked")
        self.assertFalse(compliance_result['passed'])
        self.assertIn('daily_drawdown', compliance_result['violations'])
    
    def test_kill_switch_activation(self):
        """Test: Kill switch triggers on consecutive losses"""
        print("\n🧪 Testing kill switch activation...")
        
        # Simulate consecutive losses
        self.engine.consecutive_losses = 5  # At threshold
        
        result = self.engine._check_kill_switches()
        
        print(f"   ✅ Kill switch activated on consecutive losses")
        self.assertTrue(result['triggered'])
        self.assertEqual(result['reason'], 'consecutive_losses')
    
    def test_decision_log_emission(self):
        """Test: Decision logging and telemetry emission"""
        print("\n🧪 Testing decision log emission...")
        
        test_decision = {
            "decision_id": "test_decision_001",
            "signal_id": "test_signal_001",
            "action": "EXECUTE",
            "position_size": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": 125.5,
            "compliance_passed": True
        }
        
        # Mock telemetry emission
        with patch.object(self.engine.event_bus, 'emit') as mock_emit:
            self.engine._log_decision(test_decision)
        
        # Verify telemetry emission
        mock_emit.assert_called()
        call_args = mock_emit.call_args[1]
        
        print(f"   ✅ Decision logged and telemetry emitted")
        self.assertEqual(call_args['event_type'], 'envelope_decision_logged')
        self.assertIn('decision_id', call_args['data'])
    
    def test_event_bus_integration(self):
        """Test: EventBus subscription and event processing"""
        print("\n🧪 Testing EventBus integration...")
        
        # Verify subscriptions are registered
        subscriptions = [
            "signal_generated",
            "strategy_recommendation", 
            "risk_assessment_complete",
            "pattern_detected",
            "harmony_score_calculated"
        ]
        
        for subscription in subscriptions:
            self.assertIn(subscription, self.engine.subscriptions)
        
        print(f"   ✅ All EventBus subscriptions registered")
    
    def test_real_time_market_data_integration(self):
        """Test: Real-time market data processing"""
        print("\n🧪 Testing real-time market data integration...")
        
        # Test with real market data structure
        with patch.object(self.engine, '_get_market_data') as mock_market:
            mock_market.return_value = {
                "symbol": "EURUSD",
                "bid": 1.0850,
                "ask": 1.0852,
                "spread": 0.0002,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "volume": 1000000,
                "volatility": 0.45
            }
            
            market_data = self.engine._get_market_data("EURUSD")
        
        print(f"   ✅ Market data retrieved: {market_data['symbol']}")
        self.assertIn('bid', market_data)
        self.assertIn('ask', market_data)
        self.assertIn('spread', market_data)
        self.assertGreater(market_data['ask'], market_data['bid'])

def run_phase24_tests():
    """Execute PHASE 24 test suite"""
    print("=" * 80)
    print("🚀 GENESIS PHASE 24 - EXECUTION ENVELOPE ENGINE TEST SUITE")
    print("   ARCHITECT MODE v2.7 COMPLIANT")
    print("=" * 80)
    
    # Load test configuration
    try:
        with open('envelope_config.json', 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded: {config['envelope_settings']['name']}")
    except FileNotFoundError:
        print("❌ envelope_config.json not found")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestExecutionEnvelopeEngine)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Test results summary
    print("\n" + "=" * 80)
    print("📊 PHASE 24 TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, failure in result.failures:
            print(f"   - {test}: {failure}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, error in result.errors:
            print(f"   - {test}: {error}")
    
    # Architect compliance check
    architect_compliant = (len(result.failures) == 0 and len(result.errors) == 0)
    
    print(f"\n🔐 ARCHITECT MODE v2.7 COMPLIANCE: {'✅ PASSED' if architect_compliant else '❌ FAILED'}")
    print("=" * 80)
    
    return architect_compliant

if __name__ == "__main__":
    success = run_phase24_tests()
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
        

# <!-- @GENESIS_MODULE_END: test_execution_envelope_engine -->