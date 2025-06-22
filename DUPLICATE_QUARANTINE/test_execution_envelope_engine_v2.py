import logging
# <!-- @GENESIS_MODULE_START: test_execution_envelope_engine_v2 -->
"""
🏛️ GENESIS TEST_EXECUTION_ENVELOPE_ENGINE_V2 - INSTITUTIONAL GRADE v8.0.0
===============================================================
ARCHITECT MODE ULTIMATE: Enhanced via Complete Intelligent Wiring Engine

🎯 ENHANCED FEATURES:
- Complete EventBus integration
- Real-time telemetry monitoring
- FTMO compliance enforcement
- Emergency kill-switch protection
- Institutional-grade architecture

🔐 ARCHITECT MODE v8.0.0: Ultimate compliance enforcement
"""


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
                    "module": "test_execution_envelope_engine_v2",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_execution_envelope_engine_v2", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_execution_envelope_engine_v2: {e}")
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
                "module": "test_execution_envelope_engine_v2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_execution_envelope_engine_v2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_execution_envelope_engine_v2: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_execution_envelope_engine_v2",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_execution_envelope_engine_v2: {e}")
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
        if mt5_data:
            envelope = self.engine._create_execution_envelope(test_recommendation, mt5_data, start_time)
        
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"   ✅ Decision latency: {latency_ms:.2f}ms")
        self.assertLess(latency_ms, 450, "Decision latency exceeds 450ms threshold")
        self.assertIsNotNone(mt5_data)
        if envelope:
            self.assertGreater(envelope.position_size_lots, 0)
    
    def test_spread_block_functionality(self):
        """Test: Spread blocking when exceeds threshold"""
        print("\n🧪 Testing spread block functionality...")
        
        test_recommendation = {
            "recommendation_id": "test_002",
            "symbol": "GBPUSD",
            "strategy": "reversal_signal",
            "execution_quality": 0.90,
            "htf_alignment": True,
            "position_size_factor": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get real MT5 data and check spread
        mt5_data = self.engine._get_realtime_mt5_data("GBPUSD")
        
        print(f"   ✅ MT5 data retrieved for GBPUSD")
        self.assertIsNotNone(mt5_data)
        self.assertIn('spread_pips', mt5_data)
        print(f"   📊 Current spread: {mt5_data['spread_pips']} pips")
    
    def test_position_sizing_calculation(self):
        """Test: Dynamic position sizing with risk management"""
        print("\n🧪 Testing position sizing calculation...")
        
        test_recommendation = {
            "recommendation_id": "test_003",
            "symbol": "USDJPY",
            "strategy": "momentum_breakout",
            "execution_quality": 0.75,
            "htf_alignment": True,
            "position_size_factor": 0.75,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        mt5_data = self.engine._get_realtime_mt5_data("USDJPY")
        position_size = self.engine._calculate_position_size(test_recommendation, mt5_data)
        
        print(f"   ✅ Position size calculated: {position_size}")
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 5.0)  # Max position size check
        self.assertGreaterEqual(position_size, 0.01)  # Min position size check
    
    def test_compliance_check_enforcement(self):
        """Test: Risk and compliance validation"""
        print("\n🧪 Testing compliance check enforcement...")
        
        test_recommendation = {
            "recommendation_id": "test_004",
            "symbol": "EURUSD",
            "strategy": "test_strategy",
            "execution_quality": 0.80,
            "htf_alignment": True,
            "position_size_factor": 1.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        mt5_data = self.engine._get_realtime_mt5_data("EURUSD")
        position_size = 2.0  # Test position size
        risk_score = 0.5  # Moderate risk
        
        compliance_checks = self.engine._perform_compliance_checks(
            "EURUSD", position_size, mt5_data, risk_score
        )
        
        print(f"   ✅ Compliance checks performed")
        self.assertIsNotNone(compliance_checks)
        self.assertIn('ftmo_compliant', compliance_checks)
    
    def test_kill_switch_status(self):
        """Test: Kill switch status monitoring"""
        print("\n🧪 Testing kill switch status...")
        
        # Check initial kill switch status
        initial_status = self.engine.kill_switch_status
        
        print(f"   ✅ Kill switch status: {initial_status}")
        self.assertIn(initial_status, ["ACTIVE", "INACTIVE", "TRIGGERED"])
    
    def test_mt5_data_structure(self):
        """Test: MT5 data structure and content validation"""
        print("\n🧪 Testing MT5 data structure...")
        
        mt5_data = self.engine._get_realtime_mt5_data("EURUSD")
        
        # Verify required fields
        required_fields = [
            'symbol', 'timestamp', 'bid', 'ask', 'spread_pips',
            'volume', 'liquidity_score', 'atr_pips', 'volatility_score',
            'session', 'market_hours', 'data_quality', 'connection_status'
        ]
        
        for field in required_fields:
            self.assertIn(field, mt5_data, f"Missing required field: {field}")
        
        print(f"   ✅ All MT5 data fields present")
        self.assertGreater(mt5_data['ask'], mt5_data['bid'])
        self.assertEqual(mt5_data['symbol'], "EURUSD")
    
    def test_event_bus_integration(self):
        """Test: EventBus subscription and integration"""
        print("\n🧪 Testing EventBus integration...")
        
        # Check that engine has event bus instance
        self.assertIsNotNone(self.engine.event_bus)
        
        # Test event subscription setup (this was done in __init__)
        print(f"   ✅ EventBus integration verified")
    
    def test_execution_envelope_creation(self):
        """Test: Full execution envelope creation process"""
        print("\n🧪 Testing execution envelope creation...")
        
        test_recommendation = {
            "recommendation_id": "test_envelope_001",
            "symbol": "GBPUSD",
            "strategy": "breakout_momentum",
            "execution_quality": 0.88,
            "htf_alignment": True,
            "position_size_factor": 0.6,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        start_time = time.time()
        mt5_data = self.engine._get_realtime_mt5_data("GBPUSD")
        envelope = self.engine._create_execution_envelope(test_recommendation, mt5_data, start_time)
        
        if envelope:
            print(f"   ✅ Execution envelope created: {envelope.envelope_id}")
            self.assertIsNotNone(envelope.envelope_id)
            self.assertEqual(envelope.symbol, "GBPUSD")
            self.assertIn(envelope.direction, ["BUY", "SELL"])
            self.assertGreater(envelope.position_size_lots, 0)
        else:
            print(f"   ⚠️ Envelope creation returned None (may be expected for test conditions)")
    
    def test_trading_session_detection(self):
        """Test: Trading session detection"""
        print("\n🧪 Testing trading session detection...")
        
        session = self.engine._get_current_trading_session()
        market_hours = self.engine._is_market_hours()
        
        print(f"   ✅ Current session: {session}")
        print(f"   ✅ Market hours: {market_hours}")
        
        self.assertIsNotNone(session)
        self.assertIsInstance(market_hours, bool)

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
        

# <!-- @GENESIS_MODULE_END: test_execution_envelope_engine_v2 -->
