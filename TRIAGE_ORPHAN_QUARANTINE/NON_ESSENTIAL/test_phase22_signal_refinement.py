import logging
# <!-- @GENESIS_MODULE_START: test_phase22_signal_refinement -->

from datetime import datetime\n"""

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


GENESIS PHASE 22 SIGNAL REFINEMENT ENGINE TEST SUITE
===================================================

Comprehensive test suite for the Strategic Signal Refinement Engine (SSR)
Tests all critical functionality including HTF alignment, ASIO optimization,
signal refinement, and EventBus integration.

Author: GENESIS AI AGENT v2.9
Phase: 22 - Strategic Signal Refinement Testing
Compliance: ARCHITECT MODE v3.0 STRICT
"""

import unittest
import json
import datetime
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import numpy as np

# Import the SSR Engine and related classes
from signal_refinement_engine import (
    StrategicSignalRefinementEngine,
    RawSignal,
    HTFStructureData,
    ASIOOptimizationAdvice,
    RefinedSignal,
    HTFStructureAlignment,
    SignalConfidenceLevel
)

class TestStrategicSignalRefinementEngine(unittest.TestCase):
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

            emit_telemetry("test_phase22_signal_refinement", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_phase22_signal_refinement", "position_calculated", {
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
                        "module": "test_phase22_signal_refinement",
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
                print(f"Emergency stop error in test_phase22_signal_refinement: {e}")
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
                        "module": "test_phase22_signal_refinement",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_phase22_signal_refinement: {e}")
    """Test suite for Strategic Signal Refinement Engine"""

    def setUp(self):
        """Set up test environment"""
        # Mock EventBus and dependencies to prevent actual system integration during tests
        with patch('signal_refinement_engine.get_event_bus'), \
             patch('signal_refinement_engine.subscribe_to_event'), \
             patch('signal_refinement_engine.register_route'):
            
            self.ssr_engine = StrategicSignalRefinementEngine()
            
        # Create test data
        self.test_raw_signal = RawSignal(
            symbol="EURUSD",
            signal_type="BUY",
            strength=0.8,
            timestamp=datetime.datetime.utcnow().isoformat(),
            source_module="TestModule",
            confidence_score=0.75,
            metadata={"test": "data"}
        )
        
        self.test_htf_data = HTFStructureData(
            symbol="EURUSD",
            timeframe="H4",
            trend_direction="BULLISH",
            support_level=1.0850,
            resistance_level=1.0950,
            structure_quality=0.85,
            alignment_score=0.9,
            confidence=0.88,
            timestamp=datetime.datetime.utcnow().isoformat()
        )
        
        self.test_asio_advice = ASIOOptimizationAdvice(
            symbol="EURUSD",
            optimization_type="CONFIDENCE_BOOST",
            confidence_adjustment=0.15,
            timing_adjustment=0.05,
            volume_adjustment=1.2,
            risk_adjustment=0.1,
            ml_score=0.82,
            model_version="ASIO_v1.0",
            timestamp=datetime.datetime.utcnow().isoformat()
        )

    def test_ssr_engine_initialization(self):
        """Test SSR Engine proper initialization"""
        self.assertIsNotNone(self.ssr_engine)
        self.assertIsNotNone(self.ssr_engine.logger)
        self.assertIsNotNone(self.ssr_engine.config)
        self.assertEqual(self.ssr_engine.config["htf_weight"], 0.4)
        self.assertEqual(self.ssr_engine.config["asio_weight"], 0.35)
        self.assertEqual(self.ssr_engine.config["original_weight"], 0.25)
        self.assertFalse(self.ssr_engine.running)
        
    def test_ssr_engine_start_stop(self):
        """Test SSR Engine start and stop functionality"""
        # Test start
        self.ssr_engine.start()
        self.assertTrue(self.ssr_engine.running)
        self.assertIsNotNone(self.ssr_engine.worker_thread)
        
        # Test stop
        self.ssr_engine.stop()
        self.assertFalse(self.ssr_engine.running)

    def test_htf_alignment_calculation(self):
        """Test HTF structure alignment calculation"""
        # Test strong alignment (BUY signal + BULLISH HTF + high quality)
        alignment = self.ssr_engine._calculate_htf_alignment(
            self.test_raw_signal, self.test_htf_data
        )
        self.assertEqual(alignment, HTFStructureAlignment.STRONGLY_ALIGNED)
        
        # Test strong opposition (BUY signal + BEARISH HTF + high quality)
        bearish_htf = HTFStructureData(
            symbol="EURUSD",
            timeframe="H4",
            trend_direction="BEARISH",
            support_level=1.0850,
            resistance_level=1.0950,
            structure_quality=0.85,
            alignment_score=0.9,
            confidence=0.88,
            timestamp=datetime.datetime.utcnow().isoformat()
        )
        alignment = self.ssr_engine._calculate_htf_alignment(
            self.test_raw_signal, bearish_htf
        )
        self.assertEqual(alignment, HTFStructureAlignment.STRONGLY_AGAINST)
        
        # Test weak alignment (BUY signal + BULLISH HTF + low quality)
        weak_htf = HTFStructureData(
            symbol="EURUSD",
            timeframe="H4",
            trend_direction="BULLISH",
            support_level=1.0850,
            resistance_level=1.0950,
            structure_quality=0.6,
            alignment_score=0.6,
            confidence=0.65,
            timestamp=datetime.datetime.utcnow().isoformat()
        )
        alignment = self.ssr_engine._calculate_htf_alignment(
            self.test_raw_signal, weak_htf
        )
        self.assertEqual(alignment, HTFStructureAlignment.ALIGNED)

    def test_asio_optimization_application(self):
        """Test ASIO optimization application"""
        optimization = self.ssr_engine._apply_asio_optimization(
            self.test_raw_signal, self.test_asio_advice
        )
        
        self.assertEqual(optimization["confidence_adjustment"], 0.15)
        self.assertEqual(optimization["timing_adjustment"], 0.05)
        self.assertEqual(optimization["volume_adjustment"], 1.2)
        self.assertEqual(optimization["risk_adjustment"], 0.1)
        self.assertEqual(optimization["ml_score"], 0.82)

    def test_refined_confidence_calculation(self):
        """Test refined confidence score calculation"""
        htf_alignment = HTFStructureAlignment.STRONGLY_ALIGNED
        asio_optimization = {
            "confidence_adjustment": 0.15,
            "timing_adjustment": 0.05,
            "volume_adjustment": 1.2,
            "risk_adjustment": 0.1,
            "ml_score": 0.82
        }
        
        refined_confidence = self.ssr_engine._calculate_refined_confidence(
            self.test_raw_signal, htf_alignment, asio_optimization
        )
        
        # Should be higher than original due to strong alignment and positive ASIO
        self.assertGreater(refined_confidence, self.test_raw_signal.confidence_score)
        self.assertLessEqual(refined_confidence, 1.0)
        self.assertGreaterEqual(refined_confidence, 0.0)

    def test_signal_type_determination(self):
        """Test refined signal type determination"""
        htf_alignment = HTFStructureAlignment.STRONGLY_ALIGNED
        asio_optimization = {"ml_score": 0.82}
        
        # Strong alignment should maintain signal type
        refined_type = self.ssr_engine._determine_refined_signal_type(
            self.test_raw_signal, htf_alignment, asio_optimization
        )
        self.assertEqual(refined_type, "BUY")
        
        # Strong opposition should downgrade to HOLD
        htf_alignment = HTFStructureAlignment.STRONGLY_AGAINST
        refined_type = self.ssr_engine._determine_refined_signal_type(
            self.test_raw_signal, htf_alignment, asio_optimization
        )
        self.assertEqual(refined_type, "HOLD")
        
        # Low ASIO ML score should downgrade to HOLD
        htf_alignment = HTFStructureAlignment.ALIGNED
        low_asio = {"ml_score": 0.2}
        refined_type = self.ssr_engine._determine_refined_signal_type(
            self.test_raw_signal, htf_alignment, low_asio
        )
        self.assertEqual(refined_type, "HOLD")

    def test_refinement_score_calculation(self):
        """Test refinement score calculation"""
        original_confidence = 0.75
        refined_confidence = 0.85
        
        score = self.ssr_engine._calculate_refinement_score(
            original_confidence, refined_confidence
        )
        self.assertEqual(score, 0.1)
        
        # Test negative refinement
        refined_confidence = 0.65
        score = self.ssr_engine._calculate_refinement_score(
            original_confidence, refined_confidence
        )
        self.assertEqual(score, -0.1)

    def test_signal_quality_assessment(self):
        """Test signal quality assessment"""
        # Test excellent quality
        quality = self.ssr_engine._assess_signal_quality(0.9, 0.15)
        self.assertEqual(quality, "EXCELLENT")
        
        # Test good quality
        quality = self.ssr_engine._assess_signal_quality(0.75, 0.08)
        self.assertEqual(quality, "GOOD")
        
        # Test moderate quality
        quality = self.ssr_engine._assess_signal_quality(0.6, 0.02)
        self.assertEqual(quality, "MODERATE")
        
        # Test poor quality
        quality = self.ssr_engine._assess_signal_quality(0.4, 0.0)
        self.assertEqual(quality, "POOR")
        
        # Test rejected quality
        quality = self.ssr_engine._assess_signal_quality(0.2, -0.1)
        self.assertEqual(quality, "REJECTED")

    def test_execution_priority_calculation(self):
        """Test execution priority calculation"""
        htf_alignment = HTFStructureAlignment.STRONGLY_ALIGNED
        asio_optimization = {"ml_score": 0.85}
        
        priority = self.ssr_engine._calculate_execution_priority(
            0.9, htf_alignment, asio_optimization
        )
        
        # Should be high priority (base 9 + HTF bonus 3 + ASIO bonus 2 = 14, capped at 10)
        self.assertEqual(priority, 10)
        
        # Test low priority
        htf_alignment = HTFStructureAlignment.STRONGLY_AGAINST
        asio_optimization = {"ml_score": 0.2}
        
        priority = self.ssr_engine._calculate_execution_priority(
            0.3, htf_alignment, asio_optimization
        )
        
        # Should be low priority (base 3 + HTF penalty -3 + ASIO penalty -2 = -2, capped at 1)
        self.assertEqual(priority, 1)

    def test_risk_adjusted_size_calculation(self):
        """Test risk-adjusted position size calculation"""
        refined_confidence = 0.85
        asio_optimization = {"risk_adjustment": 0.1}
        
        adjusted_size = self.ssr_engine._calculate_risk_adjusted_size(
            self.test_raw_signal, refined_confidence, asio_optimization
        )
        
        # Should be based on signal strength (0.8) * confidence (0.85) * risk multiplier (1.1)
        expected_size = 0.8 * 0.85 * 1.1
        self.assertAlmostEqual(adjusted_size, expected_size, places=3)
        
        # Test bounds
        self.assertGreaterEqual(adjusted_size, 0.1)
        self.assertLessEqual(adjusted_size, 2.0)

    def test_cache_functionality(self):
        """Test HTF and ASIO cache functionality"""
        # Test HTF caching
        self.ssr_engine._handle_htf_structure(asdict(self.test_htf_data))
        self.assertIn("EURUSD", self.ssr_engine.htf_cache)
        self.assertEqual(
            self.ssr_engine.htf_cache["EURUSD"]["data"].symbol, 
            "EURUSD"
        )
        
        # Test ASIO caching
        self.ssr_engine._handle_asio_advice(asdict(self.test_asio_advice))
        self.assertIn("EURUSD", self.ssr_engine.asio_cache)
        self.assertEqual(
            self.ssr_engine.asio_cache["EURUSD"]["advice"].symbol,
            "EURUSD"
        )

    def test_can_process_refinement(self):
        """Test refinement processing readiness check"""
        # Create refinement entry
        refinement = {
            "raw_signal": self.test_raw_signal,
            "htf_data": None,
            "asio_advice": None,
            "status": "pending",
            "created_at": datetime.datetime.utcnow()
        }
        
        # Should not be able to process without cache data
        self.assertFalse(self.ssr_engine._can_process_refinement(refinement))
        
        # Add cache data
        self.ssr_engine.htf_cache["EURUSD"] = {
            "data": self.test_htf_data,
            "timestamp": datetime.datetime.utcnow()
        }
        self.ssr_engine.asio_cache["EURUSD"] = {
            "advice": self.test_asio_advice,
            "timestamp": datetime.datetime.utcnow()
        }
        
        # Should now be able to process
        self.assertTrue(self.ssr_engine._can_process_refinement(refinement))

    def test_full_signal_refinement(self):
        """Test complete signal refinement process"""
        # Setup cache data
        self.ssr_engine.htf_cache["EURUSD"] = {
            "data": self.test_htf_data,
            "timestamp": datetime.datetime.utcnow()
        }
        self.ssr_engine.asio_cache["EURUSD"] = {
            "advice": self.test_asio_advice,
            "timestamp": datetime.datetime.utcnow()
        }
        
        # Create refinement entry
        refinement = {
            "raw_signal": self.test_raw_signal,
            "htf_data": None,
            "asio_advice": None,
            "status": "pending",
            "created_at": datetime.datetime.utcnow()
        }
        
        # Refine the signal
        refined_signal = self.ssr_engine._refine_signal(refinement)
          # Validate refined signal
        self.assertIsNotNone(refined_signal)
        if refined_signal:
            self.assertIsInstance(refined_signal, RefinedSignal)
            self.assertEqual(refined_signal.symbol, "EURUSD")
            self.assertEqual(refined_signal.original_signal_type, "BUY")
            self.assertGreaterEqual(refined_signal.refined_confidence, 0.0)
            self.assertLessEqual(refined_signal.refined_confidence, 1.0)
            self.assertIn(refined_signal.signal_quality, 
                         ["EXCELLENT", "GOOD", "MODERATE", "POOR", "REJECTED"])
            self.assertGreaterEqual(refined_signal.execution_priority, 1)
            self.assertLessEqual(refined_signal.execution_priority, 10)

    def test_performance_metrics(self):
        """Test performance metrics collection"""
        metrics = self.ssr_engine.get_performance_metrics()
        
        self.assertIn("refinement_stats", metrics)
        self.assertIn("active_refinements", metrics)
        self.assertIn("cache_status", metrics)
        self.assertIn("recent_refinements", metrics)
        self.assertIn("status", metrics)
        
        # Check initial values
        self.assertEqual(metrics["refinement_stats"]["total_signals_processed"], 0)
        self.assertEqual(metrics["active_refinements"], 0)
        self.assertEqual(metrics["cache_status"]["htf_cache_size"], 0)
        self.assertEqual(metrics["cache_status"]["asio_cache_size"], 0)

    def test_raw_signal_handling(self):
        """Test raw signal event handling"""
        # Mock the raw signal data
        signal_data = asdict(self.test_raw_signal)
        
        # Handle the raw signal
        self.ssr_engine._handle_raw_signal(signal_data)
        
        # Check that refinement was queued
        refinement_id = f"{self.test_raw_signal.symbol}_{self.test_raw_signal.timestamp}"
        self.assertIn(refinement_id, self.ssr_engine.active_refinements)
        self.assertEqual(
            self.ssr_engine.active_refinements[refinement_id]["status"], 
            "pending"
        )

    @patch('signal_refinement_engine.emit_event')
    def test_telemetry_emission(self, mock_emit):
        """Test telemetry emission"""
        # Test telemetry emission
        self.event_bus.request('data:live_feed') = {"test": "telemetry"}
        self.ssr_engine._emit_telemetry("test.telemetry", self.event_bus.request('data:live_feed'))
        
        # Verify emit_event was called correctly
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        self.assertEqual(call_args[0][0], "ModuleTelemetry")
        self.assertEqual(call_args[0][2], "SSREngine")
        
        emitted_data = call_args[0][1]
        self.assertEqual(emitted_data["module"], "SSREngine")
        self.assertEqual(emitted_data["telemetry_type"], "test.telemetry")
        self.assertEqual(emitted_data["data"], self.event_bus.request('data:live_feed'))

    @patch('signal_refinement_engine.emit_event')
    def test_refined_signal_emission(self, mock_emit):
        """Test refined signal emission"""
        # Create a test refined signal
        refined_signal = RefinedSignal(
            symbol="EURUSD",
            original_signal_type="BUY",
            refined_signal_type="BUY",
            original_confidence=0.75,
            refined_confidence=0.85,
            htf_alignment="STRONGLY_ALIGNED",
            asio_optimization={"confidence_adjustment": 0.15},
            refinement_score=0.1,
            signal_quality="EXCELLENT",
            execution_priority=9,
            risk_adjusted_position_size=1.2,
            timestamp=datetime.datetime.utcnow().isoformat(),
            refinement_metadata={"test": "metadata"}
        )
        
        # Emit the refined signal
        self.ssr_engine._emit_refined_signal(refined_signal)
        
        # Verify both RefinedSignal and telemetry were emitted
        self.assertEqual(mock_emit.call_count, 2)
          # Check RefinedSignal emission
        refined_signal_call = None
        telemetry_call = None
        
        for call in mock_emit.call_args_list:
            if call[0][0] == "RefinedSignal":
                refined_signal_call = call
            elif call[0][0] == "ModuleTelemetry":
                telemetry_call = call
        
        self.assertIsNotNone(refined_signal_call)
        self.assertIsNotNone(telemetry_call)
        
        if refined_signal_call:
            # Validate refined signal data
            refined_data = refined_signal_call[0][1]
            self.assertEqual(refined_data["symbol"], "EURUSD")
            self.assertEqual(refined_data["refined_confidence"], 0.85)

    def test_cache_cleanup(self):
        """Test cache cleanup functionality"""
        # Add old entries to cache
        old_timestamp = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
        
        self.ssr_engine.htf_cache["OLD_SYMBOL"] = {
            "data": self.test_htf_data,
            "timestamp": old_timestamp
        }
        self.ssr_engine.asio_cache["OLD_SYMBOL"] = {
            "advice": self.test_asio_advice,
            "timestamp": old_timestamp
        }
        
        # Add fresh entries
        fresh_timestamp = datetime.datetime.utcnow()
        self.ssr_engine.htf_cache["FRESH_SYMBOL"] = {
            "data": self.test_htf_data,
            "timestamp": fresh_timestamp
        }
        self.ssr_engine.asio_cache["FRESH_SYMBOL"] = {
            "advice": self.test_asio_advice,
            "timestamp": fresh_timestamp
        }
        
        # Run cleanup
        self.ssr_engine._cleanup_cache()
        
        # Check that old entries were removed
        self.assertNotIn("OLD_SYMBOL", self.ssr_engine.htf_cache)
        self.assertNotIn("OLD_SYMBOL", self.ssr_engine.asio_cache)
        
        # Check that fresh entries remain
        self.assertIn("FRESH_SYMBOL", self.ssr_engine.htf_cache)
        self.assertIn("FRESH_SYMBOL", self.ssr_engine.asio_cache)

    def test_configuration_validation(self):
        """Test SSR Engine configuration validation"""
        config = self.ssr_engine.config
        
        # Test weight distribution
        total_weight = (config["htf_weight"] + 
                       config["asio_weight"] + 
                       config["original_weight"])
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Test threshold bounds
        self.assertGreaterEqual(config["min_confidence_threshold"], 0.0)
        self.assertLessEqual(config["max_confidence_threshold"], 1.0)
        self.assertLess(config["min_confidence_threshold"], 
                       config["max_confidence_threshold"])

if __name__ == "__main__":
    print("GENESIS PHASE 22 SSR ENGINE TEST SUITE")
    print("=" * 45)
    print("Running comprehensive tests for Strategic Signal Refinement Engine...")
    print()
    
    # Run the test suite
    unittest.main(verbosity=2)

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
        

# <!-- @GENESIS_MODULE_END: test_phase22_signal_refinement -->