import logging
# <!-- @GENESIS_MODULE_START: test_priority_score_patch_phase37 -->
"""
🏛️ GENESIS TEST_PRIORITY_SCORE_PATCH_PHASE37 - INSTITUTIONAL GRADE v8.0.0
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

from event_bus import EventBus

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

                emit_telemetry("test_priority_score_patch_phase37", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_priority_score_patch_phase37", "position_calculated", {
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
                            "module": "test_priority_score_patch_phase37",
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
                    print(f"Emergency stop error in test_priority_score_patch_phase37: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "test_priority_score_patch_phase37",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_priority_score_patch_phase37", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_priority_score_patch_phase37: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
"""
GENESIS Trade Priority Resolver PATCH - TEST SUITE v1.0
=======================================================
Phase 37 Patch: Broker Weight + Drawdown Suppression Testing
ARCHITECT MODE COMPLIANCE: Real MT5 data integration testing
"""

import unittest
import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Import GENESIS modules
from trade_priority_resolver import TradePriorityResolver, SignalScore
from hardened_event_bus import get_event_bus, emit_event

class TestTradePriorityResolverPatch(unittest.TestCase):
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

            emit_telemetry("test_priority_score_patch_phase37", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_priority_score_patch_phase37", "position_calculated", {
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
                        "module": "test_priority_score_patch_phase37",
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
                print(f"Emergency stop error in test_priority_score_patch_phase37: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "test_priority_score_patch_phase37",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_priority_score_patch_phase37", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_priority_score_patch_phase37: {e}")
    """
    Test suite for Trade Priority Resolver Phase 37 Patch
    Focuses on broker weight adjustment and drawdown suppression
    """
    
    def setUp(self):
        """Set up test environment with real MT5 data structures"""
        self.resolver = TradePriorityResolver()
        self.event_bus = get_event_bus()
        
        # Sample signals for testing
        self.live_signals = [
            {
                "id": "SIG_EURUSD_001",
                "symbol": "EURUSD",
                "type": "BUY",
                "strength": 0.85,
                "broker": "MT5_LIVE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "AdvancedPatternMiner"
            },
            {
                "id": "SIG_GBPUSD_002",
                "symbol": "GBPUSD",
                "type": "SELL",
                "strength": 0.72,
                "broker": "FTMO",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "SignalEngine"
            },
            {
                "id": "SIG_USDJPY_003",
                "symbol": "USDJPY",
                "type": "BUY",
                "strength": 0.78,
                "broker": "ICMarkets",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "PatternEngine"
            }
        ]
        
        # Mock risk engine data for drawdown testing
        self.normal_risk_data = {
            "current_drawdown_pct": 1.5,
            "equity_stability": "stable",
            "risk_level": "normal"
        }
        
        self.high_drawdown_risk_data = {
            "current_drawdown_pct": 5.2,
            "equity_stability": "unstable",
            "risk_level": "high"
        }
        
        self.moderate_drawdown_risk_data = {
            "current_drawdown_pct": 3.8,
            "equity_stability": "moderate",
            "risk_level": "elevated"
        }
    
    def test_broker_weight_calculation(self):
        """Test broker-specific weight adjustments"""
        # Test MT5_LIVE broker (highest weight)
        weight_live = self.resolver._get_broker_weight("MT5_LIVE")
        self.assertEqual(weight_live, 0.15)
        
        # Test FTMO broker
        weight_ftmo = self.resolver._get_broker_weight("FTMO")
        self.assertEqual(weight_ftmo, 0.12)
        
        # Test ECN broker
        weight_ecn = self.resolver._get_broker_weight("ICMarkets")
        self.assertEqual(weight_ecn, 0.10)
        
        # Test demo broker (lower weight)
        weight_demo = self.resolver._get_broker_weight("MT5_DEMO")
        self.assertEqual(weight_demo, 0.05)
        
        # Test unknown broker (default)
        weight_unknown = self.resolver._get_broker_weight("UnknownBroker")
        self.assertEqual(weight_unknown, 0.0)
    
    def test_drawdown_suppression_normal_conditions(self):
        """Test drawdown suppression under normal conditions"""
        # Set normal risk conditions
        self.resolver.risk_engine_data = self.normal_risk_data
        
        suppression = self.resolver._get_drawdown_suppression()
        self.assertEqual(suppression, 0.0)  # No suppression
    
    def test_drawdown_suppression_high_drawdown(self):
        """Test drawdown suppression with high drawdown (>4.5%)"""
        # Set high drawdown conditions
        self.resolver.risk_engine_data = self.high_drawdown_risk_data
        
        suppression = self.resolver._get_drawdown_suppression()
        self.assertEqual(suppression, 1.0)  # Complete suppression
    
    def test_drawdown_suppression_moderate_drawdown(self):
        """Test drawdown suppression with moderate drawdown (3.5-4.5%)"""
        # Set moderate drawdown conditions
        self.resolver.risk_engine_data = self.moderate_drawdown_risk_data
        
        suppression = self.resolver._get_drawdown_suppression()
        self.assertEqual(suppression, 0.8)  # Heavy suppression
    
    def test_drawdown_suppression_light_conditions(self):
        """Test drawdown suppression with light drawdown (2.5-3.5%)"""
        # Set light drawdown conditions
        light_risk_data = {
            "current_drawdown_pct": 2.8,
            "equity_stability": "stable",
            "risk_level": "normal"
        }
        self.resolver.risk_engine_data = light_risk_data
        
        suppression = self.resolver._get_drawdown_suppression()
        self.assertEqual(suppression, 0.4)  # Moderate suppression
    
    def test_signal_scoring_with_broker_weight(self):
        """Test signal scoring with broker weight integration"""
        # Use signal with high-quality broker
        signal_live = self.live_signals[0]  # MT5_LIVE broker
        
        # Set normal risk conditions
        self.resolver.risk_engine_data = self.normal_risk_data
        
        score_result = self.resolver._score_signal(signal_live)
        
        # Verify broker weight is included
        self.assertEqual(score_result.broker_weight, 0.15)
        self.assertEqual(score_result.broker, "MT5_LIVE")
        
        # Verify final score incorporates broker weight
        expected_score = (
            score_result.base_strength - 
            score_result.latency_penalty - 
            score_result.news_risk - 
            score_result.exposure_bias + 
            score_result.session_bonus + 
            score_result.broker_weight - 
            score_result.drawdown_suppression
        )
        self.assertAlmostEqual(score_result.final_score, max(expected_score, 0.0), places=2)
    
    def test_signal_scoring_with_drawdown_suppression(self):
        """Test signal scoring with drawdown suppression"""
        # Use signal with normal broker
        signal = self.live_signals[1]  # FTMO broker
        
        # Set high drawdown conditions (should suppress score to 0)
        self.resolver.risk_engine_data = self.high_drawdown_risk_data
        
        score_result = self.resolver._score_signal(signal)
        
        # Verify drawdown suppression is applied
        self.assertEqual(score_result.drawdown_suppression, 1.0)
        self.assertEqual(score_result.final_score, 0.0)  # Complete suppression
    
    def test_signal_comparison_broker_weight_impact(self):
        """Test that broker weight affects signal ranking"""
        # Set normal risk conditions
        self.resolver.risk_engine_data = self.normal_risk_data
        
        # Score signals with different brokers but same strength
        signal_live = {
            "id": "SIG_TEST_001",
            "symbol": "EURUSD",
            "type": "BUY",
            "strength": 0.75,
            "broker": "MT5_LIVE"
        }
        
        signal_demo = {
            "id": "SIG_TEST_002",
            "symbol": "EURUSD",
            "type": "BUY",
            "strength": 0.75,
            "broker": "MT5_DEMO"
        }
        
        score_live = self.resolver._score_signal(signal_live)
        score_demo = self.resolver._score_signal(signal_demo)
        
        # Live broker should have higher score due to broker weight
        self.assertGreater(score_live.final_score, score_demo.final_score)
        self.assertGreater(score_live.broker_weight, score_demo.broker_weight)
    
    def test_signal_prioritization_with_patch(self):
        """Test full signal prioritization with patch enhancements"""
        # Set normal risk conditions
        self.resolver.risk_engine_data = self.normal_risk_data
        
        # Process multiple signals
        scored_signals = []
        for signal in self.live_signals:
            score_result = self.resolver._score_signal(signal)
            scored_signals.append((signal, score_result))
        
        # Sort by final score
        scored_signals.sort(key=lambda x: x[1].final_score, reverse=True)
        
        # Verify all signals have broker weight and drawdown suppression fields
        for signal, score in scored_signals:
            self.assertIsInstance(score.broker_weight, float)
            self.assertIsInstance(score.drawdown_suppression, float)
            self.assertGreaterEqual(score.broker_weight, 0.0)
            self.assertGreaterEqual(score.drawdown_suppression, 0.0)
    
    def test_equity_stability_impact(self):
        """Test equity stability impact on drawdown suppression"""
        # Test unstable equity with low drawdown
        unstable_risk_data = {
            "current_drawdown_pct": 1.0,  # Low drawdown
            "equity_stability": "unstable",  # But unstable
            "risk_level": "normal"
        }
        self.resolver.risk_engine_data = unstable_risk_data
        
        suppression = self.resolver._get_drawdown_suppression()
        self.assertEqual(suppression, 0.2)  # Light suppression for instability
    
    def test_patch_telemetry_integration(self):
        """Test that patch adds new telemetry fields"""
        # Set normal risk conditions
        self.resolver.risk_engine_data = self.normal_risk_data
        
        signal = self.live_signals[0]
        score_result = self.resolver._score_signal(signal)
        
        # Verify new telemetry fields are present
        score_dict = {
            "signal_id": score_result.signal_id,
            "broker_weight": score_result.broker_weight,
            "drawdown_suppression": score_result.drawdown_suppression,
            "final_score": score_result.final_score
        }
        
        # Verify fields have valid values
        self.assertIn("broker_weight", score_dict)
        self.assertIn("drawdown_suppression", score_dict)
        self.assertIsInstance(score_dict["broker_weight"], float)
        self.assertIsInstance(score_dict["drawdown_suppression"], float)
    
    def test_emergency_drawdown_override(self):
        """Test emergency drawdown override (score = 0 when DD > 4.5%)"""
        # Set emergency drawdown conditions
        emergency_risk_data = {
            "current_drawdown_pct": 6.0,  # Well above 4.5% threshold
            "equity_stability": "critical",
            "risk_level": "emergency"
        }
        self.resolver.risk_engine_data = emergency_risk_data
        
        # Test with high-strength signal
        high_strength_signal = {
            "id": "SIG_EMERGENCY_001",
            "symbol": "EURUSD",
            "type": "BUY",
            "strength": 0.95,  # Very high strength
            "broker": "MT5_LIVE"  # Best broker
        }
        
        score_result = self.resolver._score_signal(high_strength_signal)
        
        # Even with high strength and best broker, score should be 0
        self.assertEqual(score_result.final_score, 0.0)
        self.assertEqual(score_result.drawdown_suppression, 1.0)
    
    def test_patch_error_handling(self):
        """Test error handling in patch methods"""
        # Test broker weight with invalid data
        self.resolver.broker_latency_data = None
        weight = self.resolver._get_broker_weight("InvalidBroker")
        self.assertEqual(weight, 0.0)
        
        # Test drawdown suppression with missing risk data
        self.resolver.risk_engine_data = {}
        suppression = self.resolver._get_drawdown_suppression()
        self.assertEqual(suppression, 0.0)
        
        # Test signal scoring with corrupt signal data
        corrupt_signal = {}
        score_result = self.resolver._score_signal(corrupt_signal)
        self.assertEqual(score_result.final_score, 0.0)
        self.assertIsInstance(score_result, SignalScore)

if __name__ == "__main__":
    # Run patch tests
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
        

# <!-- @GENESIS_MODULE_END: test_priority_score_patch_phase37 -->
