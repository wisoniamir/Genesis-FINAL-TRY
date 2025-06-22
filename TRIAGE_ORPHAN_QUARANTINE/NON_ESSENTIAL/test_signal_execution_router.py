import logging
# <!-- @GENESIS_MODULE_START: test_signal_execution_router -->
"""
🏛️ GENESIS TEST_SIGNAL_EXECUTION_ROUTER - INSTITUTIONAL GRADE v8.0.0
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

"""
Test Suite for GENESIS Phase 71: Signal Execution Router
🔐 ARCHITECT MODE v5.0.0 - COMPLIANT TEST SCAFFOLDS
🧪 Comprehensive Testing for Trade Routing & Execution Validation

Tests trade recommendation validation, risk checking, macro conflict detection,
manual override handling, compliance validation, MT5 routing, and EventBus compliance.
"""

import unittest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import uuid

# Import the module under test
from signal_execution_router import SignalExecutionRouter, ExecutionDecision

class TestSignalExecutionRouter(unittest.TestCase):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "test_signal_execution_router",
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
                print(f"Emergency stop error in test_signal_execution_router: {e}")
                return False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """Comprehensive test suite for SignalExecutionRouter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.router = SignalExecutionRouter()
        self.test_config = {
            "max_position_size": 0.02,
            "max_correlation": 0.7,
            "max_drawdown": 0.05,
            "risk_limits": {
                "daily_loss_limit": 0.02,
                "position_limit": 5,
                "exposure_limit": 0.1
            },
            "mt5_execution": {
                "enabled": True,
                "timeout_ms": 5000,
                "retry_attempts": 3
            }
        }
        
        # Mock trade recommendation
        self.live_recommendation = {
            "recommendation_id": str(uuid.uuid4()),
            "symbol": "EURUSD",
            "direction": "long",
            "entry": 1.08520,
            "stop_loss": 1.08320,
            "take_profit": 1.08920,
            "confidence": 8.2,
            "risk_reward": 2.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self.router, 'stop_execution_loop'):
            self.router.stop_execution_loop()
        
    def test_initialization(self):
        """Test router initialization and configuration loading"""
        self.assertIsNotNone(self.router.config)
        self.assertIsNotNone(self.router.execution_history)
        self.assertIsNotNone(self.router.risk_state)
        self.assertEqual(self.router.is_running, False)
        
    def test_trade_recommendation_validation(self):
        """Test trade recommendation validation and processing"""
        # Test valid recommendation
        result = self.router.validate_recommendation(self.live_recommendation)
        
        self.assertIsNotNone(result)
        self.assertIn("is_valid", result)
        self.assertIn("validation_errors", result)
        self.assertIsInstance(result["is_valid"], bool)
        self.assertIsInstance(result["validation_errors"], list)
        
        # Test invalid recommendation (missing required fields)
        invalid_recommendation = {"symbol": "EURUSD"}
        result = self.router.validate_recommendation(invalid_recommendation)
        
        self.assertFalse(result["is_valid"])
        self.assertGreater(len(result["validation_errors"]), 0)
        
    def test_risk_limit_checking(self):
        """Test risk limit validation and position size checking"""
        # Test position size validation
        position_check = self.router.validate_position_size(
            self.live_recommendation["symbol"], 
            0.01  # 1% position size
        )
        
        self.assertIsNotNone(position_check)
        self.assertIn("within_limits", position_check)
        self.assertIn("current_exposure", position_check)
        
        # Test excessive position size
        excessive_check = self.router.validate_position_size(
            self.live_recommendation["symbol"],
            0.05  # 5% position size (over limit)
        )
        
        self.assertFalse(excessive_check["within_limits"])
        
    def test_macro_conflict_detection(self):
        """Test macro conflict detection and filtering"""
        # Mock macro environment data
        macro_data = {
            "economic_calendar": [
                {"impact": "high", "currency": "EUR", "direction": "bearish"}
            ],
            "sentiment": {"EURUSD": "conflicted"},
            "correlations": {"EURUSD": {"GBPUSD": 0.85}}
        }
        
        conflict_result = self.router.detect_macro_conflicts(
            self.live_recommendation, macro_data
        )
        
        self.assertIsNotNone(conflict_result)
        self.assertIn("conflicts_detected", conflict_result)
        self.assertIn("conflict_details", conflict_result)
        self.assertIsInstance(conflict_result["conflicts_detected"], bool)
        
    def test_manual_override_handling(self):
        """Test manual override state checking and processing"""
        # Test normal state (no override)
        override_state = self.router.check_manual_override_state()
        
        self.assertIsNotNone(override_state)
        self.assertIn("override_active", override_state)
        self.assertIn("override_type", override_state)
        
        # Test with active override
        self.router.set_manual_override("pause_all", "Manual intervention required")
        
        override_state = self.router.check_manual_override_state()
        self.assertTrue(override_state["override_active"])
        self.assertEqual(override_state["override_type"], "pause_all")
        
    def test_compliance_validation(self):
        """Test compliance checks and audit trail generation"""
        compliance_result = self.router.validate_compliance(self.live_recommendation)
        
        self.assertIsNotNone(compliance_result)
        self.assertIn("compliant", compliance_result)
        self.assertIn("compliance_score", compliance_result)
        self.assertIn("audit_trail", compliance_result)
        self.assertIsInstance(compliance_result["compliant"], bool)
        self.assertGreaterEqual(compliance_result["compliance_score"], 0)
        self.assertLessEqual(compliance_result["compliance_score"], 100)
        
    def test_mt5_order_routing(self):
        """Test MT5 order formatting and routing"""
        # Mock MT5 order generation
        with patch('signal_execution_router.MT5Connector') as mock_mt5:
            mock_mt5.return_value.place_order = Mock(return_value={"order_id": "12345", "status": "filled"})
            
            order_result = self.router.route_to_mt5(self.live_recommendation)
            
            self.assertIsNotNone(order_result)
            self.assertIn("order_submitted", order_result)
            self.assertIn("order_id", order_result)
            
    def test_execution_decision_making(self):
        """Test execution decision logic and validation flow"""
        # Test complete execution decision flow
        decision = self.router.make_execution_decision(self.live_recommendation)
        
        self.assertIsNotNone(decision)
        self.assertIsInstance(decision, ExecutionDecision)
        self.assertEqual(decision.recommendation_id, self.live_recommendation["recommendation_id"])
        self.assertEqual(decision.symbol, self.live_recommendation["symbol"])
        self.assertIn(decision.decision, ["execute", "reject", "defer"])
        
    def test_risk_exposure_tracking(self):
        """Test risk exposure calculation and tracking"""
        # Add some positions to track
        self.router.add_position({
            "symbol": "EURUSD",
            "size": 0.01,
            "direction": "long",
            "entry": 1.08520
        })
        
        exposure = self.router.calculate_risk_exposure()
        
        self.assertIsNotNone(exposure)
        self.assertIn("total_exposure", exposure)
        self.assertIn("symbol_exposure", exposure)
        self.assertIn("correlation_risk", exposure)
        self.assertGreaterEqual(exposure["total_exposure"], 0)
        
    def test_position_size_validation(self):
        """Test position size limits and validation"""
        # Test normal position size
        validation = self.router.validate_position_limits("EURUSD", 0.01)
        self.assertTrue(validation["within_limits"])
        
        # Test excessive position size
        validation = self.router.validate_position_limits("EURUSD", 0.05)
        self.assertFalse(validation["within_limits"])
        
    def test_correlation_analysis(self):
        """Test correlation analysis and risk assessment"""
        # Mock existing positions
        existing_positions = [
            {"symbol": "EURUSD", "size": 0.01, "direction": "long"},
            {"symbol": "GBPUSD", "size": 0.01, "direction": "long"}
        ]
        
        correlation_risk = self.router.analyze_correlation_risk(
            self.live_recommendation, existing_positions
        )
        
        self.assertIsNotNone(correlation_risk)
        self.assertIn("correlation_score", correlation_risk)
        self.assertIn("risk_level", correlation_risk)
        self.assertGreaterEqual(correlation_risk["correlation_score"], 0)
        self.assertLessEqual(correlation_risk["correlation_score"], 1)
        
    def test_drawdown_monitoring(self):
        """Test drawdown calculation and monitoring"""
        # Mock account history
        account_history = [
            {"timestamp": "2025-06-18T10:00:00Z", "balance": 10000},
            {"timestamp": "2025-06-18T11:00:00Z", "balance": 9950},
            {"timestamp": "2025-06-18T12:00:00Z", "balance": 9900}
        ]
        
        drawdown = self.router.calculate_current_drawdown(account_history)
        
        self.assertIsNotNone(drawdown)
        self.assertIn("current_drawdown", drawdown)
        self.assertIn("max_drawdown", drawdown)
        self.assertGreaterEqual(drawdown["current_drawdown"], 0)
        
    def test_order_format_validation(self):
        """Test MT5 order format validation"""
        # Generate MT5 order format
        order_data = self.router.format_mt5_order(self.live_recommendation)
        
        self.assertIsNotNone(order_data)
        self.assertIn("symbol", order_data)
        self.assertIn("action", order_data)
        self.assertIn("volume", order_data)
        self.assertIn("price", order_data)
        self.assertIn("sl", order_data)  # Stop loss
        self.assertIn("tp", order_data)  # Take profit
        
        # Validate required MT5 fields
        required_fields = ["symbol", "action", "volume", "price"]
        for field in required_fields:
            self.assertIn(field, order_data)
            
    @patch('signal_execution_router.EventBus')
    def test_eventbus_integration(self, mock_eventbus):
        """Test EventBus integration and event emission"""
        mock_eventbus.return_value.emit = Mock()
        
        # Process a recommendation to trigger EventBus emission
        decision = self.router.make_execution_decision(self.live_recommendation)
        
        # Verify EventBus emission
        self.assertTrue(mock_eventbus.return_value.emit.called)
        call_args = mock_eventbus.return_value.emit.call_args
        self.assertEqual(call_args[0][0], "execution_decision_made")
        
    @patch('signal_execution_router.TelemetrySync')
    def test_telemetry_emission(self, mock_telemetry):
        """Test telemetry hooks and metric emission"""
        mock_telemetry.return_value.emit = Mock()
        
        # Trigger telemetry emission
        self.router.emit_telemetry_metrics()
        
        # Verify telemetry emission
        self.assertTrue(mock_telemetry.return_value.emit.called)
        
    def test_error_handling(self):
        """Test error handling and recovery mechanisms"""
        # Test invalid recommendation input
        with self.assertRaises(ValueError):
            self.router.make_execution_decision({})
            
        # Test MT5 connection error simulation
        with patch('signal_execution_router.MT5Connector') as mock_mt5:
            mock_mt5.side_effect = ConnectionError("MT5 connection failed")
            
            # Should handle gracefully without crashing
            result = self.router.route_to_mt5(self.live_recommendation)
            self.assertIsNotNone(result)
            self.assertFalse(result["order_submitted"])
            
    def test_thread_safety(self):
        """Test thread safety of execution routing"""
        results = []
        
        def process_recommendations():
            for i in range(10):
                rec = self.live_recommendation.copy()
                rec["recommendation_id"] = str(uuid.uuid4())
                rec["symbol"] = f"PAIR{i}"
                decision = self.router.make_execution_decision(rec)
                results.append(decision)
                
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=process_recommendations)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Verify no race conditions
        self.assertEqual(len(results), 30)  # 3 threads * 10 decisions each
        
    def test_performance_metrics(self):
        """Test performance measurement and monitoring"""
        start_time = time.time()
        
        # Process 100 execution decisions
        for i in range(100):
            rec = self.live_recommendation.copy()
            rec["recommendation_id"] = str(uuid.uuid4())
            self.router.make_execution_decision(rec)
            
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (< 3 seconds)
        self.assertLess(execution_time, 3.0)
        
        # Average time per decision should be < 30ms
        avg_time = execution_time / 100
        self.assertLess(avg_time, 0.03)
        
    def test_real_time_processing(self):
        """Test real-time recommendation processing"""
        # Mock real-time recommendation stream
        recommendations = []
        for i in range(5):
            rec = self.live_recommendation.copy()
            rec["recommendation_id"] = str(uuid.uuid4())
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            recommendations.append(rec)
            
        processed_count = 0
        
        def process_callback(decision):
            nonlocal processed_count
            processed_count += 1
            
        # Process recommendation stream
        for rec in recommendations:
            decision = self.router.make_execution_decision(rec)
            process_callback(decision)
            
        self.assertEqual(processed_count, len(recommendations))
        
    def test_risk_state_consistency(self):
        """Test risk state consistency across operations"""
        initial_state = self.router.get_risk_state_snapshot()
        
        # Process multiple recommendations
        for i in range(5):
            rec = self.live_recommendation.copy()
            rec["recommendation_id"] = str(uuid.uuid4())
            self.router.make_execution_decision(rec)
            
        final_state = self.router.get_risk_state_snapshot()
        
        # Verify state consistency
        self.assertIsNotNone(initial_state)
        self.assertIsNotNone(final_state)
        self.assertEqual(type(initial_state), type(final_state))
        
    def test_execution_monitoring_alerts(self):
        """Test execution monitoring and alert generation"""
        # Mock high-risk scenario
        high_risk_rec = self.live_recommendation.copy()
        high_risk_rec["confidence"] = 4.0  # Low confidence
        high_risk_rec["risk_reward"] = 0.8  # Poor risk-reward
        
        decision = self.router.make_execution_decision(high_risk_rec)
        
        # Should generate monitoring alert for high-risk trades
        self.assertIsNotNone(decision)
        if decision.decision == "execute":
            # Check if monitoring alert was generated
            alerts = self.router.get_monitoring_alerts()
            self.assertIsNotNone(alerts)

if __name__ == "__main__":
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
        

# <!-- @GENESIS_MODULE_END: test_signal_execution_router -->
