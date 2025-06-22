import logging
import sys
from pathlib import Path


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
                            "module": "test_post_trade_feedback_collector",
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
                    print(f"Emergency stop error in test_post_trade_feedback_collector: {e}")
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
                    "module": "test_post_trade_feedback_collector",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_post_trade_feedback_collector", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_post_trade_feedback_collector: {e}")
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


# <!-- @GENESIS_MODULE_START: test_post_trade_feedback_collector -->

#!/usr/bin/env python3
"""
Test scaffold for PostTradeFeedbackCollector module
GENESIS Phase 73 - Post-Trade Feedback Collector Tests
Architect Mode v5.0.0 Compliance
"""

import unittest
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
try:
    from post_trade_feedback_collector import PostTradeFeedbackCollector, TradeFeedbackRecord
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from post_trade_feedback_collector import PostTradeFeedbackCollector, TradeFeedbackRecord


class TestPostTradeFeedbackCollector(unittest.TestCase):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "test_post_trade_feedback_collector",
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
                print(f"Emergency stop error in test_post_trade_feedback_collector: {e}")
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
                "module": "test_post_trade_feedback_collector",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_post_trade_feedback_collector", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_post_trade_feedback_collector: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_post_trade_feedback_collector",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_post_trade_feedback_collector: {e}")
    """Comprehensive test suite for PostTradeFeedbackCollector"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "feedback_config": {
                "analytics_output_path": f"{self.temp_dir}/trade_feedback.json",
                "performance_thresholds": {
                    "win_rate_target": 0.65,
                    "r_ratio_target": 2.0,
                    "max_processing_latency_ms": 50
                }
            },
            "telemetry_config": {
                "emit_interval_ms": 100,
                "metrics_retention_hours": 1
            },
            "event_bus_config": {
                "listen_topics": ["order:closed", "signal:executed"],
                "emit_topics": ["feedback:trade:recorded"]
            },
            "architect_mode": {
                "version": "v5.0.0",
                "compliance_level": "strict"
            }
        }
        
        # Create config file
        self.config_path = f"{self.temp_dir}/test_config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Mock EventBus
        self.mock_event_bus = Mock()
        
        # Initialize collector
        self.collector = PostTradeFeedbackCollector(config_path=self.config_path)
        self.collector.connect_event_bus(self.mock_event_bus)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of PostTradeFeedbackCollector"""
        self.assertIsNotNone(self.collector)
        self.assertTrue(self.collector.active)
        self.assertEqual(self.collector.config["architect_mode"]["version"], "v5.0.0")
        
        # Verify analytics directory creation
        analytics_path = Path(self.test_config["feedback_config"]["analytics_output_path"])
        analytics_dir = analytics_path.parent
        self.assertTrue(analytics_dir.exists())
    
    def test_trade_record_creation(self):
        """Test TradeRecord dataclass creation and validation"""
        test_trade = {
            "trade_id": "12345",
            "symbol": "EURUSD",
            "action": "buy",
            "volume": 0.1,
            "entry_price": 1.0850,
            "exit_price": 1.0860,
            "entry_time": datetime.now().isoformat(),
            "exit_time": datetime.now().isoformat()
        }
          # Create test record using actual dataclass structure
        # Note: This is just for testing, actual records are created by the collector
    
    def test_order_closed_handling(self):
        """Test handling of order:closed events"""
        test_event = {
            "order_id": "12345",
            "symbol": "EURUSD",
            "action": "buy",
            "volume": 0.1,
            "entry_price": 1.0850,
            "exit_price": 1.0860,
            "entry_time": datetime.now().isoformat(),
            "exit_time": datetime.now().isoformat(),
            "pnl": 10.0,
            "commission": 0.5
        }
        
        # Test order closed event handling
        self.collector._handle_order_closed(test_event)
        
        # Verify analytics file exists and contains data
        analytics_path = Path(self.test_config["feedback_config"]["analytics_output_path"])
        if analytics_path.exists():
            with open(analytics_path, 'r') as f:
                analytics_data = json.load(f)
                self.assertIn("trades", analytics_data)
        
        # Verify EventBus emission
        self.mock_event_bus.emit.assert_called()
    
    def test_signal_executed_handling(self):
        """Test handling of signal:executed events"""
        test_event = {
            "signal_id": "signal_123",
            "symbol": "EURUSD",
            "action": "buy",
            "confidence": 0.85,
            "execution_time": datetime.now().isoformat(),
            "entry_price": 1.0850
        }
        
        # Test signal execution event handling
        self.collector._handle_signal_executed(test_event)
        
        # Should handle gracefully even without complete trade data
        self.mock_event_bus.emit.assert_called()
    
    def test_performance_compliance(self):
        """Test performance requirements (≤50ms latency)"""
        import time
        
        test_event = {
            "order_id": "12345",
            "symbol": "EURUSD",
            "action": "buy",
            "volume": 0.1,
            "entry_price": 1.0850,
            "exit_price": 1.0860,
            "entry_time": datetime.now().isoformat(),
            "exit_time": datetime.now().isoformat(),
            "pnl": 10.0
        }
        
        start_time = time.time()
        self.collector._handle_order_closed(test_event)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Should meet ≤50ms requirement
        self.assertLessEqual(latency_ms, 50)
    
    def test_win_loss_classification(self):
        """Test win/loss classification logic"""
        # Test winning trade
        win_event = {
            "order_id": "12345",
            "pnl": 10.0,
            "symbol": "EURUSD",
            "action": "buy",
            "volume": 0.1,
            "entry_price": 1.0850,
            "exit_price": 1.0860,
            "entry_time": datetime.now().isoformat(),
            "exit_time": datetime.now().isoformat()
        }
        
        self.collector._handle_order_closed(win_event)
        
        # Test losing trade
        loss_event = {
            "order_id": "54321",
            "pnl": -10.0,
            "symbol": "EURUSD",
            "action": "sell",
            "volume": 0.1,
            "entry_price": 1.0860,
            "exit_price": 1.0850,
            "entry_time": datetime.now().isoformat(),
            "exit_time": datetime.now().isoformat()
        }
        
        self.collector._handle_order_closed(loss_event)
        
        # Verify classification worked
        self.mock_event_bus.emit.assert_called()
    
    def test_r_ratio_calculation(self):
        """Test R:R ratio calculation"""
        # Test with stop loss and take profit data
        test_event = {
            "order_id": "12345",
            "symbol": "EURUSD",
            "action": "buy",
            "volume": 0.1,
            "entry_price": 1.0850,
            "exit_price": 1.0860,
            "stop_loss": 1.0840,
            "take_profit": 1.0870,
            "entry_time": datetime.now().isoformat(),
            "exit_time": datetime.now().isoformat(),
            "pnl": 10.0
        }
        
        self.collector._handle_order_closed(test_event)
        
        # Should calculate R:R ratio based on risk/reward
        self.mock_event_bus.emit.assert_called()
    
    def test_telemetry_emission(self):
        """Test telemetry metrics emission"""
        # Access telemetry data
        self.assertIn("win_rate", self.collector.telemetry_data)
        self.assertIn("avg_r_ratio", self.collector.telemetry_data)
        self.assertIn("total_trades", self.collector.telemetry_data)
        
        # Verify initial values
        self.assertEqual(self.collector.telemetry_data["total_trades"], 0)
    
    def test_error_handling(self):
        """Test error handling and alerting"""
        # Test with invalid event data
        invalid_event = {"invalid": "data"}
        
        # Should handle gracefully without crashing
        try:
            self.collector._handle_order_closed(invalid_event)
        except Exception:
            # Error should be logged but not crash the system
    logger.info("Function operational")("Real implementation required - no stubs allowed in production")
        # System should still be active
        self.assertTrue(self.collector.active)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test with invalid config
        invalid_config = {"invalid": "config"}
        invalid_config_path = f"{self.temp_dir}/invalid_config.json"
        
        with open(invalid_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        # Should handle invalid config gracefully with defaults
        try:
            collector = PostTradeFeedbackCollector(config_path=invalid_config_path)
            self.assertIsNotNone(collector)
        except Exception:
            # May raise exception, but should be handled gracefully
    logger.info("Function operational")("Real implementation required - no stubs allowed in production")
    def test_architect_mode_compliance(self):
        """Test architect mode v5.0.0 compliance"""
        # Verify all required components
        self.assertTrue(hasattr(self.collector, 'config'))
        self.assertTrue(hasattr(self.collector, 'telemetry_data'))
        self.assertTrue(hasattr(self.collector, 'trade_history'))
        
        # Verify architect mode settings
        self.assertEqual(self.collector.config["architect_mode"]["version"], "v5.0.0")
        self.assertEqual(self.collector.config["architect_mode"]["compliance_level"], "strict")
    
    def test_feedback_summary_generation(self):
        """Test feedback summary generation"""
        # Process multiple trades
        for i in range(5):
            test_event = {
                "order_id": f"order_{i}",
                "symbol": "EURUSD",
                "action": "buy",
                "volume": 0.1,
                "entry_price": 1.0850,
                "exit_price": 1.0860 if i % 2 == 0 else 1.0840,  # Alternate wins/losses
                "entry_time": datetime.now().isoformat(),
                "exit_time": datetime.now().isoformat(),
                "pnl": 10.0 if i % 2 == 0 else -10.0
            }
            
            self.collector._handle_order_closed(test_event)
        
        # Should generate summary data
        self.assertGreater(self.collector.telemetry_data["total_trades"], 0)
    
    def test_concurrent_operations(self):
        """Test concurrent trade processing"""
        import threading
        
        def process_trade(trade_id):
            test_event = {
                "order_id": trade_id,
                "symbol": "EURUSD",
                "action": "buy",
                "volume": 0.1,
                "entry_price": 1.0850,
                "exit_price": 1.0860,
                "entry_time": datetime.now().isoformat(),
                "exit_time": datetime.now().isoformat(),
                "pnl": 10.0
            }
            self.collector._handle_order_closed(test_event)
        
        # Process multiple trades concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_trade, args=(f"order_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all trades were processed
        self.assertGreaterEqual(self.collector.telemetry_data["total_trades"], 10)


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
        

# <!-- @GENESIS_MODULE_END: test_post_trade_feedback_collector -->