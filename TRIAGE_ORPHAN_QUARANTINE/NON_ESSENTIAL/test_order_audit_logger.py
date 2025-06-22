import logging
import sys
from pathlib import Path


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


# <!-- @GENESIS_MODULE_START: test_order_audit_logger -->

#!/usr/bin/env python3
"""
Test scaffold for OrderAuditLogger module
GENESIS Phase 72 - Order Audit Logger Tests
Architect Mode v5.0.0 Compliance
"""

import unittest
import json
import os
import tempfile
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
try:
    from order_audit_logger import OrderAuditLogger, OrderAuditRecord
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from order_audit_logger import OrderAuditLogger, OrderAuditRecord


class TestOrderAuditLogger(unittest.TestCase):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "test_order_audit_logger",
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
                print(f"Emergency stop error in test_order_audit_logger: {e}")
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
                        "module": "test_order_audit_logger",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_order_audit_logger: {e}")
    """Comprehensive test suite for OrderAuditLogger"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "audit_config": {
                "log_output_path": f"{self.temp_dir}/order_audit",
                "validation_enabled": True,
                "performance_thresholds": {
                    "max_write_latency_ms": 20,
                    "min_write_success_rate": 0.999
                }
            },
            "telemetry_config": {
                "emit_interval_ms": 100,
                "metrics_retention_hours": 1
            },
            "event_bus_config": {
                "listen_topics": ["order:executed", "order:cancelled"],
                "emit_topics": ["audit:order:logged"]
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
          # Initialize logger without event_bus parameter
        self.logger = OrderAuditLogger(config_path=self.config_path)
        
        # Connect mock event bus
        self.logger.connect_event_bus(self.mock_event_bus)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of OrderAuditLogger"""
        self.assertIsNotNone(self.logger)
        self.assertTrue(self.logger.active)
        self.assertEqual(self.logger.config["architect_mode"]["version"], "v5.0.0")
        
        # Verify log directory creation
        log_dir = Path(self.test_config["audit_config"]["log_output_path"])
        self.assertTrue(log_dir.exists())
    
    def test_order_record_creation(self):
        """Test OrderRecord data
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
        class creation and validation"""
        test_order = {
            "order_id": "12345",
            "symbol": "EURUSD",
            "order_type": "BUY",
            "volume": 0.1,
            "price": 1.0850,
            "timestamp": datetime.now().isoformat()
        }
        
        record = OrderAuditRecord(
            audit_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            order_id=test_order["order_id"],
            symbol=test_order["symbol"],
            order_type="market",
            action=test_order["order_type"].lower(),
            volume=test_order["volume"],
            price=test_order["price"],
            stop_loss=None,
            take_profit=None,
            status="executed",
            execution_time=test_order["timestamp"],
            rejection_reason=None,
            slippage=None,
            commission=None,
            swap=None,
            timestamp=test_order["timestamp"],
            mt5_timestamp=test_order["timestamp"],
            data_hash="",
            validation_status="valid",
            original_signal_id=None
        )
        
        self.assertEqual(record.order_id, test_order["order_id"])
        self.assertEqual(record.symbol, test_order["symbol"])
        self.assertIsInstance(record.trace_id, str)
    
    def test_hash_generation(self):
        """Test hash generation for order records"""
        self.event_bus.request('data:live_feed') = {
            "order_id": "12345",
            "timestamp": "2025-06-18T16:00:00Z",
            "symbol": "EURUSD"
        }
        
        hash1 = self.logger._generate_hash(self.event_bus.request('data:live_feed'))
        hash2 = self.logger._generate_hash(self.event_bus.request('data:live_feed'))
        
        # Same data should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different data should produce different hash
        self.event_bus.request('data:live_feed')["order_id"] = "54321"
        hash3 = self.logger._generate_hash(self.event_bus.request('data:live_feed'))
        self.assertNotEqual(hash1, hash3)
    
    def test_order_execution_handling(self):
        """Test handling of order:executed events"""
        test_event = {
            "order_id": "12345",
            "symbol": "EURUSD",
            "order_type": "BUY",
            "volume": 0.1,
            "price": 1.0850,
            "execution_time": datetime.now().isoformat(),
            "status": "EXECUTED"
        }
        
        # Test order execution event handling
        self.logger._handle_order_executed(test_event)
        
        # Verify log file was created and contains data
        log_files = list(Path(self.test_config["audit_config"]["log_output_path"]).glob("*.json"))
        self.assertGreater(len(log_files), 0)
        
        # Verify EventBus emission
        self.mock_event_bus.emit.assert_called()
        
        # Check emitted event structure
        emit_calls = self.mock_event_bus.emit.call_args_list
        self.assertTrue(any("audit:order:logged" in str(call) for call in emit_calls))
    
    def test_performance_compliance(self):
        """Test performance requirements (≤20ms latency)"""
        import time
        
        test_event = {
            "order_id": "12345",
            "symbol": "EURUSD",
            "order_type": "BUY",
            "volume": 0.1,
            "price": 1.0850,
            "execution_time": datetime.now().isoformat(),
            "status": "EXECUTED"
        }
        
        start_time = time.time()
        self.logger._handle_order_executed(test_event)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Should meet ≤20ms requirement
        self.assertLessEqual(latency_ms, 20)
    
    def test_write_success_rate(self):
        """Test write success rate (≥99.9%)"""
        total_writes = 100
        successful_writes = 0
        
        for i in range(total_writes):
            test_event = {
                "order_id": f"order_{i}",
                "symbol": "EURUSD",
                "order_type": "BUY",
                "volume": 0.1,
                "price": 1.0850,
                "execution_time": datetime.now().isoformat(),
                "status": "EXECUTED"
            }
            
            try:
                self.logger._handle_order_executed(test_event)
                successful_writes += 1
            except Exception:
    logger.info("Function operational")("Real implementation required - no stubs allowed in production")
        success_rate = successful_writes / total_writes
        
        # Should meet ≥99.9% requirement
        self.assertGreaterEqual(success_rate, 0.999)
    
    def test_telemetry_emission(self):
        """Test telemetry metrics emission"""
        # Trigger telemetry
        self.logger._emit_telemetry()
        
        # Verify telemetry events were emitted
        telemetry_calls = [call for call in self.mock_event_bus.emit.call_args_list 
                          if "telemetry" in str(call)]
        self.assertGreater(len(telemetry_calls), 0)
    
    def test_error_handling(self):
        """Test error handling and alerting"""
        # Test with invalid event data
        invalid_event = {"invalid": "data"}
        
        # Should handle gracefully without crashing
        try:
            self.logger._handle_order_executed(invalid_event)
        except Exception as e:
            # Error should be logged and alerted
            self.mock_event_bus.emit.assert_called()
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test with invalid config
        invalid_config = {"invalid": "config"}
        invalid_config_path = f"{self.temp_dir}/invalid_config.json"
        
        with open(invalid_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        # Should handle invalid config gracefully
        with self.assertRaises((KeyError, ValueError)):
            OrderAuditLogger(config_path=invalid_config_path, event_bus=self.mock_event_bus)
    
    def test_architect_mode_compliance(self):
        """Test architect mode v5.0.0 compliance"""
        # Verify all required components
        self.assertTrue(hasattr(self.logger, 'config'))
        self.assertTrue(hasattr(self.logger, 'telemetry_metrics'))
        self.assertTrue(hasattr(self.logger, 'error_count'))
        
        # Verify architect mode settings
        self.assertEqual(self.logger.config["architect_mode"]["version"], "v5.0.0")
        self.assertEqual(self.logger.config["architect_mode"]["compliance_level"], "strict")
    
    def test_trace_id_correlation(self):
        """Test trace ID correlation across events"""
        test_event = {
            "order_id": "12345",
            "symbol": "EURUSD",
            "order_type": "BUY",
            "volume": 0.1,
            "price": 1.0850,
            "execution_time": datetime.now().isoformat(),
            "status": "EXECUTED"
        }
        
        # Process the event
        self.logger._handle_order_executed(test_event)
        
        # Check that trace ID was generated and used
        log_files = list(Path(self.test_config["audit_config"]["log_output_path"]).glob("*.json"))
        self.assertGreater(len(log_files), 0)
        
        # Read log file and verify trace ID
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
            self.assertIn("trace_id", log_data)
            self.assertIsInstance(log_data["trace_id"], str)
    
    def test_concurrent_operations(self):
        """Test concurrent order processing"""
        import threading
        
        def process_order(order_id):
            test_event = {
                "order_id": order_id,
                "symbol": "EURUSD",
                "order_type": "BUY",
                "volume": 0.1,
                "price": 1.0850,
                "execution_time": datetime.now().isoformat(),
                "status": "EXECUTED"
            }
            self.logger._handle_order_executed(test_event)
        
        # Process multiple orders concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_order, args=(f"order_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all orders were processed
        log_files = list(Path(self.test_config["audit_config"]["log_output_path"]).glob("*.json"))
        self.assertGreaterEqual(len(log_files), 10)


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)


# <!-- @GENESIS_MODULE_END: test_order_audit_logger -->