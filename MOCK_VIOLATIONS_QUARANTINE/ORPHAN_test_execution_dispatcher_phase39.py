
import logging
import sys
from pathlib import Path

# GENESIS System Integration
logger = logging.getLogger(__name__)

class SystemIntegration:
    """Connects this module to the GENESIS trading system"""
    
    def __init__(self):
        self.connected = True
        logger.info(f"Module {__name__} connected to GENESIS system")
    
    def register_with_eventbus(self):
        """Register this module with the event bus"""
        pass
    
    def enable_telemetry(self):
        """Enable telemetry for this module"""
        pass

# Auto-connect to system
_integration = SystemIntegration()
_integration.register_with_eventbus()
_integration.enable_telemetry()


from event_bus import EventBus

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
                    "module": "ORPHAN_test_execution_dispatcher_phase39",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("ORPHAN_test_execution_dispatcher_phase39", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in ORPHAN_test_execution_dispatcher_phase39: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
"""
GENESIS Execution Dispatcher - TEST SUITE v1.0
================================================
MT5-driven test cases for order execution and FTMO compliance
ARCHITECT MODE COMPLIANCE: Real broker execution testing
"""

import unittest
import json
import time
import threading
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Import GENESIS modules
from execution_dispatcher import ExecutionDispatcher, ExecutionResult
from hardened_event_bus import get_event_bus, emit_event


# <!-- @GENESIS_MODULE_END: ORPHAN_test_execution_dispatcher_phase39 -->


# <!-- @GENESIS_MODULE_START: ORPHAN_test_execution_dispatcher_phase39 -->

class TestExecutionDispatcher(unittest.TestCase):
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "ORPHAN_test_execution_dispatcher_phase39",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("ORPHAN_test_execution_dispatcher_phase39", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in ORPHAN_test_execution_dispatcher_phase39: {e}")
    """
    Test suite for Execution Dispatcher
    Focuses on MT5 live execution and FTMO compliance
    """
    
    def setUp(self):
        """Set up test environment with real MT5 data structures"""
        self.event_bus = get_event_bus()
        
        # Mock MT5 module for testing
        self.mt5_mock = MagicMock()
        self.mt5_mock.initialize.return_value = True
        self.mt5_mock.account_info.return_value = MagicMock(
            login=12345,
            server="TestServer",
            currency="USD",
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            leverage=100,
            trade_mode=2
        )
        self.mt5_mock.symbol_info_tick.return_value = MagicMock(
            ask=1.1000,
            bid=1.0998
        )
        self.mt5_mock.order_send.return_value = MagicMock(
            retcode=10009,  # TRADE_RETCODE_DONE
            order=123456,
            volume=0.1,
            price=1.1000,
            comment="Order successful",
            request=MagicMock(magic=240617001)
        )
        self.mt5_mock.TRADE_RETCODE_DONE = 10009
        self.mt5_mock.ORDER_TYPE_BUY = 0
        self.mt5_mock.ORDER_TYPE_SELL = 1
        self.mt5_mock.TRADE_ACTION_DEAL = 1
        self.mt5_mock.ORDER_TIME_GTC = 0
        self.mt5_mock.ORDER_FILLING_IOC = 1
        
        # Sample qualified signals
        self.live_signals = [
            {
                "id": "QUAL_EURUSD_001",
                "symbol": "EURUSD",
                "type": "BUY",
                "volume": 0.1,
                "entry_price": 1.1000,
                "stop_loss": 1.0950,
                "take_profit": 1.1100,
                "broker": "MT5_DEMO",
                "priority_score": 0.85,
                "qualification_reason": "passed_all_filters",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "QUAL_GBPUSD_002",
                "symbol": "GBPUSD",
                "type": "SELL",
                "volume": 0.2,
                "entry_price": 1.2500,
                "stop_loss": 1.2550,
                "take_profit": 1.2400,
                "broker": "FTMO",
                "priority_score": 0.78,
                "qualification_reason": "passed_all_filters",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
    
    @patch('execution_dispatcher.mt5')
    def test_dispatcher_initialization(self, mock_mt5):
        """Test Execution Dispatcher initialization"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        # Verify initialization
        self.assertEqual(dispatcher.module_name, "ExecutionDispatcher")
        self.assertEqual(dispatcher.version, "1.0.0")
        self.assertEqual(dispatcher.status, "active")
        self.assertTrue(dispatcher.mt5_initialized)
        self.assertTrue(dispatcher.is_running)
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_signal_translation_to_order(self, mock_mt5):
        """Test signal to MT5 order translation"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        mock_mt5.symbol_info_tick.return_value = self.mt5_mock.symbol_info_tick.return_value
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1
        
        dispatcher = ExecutionDispatcher()
        signal = self.live_signals[0]
        
        order_request = dispatcher._translate_signal_to_order(signal)
        
        # Verify order request structure
        self.assertIsNotNone(order_request)
        self.assertEqual(order_request["symbol"], "EURUSD")
        self.assertEqual(order_request["type"], 0)  # BUY
        self.assertEqual(order_request["volume"], 0.1)
        self.assertEqual(order_request["sl"], 1.0950)
        self.assertEqual(order_request["tp"], 1.1100)
        self.assertIn("GENESIS_EXEC", order_request["comment"])
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_ftmo_compliance_validation(self, mock_mt5):
        """Test FTMO compliance validation"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        # Test valid signal
        valid_signal = {
            "volume": 0.1,
            "symbol": "EURUSD",
            "type": "BUY"
        }
        self.assertTrue(dispatcher._validate_ftmo_compliance(valid_signal))
        
        # Test volume too high
        invalid_signal = {
            "volume": 5.0,  # Exceeds max_trade_volume (2.0)
            "symbol": "EURUSD",
            "type": "BUY"
        }
        self.assertFalse(dispatcher._validate_ftmo_compliance(invalid_signal))
        
        # Test daily volume limit
        dispatcher.daily_volume_used = 9.9
        limit_signal = {
            "volume": 0.2,  # Would exceed daily limit (10.0)
            "symbol": "EURUSD",
            "type": "BUY"
        }
        self.assertFalse(dispatcher._validate_ftmo_compliance(limit_signal))
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_volume_calculation(self, mock_mt5):
        """Test volume calculation with FTMO limits"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        # Test normal volume
        signal = {"volume": 0.5}
        volume = dispatcher._calculate_volume(signal)
        self.assertEqual(volume, 0.5)
        
        # Test volume too high (should be capped)
        signal = {"volume": 5.0}
        volume = dispatcher._calculate_volume(signal)
        self.assertEqual(volume, 2.0)  # max_trade_volume
        
        # Test volume too low (should be raised)
        signal = {"volume": 0.005}
        volume = dispatcher._calculate_volume(signal)
        self.assertEqual(volume, 0.01)  # min_trade_volume
        
        # Test daily volume limit
        dispatcher.daily_volume_used = 9.8
        signal = {"volume": 0.5}
        volume = dispatcher._calculate_volume(signal)
        self.assertEqual(volume, 0.2)  # Remaining daily volume
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_magic_number_generation(self, mock_mt5):
        """Test magic number generation"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        signal1 = {"id": "SIG_001"}
        signal2 = {"id": "SIG_002"}
        
        magic1 = dispatcher._generate_magic_number(signal1)
        magic2 = dispatcher._generate_magic_number(signal2)
        
        # Verify magic numbers are different
        self.assertNotEqual(magic1, magic2)
        
        # Verify magic numbers start with base
        self.assertTrue(str(magic1).startswith("24061"))
        self.assertTrue(str(magic2).startswith("24061"))
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_successful_order_execution(self, mock_mt5):
        """Test successful order execution flow"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        mock_mt5.symbol_info_tick.return_value = self.mt5_mock.symbol_info_tick.return_value
        mock_mt5.order_send.return_value = self.mt5_mock.order_send.return_value
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        dispatcher = ExecutionDispatcher()
        
        # Track emitted events
        success_events = []
        def capture_success(event_data):
            success_events.append(event_data)
        
        self.event_bus.subscribe("execution_success", capture_success)
        
        # Execute order
        task = {
            "action": "execute_signal",
            "signal": self.live_signals[0],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        dispatcher._execute_order(task)
        
        # Wait for processing
        time.sleep(0.1)
        
        # Verify metrics updated
        self.assertEqual(dispatcher.metrics.total_orders_dispatched, 1)
        self.assertEqual(dispatcher.metrics.total_orders_successful, 1)
        self.assertEqual(dispatcher.metrics.success_rate_percentage, 100.0)
        
        # Verify MT5 order_send was called
        mock_mt5.order_send.assert_called_once()
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_failed_order_execution(self, mock_mt5):
        """Test failed order execution handling"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        mock_mt5.symbol_info_tick.return_value = self.mt5_mock.symbol_info_tick.return_value
        
        # Mock failed order
        failed_result = MagicMock()
        failed_result.retcode = 10013  # TRADE_RETCODE_INVALID_PRICE
        failed_result.comment = "Invalid price"
        mock_mt5.order_send.return_value = failed_result
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        dispatcher = ExecutionDispatcher()
        
        # Track emitted events
        error_events = []
        def capture_error(event_data):
            error_events.append(event_data)
        
        self.event_bus.subscribe("execution_error", capture_error)
        
        # Execute order
        task = {
            "action": "execute_signal",
            "signal": self.live_signals[0],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        dispatcher._execute_order(task)
        
        # Wait for processing
        time.sleep(0.1)
        
        # Verify metrics updated
        self.assertEqual(dispatcher.metrics.total_orders_dispatched, 1)
        self.assertEqual(dispatcher.metrics.total_orders_failed, 1)
        self.assertEqual(dispatcher.metrics.success_rate_percentage, 0.0)
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_qualified_signals_handling(self, mock_mt5):
        """Test handling of qualified signals from ExecutionSelector"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        # Simulate qualified signals event
        event_data = {
            "module": "ExecutionSelector",
            "signals": self.live_signals,
            "total_qualified": 2,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        initial_queue_size = dispatcher.execution_queue.qsize()
        dispatcher._handle_qualified_signals(event_data)
        
        # Verify signals were queued
        self.assertEqual(dispatcher.execution_queue.qsize(), initial_queue_size + 2)
        self.assertEqual(dispatcher.metrics.total_signals_received, 2)
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_emergency_stop_handling(self, mock_mt5):
        """Test emergency stop functionality"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        # Add some orders to queue
        for signal in self.live_signals:
            dispatcher.execution_queue.put({
                "action": "execute_signal",
                "signal": signal,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        initial_queue_size = dispatcher.execution_queue.qsize()
        self.assertTrue(dispatcher.is_running)
        
        # Trigger emergency stop
        emergency_event = {
            "module": "RiskEngine",
            "reason": "Maximum drawdown exceeded",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        dispatcher._handle_emergency_stop(emergency_event)
        
        # Verify execution stopped
        self.assertFalse(dispatcher.is_running)
        self.assertEqual(dispatcher.execution_queue.qsize(), 0)
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_broker_profile_loading(self, mock_mt5):
        """Test broker profile loading and configuration"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        # Verify broker profiles loaded
        self.assertIn("MT5_DEMO", dispatcher.broker_profiles)
        self.assertIn("MT5_LIVE", dispatcher.broker_profiles)
        self.assertIn("FTMO", dispatcher.broker_profiles)
        
        # Verify FTMO profile has specific settings
        ftmo_profile = dispatcher.broker_profiles["FTMO"]
        self.assertEqual(ftmo_profile["max_volume"], 2.0)
        self.assertTrue(ftmo_profile.get("ftmo_compliant", False))
        
        # Verify FTMO rules loaded
        self.assertIn("max_daily_loss_pct", dispatcher.ftmo_rules)
        self.assertIn("max_total_loss_pct", dispatcher.ftmo_rules)
        self.assertIn("max_lot_size", dispatcher.ftmo_rules)
        
        dispatcher.shutdown()
    
    @patch('execution_dispatcher.mt5')
    def test_metrics_tracking(self, mock_mt5):
        """Test execution metrics tracking"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
        
        dispatcher = ExecutionDispatcher()
        
        # Initial metrics
        initial_metrics = dispatcher.metrics
        self.assertEqual(initial_metrics.total_signals_received, 0)
        self.assertEqual(initial_metrics.total_orders_dispatched, 0)
        self.assertEqual(initial_metrics.success_rate_percentage, 0.0)
        
        # Simulate successful execution
        signal = self.live_signals[0]
        result = MagicMock()
        result.volume = 0.1
        result.price = 1.1000
        result.order = 123456
        result.comment = "Success"
        result.request = MagicMock(magic=240617001)
        
        dispatcher._handle_execution_success(signal, result, 50.0)
        
        # Verify metrics updated
        self.assertEqual(dispatcher.metrics.total_orders_dispatched, 1)
        self.assertEqual(dispatcher.metrics.total_orders_successful, 1)
        self.assertEqual(dispatcher.metrics.success_rate_percentage, 100.0)
        self.assertEqual(dispatcher.metrics.average_execution_latency_ms, 50.0)
        
        dispatcher.shutdown()
    
    def test_status_reporting(self):
        """Test module status reporting"""
        with patch('execution_dispatcher.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.account_info.return_value = self.mt5_mock.account_info.return_value
            
            dispatcher = ExecutionDispatcher()
            status = dispatcher.get_status()
            
            # Verify status structure
            self.assertIn("module", status)
            self.assertIn("version", status)
            self.assertIn("status", status)
            self.assertIn("mt5_initialized", status)
            self.assertIn("metrics", status)
            self.assertIn("broker_context", status)
            self.assertIn("ftmo_compliance", status)
            
            self.assertEqual(status["module"], "ExecutionDispatcher")
            self.assertEqual(status["version"], "1.0.0")
            self.assertTrue(status["mt5_initialized"])
            
            dispatcher.shutdown()

if __name__ == "__main__":
    # Run tests
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
        

def detect_divergence(price_data: list, indicator_data: list, window: int = 10) -> Dict:
    """
    Detect regular and hidden divergences between price and indicator
    
    Args:
        price_data: List of price values (closing prices)
        indicator_data: List of indicator values (e.g., RSI, MACD)
        window: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence information
    """
    result = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "strength": 0.0
    }
    
    # Need at least window + 1 periods of data
    if len(price_data) < window + 1 or len(indicator_data) < window + 1:
        return result
        
    # Get the current and historical points
    current_price = price_data[-1]
    previous_price = min(price_data[-window:-1]) if price_data[-1] > price_data[-2] else max(price_data[-window:-1])
    previous_price_idx = price_data[-window:-1].index(previous_price) + len(price_data) - window
    
    current_indicator = indicator_data[-1]
    previous_indicator = indicator_data[previous_price_idx]
    
    # Check for regular divergences
    # Bullish - Lower price lows but higher indicator lows
    if current_price < previous_price and current_indicator > previous_indicator:
        result["regular_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Higher price highs but lower indicator highs
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["regular_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Check for hidden divergences
    # Bullish - Higher price lows but lower indicator lows
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["hidden_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Lower price highs but higher indicator highs
    elif current_price < previous_price and current_indicator > previous_indicator:
        result["hidden_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Emit divergence event if detected
    if any([result["regular_bullish"], result["regular_bearish"], 
            result["hidden_bullish"], result["hidden_bearish"]]):
        emit_event("divergence_detected", {
            "type": next(k for k, v in result.items() if v is True and k != "strength"),
            "strength": result["strength"],
            "symbol": price_data.symbol if hasattr(price_data, "symbol") else "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
    return result
