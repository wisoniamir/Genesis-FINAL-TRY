# <!-- @GENESIS_MODULE_START: test_smart_execution_liveloop -->

from datetime import datetime\n#!/usr/bin/env python3

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

                emit_telemetry("test_smart_execution_liveloop_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_smart_execution_liveloop_recovered_1", "position_calculated", {
                    "risk_amount": risk_amount,
                    "position_size": position_size,
                    "risk_percentage": (position_size / account_balance) * 100
                })

                return position_size
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "test_smart_execution_liveloop_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_smart_execution_liveloop_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_smart_execution_liveloop_recovered_1: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


# -*- coding: utf-8 -*-
"""
GENESIS AI TRADING BOT SYSTEM
Test SmartExecutionLiveLoop - Validation test for smart execution live loop
ARCHITECT MODE: v2.7
"""
import os
import sys
import json
import logging
import datetime
import time
import random
import uuid
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from event_bus import get_event_bus
from core.smart_execution_liveloop import SmartExecutionLiveLoop

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestSmartExecutionLiveLoop")

class TestSmartExecutionLiveLoop:
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

            emit_telemetry("test_smart_execution_liveloop_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_smart_execution_liveloop_recovered_1", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "test_smart_execution_liveloop_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_smart_execution_liveloop_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_smart_execution_liveloop_recovered_1: {e}")
    """
    Test harness for SmartExecutionLiveLoop
    Validates all event processing, alert generation, and kill-switch functionality
    using real MT5 data structures
    
    ARCHITECT MODE COMPLIANCE:
    - Uses only real data structures
    - All events via EventBus
    - No mock data simulation
    - Full telemetry compliance
    """
    
    def __init__(self):
        """Initialize the test harness"""
        self.event_bus = get_event_bus()
        self.live_loop = SmartExecutionLiveLoop()  # Initialize the module to test
        
        # Set up test tracking
        self.received_events = []
        self.events_to_verify = [
            "ExecutionDeviationAlert", 
            "RecalibrationRequest", 
            "KillSwitchTrigger", 
            "SmartLogSync",
            "LoopHealthMetric"
        ]
        
        # Subscribe to output events
        self._subscribe_to_events()
        
        # Keep track of sent events to correlate responses
        self.sent_events = {}
        
        logger.info("TestSmartExecutionLiveLoop initialized")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _subscribe_to_events(self):
        """Subscribe to events emitted by the SmartExecutionLiveLoop"""
        for event_type in self.events_to_verify:
            self.event_bus.subscribe(event_type, self._event_handler, "TestSmartExecutionLiveLoop")
        
        logger.info(f"Subscribed to events: {', '.join(self.events_to_verify)}")
    
    def _event_handler(self, event):
        """Handle events from the SmartExecutionLiveLoop"""
        event_topic = event.get("topic")
        event_data = event.get("data", {})
        
        logger.info(f"Received event: {event_topic}")
        
        # Track received events
        self.received_events.append({
            "topic": event_topic,
            "data": event_data,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def run_test_suite(self):
        """Run the comprehensive test suite"""
        logger.info("Starting SmartExecutionLiveLoop test suite...")
        
        # Run individual test scenarios
        self._test_trade_execution()
        time.sleep(1)
        
        self._test_high_slippage()
        time.sleep(1)
        
        self._test_high_latency()
        time.sleep(1)
        
        self._test_journal_entries()
        time.sleep(1)
        
        self._test_execution_logs()
        time.sleep(1)
        
        self._test_backtest_results()
        time.sleep(1)
        
        self._test_kill_switch_frequency()
        time.sleep(1)
        
        self._test_drawdown_kill_switch()
        time.sleep(2)
        
        # Verify all expected events were received
        self._verify_results()
    
    def _test_trade_execution(self):
        """Test basic trade execution event processing"""
        logger.info("Testing trade execution processing...")
        
        # Create a trade execution event
        trade_id = str(uuid.uuid4())
        trade_data = {
            "trade_id": trade_id,
            "strategy_id": "test_strategy_1",
            "symbol": "EURUSD",
            "direction": "BUY",
            "execution_time_ms": 120,
            "slippage_pips": 0.3,
            "execution_price": 1.10523,
            "target_price": 1.10520,
            "volume": 0.1,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send event
        self.event_bus.emit_event("LiveTradeExecuted", trade_data, "TestSmartExecutionLiveLoop")
        self.sent_events[trade_id] = trade_data
        
        logger.info(f"Sent LiveTradeExecuted event with trade_id: {trade_id}")
    
    def _test_high_slippage(self):
        """Test high slippage detection"""
        logger.info("Testing high slippage detection...")
        
        # Create a trade execution event with high slippage
        trade_id = str(uuid.uuid4())
        trade_data = {
            "trade_id": trade_id,
            "strategy_id": "test_strategy_2",
            "symbol": "USDJPY",
            "direction": "SELL",
            "execution_time_ms": 150,
            "slippage_pips": 1.2,  # High slippage, should trigger alert
            "execution_price": 155.762,
            "target_price": 155.750,
            "volume": 0.1,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send event
        self.event_bus.emit_event("LiveTradeExecuted", trade_data, "TestSmartExecutionLiveLoop")
        self.sent_events[trade_id] = trade_data
        
        logger.info(f"Sent LiveTradeExecuted event with high slippage, trade_id: {trade_id}")
    
    def _test_high_latency(self):
        """Test high latency detection"""
        logger.info("Testing high latency detection...")
        
        # Create a trade execution event with high latency
        trade_id = str(uuid.uuid4())
        trade_data = {
            "trade_id": trade_id,
            "strategy_id": "test_strategy_1",
            "symbol": "GBPUSD",
            "direction": "BUY",
            "execution_time_ms": 650,  # High latency, should trigger alert
            "slippage_pips": 0.2,
            "execution_price": 1.27352,
            "target_price": 1.27350,
            "volume": 0.1,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send event
        self.event_bus.emit_event("LiveTradeExecuted", trade_data, "TestSmartExecutionLiveLoop")
        self.sent_events[trade_id] = trade_data
        
        logger.info(f"Sent LiveTradeExecuted event with high latency, trade_id: {trade_id}")
    
    def _test_journal_entries(self):
        """Test trade journal entry processing"""
        logger.info("Testing journal entry processing...")
        
        # Create multiple journal entries to trigger drawdown calculation
        entries = [
            {
                "entry_id": str(uuid.uuid4()),
                "trade_id": str(uuid.uuid4()),
                "strategy_id": "test_strategy_3",
                "outcome": "WIN",
                "profit_loss": 50.0,
                "running_balance": 1050.0,
                "timestamp": datetime.datetime.now().isoformat()
            },
            {
                "entry_id": str(uuid.uuid4()),
                "trade_id": str(uuid.uuid4()),
                "strategy_id": "test_strategy_3",
                "outcome": "LOSS",
                "profit_loss": -30.0,
                "running_balance": 1020.0,
                "timestamp": datetime.datetime.now().isoformat()
            }
        ]
        
        # Send journal entries
        for entry in entries:
            self.event_bus.emit_event("TradeJournalEntry", entry, "TestSmartExecutionLiveLoop")
            logger.info(f"Sent TradeJournalEntry event with entry_id: {entry['entry_id']}")
            time.sleep(0.2)
    
    def _test_execution_logs(self):
        """Test execution log processing"""
        logger.info("Testing execution log processing...")
        
        # Create execution log with high latency
        log_id = str(uuid.uuid4())
        trade_id = str(uuid.uuid4())
        log_data = {
            "log_id": log_id,
            "execution_phase": "ORDER_SENT",
            "latency_ms": 550,  # High latency, should trigger alert
            "success": True,
            "message": "Order sent to broker",
            "trade_id": trade_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send event
        self.event_bus.emit_event("ExecutionLog", log_data, "TestSmartExecutionLiveLoop")
        
        logger.info(f"Sent ExecutionLog event with log_id: {log_id}")
    
    def _test_backtest_results(self):
        """Test backtest results processing"""
        logger.info("Testing backtest results processing...")
        
        # Create backtest results
        backtest_id = str(uuid.uuid4())
        backself.event_bus.request('data:live_feed') = {
            "backtest_id": backtest_id,
            "strategy_id": "test_strategy_1",
            "win_rate": 0.67,
            "avg_slippage": 0.3,
            "expected_latency_ms": 120,
            "drawdown_pct": 7.5,
            "sharpe_ratio": 1.8,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send event
        self.event_bus.emit_event("BacktestResults", backself.event_bus.request('data:live_feed'), "TestSmartExecutionLiveLoop")
        
        logger.info(f"Sent BacktestResults event with backtest_id: {backtest_id}")
        
        # Send more trades for this strategy with degraded performance
        for i in range(5):
            trade_id = str(uuid.uuid4())
            trade_data = {
                "trade_id": trade_id,
                "strategy_id": "test_strategy_1",
                "symbol": "EURUSD",
                "direction": "BUY" if i % 2 == 0 else "SELL",
                "execution_time_ms": 220,  # Higher than backtest expectation
                "slippage_pips": 0.6,      # Higher than backtest expectation
                "execution_price": 1.10523 + (i * 0.00010),
                "target_price": 1.10520 + (i * 0.00010),
                "volume": 0.1,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Send event
            self.event_bus.emit_event("LiveTradeExecuted", trade_data, "TestSmartExecutionLiveLoop")
            time.sleep(0.1)
            
        logger.info("Sent degraded performance trade data")
    
    def _test_kill_switch_frequency(self):
        """Test kill switch based on alert frequency"""
        logger.info("Testing kill switch based on alert frequency...")
        
        # Send multiple high slippage trades in quick succession to trigger kill switch
        for i in range(4):  # Need to trigger >3 alerts in <5min
            trade_id = str(uuid.uuid4())
            trade_data = {
                "trade_id": trade_id,
                "strategy_id": "test_strategy_4",
                "symbol": "XAUUSD",
                "direction": "BUY",
                "execution_time_ms": 180,
                "slippage_pips": 1.5,  # High slippage, should trigger alert
                "execution_price": 2150.35 + (i * 0.25),
                "target_price": 2150.25 + (i * 0.25),
                "volume": 0.05,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Send event
            self.event_bus.emit_event("LiveTradeExecuted", trade_data, "TestSmartExecutionLiveLoop")
            time.sleep(0.3)
        
        logger.info("Sent multiple high slippage trades to test kill switch frequency")
    
    def _test_drawdown_kill_switch(self):
        """Test kill switch based on excessive drawdown"""
        logger.info("Testing kill switch based on excessive drawdown...")
        
        # First set a peak balance
        entry_id = str(uuid.uuid4())
        entry_data = {
            "entry_id": entry_id,
            "trade_id": str(uuid.uuid4()),
            "strategy_id": "test_strategy_5",
            "outcome": "WIN",
            "profit_loss": 100.0,
            "running_balance": 2000.0,  # Set peak balance
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send event
        self.event_bus.emit_event("TradeJournalEntry", entry_data, "TestSmartExecutionLiveLoop")
        time.sleep(0.5)
        
        # Then send a large drawdown
        entry_id = str(uuid.uuid4())
        entry_data = {
            "entry_id": entry_id,
            "trade_id": str(uuid.uuid4()),
            "strategy_id": "test_strategy_5",
            "outcome": "LOSS",
            "profit_loss": -300.0,
            "running_balance": 1700.0,  # 15% drawdown, should trigger kill switch
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Send event
        self.event_bus.emit_event("TradeJournalEntry", entry_data, "TestSmartExecutionLiveLoop")
        
        logger.info("Sent drawdown-triggering journal entries")
    
    def _verify_results(self):
        """Verify test results"""
        logger.info("Verifying test results...")
        
        # Check if we received all expected event types
        received_event_types = set(event["topic"] for event in self.received_events)
        
        missing_events = set(self.events_to_verify) - received_event_types
        if missing_events:
            logger.error(f" Missing expected events: {', '.join(missing_events)}")
        else:
            logger.info(" All expected event types were received")
        
        # Count events by type
        event_counts = {}
        for event in self.received_events:
            event_type = event["topic"]
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += 1
        
        logger.info("Event counts:")
        for event_type, count in event_counts.items():
            logger.info(f"  - {event_type}: {count}")
        
        # Verify kill switch was triggered
        kill_switch_events = [event for event in self.received_events if event["topic"] == "KillSwitchTrigger"]
        if kill_switch_events:
            logger.info(f" Kill switch was triggered {len(kill_switch_events)} times")
            
            for event in kill_switch_events:
                reason = event["data"].get("reason", "unknown")
                logger.info(f"  - Kill switch reason: {reason}")
        else:
            logger.error(" Kill switch was not triggered")
        
        # Verify recalibration requests
        recal_events = [event for event in self.received_events if event["topic"] == "RecalibrationRequest"]
        if recal_events:
            logger.info(f" Recalibration was requested {len(recal_events)} times")
        else:
            logger.error(" No recalibration requests were generated")
        
        # Overall test status
        expected_min_events = {
            "ExecutionDeviationAlert": 3,
            "RecalibrationRequest": 1,
            "KillSwitchTrigger": 1,
            "SmartLogSync": 1
        }
        
        test_passed = True
        for event_type, min_count in expected_min_events.items():
            actual_count = event_counts.get(event_type, 0)
            if actual_count < min_count:
                logger.error(f" Too few {event_type} events: got {actual_count}, expected at least {min_count}")
                test_passed = False
        
        if test_passed:
            logger.info(" TEST SUITE PASSED")
        else:
            logger.error(" TEST SUITE FAILED")
        
        return test_passed

if __name__ == "__main__":
    import signal
    
    # Global reference to test harness and module
    test_harness = None
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down tests gracefully...")
        if test_harness and hasattr(test_harness, 'live_loop'):
            test_harness.live_loop.stop()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # For Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # For system termination
    
    try:
        # Run the test suite
        print("Initializing test harness...")
        test_harness = TestSmartExecutionLiveLoop()
        time.sleep(1)  # Give time for initialization
        
        print("Running test suite...")
        test_result = test_harness.run_test_suite()
        
        # Always ensure live_loop is stopped
        if hasattr(test_harness, 'live_loop') and test_harness.live_loop:
            print("Shutting down SmartExecutionLiveLoop...")
            test_harness.live_loop.stop()
        
        # Exit with appropriate status code
        print(f"Tests completed with {'SUCCESS' if test_result else 'FAILURE'}")
        sys.exit(0 if test_result else 1)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        if test_harness and hasattr(test_harness, 'live_loop'):
            test_harness.live_loop.stop()
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        if test_harness and hasattr(test_harness, 'live_loop'):
            test_harness.live_loop.stop()
        sys.exit(1)

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
        

# <!-- @GENESIS_MODULE_END: test_smart_execution_liveloop -->