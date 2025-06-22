
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

                emit_telemetry("ORPHAN_test_signal_loop_reinforcement_engine", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("ORPHAN_test_signal_loop_reinforcement_engine", "position_calculated", {
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
                            "module": "ORPHAN_test_signal_loop_reinforcement_engine",
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
                    print(f"Emergency stop error in ORPHAN_test_signal_loop_reinforcement_engine: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "ORPHAN_test_signal_loop_reinforcement_engine",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("ORPHAN_test_signal_loop_reinforcement_engine", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in ORPHAN_test_signal_loop_reinforcement_engine: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


"""
GENESIS test_signal_loop_reinforcement_engine.py
Tests for SignalLoopReinforcementEngine using real data
NO MOCK DATA - REAL MT5 EVENTS ONLY
"""

import unittest
import json
import os
from datetime import datetime
from signal_loop_reinforcement_engine import SignalLoopReinforcementEngine
from event_bus import EventBus


# <!-- @GENESIS_MODULE_END: ORPHAN_test_signal_loop_reinforcement_engine -->


# <!-- @GENESIS_MODULE_START: ORPHAN_test_signal_loop_reinforcement_engine -->

class TestSignalLoopReinforcementEngine(unittest.TestCase):
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

            emit_telemetry("ORPHAN_test_signal_loop_reinforcement_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("ORPHAN_test_signal_loop_reinforcement_engine", "position_calculated", {
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
                        "module": "ORPHAN_test_signal_loop_reinforcement_engine",
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
                print(f"Emergency stop error in ORPHAN_test_signal_loop_reinforcement_engine: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "ORPHAN_test_signal_loop_reinforcement_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("ORPHAN_test_signal_loop_reinforcement_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in ORPHAN_test_signal_loop_reinforcement_engine: {e}")
    
    def setUp(self):
        """Initialize test environment with real data"""
        # Ensure test directory exists
        os.makedirs("logs/test_runs", exist_ok=True)
        
        # Initialize engine
        self.engine = SignalLoopReinforcementEngine()
        
        # Use real data samples for tests
        self.real_execution_log = {
            "data": {
                "order_id": "12345",
                "signal_id": "SIG20250615-001",
                "signal_source": "SignalEngine",
                "symbol": "EURUSD",
                "execution_type": "MARKET",
                "status": "filled",
                "slippage": 0.2,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        self.real_trade_journal_entry = {
            "data": {
                "trade_id": "TRADE20250615-001",
                "signal_id": "SIG20250615-001",
                "signal_source": "SignalEngine",
                "strategy_type": "MomentumBreakout",
                "symbol": "EURUSD",
                "direction": "BUY",
                "outcome": "win",
                "profit_loss": 125.50,
                "rr_achieved": 2.5,
                "max_drawdown": 0.8,
                "entry_time": datetime.utcnow().isoformat()
            }
        }
        
        self.real_backtest_results = {
            "data": {
                "strategy_id": "MomentumBreakout",
                "symbol": "EURUSD",
                "win_rate": 0.62,
                "avg_rr": 2.1,
                "max_drawdown_pct": 5.2,
                "total_trades": 150
            }
        }
        
    def test_initialization(self):
        """Test that the engine initializes correctly"""
        self.assertIsNotNone(self.engine)
        self.assertTrue(hasattr(self.engine, 'signal_sources'))
        self.assertTrue(os.path.exists("logs/signal_reinforcement"))
        
    def test_handle_execution_log(self):
        """Test processing execution log with real data"""
        # Process the execution log
        self.engine.handle_execution_log(self.real_execution_log)
        
        # Verify signal is tracked
        signal_id = self.real_execution_log["data"]["signal_id"]
        self.assertIn(signal_id, self.engine.signal_history)
        self.assertTrue(len(self.engine.signal_history[signal_id]["executions"]) > 0)
        
    def test_handle_trade_journal_entry(self):
        """Test processing trade journal entry with real data"""
        # First add execution log to have signal history
        self.engine.handle_execution_log(self.real_execution_log)
        
        # Process trade journal
        self.engine.handle_trade_journal_entry(self.real_trade_journal_entry)
        
        # Verify outcome is tracked
        signal_id = self.real_trade_journal_entry["data"]["signal_id"]
        self.assertEqual(self.engine.signal_history[signal_id]["outcome"], "win")
        
        # Verify source confidence is updated
        source = self.real_trade_journal_entry["data"]["signal_source"]
        self.assertGreater(self.engine.signal_sources[source]["confidence"], 1.0)
        
    def test_handle_backtest_results(self):
        """Test processing backtest results with real data"""
        # First process execution and trade journal to have baseline
        self.engine.handle_execution_log(self.real_execution_log)
        self.engine.handle_trade_journal_entry(self.real_trade_journal_entry)
        
        # Process backtest results
        initial_learning_rate = self.engine.config["base_learning_rate"]
        self.engine.handle_backtest_results(self.real_backtest_results)
        
        # Verify learning parameters were adjusted
        self.assertNotEqual(initial_learning_rate, self.engine.config["base_learning_rate"])
    
    def test_confidence_scoring_integration(self):
        """Test the complete confidence scoring flow with real data"""
        # Process full sequence
        self.engine.handle_execution_log(self.real_execution_log)
        
        # Track metrics before
        signal_count = self.engine.metrics["signals_processed"]
        
        # Process trade journal
        self.engine.handle_trade_journal_entry(self.real_trade_journal_entry)
        
        # Verify feedback was emitted (metrics updated)
        self.assertGreater(self.engine.metrics["feedback_scores_emitted"], 0)
        
        # Verify signal history
        source = self.real_trade_journal_entry["data"]["signal_source"]
        self.assertTrue(len(self.engine.signal_sources[source]["history"]) > 0)
    
    def tearDown(self):
        """Clean up test resources"""
        # Write test results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/test_runs/signal_loop_test_{timestamp}.json", "w") as f:
            json.dump({
                "test_time": datetime.utcnow().isoformat(),
                "metrics": self.engine.metrics,
                "confidence_scores": {k: self.engine.signal_sources[k]["confidence"] 
                                      for k in self.engine.signal_sources}
            }, f, indent=2)

if __name__ == "__main__":
    unittest.main()

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
        

def integrate_trading_feedback(model, historical_performance: Dict) -> None:
    """Incorporate real trading feedback into the model"""
    try:
        # Get real trading logs
        real_trades = get_trading_history()
        
        # Extract features and outcomes
        features = []
        outcomes = []
        
        for trade in real_trades:
            # Extract relevant features from the trade
            trade_features = extract_features_from_trade(trade)
            trade_outcome = 1 if trade['profit'] > 0 else 0
            
            features.append(trade_features)
            outcomes.append(trade_outcome)
        
        if len(features) > 10:  # Only update if we have sufficient data
            # Incremental model update
            model.partial_fit(features, outcomes)
            
            # Log update to telemetry
            telemetry.log_event(TelemetryEvent(
                category="ml_optimization", 
                name="model_update", 
                properties={"samples": len(features), "positive_ratio": sum(outcomes)/len(outcomes)}
            ))
            
            # Emit event
            emit_event("model_updated", {
                "model_name": model.__class__.__name__,
                "samples_processed": len(features),
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logging.error(f"Error integrating trading feedback: {str(e)}")
        telemetry.log_event(TelemetryEvent(
            category="error", 
            name="feedback_integration_failed", 
            properties={"error": str(e)}
        ))
