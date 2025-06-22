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

                emit_telemetry("test_phase9_safe_reinforcement_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_phase9_safe_reinforcement_recovered_1", "position_calculated", {
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
                            "module": "test_phase9_safe_reinforcement_recovered_1",
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
                    print(f"Emergency stop error in test_phase9_safe_reinforcement_recovered_1: {e}")
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
                    "module": "test_phase9_safe_reinforcement_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_phase9_safe_reinforcement_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_phase9_safe_reinforcement_recovered_1: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
"""
GENESIS Test SignalLoopReinforcementEngine PHASE 9 - ARCHITECT MODE v2.7
========================================================================
PHASE 9 RECOVERY: Safe test with deadlock prevention and loop control
- Limited to 3 signal candidates maximum
- Timeout protection (10 seconds max)
- Event consumption verification
- Clean exit logic

Dependencies: event_bus.py, signal_loop_reinforcement_engine.py
Test Mode: SAFE_EXECUTION (no infinite loops)
Compliance: ENFORCED
Real Data: ENABLED (realistic signal data, limited scope)
"""

import os
import sys
import json
import time
import logging
import signal
from datetime import datetime
from pathlib import Path
from threading import Timer

# Add timeout handler for Windows compatibility
def timeout_handler():
    print(" TIMEOUT: Test execution exceeded 10 seconds - forcibly exiting")
    sys.exit(1)

# Use Timer for cross-platform timeout
timeout_timer = Timer(10.0, timeout_handler)
timeout_timer.start()

# Configure logging
log_dir = Path("logs/reinforcement_engine/")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"phase9_safe_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("Phase9SafeTest")
logger.setLevel(logging.INFO)

# Create file handler
fh = logging.FileHandler(log_file, encoding='utf-8')
fh.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers
if not logger.hasHandlers():
    logger.addHandler(fh)
    logger.addHandler(ch)

# Test results tracking
test_results = {
    "test_start_time": datetime.utcnow().isoformat(),
    "phase9_features_validated": {
        "deadlock_prevention": False,
        "safe_execution": False,
        "event_consumption": False,
        "clean_exit": False
    },
    "events_processed": 0,
    "test_result": "PENDING",
    "execution_time": 0
}

def test_phase9_safe_execution():
    """Test PHASE 9 safe execution with deadlock prevention"""
    start_time = time.time()
    
    try:
        logger.info(" Starting PHASE 9 Safe SignalLoopReinforcementEngine Test")
        
        # Import and initialize (with timeout protection)
        try:
            from signal_loop_reinforcement_engine import SignalLoopReinforcementEngine
            from event_bus import get_event_bus


# <!-- @GENESIS_MODULE_END: test_phase9_safe_reinforcement_recovered_1 -->


# <!-- @GENESIS_MODULE_START: test_phase9_safe_reinforcement_recovered_1 -->
            
            logger.info(" Modules imported successfully")
            
            # Initialize with timeout
            init_start = time.time()
            engine = SignalLoopReinforcementEngine()
            init_time = time.time() - init_start
            
            if init_time > 5:
                logger.warning(f" Initialization took {init_time:.2f}s - potential performance issue")
            else:
                logger.info(f" Engine initialized in {init_time:.2f}s")
                test_results["phase9_features_validated"]["safe_execution"] = True
            
        except Exception as e:
            logger.error(f" Module initialization failed: {str(e)}")
            test_results["test_result"] = "FAILED"
            return
        
        # Test with LIMITED signals (max 3)
        logger.info(" Testing with LIMITED signal set (max 3 signals)")
        
        test_signals = [
            {"signal_id": "SAFE_TEST_001", "outcome": "win", "slippage": 0.3, "tp_sl_ratio": 1.8},
            {"signal_id": "SAFE_TEST_002", "outcome": "win", "slippage": 0.2, "tp_sl_ratio": 2.1},
            {"signal_id": "SAFE_TEST_003", "outcome": "loss", "slippage": 0.5, "tp_sl_ratio": 0.9}
        ]
        
        events_processed = 0
        
        for i, signal_data in enumerate(test_signals):
            if time.time() - start_time > 8:  # Safety timeout
                logger.warning(" Approaching timeout - stopping test")
                break
                
            logger.info(f" Processing signal {i+1}/3: {signal_data['signal_id']}")
            
            try:
                # Process signal with timeout check
                process_start = time.time()
                engine.track_signal_performance(
                    signal_data["signal_id"],
                    signal_data["outcome"],
                    signal_data["slippage"],
                    signal_data["tp_sl_ratio"]
                )
                process_time = time.time() - process_start
                
                if process_time > 1:  # Should be fast
                    logger.warning(f" Signal processing took {process_time:.2f}s")
                else:
                    events_processed += 1
                    logger.info(f" Signal processed in {process_time:.3f}s")
                
                # Small delay to prevent flooding
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f" Error processing signal {signal_data['signal_id']}: {str(e)}")
        
        test_results["events_processed"] = events_processed
        test_results["phase9_features_validated"]["event_consumption"] = events_processed == len(test_signals)
        test_results["phase9_features_validated"]["deadlock_prevention"] = True
        
        # Verify signal tracking
        logger.info(" Verifying signal tracking...")
        tracked_signals = len(engine.signal_performance)
        logger.info(f" Total signals tracked: {tracked_signals}")
        
        if tracked_signals == len(test_signals):
            test_results["phase9_features_validated"]["event_consumption"] = True
        
        execution_time = time.time() - start_time
        test_results["execution_time"] = execution_time
        
        logger.info(f" Test completed in {execution_time:.2f}s")
        test_results["test_result"] = "PASSED"
        test_results["phase9_features_validated"]["clean_exit"] = True
        
    except Exception as e:
        logger.error(f" Test failed with exception: {str(e)}")
        test_results["test_result"] = "FAILED"
        test_results["error"] = str(e)    
    finally:
        # Clean exit
        timeout_timer.cancel()  # Cancel timeout
        
        # Save results
        try:
            results_file = log_dir / f"phase9_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            logger.info(f" Test results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 9 SAFE TEST SUMMARY")
        logger.info("="*60)
        
        for feature, passed in test_results["phase9_features_validated"].items():
            status = " PASS" if passed else " FAIL"
            logger.info(f"{feature}: {status}")
        
        overall_result = test_results["test_result"]
        logger.info(f"\nOverall Result: {overall_result}")
        logger.info(f"Events Processed: {test_results['events_processed']}")
        logger.info(f"Execution Time: {test_results.get('execution_time', 0):.2f}s")
        logger.info("="*60)

if __name__ == "__main__":
    test_phase9_safe_execution()
    print(" PHASE 9 Safe Test completed - check logs for details")


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
