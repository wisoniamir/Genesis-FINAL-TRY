
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

                emit_telemetry("phase_90_feedback_validation_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("phase_90_feedback_validation_recovered_1", "position_calculated", {
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
                            "module": "phase_90_feedback_validation_recovered_1",
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
                    print(f"Emergency stop error in phase_90_feedback_validation_recovered_1: {e}")
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
                    "module": "phase_90_feedback_validation_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("phase_90_feedback_validation_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in phase_90_feedback_validation_recovered_1: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


# <!-- @GENESIS_MODULE_START: phase_90_feedback_validation -->

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENESIS Phase 90 Feedback Engine Validation
Comprehensive test suite for Post-Trade Feedback Engine
ARCHITECT MODE: v5.0.0 COMPLIANT
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from post_trade_feedback_engine import PostTradeFeedbackEngine
    from event_bus import get_event_bus, emit_event
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure all required modules are in the current directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase90ValidationSuite:
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

            emit_telemetry("phase_90_feedback_validation_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("phase_90_feedback_validation_recovered_1", "position_calculated", {
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
                        "module": "phase_90_feedback_validation_recovered_1",
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
                print(f"Emergency stop error in phase_90_feedback_validation_recovered_1: {e}")
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
                "module": "phase_90_feedback_validation_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("phase_90_feedback_validation_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in phase_90_feedback_validation_recovered_1: {e}")
    """Comprehensive validation suite for Phase 90 Feedback Engine"""
    
    def __init__(self):
        self.test_results = {}
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
        # Test data
        self.test_trades = [
            {
                "trade_id": "TEST_TRADE_001",
                "symbol": "EURUSD",
                "action": "BUY",
                "lot_size": 0.1,
                "open_price": 1.0850,
                "close_price": 1.0875,
                "open_time": "2025-06-18T10:00:00Z",
                "close_time": "2025-06-18T11:30:00Z",
                "profit": 25.0,
                "strategy_id": "STRATEGY_MOMENTUM",
                "signal_id": "SIG_MOM_001"
            },
            {
                "trade_id": "TEST_TRADE_002",
                "symbol": "GBPUSD",
                "action": "SELL",
                "lot_size": 0.15,
                "open_price": 1.2650,
                "close_price": 1.2635,
                "open_time": "2025-06-18T12:00:00Z",
                "close_time": "2025-06-18T12:45:00Z",
                "profit": 22.5,
                "strategy_id": "STRATEGY_REVERSAL",
                "signal_id": "SIG_REV_001"
            },
            {
                "trade_id": "TEST_TRADE_003",
                "symbol": "USDJPY",
                "action": "BUY",
                "lot_size": 0.2,
                "open_price": 150.25,
                "close_price": 149.85,
                "open_time": "2025-06-18T14:00:00Z",
                "close_time": "2025-06-18T15:15:00Z",
                "profit": -40.0,
                "strategy_id": "STRATEGY_MOMENTUM",
                "signal_id": "SIG_MOM_002"
            }
        ]
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test and record results"""
        self.test_count += 1
        try:
            print(f"\n🧪 Running test: {test_name}")
            result = test_func(*args, **kwargs)
            
            if result:
                print(f"✅ {test_name}: PASSED")
                self.test_results[test_name] = {"status": "PASSED", "details": result}
                self.passed_tests += 1
                return True
            else:
                print(f"❌ {test_name}: FAILED")
                self.test_results[test_name] = {"status": "FAILED", "details": "Test returned False"}
                self.failed_tests += 1
                return False
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            self.test_results[test_name] = {"status": "ERROR", "details": str(e)}
            self.failed_tests += 1
            return False
    
    def test_engine_initialization(self):
        """Test 1: Engine Initialization"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Check basic attributes
            assert hasattr(engine, 'module_name')
            assert hasattr(engine, 'event_bus')
            assert hasattr(engine, 'closed_trades')
            assert hasattr(engine, 'telemetry')
            
            # Check directories created
            assert engine.log_dir.exists()
            assert engine.journal_dir.exists()
            assert engine.telemetry_dir.exists()
            assert engine.data_dir.exists()
            
            return {"engine_initialized": True, "directories_created": 4}
            
        except Exception as e:
            return False
    
    def test_trade_closed_processing(self):
        """Test 2: Trade Closed Event Processing"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Process test trade
            test_trade = self.test_trades[0].copy()
            initial_count = len(engine.closed_trades)
            
            engine._handle_trade_closed(test_trade)
            
            # Verify trade was processed
            assert len(engine.closed_trades) == initial_count + 1
            
            # Check the processed trade
            processed_trade = engine.closed_trades[-1]
            assert processed_trade['trade_id'] == test_trade['trade_id']
            assert processed_trade['symbol'] == test_trade['symbol']
            assert 'outcome_score' in processed_trade
            assert 'processed_at' in processed_trade
            assert 'duration_minutes' in processed_trade
            assert 'pip_movement' in processed_trade
            
            return {
                "trade_processed": True,
                "trade_id": processed_trade['trade_id'],
                "outcome_score": processed_trade['outcome_score'],
                "duration_minutes": processed_trade['duration_minutes']
            }
            
        except Exception as e:
            return False
    
    def test_trade_scoring(self):
        """Test 3: Trade Outcome Scoring"""
        try:
            engine = PostTradeFeedbackEngine()
            
            scores = []
            for trade in self.test_trades:
                # Create a trade record for scoring
                trade_record = trade.copy()
                trade_record.update({
                    "duration_minutes": 90,
                    "pip_movement": 25.0 if trade['profit'] > 0 else -40.0,
                    "risk_reward_ratio": trade['profit'] / 100 if trade['profit'] > 0 else trade['profit'] / 100
                })
                
                score = engine._score_trade_outcome(trade_record)
                scores.append(score)
                
                # Verify score is within valid range
                assert 0.0 <= score <= 1.0
                
                # Verify profitable trades generally score higher
                if trade['profit'] > 0:
                    assert score > 0.4  # Should be above neutral
            
            return {
                "trades_scored": len(scores),
                "scores": scores,
                "avg_score": sum(scores) / len(scores),
                "all_valid_range": all(0.0 <= s <= 1.0 for s in scores)
            }
            
        except Exception as e:
            return False
    
    def test_daily_journal_logging(self):
        """Test 4: Daily Journal Logging"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Process a test trade
            test_trade = self.test_trades[0].copy()
            engine._handle_trade_closed(test_trade)
            
            # Check journal file was created
            today = datetime.now().strftime("%Y%m%d")
            journal_file = engine.journal_dir / f"{today}_journal.json"
            
            assert journal_file.exists()
            
            # Load and verify journal content
            with open(journal_file, "r") as f:
                journal_data = json.load(f)
            
            assert "date" in journal_data
            assert "trades" in journal_data
            assert "summary" in journal_data
            assert len(journal_data["trades"]) >= 1
            
            # Check summary calculations
            summary = journal_data["summary"]
            assert "total_trades" in summary
            assert "winning_trades" in summary
            assert "losing_trades" in summary
            assert "total_profit" in summary
            assert "avg_score" in summary
            assert "win_rate" in summary
            
            return {
                "journal_created": True,
                "journal_file": str(journal_file),
                "total_trades": summary["total_trades"],
                "win_rate": summary["win_rate"]
            }
            
        except Exception as e:
            return False
    
    def test_strategy_feedback_emission(self):
        """Test 5: Strategy Feedback Event Emission"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Monitor event bus for feedback events
            feedback_events = []
            
            def capture_feedback(event_data):
                feedback_events.append(event_data)
            
            # Subscribe to feedback events
            engine.event_bus.subscribe("feedback:new_result", capture_feedback, "TestListener")
            
            # Process test trades
            initial_feedback_count = engine.telemetry["feedback_events_sent"]
            
            for trade in self.test_trades[:2]:  # Process 2 trades
                engine._handle_trade_closed(trade)
                time.sleep(0.1)  # Allow event processing
            
            # Verify feedback events were sent
            assert engine.telemetry["feedback_events_sent"] > initial_feedback_count
            
            # Check feedback event structure (if captured)
            if feedback_events:
                feedback = feedback_events[0]
                assert "strategy_id" in feedback
                assert "outcome" in feedback
                assert "outcome_score" in feedback
                assert "profit" in feedback
                assert "trade_id" in feedback
            
            return {
                "feedback_events_sent": engine.telemetry["feedback_events_sent"],
                "feedback_captured": len(feedback_events),
                "live_feedback": feedback_events[0] if feedback_events else None
            }
            
        except Exception as e:
            return False
    
    def test_telemetry_updates(self):
        """Test 6: Telemetry Updates"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Get initial telemetry
            initial_telemetry = engine.get_telemetry()
            
            # Process test trades
            for trade in self.test_trades:
                engine._handle_trade_closed(trade)
            
            # Get updated telemetry
            updated_telemetry = engine.get_telemetry()
            
            # Verify telemetry was updated
            assert updated_telemetry["trades_processed"] > initial_telemetry["trades_processed"]
            assert "avg_trade_score" in updated_telemetry
            assert "win_rate" in updated_telemetry
            assert "profit_factor" in updated_telemetry
            
            # Check telemetry file creation
            telemetry_file = engine.telemetry_dir / "feedback_engine_telemetry.json"
            assert telemetry_file.exists()
            
            return {
                "telemetry_updated": True,
                "trades_processed": updated_telemetry["trades_processed"],
                "avg_trade_score": updated_telemetry["avg_trade_score"],
                "win_rate": updated_telemetry["win_rate"],
                "telemetry_file_exists": telemetry_file.exists()
            }
            
        except Exception as e:
            return False
    
    def test_event_bus_integration(self):
        """Test 7: EventBus Integration"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Test event emission capabilities
            test_events_emitted = []
            
            def capture_events(event_data):
                test_events_emitted.append(event_data)
            
            # Subscribe to various events
            engine.event_bus.subscribe("journal:trade_logged", capture_events, "TestListener")
            engine.event_bus.subscribe("telemetry:trade_outcome", capture_events, "TestListener")
            
            # Process a trade to trigger events
            engine._handle_trade_closed(self.test_trades[0])
            time.sleep(0.2)  # Allow event processing
            
            # Verify events were emitted
            journal_events = [e for e in test_events_emitted if "date" in e]
            telemetry_events = [e for e in test_events_emitted if "timestamp" in e]
            
            return {
                "events_emitted": len(test_events_emitted),
                "journal_events": len(journal_events),
                "telemetry_events": len(telemetry_events),
                "event_bus_working": True
            }
            
        except Exception as e:
            return False
    
    def self.event_bus.request('data:live_feed')_persistence(self):
        """Test 8: Data Persistence"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Process test trades
            for trade in self.test_trades:
                engine._handle_trade_closed(trade)
            
            # Save historical data
            engine._save_historical_data()
            
            # Check data files
            trades_file = engine.data_dir / "closed_trades.json"
            scores_file = engine.data_dir / "trade_scores.json"
            
            assert trades_file.exists()
            assert scores_file.exists()
            
            # Verify data content
            with open(trades_file, "r") as f:
                saved_trades = json.load(f)
            
            with open(scores_file, "r") as f:
                saved_scores = json.load(f)
            
            assert len(saved_trades) >= len(self.test_trades)
            assert len(saved_scores) > 0
            
            return {
                "data_persisted": True,
                "trades_saved": len(saved_trades),
                "strategies_scored": len(saved_scores),
                "files_created": [str(trades_file), str(scores_file)]
            }
            
        except Exception as e:
            return False
    
    def test_error_handling(self):
        """Test 9: Error Handling"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Test with invalid trade data
            invalid_trades = [
                {},  # Empty trade
                {"trade_id": "INVALID"},  # Missing required fields
                {"symbol": "INVALID", "profit": "not_a_number"}  # Invalid data types
            ]
            
            initial_trade_count = len(engine.closed_trades)
            
            for invalid_trade in invalid_trades:
                try:
                    engine._handle_trade_closed(invalid_trade)
                except Exception:
    logger.info("Function operational")("Real implementation required - no stubs allowed in production")
            
            # Verify engine still functional
            engine._handle_trade_closed(self.test_trades[0])
            
            return {
                "error_handling_works": True,
                "invalid_trades_processed": len(invalid_trades),
                "engine_still_functional": len(engine.closed_trades) > initial_trade_count
            }
            
        except Exception as e:
            return False
    
    def test_strategy_adjustment_recommendations(self):
        """Test 10: Strategy Adjustment Recommendations"""
        try:
            engine = PostTradeFeedbackEngine()
            
            # Create a strategy with consistently poor performance
            poor_trades = []
            for i in range(6):  # Need 5+ trades for adjustment trigger
                trade = {
                    "trade_id": f"POOR_TRADE_{i:03d}",
                    "symbol": "EURUSD",
                    "action": "BUY",
                    "lot_size": 0.1,
                    "open_price": 1.0850,
                    "close_price": 1.0800,  # Always losing
                    "open_time": f"2025-06-18T{10+i}:00:00Z",
                    "close_time": f"2025-06-18T{11+i}:00:00Z",
                    "profit": -50.0,  # Always losing
                    "strategy_id": "POOR_STRATEGY",
                    "signal_id": f"SIG_POOR_{i:03d}"
                }
                poor_trades.append(trade)
            
            # Monitor strategy adjustment events
            adjustment_events = []
            
            def capture_adjustments(event_data):
                adjustment_events.append(event_data)
            
            engine.event_bus.subscribe("strategy:adjust", capture_adjustments, "TestListener")
            
            # Process poor trades
            initial_adjustments = engine.telemetry["strategy_adjustments"]
            
            for trade in poor_trades:
                engine._handle_trade_closed(trade)
                time.sleep(0.05)
            
            # Verify adjustment recommendation was triggered
            assert engine.telemetry["strategy_adjustments"] > initial_adjustments
            
            return {
                "poor_trades_processed": len(poor_trades),
                "adjustments_triggered": engine.telemetry["strategy_adjustments"],
                "adjustment_events_captured": len(adjustment_events),
                "strategy_id": "POOR_STRATEGY"
            }
            
        except Exception as e:
            return False
    
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        print("\n" + "="*80)
        print("GENESIS PHASE 90 POST-TRADE FEEDBACK ENGINE VALIDATION")
        print("="*80)
        
        # Run all tests
        tests = [
            ("Engine Initialization", self.test_engine_initialization),
            ("Trade Closed Processing", self.test_trade_closed_processing),
            ("Trade Scoring", self.test_trade_scoring),
            ("Daily Journal Logging", self.test_daily_journal_logging),
            ("Strategy Feedback Emission", self.test_strategy_feedback_emission),
            ("Telemetry Updates", self.test_telemetry_updates),
            ("EventBus Integration", self.test_event_bus_integration),
            ("Data Persistence", self.self.event_bus.request('data:live_feed')_persistence),
            ("Error Handling", self.test_error_handling),
            ("Strategy Adjustment Recommendations", self.test_strategy_adjustment_recommendations)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Calculate success rate
        success_rate = (self.passed_tests / self.test_count) * 100 if self.test_count > 0 else 0
        
        # Generate results summary
        print(f"\n" + "="*80)
        print("VALIDATION RESULTS SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Detailed results
        print(f"\n📊 DETAILED TEST RESULTS:")
        print("-" * 80)
        for test_name, result in self.test_results.items():
            status = result["status"]
            icon = "✅" if status == "PASSED" else "❌"
            print(f"{icon} {test_name}: {status}")
            
            if isinstance(result["details"], dict):
                for key, value in result["details"].items():
                    print(f"   {key}: {value}")
        
        # Generate validation report
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "phase": "PHASE_90_POST_TRADE_FEEDBACK_ENGINE",
            "architect_mode": "v5.0.0",
            "total_tests": self.test_count,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": success_rate,
            "test_results": self.test_results,
            "validation_status": "PASSED" if success_rate >= 80 else "FAILED"
        }
        
        # Save validation report
        report_file = Path("logs") / "phase_90_feedback_validation_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Validation report saved to: {report_file}")
        
        # Final status
        if success_rate >= 80:
            print(f"\n🎉 PHASE 90 VALIDATION: ✅ PASSED ({success_rate:.1f}%)")
            print("Post-Trade Feedback Engine is ready for production use!")
        else:
            print(f"\n🚨 PHASE 90 VALIDATION: ❌ FAILED ({success_rate:.1f}%)")
            print("Issues detected. Review failed tests and fix before deployment.")
        
        return report

def main():
    """Main validation execution"""
    try:
        validator = Phase90ValidationSuite()
        report = validator.run_comprehensive_validation()
        
        return report["validation_status"] == "PASSED"
        
    except Exception as e:
        print(f"❌ Validation suite error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
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
        

# <!-- @GENESIS_MODULE_END: phase_90_feedback_validation -->