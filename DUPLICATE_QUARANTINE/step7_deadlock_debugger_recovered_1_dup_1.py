# <!-- @GENESIS_MODULE_START: step7_deadlock_debugger -->

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

                emit_telemetry("step7_deadlock_debugger_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("step7_deadlock_debugger_recovered_1", "position_calculated", {
                    "risk_amount": risk_amount,
                    "position_size": position_size,
                    "risk_percentage": (position_size / account_balance) * 100
                })

                return position_size
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "step7_deadlock_debugger_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("step7_deadlock_debugger_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in step7_deadlock_debugger_recovered_1: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


# -*- coding: utf-8 -*-
"""
🔍 GENESIS AI AGENT — STEP 7 DEADLOCK DEBUGGER
ARCHITECT LOCK-IN v2.7 - DIRECT EMISSION TEST

This debugger will identify exactly where the SmartExecutionMonitor deadlock is occurring
by testing each threshold individually and logging all event emissions in real-time.

PURPOSE: Isolate the exact point of failure in the Step 7 validation process.
"""

import os
import sys
import json
import time
import logging
import datetime
import threading
from pathlib import Path
from typing import Dict, Any, List

# Configure detailed logging
log_dir = Path("logs/smart_monitor")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    handlers=[
        logging.FileHandler(log_dir / f"deadlock_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DeadlockDebugger")

# Import GENESIS modules
from event_bus import get_event_bus, emit_event, subscribe_to_event, register_route
from smart_execution_monitor import SmartExecutionMonitor

class Step7DeadlockDebugger:
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

            emit_telemetry("step7_deadlock_debugger_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("step7_deadlock_debugger_recovered_1", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "step7_deadlock_debugger_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("step7_deadlock_debugger_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in step7_deadlock_debugger_recovered_1: {e}")
    """
    Debugger to identify exactly where the Step 7 SmartExecutionMonitor validation deadlocks
    """
    
    def __init__(self):
        """Initialize debugger with detailed tracking"""
        self.event_bus = get_event_bus()
        self.monitor = None
        self.events_emitted = []
        self.events_received = []
        self.threshold_tests = {
            "slippage": {"tested": False, "triggered": False},
            "latency": {"tested": False, "triggered": False},
            "drawdown": {"tested": False, "triggered": False},
            "pattern_decay": {"tested": False, "triggered": False}
        }
        
        self.setup_event_tracking()
        
        logger.info(json.dumps({
            "event": "debugger_initialized",
            "timestamp": datetime.datetime.now().isoformat()
        }))
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def setup_event_tracking(self):
        """Setup comprehensive event tracking to catch all emissions"""
        
        # Track all SmartExecutionMonitor outputs
        subscribe_to_event("ExecutionDeviationAlert", self.track_deviation_alert, "DeadlockDebugger")
        subscribe_to_event("KillSwitchTrigger", self.track_kill_switch, "DeadlockDebugger")
        subscribe_to_event("RecalibrationRequest", self.track_recalibration, "DeadlockDebugger")
        subscribe_to_event("SmartLogSync", self.track_smart_log, "DeadlockDebugger")
        subscribe_to_event("ModuleTelemetry", self.track_telemetry, "DeadlockDebugger")
        subscribe_to_event("ModuleError", self.track_error, "DeadlockDebugger")
        
        # Register routes
        register_route("LiveTradeExecuted", "DeadlockDebugger", "SmartExecutionMonitor")
        register_route("TradeJournalEntry", "DeadlockDebugger", "SmartExecutionMonitor")
        register_route("PatternDetected", "DeadlockDebugger", "SmartExecutionMonitor")
        register_route("ModuleTelemetry", "DeadlockDebugger", "SmartExecutionMonitor")
        
        logger.info(json.dumps({
            "event": "event_tracking_setup",
            "subscribers_registered": 6,
            "routes_registered": 4
        }))
    
    def track_deviation_alert(self, event):
        """Track ExecutionDeviationAlert events"""
        self.events_received.append({
            "type": "ExecutionDeviationAlert",
            "timestamp": datetime.datetime.now().isoformat(),
            "data": event
        })
        
        # Determine which threshold was triggered
        details = event.get("data", {}).get("details", {})
        if "slippage" in details:
            self.threshold_tests["slippage"]["triggered"] = True
        if "latency" in details:
            self.threshold_tests["latency"]["triggered"] = True
            
        logger.info(json.dumps({
            "event": "deviation_alert_tracked",
            "details": details,
            "total_events_received": len(self.events_received)
        }))
    
    def track_kill_switch(self, event):
        """Track KillSwitchTrigger events"""
        self.events_received.append({
            "type": "KillSwitchTrigger",
            "timestamp": datetime.datetime.now().isoformat(),
            "data": event
        })
        
        self.threshold_tests["drawdown"]["triggered"] = True
        
        logger.critical(json.dumps({
            "event": "kill_switch_tracked",
            "reason": event.get("data", {}).get("reason"),
            "total_events_received": len(self.events_received)
        }))
    
    def track_recalibration(self, event):
        """Track RecalibrationRequest events"""
        self.events_received.append({
            "type": "RecalibrationRequest",
            "timestamp": datetime.datetime.now().isoformat(),
            "data": event
        })
        
        self.threshold_tests["pattern_decay"]["triggered"] = True
        
        logger.info(json.dumps({
            "event": "recalibration_tracked",
            "strategy_id": event.get("data", {}).get("strategy_id"),
            "total_events_received": len(self.events_received)
        }))
    
    def track_smart_log(self, event):
        """Track SmartLogSync events"""
        self.events_received.append({
            "type": "SmartLogSync",
            "timestamp": datetime.datetime.now().isoformat(),
            "data": event
        })
        
        logger.info(json.dumps({
            "event": "smart_log_tracked",
            "total_events_received": len(self.events_received)
        }))
    
    def track_telemetry(self, event):
        """Track ModuleTelemetry events"""
        if event.get("data", {}).get("module") == "SmartExecutionMonitor":
            self.events_received.append({
                "type": "ModuleTelemetry",
                "timestamp": datetime.datetime.now().isoformat(),
                "data": event
            })
            
            logger.debug(json.dumps({
                "event": "telemetry_tracked",
                "module": event.get("data", {}).get("module"),
                "action": event.get("data", {}).get("action")
            }))
    
    def track_error(self, event):
        """Track ModuleError events"""
        self.events_received.append({
            "type": "ModuleError",
            "timestamp": datetime.datetime.now().isoformat(),
            "data": event
        })
        
        logger.error(json.dumps({
            "event": "error_tracked",
            "module": event.get("data", {}).get("module"),
            "error": event.get("data", {}).get("error")
        }))
    
    def start_smart_monitor(self):
        """Start SmartExecutionMonitor and confirm it's operational"""
        logger.info(json.dumps({
            "event": "starting_smart_monitor",
            "mode": "debug"
        }))
        
        try:
            self.monitor = SmartExecutionMonitor()
            
            # Verify initialization
            assert self.monitor is not None, "SmartExecutionMonitor failed to initialize"
            assert self.monitor.event_bus is not None, "EventBus connection failed"
            
            logger.info(json.dumps({
                "event": "smart_monitor_started",
                "status": "operational",
                "event_bus_connected": True
            }))
            
            # Allow time for full initialization
            time.sleep(3)
            return True
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "smart_monitor_start_failed",
                "error": str(e)
            }))
            return False
    
    def test_slippage_threshold_direct(self):
        """Test slippage threshold with guaranteed trigger"""
        logger.info(json.dumps({
            "event": "testing_slippage_threshold",
            "threshold": "0.7_pips",
            "real_value": "2.5_pips"
        }))
        
        # Create trade with extreme slippage to guarantee trigger
        trade_data = {
            "trade_id": f"slippage_debug_{int(time.time())}",
            "symbol": "EURUSD",
            "direction": "BUY",
            "entry_price": 1.0850,
            "requested_price": 1.0825,
            "slippage": 2.5,  # 2.5 pips >> 0.7 threshold
            "lot_size": 1.0,
            "execution_time": datetime.datetime.now().isoformat(),
            "execution_latency_ms": 150,
            "strategy_id": "debug_slippage_strategy",
            "profit": 0,
            "account_balance": 100000.0
        }
        
        # Emit event and track
        emit_event("LiveTradeExecuted", trade_data, "DeadlockDebugger")
        self.events_emitted.append({"type": "LiveTradeExecuted", "data": trade_data})
        self.threshold_tests["slippage"]["tested"] = True
        
        logger.info(json.dumps({
            "event": "slippage_test_emitted",
            "slippage_value": 2.5,
            "threshold": 0.7,
            "should_trigger": True
        }))
        
        # Wait and check
        time.sleep(3)
        return self.threshold_tests["slippage"]["triggered"]
    
    def test_latency_threshold_direct(self):
        """Test latency threshold with guaranteed trigger"""
        logger.info(json.dumps({
            "event": "testing_latency_threshold",
            "threshold": "350ms",
            "real_value": "1500ms"
        }))
        
        # Create trade with extreme latency
        trade_data = {
            "trade_id": f"latency_debug_{int(time.time())}",
            "symbol": "GBPUSD",
            "direction": "SELL",
            "entry_price": 1.2650,
            "requested_price": 1.2650,
            "slippage": 0.1,  # Low slippage
            "lot_size": 0.5,
            "execution_time": datetime.datetime.now().isoformat(),
            "execution_latency_ms": 1500,  # 1500ms >> 350ms threshold
            "strategy_id": "debug_latency_strategy",
            "profit": 0,
            "account_balance": 100000.0
        }
        
        # Emit event and track
        emit_event("LiveTradeExecuted", trade_data, "DeadlockDebugger")
        self.events_emitted.append({"type": "LiveTradeExecuted", "data": trade_data})
        self.threshold_tests["latency"]["tested"] = True
        
        logger.info(json.dumps({
            "event": "latency_test_emitted",
            "latency_value": 1500,
            "threshold": 350,
            "should_trigger": True
        }))
        
        # Wait and check
        time.sleep(3)
        return self.threshold_tests["latency"]["triggered"]
    
    def test_drawdown_threshold_direct(self):
        """Test drawdown threshold with guaranteed trigger"""
        logger.info(json.dumps({
            "event": "testing_drawdown_threshold",
            "threshold": "12.5%",
            "real_value": "25%"
        }))
        
        # Create severe losing trade scenario
        starting_balance = 100000.0
        
        # Single massive loss to trigger drawdown
        trade_data = {
            "trade_id": f"drawdown_debug_{int(time.time())}",
            "symbol": "USDCAD",
            "direction": "BUY",
            "entry_price": 1.3550,
            "exit_price": 1.3200,  # Massive loss
            "slippage": 0.3,
            "lot_size": 10.0,  # Large lot size
            "execution_time": datetime.datetime.now().isoformat(),
            "execution_latency_ms": 180,
            "strategy_id": "debug_drawdown_strategy",
            "profit": -25000,  # 25% loss
            "account_balance": starting_balance - 25000,  # 75k remaining
            "drawdown_pct": 25.0  # Explicit 25% drawdown
        }
        
        # Emit event and track
        emit_event("LiveTradeExecuted", trade_data, "DeadlockDebugger")
        self.events_emitted.append({"type": "LiveTradeExecuted", "data": trade_data})
        self.threshold_tests["drawdown"]["tested"] = True
        
        logger.info(json.dumps({
            "event": "drawdown_test_emitted",
            "drawdown_value": 25.0,
            "threshold": 12.5,
            "should_trigger": True
        }))
        
        # Wait and check
        time.sleep(3)
        return self.threshold_tests["drawdown"]["triggered"]
    
    def test_pattern_decay_direct(self):
        """Test pattern edge decay with guaranteed trigger"""
        logger.info(json.dumps({
            "event": "testing_pattern_decay_threshold",
            "threshold": "7_sessions",
            "real_value": "15_sessions"
        }))
        
        # Create pattern with severe edge decay
        pattern_data = {
            "pattern_id": "debug_pattern_001",
            "symbol": "EURUSD",
            "pattern_type": "breakout",
            "confidence": 0.25,  # Very low confidence
            "strategy_id": "debug_pattern_strategy",
            "sessions_since_last_win": 15,  # 15 >> 7 threshold
            "win_rate_last_10": 0.10,  # 10% win rate (terrible)
            "edge_decay_sessions": 15,
            "requires_recalibration": True,
            "detected_at": datetime.datetime.now().isoformat()
        }
        
        # Emit event and track
        emit_event("PatternDetected", pattern_data, "DeadlockDebugger")
        self.events_emitted.append({"type": "PatternDetected", "data": pattern_data})
        self.threshold_tests["pattern_decay"]["tested"] = True
        
        logger.info(json.dumps({
            "event": "pattern_decay_test_emitted",
            "sessions_since_win": 15,
            "threshold": 7,
            "should_trigger": True
        }))
        
        # Wait and check
        time.sleep(3)
        return self.threshold_tests["pattern_decay"]["triggered"]
    
    def run_comprehensive_debug(self):
        """Run comprehensive debug to identify deadlock source"""
        logger.info(json.dumps({
            "event": "comprehensive_debug_started",
            "timestamp": datetime.datetime.now().isoformat()
        }))
        
        # Step 1: Start SmartExecutionMonitor
        if not self.start_smart_monitor():
            logger.error("Failed to start SmartExecutionMonitor")
            return False
        
        # Step 2: Test each threshold individually
        results = {}
        
        # Test slippage
        logger.info("=" * 60)
        logger.info("TESTING SLIPPAGE THRESHOLD")
        logger.info("=" * 60)
        results["slippage"] = self.test_slippage_threshold_direct()
        time.sleep(2)
        
        # Test latency  
        logger.info("=" * 60)
        logger.info("TESTING LATENCY THRESHOLD")
        logger.info("=" * 60)
        results["latency"] = self.test_latency_threshold_direct()
        time.sleep(2)
        
        # Test drawdown
        logger.info("=" * 60)
        logger.info("TESTING DRAWDOWN THRESHOLD")
        logger.info("=" * 60)
        results["drawdown"] = self.test_drawdown_threshold_direct()
        time.sleep(2)
        
        # Test pattern decay
        logger.info("=" * 60)
        logger.info("TESTING PATTERN DECAY THRESHOLD")
        logger.info("=" * 60)
        results["pattern_decay"] = self.test_pattern_decay_direct()
        time.sleep(2)
        
        # Step 3: Generate debug report
        self.generate_debug_report(results)
        
        return any(results.values())
    
    def generate_debug_report(self, results):
        """Generate comprehensive debug report"""
        report = {
            "debug_summary": {
                "timestamp": datetime.datetime.now().isoformat(),
                "events_emitted": len(self.events_emitted),
                "events_received": len(self.events_received),
                "total_tests": 4,
                "tests_passed": sum(results.values())
            },
            "threshold_results": {
                "slippage": {
                    "tested": self.threshold_tests["slippage"]["tested"],
                    "triggered": results.get("slippage", False),
                    "status": "PASS" if results.get("slippage", False) else "FAIL"
                },
                "latency": {
                    "tested": self.threshold_tests["latency"]["tested"],
                    "triggered": results.get("latency", False),
                    "status": "PASS" if results.get("latency", False) else "FAIL"
                },
                "drawdown": {
                    "tested": self.threshold_tests["drawdown"]["tested"],
                    "triggered": results.get("drawdown", False),
                    "status": "PASS" if results.get("drawdown", False) else "FAIL"
                },
                "pattern_decay": {
                    "tested": self.threshold_tests["pattern_decay"]["tested"],
                    "triggered": results.get("pattern_decay", False),
                    "status": "PASS" if results.get("pattern_decay", False) else "FAIL"
                }
            },
            "events_emitted": self.events_emitted,
            "events_received": self.events_received,
            "deadlock_analysis": self.analyze_deadlock(results)
        }
        
        # Save debug report
        report_file = log_dir / f"deadlock_debug_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        logger.info(json.dumps({
            "event": "debug_report_generated",
            "report_file": str(report_file),
            "tests_passed": sum(results.values()),
            "total_tests": 4,
            "events_received": len(self.events_received),
            "deadlock_identified": not any(results.values())
        }))
        
        # Print clear summary
        print("\n" + "="*80)
        print("🔍 STEP 7 DEADLOCK DEBUG RESULTS")
        print("="*80)
        for test, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test.upper()} THRESHOLD TEST: {status}")
        print(f"EVENTS EMITTED: {len(self.events_emitted)}")
        print(f"EVENTS RECEIVED: {len(self.events_received)}")
        print("="*80)
        
        return report
    
    def analyze_deadlock(self, results):
        """Analyze where the deadlock is occurring"""
        analysis = {
            "deadlock_detected": not any(results.values()),
            "probable_causes": [],
            "recommendations": []
        }
        
        if not any(results.values()):
            analysis["probable_causes"].extend([
                "SmartExecutionMonitor not processing events",
                "Threshold logic not working",
                "Event emission system failure",
                "Route registration issues"
            ])
            
            analysis["recommendations"].extend([
                "Check SmartExecutionMonitor._check_deviations method",
                "Verify event_bus.emit_event calls in SmartExecutionMonitor",
                "Test with simpler threshold values",
                "Check route registration between modules"
            ])
        
        # Specific analysis
        if len(self.events_emitted) > 0 and len(self.events_received) == 0:
            analysis["probable_causes"].append("Complete event processing failure")
            analysis["recommendations"].append("Check EventBus routing configuration")
        
        return analysis

def main():
    """Main execution function for deadlock debugging"""
    logger.info(json.dumps({
        "event": "deadlock_debugger_started",
        "timestamp": datetime.datetime.now().isoformat()
    }))
    
    try:
        debugger = Step7DeadlockDebugger()
        success = debugger.run_comprehensive_debug()
        
        if success:
            logger.info("🎯 DEADLOCK DEBUG COMPLETED - ISSUES IDENTIFIED")
        else:
            logger.error("❌ DEADLOCK CONFIRMED - SYSTEM NOT EMITTING EVENTS")
        
        return success
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "deadlock_debug_exception",
            "error": str(e)
        }))
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)

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
        

# <!-- @GENESIS_MODULE_END: step7_deadlock_debugger -->