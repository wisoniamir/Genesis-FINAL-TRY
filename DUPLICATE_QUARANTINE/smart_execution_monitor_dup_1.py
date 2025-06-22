# <!-- @GENESIS_MODULE_START: smart_execution_monitor -->

from datetime import datetime\n#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 🔗 GENESIS EventBus Integration - Auto-injected by Orphan Recovery Engine
from datetime import datetime
import json

class SmartExecutionMonitorEventBusIntegration:
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

            emit_telemetry("smart_execution_monitor", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("smart_execution_monitor", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    """EventBus integration for smart_execution_monitor"""
    
    def __init__(self):
        self.module_id = "smart_execution_monitor"
        self.event_routes = []
        
    def emit_event(self, event_type, data):
        """Emit event to EventBus"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_id,
            "event_type": event_type,
            "data": data
        }
        print(f"🔗 EVENTBUS EMIT: {event}")
        
    def emit_telemetry(self, metric_name, value):
        """Emit telemetry data"""
        telemetry = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_id,
            "metric": metric_name,
            "value": value
        }
        print(f"📊 TELEMETRY: {telemetry}")

# Auto-instantiate EventBus integration
smart_execution_monitor_eventbus = SmartExecutionMonitorEventBusIntegration()

"""
GENESIS AI TRADING BOT SYSTEM
SmartExecutionMonitor - Central monitoring system for trade execution
ARCHITECT MODE: v2.7
"""
import os
import json
import logging
import datetime
import statistics
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

from event_bus import get_event_bus, register_route

class SmartExecutionMonitor:
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

            emit_telemetry("smart_execution_monitor", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("smart_execution_monitor", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    """
    Central execution monitor that checks whether the live execution metrics deviate 
    from GENESIS backtest benchmarks or FTMO compliance rules.
    
    Responsibilities:
    - Compare live vs backtest performance metrics
    - Monitor for deviations beyond thresholds
    - Trigger recalibration when needed
    - Emit kill-switch signals when risky conditions detected
    - Log performance metrics for monitoring
    """
    
    # Configuration parameters
    SEM_CONFIG = {
        "max_allowed_slippage": 0.7,      # Maximum allowed slippage in pips
        "min_win_rate_threshold": 0.58,   # Minimum acceptable win rate
        "max_dd_threshold_pct": 12.5,     # Maximum drawdown threshold percentage
        "min_rr_threshold": 1.8,          # Minimum risk-reward ratio
        "latency_warning_ms": 350,        # Latency warning threshold in milliseconds
        "pattern_edge_decay_window": 7    # Days to analyze pattern edge decay
    }
    
    def __init__(self):
        """Initialize the SmartExecutionMonitor with required components"""        
        # Configure logging first
        self._setup_logging()
        
        # Setup EventBus connection using singleton pattern (ARCHITECT MODE COMPLIANCE)
        self.event_bus = get_event_bus()        
        
        # Initialize monitoring storage
        self.live_trades = []
        self.backtest_results = {}
        self.journal_entries = []
        self.patterns_detected = []
        self.telemetry_data = defaultdict(list)
        
        # Performance metrics tracking
        self.metrics = {
            "win_rate_live": 0.0,
            "win_rate_backtest": 0.0,
            "drawdown_live": 0.0,
            "drawdown_backtest": 0.0,
            "rr_ratio_live": 0.0,
            "rr_ratio_backtest": 0.0,
            "avg_slippage": 0.0,
            "pattern_efficiency": {},
            "execution_latency_ms": [],
        }
          # Status tracking
        self.strategies_under_review = set()
        self.kill_switch_activated = False
        
        # PHASE 16 PATCH: Add loop protection counter
        self.kill_switch_count = 0
        self.MAX_KILL_SWITCH_CYCLES = 5  # Limit how many times the loop can emit triggers
        
        # Register EventBus subscribers
        self._subscribe_to_events()
          # Register EventBus routes for compliance tracking (ARCHITECT MODE)
        self._register_compliance_routes()
        
        self.logger.info("SmartExecutionMonitor initialized")
        # Emit telemetry that module is active
        self._emit_telemetry("initialization", "SmartExecutionMonitor started successfully")

    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _subscribe_to_events(self):
        """Subscribe to all required events"""
        self.event_bus.subscribe("LiveTradeExecuted", self.handle_live_trade, "SmartExecutionMonitor")
        self.event_bus.subscribe("BacktestResults", self.handle_backtest_results, "SmartExecutionMonitor")
        self.event_bus.subscribe("TradeJournalEntry", self.handle_journal_entry, "SmartExecutionMonitor")
        self.event_bus.subscribe("ModuleTelemetry", self.handle_telemetry, "SmartExecutionMonitor")
        self.event_bus.subscribe("PatternDetected", self.handle_pattern, "SmartExecutionMonitor")
        
        # PHASE 16 PATCH: Subscribe to feedback acknowledgment events for loop reset
        self.event_bus.subscribe("RecalibrationSuccessful", self.on_feedback_ack, "SmartExecutionMonitor")
        self.event_bus.subscribe("LogSyncComplete", self.on_feedback_ack, "SmartExecutionMonitor")
        
        self.logger.info("Event subscriptions registered successfully")
    
    def _setup_logging(self):
        """Configure structured logging for the module"""
        # Ensure log directory exists
        log_dir = Path("logs/smart_monitor")
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Configure logger
        self.logger = logging.getLogger("SmartExecutionMonitor")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"smart_monitor_{timestamp}.jsonl"
        file_handler = logging.FileHandler(log_file)
        
        # Create formatter
        formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')
        file_handler.setFormatter(formatter)
          # Add handlers
        self.logger.addHandler(file_handler)
    
    def _register_compliance_routes(self):
        """Register all EventBus routes for ARCHITECT MODE compliance tracking"""
        # Register input routes (what this module consumes)
        register_route("LiveTradeExecuted", "ExecutionEngine", "SmartExecutionMonitor")
        register_route("BacktestResults", "BacktestEngine", "SmartExecutionMonitor")
        register_route("TradeJournalEntry", "TradeJournalEngine", "SmartExecutionMonitor")
        register_route("ModuleTelemetry", "TelemetryCollector", "SmartExecutionMonitor")
        register_route("PatternDetected", "PatternEngine", "SmartExecutionMonitor")
        
        # Register output routes (what this module produces)
        register_route("ExecutionDeviationAlert", "SmartExecutionMonitor", "DashboardEngine")
        register_route("KillSwitchTrigger", "SmartExecutionMonitor", "RiskEngine")
        register_route("RecalibrationRequest", "SmartExecutionMonitor", "StrategyRecommenderEngine")
        register_route("SmartLogSync", "SmartExecutionMonitor", "DashboardEngine")
        register_route("ModuleTelemetry", "SmartExecutionMonitor", "TelemetryCollector")
        register_route("ModuleError", "SmartExecutionMonitor", "TelemetryCollector")
        
        self.logger.info("ARCHITECT MODE: All EventBus routes registered for compliance")
    
    def register_event_handlers(self):
        """Register all event handlers with the EventBus - ALREADY DONE IN _subscribe_to_events()"""
        # This method is redundant - events already subscribed in _subscribe_to_events()
        self.logger.info(json.dumps({"event": "event_handlers_already_registered"}))
    
    def handle_live_trade(self, event_data):
        """Process live trade execution data"""
        self.logger.info(json.dumps({"event": "live_trade_received", "data": str(event_data)}))
        
        # Record trade for analysis
        self.live_trades.append(event_data)
        
        # Calculate metrics
        self._update_live_metrics()
        
        # Check for deviations
        self._check_deviations(event_data.get("strategy_id", "unknown"))
        
    def handle_backtest_results(self, event_data):
        """Process backtest results data"""
        self.logger.info(json.dumps({"event": "backtest_results_received", "strategy_id": event_data.get("strategy_id", "unknown")}))
        
        # Store backtest results
        strategy_id = event_data.get("strategy_id", "unknown")
        self.backtest_results[strategy_id] = event_data
        
        # Update backtest metrics
        self._update_backtest_metrics(strategy_id)
    
    def handle_journal_entry(self, event_data):
        """Process trade journal entry data"""
        self.logger.info(json.dumps({"event": "journal_entry_received", "entry_id": event_data.get("entry_id", "unknown")}))
        
        # Store journal entry
        self.journal_entries.append(event_data)
        
        # Update metrics that depend on journal data
        self._update_from_journal()
    
    def handle_telemetry(self, event_data):
        """Process telemetry data from various modules"""
        module = event_data.get("module", "unknown")
        
        # Store telemetry data
        self.telemetry_data[module].append({
            "timestamp": event_data.get("timestamp", datetime.datetime.now().isoformat()),
            "metrics": event_data.get("metrics", {}),
            "status": event_data.get("status", "unknown")
        })
        
        # Check for telemetry-based issues
        self._check_telemetry_issues(module, event_data)
    
    def handle_pattern(self, event_data):
        """Process pattern detection data"""
        pattern_id = event_data.get("pattern_id", "unknown")
        self.logger.info(json.dumps({"event": "pattern_detected", "pattern_id": pattern_id}))
        
        # Store pattern data
        self.patterns_detected.append(event_data)
        
        # Update pattern efficiency metrics
        self._update_pattern_efficiency(pattern_id, event_data)
    
    def on_feedback_ack(self, event):
        """PHASE 16 PATCH: Handle feedback acknowledgment events to reset loop counter"""
        event_data = event.get("data", event)
        if event_data.get("topic") in ["RecalibrationSuccessful", "LogSyncComplete"]:
            self.kill_switch_count = 0  # Reset loop counter when feedback is received
            self.logger.info(json.dumps({
                "event": "loop_counter_reset",
                "trigger": event_data.get("topic"),
                "source": "SmartExecutionMonitor_PHASE16_PATCH"
            }))
    
    def _update_pattern_efficiency(self, pattern_id, pattern_data):
        """Track and update pattern efficiency over time"""
        if pattern_id not in self.metrics["pattern_efficiency"]:
            self.metrics["pattern_efficiency"][pattern_id] = {
                "detections": 0,
                "successful_trades": 0,
                "efficiency": 0.0
            }
        
        # Increment detection count
        self.metrics["pattern_efficiency"][pattern_id]["detections"] += 1
        
        # Check if this pattern resulted in a successful trade
        # This requires matching trades to patterns, which depends on having trade IDs
        # that reference pattern IDs in your system
        if pattern_data.get("led_to_successful_trade", False):
            self.metrics["pattern_efficiency"][pattern_id]["successful_trades"] += 1
        
        # Calculate efficiency
        detections = self.metrics["pattern_efficiency"][pattern_id]["detections"]
        successes = self.metrics["pattern_efficiency"][pattern_id]["successful_trades"]
        
        if detections > 0:
            self.metrics["pattern_efficiency"][pattern_id]["efficiency"] = successes / detections
    
    def _check_deviations(self, strategy_id):
        """Check for deviations in key metrics and take action if needed"""
        
        # PHASE 16 PATCH: Check kill switch emission limit to prevent infinite loops
        if self.kill_switch_count >= self.MAX_KILL_SWITCH_CYCLES:
            self.logger.warning(json.dumps({
                "event": "max_kill_switch_cycles_reached",
                "kill_switch_count": self.kill_switch_count,
                "max_cycles": self.MAX_KILL_SWITCH_CYCLES,
                "source": "SmartExecutionMonitor_PHASE16_PATCH"
            }))
            # Emit termination signal to break the loop
            self.event_bus.emit_event("TerminateMonitorLoop", {
                "timestamp": datetime.datetime.now().isoformat(),
                "reason": "max_kill_switch_cycles_reached",
                "source": "SmartExecutionMonitor"
            }, "SmartExecutionMonitor")
            return
        
        deviation_detected = False
        deviation_details = {}
        
        # Check win rate deviation
        if (self.metrics["win_rate_live"] < self.metrics["win_rate_backtest"] * 0.9 or 
            self.metrics["win_rate_live"] < self.SEM_CONFIG["min_win_rate_threshold"]):
            deviation_detected = True
            deviation_details["win_rate"] = {
                "live": self.metrics["win_rate_live"],
                "backtest": self.metrics["win_rate_backtest"],
                "threshold": self.SEM_CONFIG["min_win_rate_threshold"]
            }
        
        # Check drawdown deviation
        if self.metrics["drawdown_live"] > self.metrics["drawdown_backtest"] * 1.1 or self.metrics["drawdown_live"] > self.SEM_CONFIG["max_dd_threshold_pct"]:
            deviation_detected = True
            deviation_details["drawdown"] = {
                "live": self.metrics["drawdown_live"],
                "backtest": self.metrics["drawdown_backtest"],
                "threshold": self.SEM_CONFIG["max_dd_threshold_pct"]
            }
        
        # Check risk-reward deviation
        if self.metrics["rr_ratio_live"] < self.metrics["rr_ratio_backtest"] * 0.9 or self.metrics["rr_ratio_live"] < self.SEM_CONFIG["min_rr_threshold"]:
            deviation_detected = True
            deviation_details["risk_reward"] = {
                "live": self.metrics["rr_ratio_live"],
                "backtest": self.metrics["rr_ratio_backtest"],
                "threshold": self.SEM_CONFIG["min_rr_threshold"]
            }
        
        # Check slippage
        if self.metrics["avg_slippage"] > self.SEM_CONFIG["max_allowed_slippage"]:
            deviation_detected = True
            deviation_details["slippage"] = {
                "avg": self.metrics["avg_slippage"],
                "threshold": self.SEM_CONFIG["max_allowed_slippage"]
            }
        
        # Take action if deviation detected
        if deviation_detected:
            self.strategies_under_review.add(strategy_id)
            self.logger.warning(json.dumps({
                "event": "deviation_detected",
                "strategy_id": strategy_id,
                "details": deviation_details
            }))
            
            # Emit deviation alert
            self._emit_deviation_alert(strategy_id, deviation_details)
            
            # Check if we need to trigger kill switch
            self._check_kill_switch(strategy_id, deviation_details)
            
            # Request recalibration
            self._request_recalibration(strategy_id, deviation_details)
    def _check_telemetry_issues(self, module, telemetry_data):
        """Check for issues based on telemetry data"""
        # Extract latency information if available
        latency = telemetry_data.get("metrics", {}).get("execution_latency_ms")
        
        if latency and latency > self.SEM_CONFIG["latency_warning_ms"]:
            self.logger.warning(json.dumps({
                "event": "high_latency_detected",
                "module": module,
                "latency_ms": latency,
                "threshold_ms": self.SEM_CONFIG["latency_warning_ms"]
            }))
            
            # Emit telemetry-based alert
            self._emit_deviation_alert("system", {
                "latency": {
                    "module": module,
                    "value_ms": latency,
                    "threshold_ms": self.SEM_CONFIG["latency_warning_ms"]
                }
            })
            
        # Check pattern edge decay if present in telemetry
        pattern_data = telemetry_data.get("metrics", {}).get("pattern_performance", {})
        if pattern_data and "edge_decay_sessions" in pattern_data:
            pattern_id = pattern_data.get("pattern_id", "unknown")
            edge_decay_sessions = pattern_data.get("edge_decay_sessions", 0)
            
            if edge_decay_sessions > self.SEM_CONFIG["pattern_edge_decay_window"]:
                self.logger.warning(json.dumps({
                    "event": "pattern_edge_decay_detected",
                    "pattern_id": pattern_id,
                    "edge_decay_sessions": edge_decay_sessions,
                    "threshold_sessions": self.SEM_CONFIG["pattern_edge_decay_window"]
                }))
                
                # Request pattern recalibration
                self._request_recalibration(f"pattern_{pattern_id}", {
                    "pattern_edge_decay": {
                        "pattern_id": pattern_id,
                        "edge_decay_sessions": edge_decay_sessions,
                        "threshold": self.SEM_CONFIG["pattern_edge_decay_window"]
                    }
                })
    
    def _emit_deviation_alert(self, strategy_id, details):
        """Emit deviation alert via EventBus"""
        alert_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "details": details,
            "severity": self._calculate_severity(details)
        }
        
        self.event_bus.emit_event("ExecutionDeviationAlert", alert_data, "SmartExecutionMonitor")
        self.logger.info(json.dumps({"event": "deviation_alert_emitted", "strategy_id": strategy_id}))
    
    def _calculate_severity(self, deviation_details):
        """Calculate severity level based on deviation details"""
        severity_score = 0
        
        # Add points based on which metrics are deviating
        if "win_rate" in deviation_details:
            severity_score += 2
        if "drawdown" in deviation_details:
            severity_score += 3
        if "risk_reward" in deviation_details:
            severity_score += 2
        if "slippage" in deviation_details:
            severity_score += 1
        
        # Map score to severity level
        if severity_score >= 5:
            return "critical"
        elif severity_score >= 3:
            return "high"
        elif severity_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _check_kill_switch(self, strategy_id, deviation_details):
        """Check if kill switch should be triggered"""
        # Determine if deviations are severe enough for kill switch
        trigger_kill_switch = False
        
        # Check drawdown breach - most critical for FTMO
        if "drawdown" in deviation_details and deviation_details["drawdown"]["live"] > self.SEM_CONFIG["max_dd_threshold_pct"]:
            trigger_kill_switch = True
        
        # Check if multiple critical metrics are failing
        critical_failures = 0
        if "win_rate" in deviation_details and deviation_details["win_rate"]["live"] < self.SEM_CONFIG["min_win_rate_threshold"] * 0.8:
            critical_failures += 1
        
        if "risk_reward" in deviation_details and deviation_details["risk_reward"]["live"] < self.SEM_CONFIG["min_rr_threshold"] * 0.7:
            critical_failures += 1
        
        if critical_failures >= 2:
            trigger_kill_switch = True        
        # Trigger kill switch if needed
        if trigger_kill_switch:
            self._emit_kill_switch(strategy_id, deviation_details)
    
    def _emit_kill_switch(self, strategy_id, details):
        """Emit kill switch trigger via EventBus"""
        
        # PHASE 16 PATCH: Increment kill switch counter
        self.kill_switch_count += 1
        
        kill_switch_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "reason": "Performance metrics breach",
            "details": details,
            "triggered_by": "SmartExecutionMonitor",
            "kill_switch_count": self.kill_switch_count  # PHASE 16 PATCH: Include counter in data
        }
        
        self.kill_switch_activated = True
        self.event_bus.emit_event("KillSwitchTrigger", kill_switch_data, "SmartExecutionMonitor")
        
        self.logger.critical(json.dumps({
            "event": "kill_switch_triggered",
            "strategy_id": strategy_id,
            "kill_switch_count": self.kill_switch_count,  # PHASE 16 PATCH: Log counter
            "details": details
        }))
    
    def _request_recalibration(self, strategy_id, details):
        """Request strategy recalibration via EventBus"""
        
        # PHASE 16 PATCH: Log emission for monitoring but still allow it to proceed (capped by kill switch logic)
        self.logger.info(json.dumps({
            "event": "recalibration_request_emitted",
            "strategy_id": strategy_id,
            "kill_switch_count": self.kill_switch_count,
            "max_cycles": self.MAX_KILL_SWITCH_CYCLES,
            "source": "SmartExecutionMonitor_PHASE16_PATCH"
        }))
        
        recal_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "metrics": details,
            "severity": self._calculate_severity(details),
            "kill_switch_count": self.kill_switch_count  # PHASE 16 PATCH: Include counter for tracking
        }
        
        self.event_bus.emit_event("RecalibrationRequest", recal_data, "SmartExecutionMonitor")
        
        # Emit SmartLogSync for dashboard
        self._emit_smart_log_sync(strategy_id, "recalibration_requested", details)
        
        self.logger.info(json.dumps({
            "event": "recalibration_requested",
            "strategy_id": strategy_id,
            "kill_switch_count": self.kill_switch_count  # PHASE 16 PATCH: Log counter
        }))
    
    def _emit_smart_log_sync(self, strategy_id, event_type, details):
        """Emit SmartLogSync event for dashboard"""
        
        # PHASE 16 PATCH: Log emission for monitoring
        self.logger.debug(json.dumps({
            "event": "smart_log_sync_emitted",
            "strategy_id": strategy_id,
            "event_type": event_type,
            "kill_switch_count": self.kill_switch_count,
            "source": "SmartExecutionMonitor_PHASE16_PATCH"
        }))
        
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "event_type": event_type,
            "details": details,
            "source": "SmartExecutionMonitor",
            "kill_switch_count": self.kill_switch_count  # PHASE 16 PATCH: Include counter for tracking
        }
        
        self.event_bus.emit_event("SmartLogSync", log_data, "SmartExecutionMonitor")
    
    def _emit_telemetry(self, action, message):
        """Emit telemetry data via EventBus"""
        telemetry_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "module": "SmartExecutionMonitor",
            "action": action,
            "message": message,
            "metrics": {
                "strategies_monitored": len(self.backtest_results),
                "strategies_under_review": len(self.strategies_under_review),
                "kill_switch_activated": self.kill_switch_activated,
                "win_rate_live": self.metrics["win_rate_live"],
                "drawdown_live": self.metrics["drawdown_live"],
                "avg_slippage": self.metrics["avg_slippage"]
            }
        }
        
        self.event_bus.emit_event("ModuleTelemetry", telemetry_data, "SmartExecutionMonitor")
    
    def _calculate_equity_curve(self, trades):
        """Calculate equity curve from trade history"""
        assert trades is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: smart_execution_monitor -->