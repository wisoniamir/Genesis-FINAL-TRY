# <!-- @GENESIS_MODULE_START: smart_feedback_sync -->

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

                emit_telemetry("smart_feedback_sync_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("smart_feedback_sync_recovered_1", "position_calculated", {
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
                            "module": "smart_feedback_sync_recovered_1",
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
                    print(f"Emergency stop error in smart_feedback_sync_recovered_1: {e}")
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
                    "module": "smart_feedback_sync_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("smart_feedback_sync_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in smart_feedback_sync_recovered_1: {e}")
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


"""
<!-- @GENESIS_MODULE_START: smart_feedback_sync -->
📡 GENESIS AI TRADING SYSTEM - SMART FEEDBACK SYNC v1.0
🔐 ARCHITECT MODE v3.0 - PHASE 19 FEEDBACK INGESTION LOOP

This module ingests live feedback from signal confidence rating and trade feedback systems,
detects execution drift, and triggers adaptive recalibration requests.

STRICT COMPLIANCE RULES:
- Real MT5 data only - no real/execute inputs
- EventBus-only communication - no local handlers
- Full telemetry integration with structured logging
- Institutional-grade feedback processing with learning modes
<!-- @GENESIS_MODULE_END: smart_feedback_sync -->
"""

import json
import datetime
import os
import logging
import time
import threading
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from statistics import mean, stdev
from hardened_event_bus import get_event_bus, emit_event, subscribe_to_event, register_route

class SmartFeedbackSync:
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

            emit_telemetry("smart_feedback_sync_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("smart_feedback_sync_recovered_1", "position_calculated", {
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
                        "module": "smart_feedback_sync_recovered_1",
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
                print(f"Emergency stop error in smart_feedback_sync_recovered_1: {e}")
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
                "module": "smart_feedback_sync_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("smart_feedback_sync_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in smart_feedback_sync_recovered_1: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "smart_feedback_sync_recovered_1",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in smart_feedback_sync_recovered_1: {e}")
    """
    Smart Feedback Synchronization Engine for GENESIS Trading System
    
    Ingests live feedback from:
    - SignalConfidenceRatingEngine 
    - LiveTradeFeedbackInjector
    
    Processes telemetry signals:
    - ExecutionDeviationAlert
    - RecalibrationRequest  
    - TerminateMonitorLoop
    
    Routes adaptive adjustments to:
    - AdaptiveExecutionResolver
    - TelemetryCollector
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        self.module_name = "SmartFeedbackSync"
        self.learning_mode = True
        self.halt_on_signal_loss = True
        
        # Real-time feedback storage (sliding windows)
        self.signal_confidence_window = deque(maxlen=100)  # Last 100 confidence scores
        self.trade_feedback_window = deque(maxlen=50)      # Last 50 trade executions
        self.deviation_alerts = deque(maxlen=20)           # Last 20 deviation alerts
        
        # Drift detection parameters
        self.confidence_drift_threshold = 0.15
        self.execution_drift_threshold = 0.20
        self.recalibration_cooldown = 300  # 5 minutes
        self.last_recalibration = 0
        
        # Learning mode parameters
        self.drift_patterns = defaultdict(list)
        self.adaptation_history = []
        self.signal_loss_counter = 0
        self.max_signal_loss = 5
        
        # Thread safety
        self.lock = threading.Lock()
        self.processing_active = True
        
        # Logging and telemetry
        self._setup_logging()
        self._setup_telemetry()
        self._subscribe_to_events()
        
        # Start background processing
        self.feedback_processor = threading.Thread(target=self._process_feedback_loop, daemon=True)
        self.feedback_processor.start()
        
        self._emit_module_ready()
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _setup_logging(self):
        """Setup structured JSONL logging for institutional compliance"""
        os.makedirs("logs/smart_feedback_sync", exist_ok=True)
        
        self.logger = logging.getLogger(f"SmartFeedbackSync")
        self.logger.setLevel(logging.INFO)
        
        # JSONL handler for structured logs
        jsonl_handler = logging.FileHandler(f"logs/smart_feedback_sync/feedback_sync_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl")
        jsonl_formatter = logging.Formatter('{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":%(message)s}')
        jsonl_handler.setFormatter(jsonl_formatter)
        self.logger.addHandler(jsonl_handler)
    
    def _setup_telemetry(self):
        """Setup telemetry data storage and metrics collection"""
        os.makedirs("data/smart_feedback_sync", exist_ok=True)
        
        self.telemetry_data = {
            "session_id": f"feedback_sync_{int(time.time())}",
            "start_time": datetime.datetime.now().isoformat(),
            "signals_processed": 0,
            "drift_detections": 0,
            "recalibrations_triggered": 0,
            "learning_adaptations": 0,
            "signal_loss_events": 0
        }
    
    def _subscribe_to_events(self):
        """Subscribe to EventBus events for feedback ingestion"""
        # Subscribe to signal confidence feedback
        subscribe_to_event("SignalConfidenceRated", self._on_signal_confidence_rated, self.module_name)
        subscribe_to_event("FeedbackSync", self._on_feedback_sync, self.module_name)
        
        # Subscribe to trade execution feedback  
        subscribe_to_event("TradeCompletionFeedback", self._on_trade_completion_feedback, self.module_name)
        subscribe_to_event("LiveExecutionFeedback", self._on_live_execution_feedback, self.module_name)
        
        # Subscribe to telemetry signals
        subscribe_to_event("ExecutionDeviationAlert", self._on_execution_deviation_alert, self.module_name)
        subscribe_to_event("RecalibrationRequest", self._on_recalibration_request, self.module_name)
        subscribe_to_event("TerminateMonitorLoop", self._on_terminate_monitor_loop, self.module_name)
        
        # Register EventBus routes
        self._register_eventbus_routes()
    
    def _register_eventbus_routes(self):
        """Register all EventBus routes for smart feedback sync"""
        routes = [
            ("SignalConfidenceRated", "SignalConfidenceRatingEngine", "SmartFeedbackSync"),
            ("FeedbackSync", "SignalConfidenceRatingEngine", "SmartFeedbackSync"),
            ("TradeCompletionFeedback", "LiveTradeFeedbackInjector", "SmartFeedbackSync"), 
            ("LiveExecutionFeedback", "SmartExecutionLiveLoop", "SmartFeedbackSync"),
            ("ExecutionDeviationAlert", "SmartExecutionMonitor", "SmartFeedbackSync"),
            ("RecalibrationRequest", "SmartExecutionMonitor", "SmartFeedbackSync"),
            ("TerminateMonitorLoop", "SmartExecutionMonitor", "SmartFeedbackSync"),
            ("DriftDetected", "SmartFeedbackSync", "AdaptiveExecutionResolver"),
            ("RecalibrationNeeded", "SmartFeedbackSync", "AdaptiveExecutionResolver"),
            ("FeedbackSyncMetric", "SmartFeedbackSync", "TelemetryCollector"),
            ("LearningAdaptation", "SmartFeedbackSync", "TelemetryCollector")
        ]
        
        for topic, producer, consumer in routes:
            register_route(topic, producer, consumer)
    
    def _on_signal_confidence_rated(self, event_data):
        """Handle signal confidence rating events"""
        with self.lock:
            try:
                data = event_data.get("data", {}) if isinstance(event_data, dict) else event_data
                confidence_score = data.get("confidence_score", 0.0)
                signal_id = data.get("signal_id", "unknown")
                timestamp = data.get("timestamp", datetime.datetime.now().isoformat())
                
                # Store in sliding window
                self.signal_confidence_window.append({
                    "signal_id": signal_id,
                    "confidence": confidence_score,
                    "timestamp": timestamp
                })
                
                self.telemetry_data["signals_processed"] += 1
                self.signal_loss_counter = 0  # Reset signal loss counter
                
                self.logger.info(json.dumps({
                    "event": "signal_confidence_rated",
                    "signal_id": signal_id,
                    "confidence": confidence_score,
                    "window_size": len(self.signal_confidence_window)
                }))
                
                # Check for confidence drift
                self._check_confidence_drift()
                
            except Exception as e:
                self._handle_error("signal_confidence_processing", str(e))
    
    def _on_feedback_sync(self, event_data):
        """Handle feedback sync events from signal confidence engine"""
        with self.lock:
            try:
                data = event_data.get("data", {}) if isinstance(event_data, dict) else event_data
                sync_data = data.get("sync_data", {})
                source_module = data.get("source_module", "unknown")
                
                self.logger.info(json.dumps({
                    "event": "feedback_sync_received",
                    "source": source_module,
                    "sync_data": sync_data
                }))
                
                # Process sync data for learning adaptations
                if self.learning_mode:
                    self._process_learning_adaptation(sync_data)
                
            except Exception as e:
                self._handle_error("feedback_sync_processing", str(e))
    
    def _on_trade_completion_feedback(self, event_data):
        """Handle trade completion feedback events"""
        with self.lock:
            try:
                data = event_data.get("data", {}) if isinstance(event_data, dict) else event_data
                trade_id = data.get("trade_id", "unknown")
                execution_quality = data.get("execution_quality", 0.0)
                slippage = data.get("slippage", 0.0)
                timing_deviation = data.get("timing_deviation", 0.0)
                timestamp = data.get("timestamp", datetime.datetime.now().isoformat())
                
                # Store in sliding window
                self.trade_feedback_window.append({
                    "trade_id": trade_id,
                    "execution_quality": execution_quality,
                    "slippage": slippage,
                    "timing_deviation": timing_deviation,
                    "timestamp": timestamp
                })
                
                self.signal_loss_counter = 0  # Reset signal loss counter
                
                self.logger.info(json.dumps({
                    "event": "trade_completion_feedback",
                    "trade_id": trade_id,
                    "execution_quality": execution_quality,
                    "slippage": slippage,
                    "window_size": len(self.trade_feedback_window)
                }))
                
                # Check for execution drift
                self._check_execution_drift()
                
            except Exception as e:
                self._handle_error("trade_feedback_processing", str(e))
    
    def _on_live_execution_feedback(self, event_data):
        """Handle live execution feedback from smart execution loop"""
        with self.lock:
            try:
                data = event_data.get("data", {}) if isinstance(event_data, dict) else event_data
                execution_metrics = data.get("execution_metrics", {})
                deviation_score = data.get("deviation_score", 0.0)
                
                # Add to trade feedback for analysis
                self.trade_feedback_window.append({
                    "trade_id": execution_metrics.get("trade_id", "live"),
                    "execution_quality": 1.0 - deviation_score,  # Convert deviation to quality
                    "slippage": execution_metrics.get("slippage", 0.0),
                    "timing_deviation": execution_metrics.get("timing_deviation", 0.0),
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                self.signal_loss_counter = 0
                
                self.logger.info(json.dumps({
                    "event": "live_execution_feedback",
                    "deviation_score": deviation_score,
                    "execution_metrics": execution_metrics
                }))
                
            except Exception as e:
                self._handle_error("live_execution_feedback_processing", str(e))
    
    def _on_execution_deviation_alert(self, event_data):
        """Handle execution deviation alerts"""
        with self.lock:
            try:
                data = event_data.get("data", {}) if isinstance(event_data, dict) else event_data
                alert_severity = data.get("severity", "medium")
                deviation_type = data.get("deviation_type", "unknown")
                deviation_value = data.get("deviation_value", 0.0)
                timestamp = data.get("timestamp", datetime.datetime.now().isoformat())
                
                # Store deviation alert
                self.deviation_alerts.append({
                    "severity": alert_severity,
                    "type": deviation_type,
                    "value": deviation_value,
                    "timestamp": timestamp
                })
                
                self.telemetry_data["drift_detections"] += 1
                
                self.logger.info(json.dumps({
                    "event": "execution_deviation_alert",
                    "severity": alert_severity,
                    "deviation_type": deviation_type,
                    "deviation_value": deviation_value
                }))
                
                # Trigger adaptive response if severe
                if alert_severity in ["high", "critical"]:
                    self._trigger_adaptive_response("execution_deviation", {
                        "severity": alert_severity,
                        "type": deviation_type,
                        "value": deviation_value
                    })
                
            except Exception as e:
                self._handle_error("execution_deviation_processing", str(e))
    
    def _on_recalibration_request(self, event_data):
        """Handle recalibration requests"""
        with self.lock:
            try:
                data = event_data.get("data", {}) if isinstance(event_data, dict) else event_data
                request_reason = data.get("reason", "unknown")
                urgency = data.get("urgency", "medium")
                
                self.logger.info(json.dumps({
                    "event": "recalibration_request",
                    "reason": request_reason,
                    "urgency": urgency
                }))
                
                # Process recalibration if not in cooldown
                current_time = time.time()
                if current_time - self.last_recalibration > self.recalibration_cooldown:
                    self._trigger_recalibration(request_reason, urgency)
                    self.last_recalibration = current_time
                else:
                    self.logger.info(json.dumps({
                        "event": "recalibration_cooldown",
                        "time_remaining": self.recalibration_cooldown - (current_time - self.last_recalibration)
                    }))
                
            except Exception as e:
                self._handle_error("recalibration_request_processing", str(e))
    
    def _on_terminate_monitor_loop(self, event_data):
        """Handle monitor loop termination signals"""
        with self.lock:
            try:
                data = event_data.get("data", {}) if isinstance(event_data, dict) else event_data
                termination_reason = data.get("reason", "unknown")
                
                self.logger.info(json.dumps({
                    "event": "monitor_loop_termination",
                    "reason": termination_reason
                }))
                
                # Emit final metrics before potential shutdown
                self._emit_final_metrics()
                
            except Exception as e:
                self._handle_error("monitor_termination_processing", str(e))
    
    def _check_confidence_drift(self):
        """Check for drift in signal confidence patterns"""
        if len(self.signal_confidence_window) < 10:
            return
        
        recent_confidences = [s["confidence"] for s in list(self.signal_confidence_window)[-10:]]
        historical_confidences = [s["confidence"] for s in list(self.signal_confidence_window)[:-10]]
        
        if len(historical_confidences) < 5:
            return
        
        recent_mean = mean(recent_confidences)
        historical_mean = mean(historical_confidences)
        drift_magnitude = abs(recent_mean - historical_mean)
        
        if drift_magnitude > self.confidence_drift_threshold:
            self._emit_drift_detected("confidence_drift", {
                "recent_mean": recent_mean,
                "historical_mean": historical_mean,
                "drift_magnitude": drift_magnitude,
                "threshold": self.confidence_drift_threshold
            })
    
    def _check_execution_drift(self):
        """Check for drift in execution quality patterns"""
        if len(self.trade_feedback_window) < 5:
            return
        
        recent_quality = [t["execution_quality"] for t in list(self.trade_feedback_window)[-5:]]
        historical_quality = [t["execution_quality"] for t in list(self.trade_feedback_window)[:-5]]
        
        if len(historical_quality) < 3:
            return
        
        recent_mean = mean(recent_quality)
        historical_mean = mean(historical_quality)
        drift_magnitude = abs(recent_mean - historical_mean)
        
        if drift_magnitude > self.execution_drift_threshold:
            self._emit_drift_detected("execution_drift", {
                "recent_mean": recent_mean,
                "historical_mean": historical_mean,
                "drift_magnitude": drift_magnitude,
                "threshold": self.execution_drift_threshold
            })
    
    def _emit_drift_detected(self, drift_type, drift_data):
        """Emit drift detection event to adaptive execution resolver"""
        event_data = {
            "drift_type": drift_type,
            "drift_data": drift_data,
            "timestamp": datetime.datetime.now().isoformat(),
            "module": self.module_name,
            "learning_mode": self.learning_mode
        }
        
        emit_event("DriftDetected", event_data, self.module_name)
        
        self.logger.info(json.dumps({
            "event": "drift_detected",
            "drift_type": drift_type,
            "drift_data": drift_data
        }))
    
    def _trigger_adaptive_response(self, response_type, response_data):
        """Trigger adaptive response through execution resolver"""
        event_data = {
            "response_type": response_type,
            "response_data": response_data,
            "timestamp": datetime.datetime.now().isoformat(),
            "module": self.module_name,
            "urgency": "high"
        }
        
        emit_event("RecalibrationNeeded", event_data, self.module_name)
        
        self.logger.info(json.dumps({
            "event": "adaptive_response_triggered",
            "response_type": response_type,
            "response_data": response_data
        }))
    
    def _trigger_recalibration(self, reason, urgency):
        """Trigger system recalibration"""
        recalibration_data = {
            "reason": reason,
            "urgency": urgency,
            "timestamp": datetime.datetime.now().isoformat(),
            "module": self.module_name,
            "signal_window_size": len(self.signal_confidence_window),
            "trade_window_size": len(self.trade_feedback_window)
        }
        
        emit_event("RecalibrationNeeded", recalibration_data, self.module_name)
        self.telemetry_data["recalibrations_triggered"] += 1
        
        self.logger.info(json.dumps({
            "event": "recalibration_triggered",
            "recalibration_data": recalibration_data
        }))
    
    def _process_learning_adaptation(self, sync_data):
        """Process learning adaptations in feedback patterns"""
        try:
            adaptation_data = {
                "sync_data": sync_data,
                "timestamp": datetime.datetime.now().isoformat(),
                "confidence_window_size": len(self.signal_confidence_window),
                "trade_window_size": len(self.trade_feedback_window)
            }
            
            self.adaptation_history.append(adaptation_data)
            self.telemetry_data["learning_adaptations"] += 1
            
            # Emit learning adaptation event
            emit_event("LearningAdaptation", adaptation_data, self.module_name)
            
            self.logger.info(json.dumps({
                "event": "learning_adaptation",
                "adaptation_data": adaptation_data
            }))
            
        except Exception as e:
            self._handle_error("learning_adaptation", str(e))
    
    def _process_feedback_loop(self):
        """Background feedback processing loop"""
        while self.processing_active:
            try:
                # Check for signal loss
                self._check_signal_loss()
                
                # Emit periodic metrics
                self._emit_feedback_metrics()
                
                # Save telemetry data
                self._save_telemetry_data()
                
                time.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                self._handle_error("feedback_loop", str(e))
                time.sleep(60)  # Wait longer on error
    
    def _check_signal_loss(self):
        """Check for signal loss and handle halt condition"""
        if not self.signal_confidence_window and not self.trade_feedback_window:
            self.signal_loss_counter += 1
            self.telemetry_data["signal_loss_events"] += 1
            
            self.logger.warning(json.dumps({
                "event": "signal_loss_detected",
                "loss_count": self.signal_loss_counter,
                "max_allowed": self.max_signal_loss
            }))
            
            if self.halt_on_signal_loss and self.signal_loss_counter >= self.max_signal_loss:
                self._emit_signal_loss_halt()
    
    def _emit_signal_loss_halt(self):
        """Emit signal loss halt event"""
        halt_data = {
            "reason": "signal_loss_threshold_exceeded",
            "signal_loss_count": self.signal_loss_counter,
            "max_allowed": self.max_signal_loss,
            "timestamp": datetime.datetime.now().isoformat(),
            "module": self.module_name
        }
        
        emit_event("RecalibrationNeeded", halt_data, self.module_name)
        
        self.logger.critical(json.dumps({
            "event": "signal_loss_halt",
            "halt_data": halt_data
        }))
    
    def _emit_feedback_metrics(self):
        """Emit periodic feedback sync metrics"""
        metrics_data = {
            "signals_processed": self.telemetry_data["signals_processed"],
            "drift_detections": self.telemetry_data["drift_detections"], 
            "recalibrations_triggered": self.telemetry_data["recalibrations_triggered"],
            "learning_adaptations": self.telemetry_data["learning_adaptations"],
            "signal_loss_events": self.telemetry_data["signal_loss_events"],
            "confidence_window_size": len(self.signal_confidence_window),
            "trade_window_size": len(self.trade_feedback_window),
            "deviation_alerts_count": len(self.deviation_alerts),
            "timestamp": datetime.datetime.now().isoformat(),
            "module": self.module_name
        }
        
        emit_event("FeedbackSyncMetric", metrics_data, self.module_name)
    
    def _emit_final_metrics(self):
        """Emit final metrics before shutdown"""
        final_metrics = {
            **self.telemetry_data,
            "end_time": datetime.datetime.now().isoformat(),
            "total_runtime": time.time() - time.mktime(datetime.datetime.fromisoformat(self.telemetry_data["start_time"]).timetuple()),
            "final_confidence_window_size": len(self.signal_confidence_window),
            "final_trade_window_size": len(self.trade_feedback_window)
        }
        
        emit_event("FeedbackSyncMetric", final_metrics, self.module_name)
        
        self.logger.info(json.dumps({
            "event": "final_metrics",
            "metrics": final_metrics
        }))
    
    def _save_telemetry_data(self):
        """Save telemetry data to disk"""
        try:
            filename = f"data/smart_feedback_sync/feedback_sync_telemetry_{datetime.datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(self.telemetry_data, f, indent=2)
        except Exception as e:
            self._handle_error("telemetry_save", str(e))
    
    def _emit_module_ready(self):
        """Emit module ready event"""
        ready_data = {
            "module": self.module_name,
            "status": "ready",
            "learning_mode": self.learning_mode,
            "halt_on_signal_loss": self.halt_on_signal_loss,
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0"
        }
        
        emit_event("ModuleTelemetry", ready_data, self.module_name)
        
        self.logger.info(json.dumps({
            "event": "module_ready",
            "ready_data": ready_data
        }))
    
    def _handle_error(self, context, error_message):
        """Handle and log errors with telemetry"""
        error_data = {
            "context": context,
            "error": error_message,
            "timestamp": datetime.datetime.now().isoformat(),
            "module": self.module_name
        }
        
        emit_event("ModuleError", error_data, self.module_name)
        
        self.logger.error(json.dumps({
            "event": "error",
            "error_data": error_data
        }))

if __name__ == "__main__":
    # Initialize Smart Feedback Sync for standalone testing
    smart_feedback_sync = SmartFeedbackSync()
    
    print("🚀 GENESIS Smart Feedback Sync - Phase 19 Initialized")
    print(f"📡 Learning Mode: {smart_feedback_sync.learning_mode}")
    print(f"🛑 Halt on Signal Loss: {smart_feedback_sync.halt_on_signal_loss}")
    print("📊 EventBus routes registered for feedback ingestion")
    
    try:
        # Keep the module running for testing
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Smart Feedback Sync shutting down...")
        smart_feedback_sync.processing_active = False

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
        

# <!-- @GENESIS_MODULE_END: smart_feedback_sync -->