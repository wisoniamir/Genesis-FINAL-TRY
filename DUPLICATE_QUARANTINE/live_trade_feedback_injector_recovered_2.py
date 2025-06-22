# <!-- @GENESIS_MODULE_START: live_trade_feedback_injector -->

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENESIS Live Trade Feedback Injection Engine v2.7 - PHASE 12
Real-time trade outcome feedback injection into signal learning ecosystem
ARCHITECT MODE: v2.7 - STRICT COMPLIANCE

PHASE 12 OBJECTIVE:
Parse MT5 trade fills, match with signal fingerprints, inject into learning system

INPUTS CONSUMED:
- ExecutionSnapshot: Real MT5 trade execution data
- SL_HitEvent: Stop loss triggered events  
- TP_HitEvent: Take profit triggered events
- TradeFillEvent: Trade completion events

OUTPUTS EMITTED:
- TradeOutcomeFeedback: Trade result linked to signal
- ReinforceSignalMemory: Signal bias score updates
- PnLScoreUpdate: Performance scoring updates
- TradeMetaLogEntry: Metadata for analysis

VALIDATION REQUIREMENTS:
✅ Real MT5 data only (no real/execute)
✅ EventBus communication only
✅ Signal fingerprint matching
✅ Bias score modification
✅ Telemetry integration

NO real DATA - NO ISOLATED FUNCTIONS - STRICT COMPLIANCE
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock
from pathlib import Path

from event_bus import get_event_bus, emit_event, register_route

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradeFeedbackInjector:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "live_trade_feedback_injector_recovered_2",
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
                print(f"Emergency stop error in live_trade_feedback_injector_recovered_2: {e}")
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
    """
    GENESIS Live Trade Feedback Injection Engine - PHASE 12
    
    PHASE 12 Architecture Compliance:
    - ✅ Real MT5 trade outcome processing
    - ✅ Signal fingerprint matching
    - ✅ Dynamic bias score adjustment
    - ✅ EventBus only communication
    - ✅ Telemetry hooks enabled
    - ✅ No isolated functions
    """
    
    def __init__(self):
        """Initialize Live Trade Feedback Injection Engine"""
        self.module_name = "LiveTradeFeedbackInjector"
        self.event_bus = get_event_bus()
        
        # Signal tracking and matching
        self.signal_fingerprints = {}  # signal_id -> execution_details
        self.pending_executions = {}   # execution_id -> execution_data
        self.completed_trades = deque(maxlen=1000)  # Trade outcome history
        
        # Signal bias scores and reinforcement data
        self.signal_bias_scores = defaultdict(lambda: 1.0)  # signal_id -> bias_score
        self.signal_performance = defaultdict(list)  # signal_id -> [outcomes]
        self.pnl_tracking = defaultdict(list)  # signal_id -> [pnl_values]
        
        # Thread safety
        self.lock = Lock()
          # Telemetry
        self.telemetry = {
            "executions_processed": 0,
            "signals_matched": 0,
            "bias_adjustments": 0,
            "feedback_injections": 0,
            "pnl_updates": 0,
            "avg_processing_time_ms": 0.0
        }
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
        # Initialize logging directory
        self.log_dir = Path("logs/live_trade_feedback")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load signal fingerprints
        self._load_signal_fingerprints()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("✅ LiveTradeFeedbackInjector initialized - PHASE 12 READY")

    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _load_signal_fingerprints(self):
        """Load existing signal fingerprints from file system"""
        try:
            fingerprint_file = "signal_fingerprints.json"
            if os.path.exists(fingerprint_file):
                with open(fingerprint_file, 'r') as f:
                    self.signal_fingerprints = json.load(f)
                logger.info(f"✅ Loaded {len(self.signal_fingerprints)} signal fingerprints")
            else:
                logger.info("ℹ️ No existing signal fingerprints found - starting fresh")
                
        except Exception as e:
            logger.error(f"❌ Error loading signal fingerprints: {str(e)}")
            self._emit_error("FINGERPRINT_LOAD_ERROR", str(e))

    def _register_event_handlers(self):
        """Register EventBus event handlers"""
        try:
            # Register event consumption routes
            register_route("ExecutionSnapshot", self.module_name, "LiveTradeFeedbackInjector")
            register_route("SL_HitEvent", self.module_name, "LiveTradeFeedbackInjector") 
            register_route("TP_HitEvent", self.module_name, "LiveTradeFeedbackInjector")
            register_route("TradeFillEvent", self.module_name, "LiveTradeFeedbackInjector")
            
            # Subscribe to events
            self.event_bus.subscribe("ExecutionSnapshot", self._handle_execution_snapshot, self.module_name)
            self.event_bus.subscribe("SL_HitEvent", self._handle_sl_hit, self.module_name)
            self.event_bus.subscribe("TP_HitEvent", self._handle_tp_hit, self.module_name)
            self.event_bus.subscribe("TradeFillEvent", self._handle_trade_fill, self.module_name)
            
            logger.info("✅ Event handlers registered successfully")
              except Exception as e:
            logger.error(f"❌ Error registering event handlers: {str(e)}")
            self._emit_error("EVENT_REGISTRATION_ERROR", str(e))
            def _handle_execution_snapshot(self, event_data):
        """
        PHASE 12: Handle ExecutionSnapshot events from ExecutionEngine
        
        Args:
            event_data (dict): ExecutionSnapshot event data
        """
        start_time = time.time()
        
        try:
            # Extract actual data from event wrapper
            actual_data = event_data.get('data', event_data)
            
            execution_id = actual_data.get("execution_id")
            signal_id = actual_data.get("signal_id")
            
            logger.info(f"🔍 Processing ExecutionSnapshot: execution_id='{execution_id}', signal_id='{signal_id}'")
            
            assert execution_id or not signal_id:
                logger.warning(f"⚠️ ExecutionSnapshot missing execution_id or signal_id: execution_id='{execution_id}', signal_id='{signal_id}'")
                return
            
            with self.lock:
                # Store execution data for matching
                self.pending_executions[execution_id] = {
                    "timestamp": event_data.get("timestamp"),
                    "signal_id": signal_id,
                    "symbol": event_data.get("symbol"),
                    "direction": event_data.get("direction"),
                    "entry_price": event_data.get("entry_price"),
                    "volume": event_data.get("volume"),
                    "stop_loss": event_data.get("stop_loss"),
                    "take_profit": event_data.get("take_profit"),
                    "position_id": event_data.get("position_id"),
                    "execution_snapshot_received": True
                }
                
                # Update signal fingerprint
                self.signal_fingerprints[signal_id] = {
                    "last_execution": execution_id,
                    "symbol": event_data.get("symbol"),
                    "timestamp": event_data.get("timestamp"),
                    "execution_count": self.signal_fingerprints.get(signal_id, {}).get("execution_count", 0) + 1
                }
                
                self.telemetry["executions_processed"] += 1
                  # Processing time tracking
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.telemetry["avg_processing_time_ms"] = round(sum(self.processing_times) / len(self.processing_times), 2)
            
            logger.info(f"✅ ExecutionSnapshot processed: {execution_id} -> {signal_id}")
            self._emit_telemetry("execution_snapshot_processed", {"execution_id": execution_id, "signal_id": signal_id})
            
        except Exception as e:
            logger.error(f"❌ Error handling ExecutionSnapshot: {str(e)}")
            self._emit_error("EXECUTION_SNAPSHOT_HANDLER_ERROR", str(e))

    def _handle_sl_hit(self, event_data):
        """
        Handle Stop Loss hit events
        
        Args:
            event_data (dict): SL_HitEvent data
        """
        try:
            execution_id = event_data.get("execution_id")
            position_id = event_data.get("position_id")
            
            if execution_id in self.pending_executions:
                trade_outcome = self._process_trade_outcome(execution_id, "STOP_LOSS", event_data)
                self._inject_trade_feedback(trade_outcome)
                
            logger.info(f"✅ SL_HitEvent processed for execution {execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling SL_HitEvent: {str(e)}")
            self._emit_error("SL_HIT_HANDLER_ERROR", str(e))

    def _handle_tp_hit(self, event_data):
        """
        Handle Take Profit hit events
        
        Args:
            event_data (dict): TP_HitEvent data
        """
        try:
            execution_id = event_data.get("execution_id")
            position_id = event_data.get("position_id")
            
            if execution_id in self.pending_executions:
                trade_outcome = self._process_trade_outcome(execution_id, "TAKE_PROFIT", event_data)
                self._inject_trade_feedback(trade_outcome)
                
            logger.info(f"✅ TP_HitEvent processed for execution {execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling TP_HitEvent: {str(e)}")
            self._emit_error("TP_HIT_HANDLER_ERROR", str(e))

    def _handle_trade_fill(self, event_data):
        """
        Handle general trade fill events
        
        Args:
            event_data (dict): TradeFillEvent data
        """
        try:
            execution_id = event_data.get("execution_id")
            outcome = event_data.get("outcome", "UNKNOWN")
            
            if execution_id in self.pending_executions:
                trade_outcome = self._process_trade_outcome(execution_id, outcome, event_data)
                self._inject_trade_feedback(trade_outcome)
                
            logger.info(f"✅ TradeFillEvent processed for execution {execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling TradeFillEvent: {str(e)}")
            self._emit_error("TRADE_FILL_HANDLER_ERROR", str(e))

    def _process_trade_outcome(self, execution_id, outcome, event_data):
        """
        Process trade outcome and calculate performance metrics
        
        Args:
            execution_id (str): Execution identifier
            outcome (str): Trade outcome (STOP_LOSS, TAKE_PROFIT, etc.)
            event_data (dict): Event data
            
        Returns:
            dict: Processed trade outcome data
        """        try:
            with self.lock:
                execution_data = self.pending_executions.get(execution_id, {})
                signal_id = execution_data.get("signal_id")
                
                if not signal_id:
                    logger.warning(f"⚠️ No signal_id found for execution {execution_id}")
                    self._emit_error_event("missing_signal_id", {
                        "execution_id": execution_id,
                        "error": "No signal_id found for execution",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    raise ValueError(f"ARCHITECT_MODE_COMPLIANCE: No signal_id found for execution {execution_id}")
                
                # Calculate PnL and performance metrics
                entry_price = execution_data.get("entry_price", 0.0)
                exit_price = event_data.get("exit_price", entry_price)
                volume = execution_data.get("volume", 0.0)
                direction = execution_data.get("direction", "long")
                
                # Calculate PnL
                if direction == "long":
                    pnl = (exit_price - entry_price) * volume
                else:
                    pnl = (entry_price - exit_price) * volume
                
                # Determine win/loss
                is_win = (outcome == "TAKE_PROFIT") or (pnl > 0)
                
                trade_outcome = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "execution_id": execution_id,
                    "signal_id": signal_id,
                    "symbol": execution_data.get("symbol"),
                    "direction": direction,
                    "outcome": outcome,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "volume": volume,
                    "pnl": pnl,
                    "is_win": is_win,
                    "duration_minutes": self._calculate_trade_duration(execution_data),
                    "stop_loss": execution_data.get("stop_loss"),
                    "take_profit": execution_data.get("take_profit")
                }
                
                # Add to completed trades
                self.completed_trades.append(trade_outcome)
                
                # Update signal performance tracking
                self.signal_performance[signal_id].append({
                    "outcome": outcome,
                    "pnl": pnl,
                    "is_win": is_win,
                    "timestamp": trade_outcome["timestamp"]
                })
                
                # Update PnL tracking
                self.pnl_tracking[signal_id].append(pnl)
                
                # Remove from pending                if execution_id in self.pending_executions:
                    del self.pending_executions[execution_id]
                
                self.telemetry["signals_matched"] += 1
                
                return trade_outcome
                
        except Exception as e:
            logger.error(f"❌ Error processing trade outcome: {str(e)}")
            self._emit_error("TRADE_OUTCOME_PROCESSING_ERROR", str(e))
            raise RuntimeError(f"ARCHITECT_MODE_COMPLIANCE: Trade outcome processing failed - {e}")

    def _inject_trade_feedback(self, trade_outcome):
        """
        PHASE 12: Inject trade feedback into signal learning ecosystem
        
        Args:
            trade_outcome (dict): Processed trade outcome data
        """
        try:
            if not trade_outcome:
                return
                
            signal_id = trade_outcome["signal_id"]
            is_win = trade_outcome["is_win"]
            pnl = trade_outcome["pnl"]
            
            with self.lock:
                # Calculate new bias score
                current_bias = self.signal_bias_scores[signal_id]
                
                if is_win:
                    # Boost successful signals
                    new_bias = min(current_bias * 1.15, 2.0)  # Cap at 2.0
                else:
                    # Penalize unsuccessful signals
                    new_bias = max(current_bias * 0.85, 0.1)  # Floor at 0.1
                
                self.signal_bias_scores[signal_id] = new_bias
                self.telemetry["bias_adjustments"] += 1
                
            # Emit TradeOutcomeFeedback
            self._emit_trade_outcome_feedback(trade_outcome, current_bias, new_bias)
            
            # Emit ReinforceSignalMemory
            self._emit_reinforce_signal_memory(signal_id, new_bias, is_win)
            
            # Emit PnLScoreUpdate
            self._emit_pnl_score_update(signal_id, pnl, trade_outcome)
            
            # Emit TradeMetaLogEntry
            self._emit_trade_meta_log(trade_outcome)
            
            self.telemetry["feedback_injections"] += 1
            
            logger.info(f"✅ Trade feedback injected for signal {signal_id}: bias {current_bias:.3f} -> {new_bias:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error injecting trade feedback: {str(e)}")
            self._emit_error("FEEDBACK_INJECTION_ERROR", str(e))

    def _emit_trade_outcome_feedback(self, trade_outcome, old_bias, new_bias):
        """Emit TradeOutcomeFeedback event"""
        try:
            feedback_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "module": self.module_name,
                "signal_id": trade_outcome["signal_id"],
                "execution_id": trade_outcome["execution_id"],
                "outcome": trade_outcome["outcome"],
                "pnl": trade_outcome["pnl"],
                "is_win": trade_outcome["is_win"],
                "bias_score_before": old_bias,
                "bias_score_after": new_bias,
                "bias_adjustment": new_bias - old_bias,
                "symbol": trade_outcome["symbol"],
                "direction": trade_outcome["direction"]
            }
            
            emit_event("TradeOutcomeFeedback", feedback_data)
            logger.info(f"✅ TradeOutcomeFeedback emitted for signal {trade_outcome['signal_id']}")
            
        except Exception as e:
            logger.error(f"❌ Error emitting TradeOutcomeFeedback: {str(e)}")

    def _emit_reinforce_signal_memory(self, signal_id, new_bias, is_win):
        """Emit ReinforceSignalMemory event"""
        try:
            memory_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "module": self.module_name,
                "signal_id": signal_id,
                "new_bias_score": new_bias,
                "reinforcement_type": "POSITIVE" if is_win else "NEGATIVE",
                "learning_action": "BOOST" if is_win else "PENALIZE",
                "memory_update": True
            }
            
            emit_event("ReinforceSignalMemory", memory_data)
            logger.info(f"✅ ReinforceSignalMemory emitted for signal {signal_id}")
            
        except Exception as e:
            logger.error(f"❌ Error emitting ReinforceSignalMemory: {str(e)}")

    def _emit_pnl_score_update(self, signal_id, pnl, trade_outcome):
        """Emit PnLScoreUpdate event"""
        try:
            # Calculate aggregate PnL metrics
            signal_pnls = self.pnl_tracking[signal_id]
            total_pnl = sum(signal_pnls)
            avg_pnl = total_pnl / len(signal_pnls) if signal_pnls else 0
            
            pnl_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "module": self.module_name,
                "signal_id": signal_id,
                "trade_pnl": pnl,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "trade_count": len(signal_pnls),
                "execution_id": trade_outcome["execution_id"],
                "symbol": trade_outcome["symbol"]
            }
            
            emit_event("PnLScoreUpdate", pnl_data)
            self.telemetry["pnl_updates"] += 1
            logger.info(f"✅ PnLScoreUpdate emitted for signal {signal_id}")
            
        except Exception as e:
            logger.error(f"❌ Error emitting PnLScoreUpdate: {str(e)}")

    def _emit_trade_meta_log(self, trade_outcome):
        """Emit TradeMetaLogEntry event"""
        try:
            meta_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "module": self.module_name,
                "execution_id": trade_outcome["execution_id"],
                "signal_id": trade_outcome["signal_id"],
                "meta_type": "TRADE_COMPLETION",
                "outcome": trade_outcome["outcome"],
                "duration_minutes": trade_outcome["duration_minutes"],
                "entry_price": trade_outcome["entry_price"],
                "exit_price": trade_outcome["exit_price"],
                "volume": trade_outcome["volume"],
                "stop_loss": trade_outcome["stop_loss"],
                "take_profit": trade_outcome["take_profit"],
                "pnl": trade_outcome["pnl"],
                "is_win": trade_outcome["is_win"]
            }
            
            emit_event("TradeMetaLogEntry", meta_data)
            logger.info(f"✅ TradeMetaLogEntry emitted for execution {trade_outcome['execution_id']}")
            
        except Exception as e:
            logger.error(f"❌ Error emitting TradeMetaLogEntry: {str(e)}")

    def _calculate_trade_duration(self, execution_data):
        """Calculate trade duration in minutes"""
        try:
            start_time = datetime.fromisoformat(execution_data.get("timestamp", ""))
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() / 60
            return round(duration, 2)
        except is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: live_trade_feedback_injector -->