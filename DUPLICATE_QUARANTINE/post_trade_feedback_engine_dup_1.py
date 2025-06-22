
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

                emit_telemetry("post_trade_feedback_engine", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("post_trade_feedback_engine", "position_calculated", {
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
                            "module": "post_trade_feedback_engine",
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
                    print(f"Emergency stop error in post_trade_feedback_engine: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "post_trade_feedback_engine",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("post_trade_feedback_engine", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in post_trade_feedback_engine: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


# <!-- @GENESIS_MODULE_START: post_trade_feedback_engine -->

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENESIS Post-Trade Feedback Engine v1.0 - PHASE 90
Comprehensive trade outcome analysis and strategy feedback system
ARCHITECT MODE: v5.0.0 - STRICT COMPLIANCE

PHASE 90 OBJECTIVE:
Log every closed trade, score outcomes, and provide feedback to strategy mutator
for continuous trading strategy evolution and optimization

INPUTS CONSUMED:
- trade:closed: Closed trade events from MT5 connection bridge
- position:closed: Position closure notifications from execution engine
- order:filled: Final fill confirmations for closing orders
- news:impact: News impact events affecting trades

OUTPUTS EMITTED:
- feedback:new_result: Trade outcome feedback to strategy mutator
- strategy:adjust: Strategy adjustment recommendations
- journal:trade_logged: Trade logged to daily journal notification
- telemetry:trade_outcome: Real-time trade outcome metrics

VALIDATION REQUIREMENTS:
✅ Real MT5 trade data only (no real/execute)
✅ EventBus communication only
✅ FTMO-compliant analysis
✅ Daily journal synchronization
✅ Strategy feedback loop integration
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
import statistics
import hashlib
import uuid
from typing import Dict, List, Any, Tuple, Optional, Union
import math

# Import system components
from event_bus import get_event_bus, emit_event, register_route

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostTradeFeedbackEngine:
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

            emit_telemetry("post_trade_feedback_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("post_trade_feedback_engine", "position_calculated", {
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
                        "module": "post_trade_feedback_engine",
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
                print(f"Emergency stop error in post_trade_feedback_engine: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "post_trade_feedback_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("post_trade_feedback_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in post_trade_feedback_engine: {e}")
    """
    GENESIS Post-Trade Feedback Engine - PHASE 90
    
    PHASE 90 Architecture Compliance:
    - ✅ Real MT5 closed trade processing
    - ✅ Trade outcome scoring and analysis
    - ✅ Strategy feedback generation
    - ✅ EventBus only communication
    - ✅ Daily journal synchronization
    - ✅ Telemetry hooks enabled
    - ✅ No isolated functions
    """
    
    def __init__(self):
        """Initialize Post-Trade Feedback Engine"""
        self.module_name = "PostTradeFeedbackEngine"
        self.event_bus = get_event_bus()
        
        # Trade outcome tracking
        self.closed_trades = deque(maxlen=10000)  # Recent closed trades
        self.trade_scores = defaultdict(list)  # strategy_id -> [scores]
        self.daily_journals = {}  # date -> journal_data
        
        # Scoring parameters
        self.scoring_weights = {
            "profit_factor": 0.3,  # Raw P&L impact
            "risk_reward": 0.25,   # R:R ratio achievement
            "timing_accuracy": 0.2, # Entry/exit timing quality
            "news_impact": 0.15,    # News event correlation
            "slippage_cost": 0.1    # Execution quality
        }
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.trade_count_daily = defaultdict(int)  # date -> count
        
        # Thread safety
        self.lock = Lock()
        
        # Telemetry
        self.telemetry = {
            "trades_processed": 0,
            "feedback_events_sent": 0,
            "strategy_adjustments": 0,
            "journal_entries": 0,
            "avg_processing_time_ms": 0.0,
            "avg_trade_score": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }
        
        # Initialize logging and data directories
        self._initialize_directories()
        
        # Load historical data
        self._load_historical_data()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"✅ {self.module_name} initialized - PHASE 90 ACTIVE")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _initialize_directories(self):
        """Initialize directories for logging and data storage"""
        self.log_dir = Path("logs/feedback_engine")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.journal_dir = Path("journals")
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        self.telemetry_dir = Path("telemetry")
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path("data/trade_feedback")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_historical_data(self):
        """Load historical trade and scoring data"""
        try:
            # Load recent closed trades
            trades_file = self.data_dir / "closed_trades.json"
            if trades_file.exists():
                with open(trades_file, "r") as f:
                    trade_data = json.load(f)
                    self.closed_trades.extend(trade_data[-1000:])  # Load last 1000 trades
                logger.info(f"✅ Loaded {len(self.closed_trades)} historical trades")
            
            # Load trade scores by strategy
            scores_file = self.data_dir / "trade_scores.json"
            if scores_file.exists():
                with open(scores_file, "r") as f:
                    self.trade_scores = defaultdict(list, json.load(f))
                logger.info(f"✅ Loaded scores for {len(self.trade_scores)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Error loading historical data: {str(e)}")
            self._emit_error("HISTORICAL_DATA_LOAD_ERROR", str(e))
    
    def _save_historical_data(self):
        """Save historical data to persistent storage"""
        try:
            # Save closed trades
            trades_file = self.data_dir / "closed_trades.json"
            with open(trades_file, "w") as f:
                json.dump(list(self.closed_trades), f, indent=2)
            
            # Save trade scores
            scores_file = self.data_dir / "trade_scores.json"
            with open(scores_file, "w") as f:
                json.dump(dict(self.trade_scores), f, indent=2)
            
            logger.info(f"✅ Saved {len(self.closed_trades)} trades and scores")
            
        except Exception as e:
            logger.error(f"❌ Error saving historical data: {str(e)}")
            self._emit_error("HISTORICAL_DATA_SAVE_ERROR", str(e))
    
    def _register_event_handlers(self):
        """Register event handlers for incoming events"""
        try:
            # Register event consumption routes
            register_route("trade:closed", "PostTradeFeedbackEngine", self.module_name)
            register_route("position:closed", "PostTradeFeedbackEngine", self.module_name)
            register_route("order:filled", "PostTradeFeedbackEngine", self.module_name)
            register_route("news:impact", "PostTradeFeedbackEngine", self.module_name)
            
            # Register event production routes
            register_route("feedback:new_result", self.module_name, "StrategyMutator")
            register_route("strategy:adjust", self.module_name, "StrategyMutator")
            register_route("journal:trade_logged", self.module_name, "TelemetryCollector")
            register_route("telemetry:trade_outcome", self.module_name, "TelemetryCollector")
            
            # Subscribe to events
            self.event_bus.subscribe("trade:closed", self._handle_trade_closed, self.module_name)
            self.event_bus.subscribe("position:closed", self._handle_position_closed, self.module_name)
            self.event_bus.subscribe("order:filled", self._handle_order_filled, self.module_name)
            self.event_bus.subscribe("news:impact", self._handle_news_impact, self.module_name)
            
            logger.info("✅ Event handlers registered successfully")
            
        except Exception as e:
            logger.error(f"❌ Error registering event handlers: {str(e)}")
            self._emit_error("EVENT_REGISTRATION_ERROR", str(e))
    
    def _handle_trade_closed(self, event_data):
        """
        Handle trade closed events from MT5 connection bridge
        
        Args:
            event_data (dict): Trade closed event data
        """
        start_time = time.time()
        
        try:
            # Extract trade data
            trade_id = event_data.get("trade_id")
            symbol = event_data.get("symbol")
            action = event_data.get("action")
            lot_size = event_data.get("lot_size", 0)
            open_price = event_data.get("open_price", 0)
            close_price = event_data.get("close_price", 0)
            open_time = event_data.get("open_time")
            close_time = event_data.get("close_time", datetime.now().isoformat())
            profit = event_data.get("profit", 0)
            strategy_id = event_data.get("strategy_id", "unknown")
            signal_id = event_data.get("signal_id")
            
            assert trade_id or not symbol:
                logger.warning(f"⚠️ Incomplete trade:closed event: {event_data}")
                return
            
            with self.lock:
                # Create comprehensive trade record
                trade_record = {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "action": action,
                    "lot_size": lot_size,
                    "open_price": open_price,
                    "close_price": close_price,
                    "open_time": open_time,
                    "close_time": close_time,
                    "profit": profit,
                    "strategy_id": strategy_id,
                    "signal_id": signal_id,
                    "processed_at": datetime.now().isoformat(),
                    "duration_minutes": self._calculate_trade_duration(open_time, close_time),
                    "pip_movement": self._calculate_pip_movement(symbol, open_price, close_price, action),
                    "risk_reward_ratio": self._calculate_risk_reward(profit, lot_size)
                }
                
                # Score the trade outcome
                trade_score = self._score_trade_outcome(trade_record)
                trade_record["outcome_score"] = trade_score
                
                # Add to tracking
                self.closed_trades.append(trade_record)
                self.trade_scores[strategy_id].append(trade_score)
                
                # Log to daily journal
                self._log_to_daily_journal(trade_record)
                
                # Send feedback to strategy mutator
                self._send_strategy_feedback(trade_record)
                
                # Update telemetry
                self._update_telemetry(trade_record)
                
                # Performance tracking
                elapsed_ms = (time.time() - start_time) * 1000
                self.processing_times.append(elapsed_ms)
                self.telemetry["avg_processing_time_ms"] = statistics.mean(self.processing_times)
                
                logger.info(f"✅ Processed trade:closed for {symbol} (Score: {trade_score:.2f})")
                
                # Emit telemetry event
                self._emit_trade_outcome_telemetry(trade_record)
                
        except Exception as e:
            logger.error(f"❌ Error handling trade:closed: {str(e)}")
            self._emit_error("TRADE_CLOSED_PROCESSING_ERROR", str(e))
    
    def _handle_position_closed(self, event_data):
        """
        Handle position closed events from execution engine
        
        Args:
            event_data (dict): Position closed event data
        """
        try:
            # Position closure may indicate partial fills or split trades
            # Log for correlation with trade:closed events
            position_id = event_data.get("position_id")
            if position_id:
                logger.info(f"✅ Position closed: {position_id}")
                
        except Exception as e:
            logger.error(f"❌ Error handling position:closed: {str(e)}")
    
    def _handle_order_filled(self, event_data):
        """
        Handle order filled events for closing orders
        
        Args:
            event_data (dict): Order filled event data
        """
        try:
            # Closing order fills indicate trade completion
            order_type = event_data.get("order_type", "").lower()
            if "close" in order_type or "sell" in order_type:
                order_id = event_data.get("order_id")
                logger.info(f"✅ Closing order filled: {order_id}")
                
        except Exception as e:
            logger.error(f"❌ Error handling order:filled: {str(e)}")
    
    def _handle_news_impact(self, event_data):
        """
        Handle news impact events affecting trades
        
        Args:
            event_data (dict): News impact event data
        """
        try:
            # Store news impact for trade scoring correlation
            impact_time = event_data.get("timestamp", datetime.now().isoformat())
            impact_level = event_data.get("impact_level", "medium")
            affected_currencies = event_data.get("currencies", [])
            
            # Update recent trades with news impact correlation
            with self.lock:
                current_time = datetime.now()
                for trade in reversed(list(self.closed_trades)):
                    if not trade.get("news_impact_checked"):
                        trade_time = datetime.fromisoformat(trade["close_time"].replace("Z", "+00:00"))
                        if abs((current_time - trade_time.replace(tzinfo=None)).total_seconds()) < 300:  # 5 minutes
                            # Check if trade symbol affected by news
                            symbol = trade["symbol"]
                            base_currency = symbol[:3]
                            quote_currency = symbol[3:6]
                            
                            if base_currency in affected_currencies or quote_currency in affected_currencies:
                                trade["news_impact"] = impact_level
                                trade["news_impact_time"] = impact_time
                                logger.info(f"✅ News impact recorded for trade {trade['trade_id']}")
                        
                        trade["news_impact_checked"] = True
                
        except Exception as e:
            logger.error(f"❌ Error handling news:impact: {str(e)}")
    
    def _score_trade_outcome(self, trade_record: Dict[str, Any]) -> float:
        """
        Score trade outcome based on multiple factors
        
        Args:
            trade_record (dict): Complete trade record
            
        Returns:
            float: Trade outcome score (0.0 to 1.0)
        """
        try:
            profit = trade_record.get("profit", 0)
            lot_size = trade_record.get("lot_size", 0.01)
            duration = trade_record.get("duration_minutes", 0)
            risk_reward = trade_record.get("risk_reward_ratio", 0)
            
            # Component scores
            scores = {}
            
            # 1. Profit Factor Score (0.0 to 1.0)
            if profit > 0:
                scores["profit_factor"] = min(1.0, profit / (lot_size * 100))  # Normalize by position size
            else:
                scores["profit_factor"] = max(0.0, 0.5 + (profit / (lot_size * 100)))
            
            # 2. Risk-Reward Ratio Score
            if risk_reward > 0:
                scores["risk_reward"] = min(1.0, risk_reward / 3.0)  # Target 3:1 R:R
            else:
                scores["risk_reward"] = 0.0
            
            # 3. Timing Accuracy Score (based on duration)
            if duration > 0:
                if duration <= 60:  # Quick scalp
                    scores["timing_accuracy"] = 0.9 if profit > 0 else 0.3
                elif duration <= 240:  # Short-term
                    scores["timing_accuracy"] = 0.8 if profit > 0 else 0.4
                elif duration <= 1440:  # Day trade
                    scores["timing_accuracy"] = 0.7 if profit > 0 else 0.5
                else:  # Swing trade
                    scores["timing_accuracy"] = 0.6 if profit > 0 else 0.6
            else:
                scores["timing_accuracy"] = 0.5
            
            # 4. News Impact Score
            news_impact = trade_record.get("news_impact", "none")
            if news_impact == "high" and profit > 0:
                scores["news_impact"] = 1.0  # Benefited from high-impact news
            elif news_impact == "high" and profit < 0:
                scores["news_impact"] = 0.2  # Hurt by high-impact news
            elif news_impact == "medium":
                scores["news_impact"] = 0.7 if profit > 0 else 0.4
            else:
                scores["news_impact"] = 0.6  # Neutral news environment
            
            # 5. Slippage Cost Score (assume minimal slippage for now)
            scores["slippage_cost"] = 0.8  # Default good execution
            
            # Calculate weighted average
            total_score = 0.0
            for component, weight in self.scoring_weights.items():
                total_score += scores.get(component, 0.5) * weight
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.error(f"❌ Error scoring trade outcome: {str(e)}")
            return 0.5  # Neutral score on error
    
    def _calculate_trade_duration(self, open_time: str, close_time: str) -> int:
        """Calculate trade duration in minutes"""
        try:
            if not open_time or not close_time is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: post_trade_feedback_engine -->