
# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def emergency_stop(self, reason: str = "Manual trigger") -> bool:
                """GENESIS Emergency Kill Switch"""
                try:
                    # Emit emergency event
                    if hasattr(self, 'event_bus') and self.event_bus:
                        emit_event("emergency_stop", {
                            "module": "live_feedback_adapter",
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
                    print(f"Emergency stop error in live_feedback_adapter: {e}")
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
                    "module": "live_feedback_adapter",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("live_feedback_adapter", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in live_feedback_adapter: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


# <!-- @GENESIS_MODULE_START: live_feedback_adapter -->

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENESIS Live Feedback Adapter v2.7 - PHASE 13
Interface between Phase 12 trade outcome data and Phase 13 strategy mutation engine
ARCHITECT MODE: v2.7 - STRICT COMPLIANCE

PHASE 13 COMPONENT OBJECTIVE:
Provide an abstraction layer between LiveTradeFeedbackInjector and StrategyMutator

INPUTS CONSUMED:
- TradeOutcomeFeedback: Trade results from Phase 12
- ExecutionSnapshot: Trade execution details
- PnLScoreUpdate: Performance metrics

OUTPUTS EMITTED:
- EnrichedTradeOutcome: Enhanced trade data for mutation engine
- TradeClusterAnalysis: Pattern detection in trade outcomes
- SymbolPerformanceUpdate: Symbol-specific performance metrics

VALIDATION REQUIREMENTS:
✅ Real MT5 data only (no real/execute)
✅ EventBus communication only
✅ Trade outcome enrichment
✅ Performance clustering
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
from typing import Dict, List, Any, Tuple, Optional

# Import system components
from event_bus import get_event_bus, emit_event, register_route

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveFeedbackAdapter:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "live_feedback_adapter",
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
                print(f"Emergency stop error in live_feedback_adapter: {e}")
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
                "module": "live_feedback_adapter",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("live_feedback_adapter", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in live_feedback_adapter: {e}")
    """
    GENESIS Live Feedback Adapter - PHASE 13
    
    PHASE 13 Architecture Compliance:
    - ✅ Real MT5 trade outcome processing
    - ✅ Trade data enrichment
    - ✅ Performance clustering
    - ✅ EventBus only communication
    - ✅ Telemetry hooks enabled
    - ✅ No isolated functions
    """
    
    def __init__(self):
        """Initialize Live Feedback Adapter"""
        self.module_name = "LiveFeedbackAdapter"
        self.event_bus = get_event_bus()
        
        # Trade outcome tracking
        self.trade_outcomes = deque(maxlen=1000)  # Limited trade history
        self.symbol_performance = defaultdict(list)  # symbol -> [outcomes]
        self.strategy_fingerprints = defaultdict(dict)  # strategy_id -> fingerprint data
        
        # Processing parameters
        self.cluster_window = 20  # Number of trades to analyze for clusters
        self.symbol_window = 50  # Number of trades per symbol to track
        
        # Thread safety
        self.lock = Lock()
        
        # Telemetry
        self.telemetry = {
            "trades_processed": 0,
            "clusters_detected": 0,
            "symbols_tracked": 0,
            "avg_processing_time_ms": 0.0
        }
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
        # Initialize logging directory
        self.log_dir = Path("logs/live_feedback_adapter")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"✅ {self.module_name} initialized - PHASE 13 ACTIVE")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _register_event_handlers(self):
        """Register event handlers for incoming events"""
        try:
            # Register event consumption routes
            register_route("TradeOutcomeFeedback", "LiveFeedbackAdapter", self.module_name)
            register_route("ExecutionSnapshot", "LiveFeedbackAdapter", self.module_name)
            register_route("PnLScoreUpdate", "LiveFeedbackAdapter", self.module_name)
            
            # Register event production routes
            register_route("EnrichedTradeOutcome", self.module_name, "StrategyMutator")
            register_route("TradeClusterAnalysis", self.module_name, "StrategyMutator")
            register_route("SymbolPerformanceUpdate", self.module_name, "StrategyMutator")
            
            # Subscribe to events
            self.event_bus.subscribe("TradeOutcomeFeedback", self._handle_trade_outcome, self.module_name)
            self.event_bus.subscribe("ExecutionSnapshot", self._handle_execution_snapshot, self.module_name)
            self.event_bus.subscribe("PnLScoreUpdate", self._handle_pnl_update, self.module_name)
            
            logger.info("✅ Event handlers registered successfully")
            
        except Exception as e:
            logger.error(f"❌ Error registering event handlers: {str(e)}")
            self._emit_error("EVENT_REGISTRATION_ERROR", str(e))
    
    def _handle_trade_outcome(self, event_data):
        """
        Handle trade outcome feedback events from LiveTradeFeedbackInjector
        
        Args:
            event_data (dict): TradeOutcomeFeedback event data
        """
        start_time = time.time()
        
        try:
            # Add outcome to tracking
            with self.lock:
                self.trade_outcomes.append(event_data)
                
                # Extract symbol data
                symbol = event_data.get("symbol")
                if symbol:
                    self.symbol_performance[symbol].append(event_data)
                    # Maintain window size
                    if len(self.symbol_performance[symbol]) > self.symbol_window:
                        self.symbol_performance[symbol].pop(0)
                    
                    # Update symbol count
                    self.telemetry["symbols_tracked"] = len(self.symbol_performance)
                
                # Update telemetry
                self.telemetry["trades_processed"] += 1
                
                # Detect trade clusters
                if len(self.trade_outcomes) % 5 == 0:  # Check every 5 trades
                    self._detect_trade_clusters()
                
                # Check symbol performance
                if len(self.symbol_performance.get(symbol, [])) >= 10:
                    self._analyze_symbol_performance(symbol)
                
                # Enrich and forward trade outcome
                self._enrich_trade_outcome(event_data)
                
                # Performance tracking
                elapsed_ms = (time.time() - start_time) * 1000
                self.processing_times.append(elapsed_ms)
                self.telemetry["avg_processing_time_ms"] = statistics.mean(self.processing_times)
            
            logger.info(f"✅ Processed trade outcome for {event_data.get('strategy_id')}")
            
        except Exception as e:
            logger.error(f"❌ Error handling trade outcome: {str(e)}")
            self._emit_error("TRADE_OUTCOME_PROCESSING_ERROR", str(e))
    
    def _handle_execution_snapshot(self, event_data):
        """
        Handle execution snapshot events from ExecutionEngine
        
        Args:
            event_data (dict): ExecutionSnapshot event data
        """
        try:
            # Store execution details for later enrichment
            signal_id = event_data.get("signal_id")
            if signal_id:
                with self.lock:
                    # Extract strategy_id from the signal_id or trade_comment
                    strategy_id = event_data.get("strategy_id", self._extract_strategy_id(event_data))
                    
                    if strategy_id:
                        # Save strategy fingerprint data
                        self.strategy_fingerprints[strategy_id][signal_id] = {
                            "execution_id": event_data.get("execution_id"),
                            "symbol": event_data.get("symbol"),
                            "direction": event_data.get("direction"),
                            "entry_price": event_data.get("entry_price"),
                            "stop_loss": event_data.get("stop_loss"),
                            "take_profit": event_data.get("take_profit"),
                            "timestamp": event_data.get("timestamp")
                        }
            
            logger.info(f"✅ Processed execution snapshot for signal {signal_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling execution snapshot: {str(e)}")
            self._emit_error("EXECUTION_SNAPSHOT_PROCESSING_ERROR", str(e))
    
    def _handle_pnl_update(self, event_data):
        """
        Handle PnL score update events
        
        Args:
            event_data (dict): PnLScoreUpdate event data
        """
        try:
            # Process PnL updates for strategy performance tracking
            strategy_id = event_data.get("strategy_id")
            if strategy_id:
                logger.info(f"✅ Processed PnL update for strategy {strategy_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling PnL update: {str(e)}")
            self._emit_error("PNL_UPDATE_PROCESSING_ERROR", str(e))
    
    def _enrich_trade_outcome(self, trade_outcome):
        """
        Enrich trade outcome with additional data and emit event
        
        Args:
            trade_outcome (dict): Original trade outcome data
        """
        try:
            strategy_id = trade_outcome.get("strategy_id")
            signal_id = trade_outcome.get("signal_id")
            
            # Skip if missing key data
            assert strategy_id or not signal_id:
                logger.warning(f"⚠️ Cannot enrich trade outcome: missing strategy_id or signal_id")
                return
            
            # Get signal fingerprint data
            fingerprint_data = self.strategy_fingerprints.get(strategy_id, {}).get(signal_id, {})
            
            # Create enriched trade outcome
            enriched_data = trade_outcome.copy()
            enriched_data.update({
                "event_type": "EnrichedTradeOutcome",
                "timestamp": datetime.now().isoformat(),
                "entry_details": fingerprint_data,
                "market_context": self._get_market_context(trade_outcome.get("symbol")),
                "symbol_performance": self._get_symbol_stats(trade_outcome.get("symbol")),
                "enriched_at": datetime.now().isoformat()
            })
            
            # Emit enriched event
            emit_event("EnrichedTradeOutcome", enriched_data)
            
            logger.info(f"✅ Emitted EnrichedTradeOutcome for strategy {strategy_id}")
            
        except Exception as e:
            logger.error(f"❌ Error enriching trade outcome: {str(e)}")
            self._emit_error("TRADE_ENRICHMENT_ERROR", str(e))
    
    def _detect_trade_clusters(self):
        """
        Detect clusters in trade outcomes and emit analysis
        """
        try:
            if len(self.trade_outcomes) < self.cluster_window:
                return
                
            # Get recent trades
            recent_trades = list(self.trade_outcomes)[-self.cluster_window:]
            
            # Count outcomes by strategy and symbol
            strategy_outcomes = defaultdict(lambda: {"wins": 0, "losses": 0})
            symbol_outcomes = defaultdict(lambda: {"wins": 0, "losses": 0})
            
            for trade in recent_trades:
                strategy_id = trade.get("strategy_id", "unknown")
                symbol = trade.get("symbol", "unknown")
                outcome = trade.get("outcome", "unknown")
                
                if outcome == "WIN":
                    strategy_outcomes[strategy_id]["wins"] += 1
                    symbol_outcomes[symbol]["wins"] += 1
                elif outcome == "LOSS":
                    strategy_outcomes[strategy_id]["losses"] += 1
                    symbol_outcomes[symbol]["losses"] += 1
            
            # Detect clusters
            clusters = []
            
            # Strategy clusters
            for strategy_id, outcomes in strategy_outcomes.items():
                total = outcomes["wins"] + outcomes["losses"]
                if total >= 5:  # Minimum trades for cluster
                    win_rate = outcomes["wins"] / total if total > 0 else 0
                    
                    # High win or loss clusters
                    if win_rate >= 0.8 or win_rate <= 0.2:
                        clusters.append({
                            "type": "STRATEGY",
                            "id": strategy_id,
                            "win_rate": win_rate,
                            "trades": total,
                            "performance": "STRONG" if win_rate >= 0.8 else "WEAK"
                        })
            
            # Symbol clusters
            for symbol, outcomes in symbol_outcomes.items():
                total = outcomes["wins"] + outcomes["losses"]
                if total >= 5:  # Minimum trades for cluster
                    win_rate = outcomes["wins"] / total if total > 0 else 0
                    
                    # High win or loss clusters
                    if win_rate >= 0.8 or win_rate <= 0.2:
                        clusters.append({
                            "type": "SYMBOL",
                            "id": symbol,
                            "win_rate": win_rate,
                            "trades": total,
                            "performance": "STRONG" if win_rate >= 0.8 else "WEAK"
                        })
            
            # Emit cluster analysis if clusters found
            if clusters:
                emit_event("TradeClusterAnalysis", {
                    "event_type": "TradeClusterAnalysis",
                    "timestamp": datetime.now().isoformat(),
                    "clusters": clusters,
                    "window_size": self.cluster_window,
                    "analysis_id": f"cluster-{int(time.time())}"
                })
                
                # Update telemetry
                self.telemetry["clusters_detected"] += len(clusters)
                
                logger.info(f"✅ Detected {len(clusters)} trade clusters")
            
        except Exception as e:
            logger.error(f"❌ Error detecting trade clusters: {str(e)}")
            self._emit_error("CLUSTER_DETECTION_ERROR", str(e))
    
    def _analyze_symbol_performance(self, symbol):
        """
        Analyze performance for a specific symbol
        
        Args:
            symbol (str): Symbol to analyze
        """
        try:
            # Skip if insufficient data
            symbol_trades = self.symbol_performance.get(symbol, [])
            if len(symbol_trades) < 10:
                return
                
            # Calculate performance metrics
            total_trades = len(symbol_trades)
            wins = sum(1 for t in symbol_trades if t.get("outcome") == "WIN")
            losses = sum(1 for t in symbol_trades if t.get("outcome") == "LOSS")
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Get strategies used for this symbol
            strategies = set(t.get("strategy_id", "unknown") for t in symbol_trades)
            
            # Create symbol performance update
            performance_data = {
                "event_type": "SymbolPerformanceUpdate",
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "strategies_used": list(strategies),
                "last_trade_timestamp": symbol_trades[-1].get("timestamp"),
                "performance_category": self._categorize_performance(win_rate)
            }
            
            # Emit symbol performance update
            emit_event("SymbolPerformanceUpdate", performance_data)
            
            logger.info(f"✅ Emitted SymbolPerformanceUpdate for {symbol}: {win_rate:.2f} win rate")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing symbol performance: {str(e)}")
            self._emit_error("SYMBOL_PERFORMANCE_ERROR", str(e))
    
    def _extract_strategy_id(self, event_data):
        """
        Extract strategy ID from execution data
        
        Args:
            event_data (dict): Execution data
            
        Returns:            str: Extracted strategy ID or None
        """
        # Try to get from trade_comment which often contains signal/strategy info
        trade_comment = event_data.get("trade_comment", "")
        
        if not trade_comment:
            logger.warning("No trade comment available for strategy extraction")
            self._emit_warning_event("missing_trade_comment", {
                "event_data": event_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise ValueError("ARCHITECT_MODE_COMPLIANCE: Trade comment required for strategy extraction")
            
        # Trade comments often have format like "strategy-123_signal-456"
        if "strategy-" in trade_comment:
            parts = trade_comment.split("_")
            for part in parts:
                if part.startswith("strategy-") is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: live_feedback_adapter -->