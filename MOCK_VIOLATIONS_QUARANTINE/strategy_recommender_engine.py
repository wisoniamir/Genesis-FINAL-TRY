# <!-- @GENESIS_MODULE_START: strategy_recommender_engine -->

"""
GENESIS StrategyRecommenderEngine Module v2.7 - Signal Recommendation Engine
Advanced strategy evaluation and recommendation based on validated signals
NO real DATA - NO ISOLATED FUNCTIONS - STRICT COMPLIANCE

Dependencies: event_bus.py, json, datetime, os, collections, numpy, pandas
Consumes: ValidatedSniperSignal, MacroUpdateEvent, RiskExposureUpdate
Emits: StrategyRecommendation, ModuleTelemetry, ModuleError
Telemetry: ENABLED
Compliance: ENFORCED
Event-driven: All processing triggered by EventBus
"""

import os
import json
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock
from event_bus import emit_event, subscribe_to_event, register_route

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyRecommenderEngine:
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

            emit_telemetry("strategy_recommender_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("strategy_recommender_engine", "position_calculated", {
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
                        "module": "strategy_recommender_engine",
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
                print(f"Emergency stop error in strategy_recommender_engine: {e}")
                return False
    """
    GENESIS StrategyRecommenderEngine v2.7 - Strategy Recommendation System
    
    Takes ValidatedSniperSignal events, evaluates against macro conditions,
    risk exposure metrics, and strategy rules to produce actionable
    StrategyRecommendation events for execution.
    
    Architecture Compliance:
    - ✅ EventBus only communication
    - ✅ Real data processing (no real/dummy data)
    - ✅ Telemetry hooks enabled
    - ✅ No isolated functions
    - ✅ Registered in all system files
    - ✅ JSONL-based logging with actionable recommendations
    """
    
    def __init__(self):
        """Initialize StrategyRecommenderEngine with configuration and event subscriptions"""
        # Thread safety
        self.lock = Lock()
        
        # Set up log directories
        self.output_path = "logs/strategy_recommender/"
        os.makedirs(self.output_path, exist_ok=True)
        
        # Configuration for kill zones (trading windows)
        self.killzone_hours = [1, 8, 13, 20]  # London and NY open/close
        self.killzone_window_minutes = 60  # Width of trading window
        
        # Macro state tracking
        self.macro_state = {
            "last_updated": datetime.utcnow().isoformat(),
            "trend_bias": {},  # Per instrument trend bias (bullish/bearish/neutral)
            "volatility_regime": {},  # Per instrument volatility regime (high/normal/low)
            "risk_sentiment": "neutral",  # Overall market risk sentiment
            "economic_calendar": []  # Upcoming high-impact events
        }
        
        # Risk exposure tracking
        self.risk_state = {
            "last_updated": datetime.utcnow().isoformat(),
            "current_exposure": {},  # Per instrument exposure
            "portfolio_heat": 0.0,  # Percentage of max portfolio heat used
            "daily_drawdown": 0.0,  # Current daily drawdown percentage
            "max_positions": 5,  # Maximum concurrent positions
            "active_positions": 0,  # Current number of active positions
            "risk_limits_ok": True  # Whether we're within risk limits
        }
        
        # Performance metrics for strategy recommendations
        self.performance = {
            "signals_processed": 0,
            "recommendations_made": 0,
            "rejected": 0,
            "rejection_reasons": defaultdict(int)
        }
        
        # Signal history tracking
        self.signal_history = defaultdict(list)
        
        # Subscribe to events via EventBus
        self._register_event_handlers()
        
        logger.info("✅ StrategyRecommenderEngine initialized - Ready for signal validation")
        self._emit_telemetry("initialization", {"status": "initialized"})
        
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _register_event_handlers(self):
        """Register all event handlers with the EventBus"""
        # Main signal input
        subscribe_to_event("ValidatedSniperSignal", self.on_validated_signal)
        
        # Risk and macro state updates
        subscribe_to_event("MacroUpdateEvent", self.on_macro_update)
        subscribe_to_event("RiskExposureUpdate", self.on_risk_update)
        
        # Phase 57-58 integration: ML and Pattern Learning
        subscribe_to_event("ModelVersionUpdate", self.on_model_version_update)
        subscribe_to_event("PatternRecommendation", self.on_pattern_recommendation)
        
        # Add routes to the registry
        self._register_event_routes()
        logger.info("✅ Registered event handlers including Phase 57-58 ML and Pattern Learning integration")
        
    def _register_event_routes(self):
        """Register all event routes for compliance tracking"""
        register_route("ValidatedSniperSignal", "SignalValidator", "StrategyRecommenderEngine")
        register_route("MacroUpdateEvent", "*", "StrategyRecommenderEngine")
        register_route("RiskExposureUpdate", "RiskEngine", "StrategyRecommenderEngine")
        register_route("StrategyRecommendation", "StrategyRecommenderEngine", "ExecutionEngine")
        register_route("ModuleTelemetry", "StrategyRecommenderEngine", "TelemetryCollector")
        register_route("ModuleError", "StrategyRecommenderEngine", "TelemetryCollector")
        
        # Phase 57-58 integration routes
        register_route("ModelVersionUpdate", "MLRetrainingLoop", "StrategyRecommenderEngine")
        register_route("PatternRecommendation", "PatternLearningEngine", "StrategyRecommenderEngine")
        register_route("PredictionAccuracy", "StrategyRecommenderEngine", "MLRetrainingLoop")
        
    def on_validated_signal(self, event_data):
        """
        Process validated sniper signals and determine if they should become recommendations
        
        ARCHITECTURE COMPLIANCE:
        - ✅ Real data only (no real/execute signals)
        - ✅ Event-driven via EventBus
        - ✅ Telemetry hooks
        """
        with self.lock:
            try:
                # Extract signal data
                signal = event_data.get("data", {})
                
                # Track for metrics
                self.performance["signals_processed"] += 1
                symbol = signal.get("symbol", "UNKNOWN")
                
                logger.info(f"Processing validated signal for {symbol}")
                
                # Store in signal history
                self.signal_history[symbol].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "signal_data": signal
                })
                
                # Maintain max 100 signals per symbol
                if len(self.signal_history[symbol]) > 100:
                    self.signal_history[symbol] = self.signal_history[symbol][-100:]
                
                # Apply recommendation filters
                result = self._evaluate_signal_for_recommendation(signal)
                
                if result["recommend"]:
                    self._emit_strategy_recommendation(result["recommendation"])
                else:
                    logger.info(f"Signal rejected: {result['rejection_reason']}")
                    self.performance["rejected"] += 1
                    self.performance["rejection_reasons"][result["rejection_reason"]] += 1
                    self._emit_telemetry("signal_rejected", {
                        "symbol": symbol,
                        "reason": result["rejection_reason"]
                    })
                    
            except Exception as e:
                logger.error(f"Error processing validated signal: {str(e)}")
                self._emit_error("signal_processing_error", str(e))
                
    def on_macro_update(self, event_data):
        """Process macro environment updates"""
        with self.lock:
            try:
                macro_data = event_data.get("data", {})
                self.macro_state["last_updated"] = datetime.utcnow().isoformat()
                
                # Update trend biases
                if "trend_bias" in macro_data:
                    self.macro_state["trend_bias"].update(macro_data["trend_bias"])
                
                # Update volatility regime
                if "volatility_regime" in macro_data:
                    self.macro_state["volatility_regime"].update(macro_data["volatility_regime"])
                
                # Update overall risk sentiment
                if "risk_sentiment" in macro_data:
                    self.macro_state["risk_sentiment"] = macro_data["risk_sentiment"]
                
                # Update economic calendar
                if "economic_calendar" in macro_data:
                    self.macro_state["economic_calendar"] = macro_data["economic_calendar"]
                    
                logger.info(f"Macro state updated: {self.macro_state['risk_sentiment']} risk sentiment")
                self._emit_telemetry("macro_state_updated", {"risk_sentiment": self.macro_state["risk_sentiment"]})
                
            except Exception as e:
                logger.error(f"Error processing macro update: {str(e)}")
                self._emit_error("macro_update_error", str(e))
                
    def on_risk_update(self, event_data):
        """Process risk exposure updates"""
        with self.lock:
            try:
                risk_data = event_data.get("data", {})
                self.risk_state["last_updated"] = datetime.utcnow().isoformat()
                
                # Update exposure data
                if "current_exposure" in risk_data:
                    self.risk_state["current_exposure"].update(risk_data["current_exposure"])
                
                # Update portfolio heat
                if "portfolio_heat" in risk_data:
                    self.risk_state["portfolio_heat"] = risk_data["portfolio_heat"]
                
                # Update drawdown
                if "daily_drawdown" in risk_data:
                    self.risk_state["daily_drawdown"] = risk_data["daily_drawdown"]
                
                # Update position counts
                if "active_positions" in risk_data:
                    self.risk_state["active_positions"] = risk_data["active_positions"]
                
                # Update risk limits status
                if "risk_limits_ok" in risk_data:
                    self.risk_state["risk_limits_ok"] = risk_data["risk_limits_ok"]
                    
                logger.info(f"Risk state updated: Heat {self.risk_state['portfolio_heat']:.2f}%, DD {self.risk_state['daily_drawdown']:.2f}%")
                self._emit_telemetry("risk_state_updated", {
                    "portfolio_heat": self.risk_state["portfolio_heat"],
                    "risk_limits_ok": self.risk_state["risk_limits_ok"]
                })
                
            except Exception as e:
                logger.error(f"Error processing risk update: {str(e)}")
                self._emit_error("risk_update_error", str(e))
                
    def on_model_version_update(self, event_data):
        """Handle ML model version updates from Phase 57"""
        try:
            logger.info(f"🔄 ML Model updated: {event_data.get('new_version', 'unknown')}")
            
            # Update internal model reference if applicable
            new_version = event_data.get("new_version", "")
            performance_scores = event_data.get("performance_scores", {})
            
            # Log model performance improvement
            self._emit_telemetry("ml_model_updated", {
                "new_version": new_version,
                "performance_scores": performance_scores,
                "timestamp": event_data.get("timestamp")
            })
            
        except Exception as e:
            logger.error(f"Error handling model version update: {e}")
            self._emit_error("model_update_error", str(e))
            
    def on_pattern_recommendation(self, event_data):
        """Handle pattern recommendations from Phase 58"""
        try:
            logger.info(f"📈 Received pattern recommendations: {len(event_data.get('top_patterns', []))}")
            
            # Update strategy weights based on pattern recommendations
            top_patterns = event_data.get("top_patterns", [])
            
            for pattern in top_patterns[:5]:  # Use top 5 patterns
                pattern_id = pattern.get("pattern_id", "")
                success_rate = pattern.get("success_rate", 0.0)
                category = pattern.get("category", "")
                
                # Boost confidence for strategies matching successful patterns
                if success_rate > 0.7:  # High success patterns
                    logger.info(f"🎯 Boosting confidence for {category} patterns: {pattern_id}")
                    
            self._emit_telemetry("pattern_recommendations_processed", {
                "patterns_received": len(top_patterns),
                "high_success_patterns": sum(1 for p in top_patterns if p.get("success_rate", 0) > 0.7)
            })
            
        except Exception as e:
            logger.error(f"Error handling pattern recommendations: {e}")
            self._emit_error("pattern_recommendation_error", str(e))
            
    def _evaluate_signal_for_recommendation(self, signal):
        """
        Evaluate signal against multiple criteria for recommendation
        
        Returns dict with:
        - recommend: boolean
        - recommendation: dict with recommendation data if approved
        - rejection_reason: string with reason if rejected
        """
        symbol = signal.get("symbol", "UNKNOWN")
        pattern = signal.get("pattern", "UNKNOWN")
        confidence = signal.get("confluence", 0)
        price = signal.get("entry_price", 0.0)
        
        result = {
            "recommend": False,
            "rejection_reason": "",
            "recommendation": {}
        }
        
        # Check if it's an OB pattern as required
        if pattern != "OBCompression":
            result["rejection_reason"] = "pattern_not_ob_compression"
            return result
        
        # Check confidence level (must be >= 7)
        if confidence < 7:
            result["rejection_reason"] = "confidence_too_low"
            return result
        
        # Check if we're within risk limits
        assert self.risk_state["risk_limits_ok"]:
            result["rejection_reason"] = "risk_limits_exceeded" 
            return result
        
        # Check if we're at max positions
        if self.risk_state["active_positions"] >= self.risk_state["max_positions"]:
            result["rejection_reason"] = "max_positions_reached"
            return result
        
        # Check if we're in a kill zone (trading window)
        if not self._is_in_killzone():
            result["rejection_reason"] = "outside_killzone_hours"
            return result
        
        # Check macro alignment
        macro_aligned = self._check_macro_alignment(symbol)
        
        # Build recommendation payload
        # Calculate entry zone (entry ± 0.2%)
        entry_price = float(price)
        entry_zone_low = entry_price * 0.998
        entry_zone_high = entry_price * 1.002
        
        # Determine direction based on pattern
        # In a real system this would be more sophisticated
        direction = "long"  # Default to long for OB compression
        
        # Calculate stop loss (2% below entry for longs)
        sl_price = entry_price * 0.98
        
        # Calculate take profit (R:R of at least 3:1)
        tp_price = entry_price * 1.06  # 6% target = 3:1 on a 2% stop
        rr_ratio = (tp_price - entry_price) / (entry_price - sl_price)
        
        # Create recommendation payload
        recommendation = {
            "symbol": symbol,
            "direction": direction,
            "confidence_score": confidence * 10,  # Scale 0-10 to 0-100
            "macro_alignment": macro_aligned,
            "pattern_source": pattern,
            "risk_approved": self.risk_state["risk_limits_ok"],
            "entry_zone": f"{entry_zone_low:.2f}–{entry_zone_high:.2f}",
            "sl": f"{sl_price:.2f}",
            "tp": f"{tp_price:.2f}",
            "rr_ratio": round(rr_ratio, 1),
            "validity": "intraday",
            "generated_by": "StrategyRecommenderEngine",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result["recommend"] = True
        result["recommendation"] = recommendation
        return result
        
    def _is_in_killzone(self):
        """Check if current time is within a designated killzone (trading window)"""
        current_hour = datetime.utcnow().hour
        
        # Check if we're within any of the killzones (trading windows)
        for hour in self.killzone_hours:
            start_hour = hour
            end_hour = (hour + self.killzone_window_minutes // 60) % 24
            
            # Handle hour wrap-around at midnight
            if start_hour <= end_hour:
                if start_hour <= current_hour < end_hour is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: strategy_recommender_engine -->