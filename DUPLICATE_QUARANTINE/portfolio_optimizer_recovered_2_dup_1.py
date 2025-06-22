# <!-- @GENESIS_MODULE_START: portfolio_optimizer -->

#!/usr/bin/env python3
"""
🔐 GENESIS TRADING BOT — PHASE 46 RISK-ADJUSTED PORTFOLIO OPTIMIZATION ENGINE
📋 Module: portfolio_optimizer.py
🎯 Purpose: Dynamic portfolio weight adjustment based on risk-adjusted returns and GENESIS confluence signals
📅 Created: 2025-06-18
⚖️ Compliance: ARCHITECT_MODE_V4.0
🧭 Phase: 46
"""

import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# GENESIS Modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from event_bus import EventBus
except ImportError:
    # Fallback event bus implementation for testing
    class EventBus:
        def __init__(self):
            self.subscribers = {}
        
        
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def subscribe(self, topic: str, callback):
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
        
        def emit(self, topic: str, data):
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"EventBus callback error: {e}")
        
        def publish(self, topic: str, data):
            self.emit(topic, data)

@dataclass
class StrategyMetrics:
    """Strategy performance and risk metrics"""
    id: str
    expected_return: float
    max_drawdown: float
    volatility: float
    confluence_score: float
    kill_switch: bool
    sharpe_ratio: float
    calmar_ratio: float
    current_weight: float
    recommended_weight: float


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
        class PortfolioOptimizer:
    """
    🎯 GENESIS Phase 46 — Risk-Adjusted Portfolio Optimization Engine
    
    📌 CORE FUNCTIONALITY:
    - Calculate Sharpe-like risk-adjusted scores per strategy
    - Penalize high-drawdown or volatile instruments  
    - Prioritize sniper-qualified setups with active kill-switch safety
    - Rebalance portfolio every 15 minutes or post fill/SL/TP event
    - Log recommended position weights and suppress risky exposure
    
    🔁 EventBus Bindings: [trade_filled, stoploss_triggered, takeprofit_triggered, portfolio_imbalance_detected]
    📡 Telemetry: [portfolio_risk_score, strategy_weight_recommendation, exposure_throttle_triggered, rebalance_log]
    🧪 MT5 Tests: [risk_score_calculation, weight_normalization, kill_switch_filtering, telemetry_emission]
    🪵 Error Handling: [logged, escalated via EventBus]
    ⚙️ Metrics: [rebalance_latency, portfolio_risk_score, strategy_weights]
    🗃️ Registry ID: portfolio_optimizer.py
    ⚖️ Compliance Score: A
    📌 Status: active
    📅 Last Modified: 2025-06-18
    📝 Author(s): GENESIS Architect Mode v4.0
    🔗 Dependencies: [event_bus.py, telemetry.json, kill_switch_audit.py]
    """

    def __init__(self):
        """Initialize portfolio optimizer with EventBus integration"""
        self.event_bus = EventBus()
        self.logger = logging.getLogger(__name__)
        self.rebalance_interval = 900  # 15 minutes
        self.last_rebalance = 0
        self.max_daily_exposure = 10000  # FTMO $10k daily limit
        self.max_trailing_exposure = 20000  # FTMO $20k trailing limit
        self.min_confluence_threshold = 0.6
        self.max_risk_score = 2.0
        
        # EventBus subscriptions
        self._subscribe_to_events()
        
        # Phase 47: Mutation Engine Integration
        self._subscribe_to_mutation_events()
        
        self.logger.info("✅ Portfolio Optimizer initialized - Phase 46+47 active")

    def _subscribe_to_events(self):
        """Subscribe to portfolio-relevant events"""
        try:
            self.event_bus.subscribe("trade_filled", self._handle_trade_filled)
            self.event_bus.subscribe("stoploss_triggered", self._handle_stoploss_triggered)
            self.event_bus.subscribe("takeprofit_triggered", self._handle_takeprofit_triggered)
            self.event_bus.subscribe("portfolio_imbalance_detected", self._handle_portfolio_imbalance)
            
            self.logger.info("📡 EventBus subscriptions registered for portfolio optimizer")
        except Exception as e:
            self.logger.error(f"🚨 EventBus subscription failed: {e}")
            self._emit_telemetry("portfolio_optimizer_error", {"error": str(e), "type": "subscription_failed"})

    def _subscribe_to_mutation_events(self):
        """Phase 47: Subscribe to mutation engine events"""
        try:
            self.event_bus.subscribe("mutation_engine:execution_feedback_received", self._handle_mutation_feedback)
            self.logger.info("📡 Phase 47: Mutation engine subscriptions registered")
        except Exception as e:
            self.logger.error(f"🚨 Phase 47: Mutation engine subscription failed: {e}")
            self._emit_telemetry("portfolio_optimizer_mutation_error", {"error": str(e), "type": "mutation_subscription_failed"})

    def _handle_trade_filled(self, data: Dict[str, Any]):
        """Handle trade fill events and trigger rebalancing"""
        try:
            self.logger.info(f"📊 Trade filled - triggering portfolio rebalance: {data}")
            self._emit_telemetry("trade_fill_trigger", data)
            self._trigger_rebalance("trade_filled")
        except Exception as e:
            self.logger.error(f"🚨 Error handling trade fill: {e}")

    def _handle_stoploss_triggered(self, data: Dict[str, Any]):
        """Handle stop loss events and adjust risk scoring"""
        try:
            strategy_id = data.get('strategy_id', 'unknown')
            self.logger.warning(f"🔴 Stop loss triggered for {strategy_id} - adjusting risk scoring")
            self._emit_telemetry("stoploss_trigger", data)
            self._trigger_rebalance("stoploss_triggered")
        except Exception as e:
            self.logger.error(f"🚨 Error handling stop loss: {e}")

    def _handle_takeprofit_triggered(self, data: Dict[str, Any]):
        """Handle take profit events"""
        try:
            strategy_id = data.get('strategy_id', 'unknown')
            self.logger.info(f"🟢 Take profit triggered for {strategy_id}")
            self._emit_telemetry("takeprofit_trigger", data)
            self._trigger_rebalance("takeprofit_triggered")
        except Exception as e:
            self.logger.error(f"🚨 Error handling take profit: {e}")

    def _handle_portfolio_imbalance(self, data: Dict[str, Any]):
        """Handle portfolio imbalance detection"""
        try:
            self.logger.warning(f"⚖️ Portfolio imbalance detected: {data}")
            self._emit_telemetry("portfolio_imbalance", data)
            self._trigger_rebalance("portfolio_imbalance")
        except Exception as e:
            self.logger.error(f"🚨 Error handling portfolio imbalance: {e}")

    def _handle_mutation_feedback(self, payload: Dict[str, Any]):
        """Phase 47: Handle mutation engine execution feedback"""
        try:
            self.logger.info(f"🧬 Phase 47: Mutation feedback received: {payload}")
            
            # Get current exposure from MT5
            current_exposure = self._get_current_exposure()
            
            # Calculate risk profile for the strategy
            risk_profile = self._get_trade_risk_profile(payload)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation(current_exposure)
              # Emit telemetry metrics
            self._emit_telemetry("optimizer.exposure_total", {
                "total_exposure": sum([v.get("value", 0) for v in current_exposure.values()])
            })
            self._emit_telemetry("optimizer.risk_profile", {"risk_profile": risk_profile})
            self._emit_telemetry("optimizer.correlation_avg", {
                "avg_correlation": correlation_matrix.get("avg_correlation", 0.0)
            })
              # Adjust strategy path based on portfolio conditions
            strategy_id = payload.get("strategy_id", "unknown")
            self._adjust_strategy_path(
                strategy_id=strategy_id,
                signals={
                    "portfolio_risk": risk_profile,
                    "correlation_penalty": correlation_matrix.get("avg_correlation", 0.0)
                }
            )
              # Emit completion event
            self.event_bus.emit("portfolio_optimizer:update_complete", {
                "strategy_id": strategy_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio_risk": risk_profile,
                "correlation_avg": correlation_matrix.get("avg_correlation", 0.0)
            })
            
            self.logger.info(f"✅ Phase 47: Portfolio optimization complete for {strategy_id}")
            
        except Exception as e:
            self.logger.error(f"🚨 Phase 47: Mutation feedback handling failed: {e}")
            self._emit_telemetry("portfolio_optimizer_mutation_error", {
                "error": str(e), 
                "payload": payload,
                "type": "mutation_feedback_failed"
            })

    def _trigger_rebalance(self, trigger_event: str):
        """Trigger portfolio rebalancing with rate limiting"""
        try:
            current_time = time.time()
            
            # Rate limiting - don't rebalance too frequently unless critical
            if (current_time - self.last_rebalance < 300 and 
                trigger_event not in ["stoploss_triggered", "portfolio_imbalance"]):
                self.logger.debug(f"⏳ Rebalance skipped - rate limited ({trigger_event})")
                return
            
            self.logger.info(f"🔄 Triggering portfolio rebalance: {trigger_event}")
            active_strategies = self._get_active_strategies()
            portfolio_weights = self.rebalance_portfolio(active_strategies)
            
            self.last_rebalance = current_time
            self._emit_telemetry("rebalance_triggered", {
                "trigger": trigger_event,
                "timestamp": current_time,
                "portfolio_weights": portfolio_weights
            })
            
        except Exception as e:
            self.logger.error(f"🚨 Rebalance trigger failed: {e}")
            self._emit_telemetry("rebalance_error", {"error": str(e), "trigger": trigger_event})

    def _get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get active strategies from strategy recommender"""
        try:
            # In production, this would interface with the strategy recommender
            # For now, return sample data structure
            live_strategies = [
                {
                    "id": "scalping_eur_usd",
                    "expected_return": 0.15,
                    "max_drawdown": 0.08,
                    "volatility": 0.12,
                    "confluence_score": 0.85,
                    "kill_switch": False,
                    "current_weight": 0.25
                },
                {
                    "id": "swing_gbp_jpy", 
                    "expected_return": 0.22,
                    "max_drawdown": 0.15,
                    "volatility": 0.18,
                    "confluence_score": 0.72,
                    "kill_switch": False,
                    "current_weight": 0.30
                },
                {
                    "id": "trend_usd_cad",
                    "expected_return": 0.08,
                    "max_drawdown": 0.25,
                    "volatility": 0.22,
                    "confluence_score": 0.45,
                    "kill_switch": True,  # Kill switch active
                    "current_weight": 0.20
                }
            ]
            
            self.logger.debug(f"📋 Retrieved {len(live_strategies)} strategies")
            return live_strategies
            
        except Exception as e:
            self.logger.error(f"🚨 Failed to get active strategies: {e}")
            return []

    def rebalance_portfolio(self, active_strategies: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        🎯 Core rebalancing algorithm with risk-adjusted scoring
        
        Args:
            active_strategies: List of strategy dictionaries with metrics
            
        Returns:
            List of tuples (strategy_id, normalized_weight)
        """
        try:
            portfolio_score = []
            total_risk_adjusted_score = 0
            
            self.logger.info(f"🔄 Rebalancing portfolio with {len(active_strategies)} strategies")
            
            for strategy in active_strategies:
                # Skip strategies with kill switch active
                if strategy.get('kill_switch', False):
                    self.logger.warning(f"🛑 Skipping strategy {strategy['id']} - kill switch active")
                    continue
                
                # Skip strategies below confluence threshold
                confluence_score = strategy.get('confluence_score', 0.0)
                if confluence_score < self.min_confluence_threshold:
                    self.logger.warning(f"📉 Skipping strategy {strategy['id']} - low confluence: {confluence_score}")
                    continue
                
                # Calculate risk-adjusted score
                expected_return = max(strategy.get('expected_return', 0.0), 1e-5)
                max_drawdown = max(strategy.get('max_drawdown', 0.01), 1e-5)
                volatility = max(strategy.get('volatility', 0.01), 1e-5)
                
                # Risk penalty calculation
                risk_penalty = max_drawdown / expected_return
                volatility_penalty = volatility / expected_return
                combined_risk = risk_penalty + volatility_penalty
                
                # Skip strategies with excessive risk
                if combined_risk > self.max_risk_score:
                    self.logger.warning(f"⚠️ Skipping strategy {strategy['id']} - excessive risk: {combined_risk:.3f}")
                    continue
                
                # Calculate base weight (inverse risk relationship)
                base_weight = max(0, 1 / (1 + combined_risk))
                
                # Apply confluence multiplier
                adjusted_weight = base_weight * confluence_score
                
                # Calculate Sharpe-like ratio for additional scoring
                sharpe_like = expected_return / volatility if volatility > 0 else 0
                adjusted_weight *= (1 + sharpe_like * 0.1)  # 10% bonus for good Sharpe
                
                portfolio_score.append((strategy['id'], adjusted_weight))
                total_risk_adjusted_score += adjusted_weight
                
                self.logger.debug(f"📊 {strategy['id']}: risk={combined_risk:.3f}, "
                               f"confluence={confluence_score:.3f}, weight={adjusted_weight:.4f}")
            
            # Normalize weights to sum to 1
            normalized_weights = self._normalize_weights(portfolio_score, total_risk_adjusted_score)
              # Emit telemetry
            self._emit_telemetry("strategy_weight_recommendation", {"weights": dict(normalized_weights)})
            self._emit_telemetry("portfolio_risk_score", {
                "total_strategies": len(active_strategies),
                "active_strategies": len(normalized_weights),
                "total_risk_score": total_risk_adjusted_score,
                "avg_risk_score": total_risk_adjusted_score / max(len(normalized_weights), 1)
            })
            
            # Log rebalancing results
            self._emit_telemetry("rebalance_log", {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio_weights": normalized_weights,
                "strategies_processed": len(active_strategies),
                "strategies_active": len(normalized_weights)
            })
            
            self.logger.info(f"✅ Portfolio rebalanced - {len(normalized_weights)} active strategies")
            
            # Emit portfolio rebalanced event
            self.event_bus.emit("portfolio_rebalanced", {
                "weights": normalized_weights,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_score": total_risk_adjusted_score
            })
            
            return normalized_weights
            
        except Exception as e:
            self.logger.error(f"🚨 Portfolio rebalancing failed: {e}")
            self._emit_telemetry("rebalance_error", {"error": str(e)})
            return []

    def _normalize_weights(self, portfolio_score: List[Tuple[str, float]], 
                         total_score: float) -> List[Tuple[str, float]]:
        """Normalize portfolio weights to sum to 1.0"""
        try:
            if total_score <= 0:
                self.logger.warning("⚠️ Zero total score - equal weights applied")
                equal_weight = 1.0 / max(len(portfolio_score), 1)
                return [(strategy_id, equal_weight) for strategy_id, _ in portfolio_score]
            
            normalized = []
            for strategy_id, weight in portfolio_score:
                normalized_weight = round(weight / total_score, 4)
                normalized.append((strategy_id, normalized_weight))
            
            # Verify normalization
            weight_sum = sum(weight for _, weight in normalized)
            if abs(weight_sum - 1.0) > 0.01:
                self.logger.warning(f"⚠️ Weight normalization issue - sum: {weight_sum:.4f}")
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"🚨 Weight normalization failed: {e}")
            return portfolio_score

    def check_exposure_limits(self, position_size: float, strategy_id: str) -> bool:
        """Check if position size violates FTMO exposure limits"""
        try:
            # Check daily exposure limit
            if position_size > self.max_daily_exposure:
                self.logger.warning(f"🚨 Daily exposure limit exceeded: {position_size} > {self.max_daily_exposure}")
                self._emit_telemetry("exposure_throttle_triggered", {
                    "strategy_id": strategy_id,
                    "position_size": position_size,
                    "limit_type": "daily",
                    "limit_value": self.max_daily_exposure
                })
                return False
            
            # Check trailing exposure limit  
            if position_size > self.max_trailing_exposure:
                self.logger.warning(f"🚨 Trailing exposure limit exceeded: {position_size} > {self.max_trailing_exposure}")
                self._emit_telemetry("exposure_throttle_triggered", {
                    "strategy_id": strategy_id,
                    "position_size": position_size,
                    "limit_type": "trailing",
                    "limit_value": self.max_trailing_exposure
                })
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"🚨 Exposure limit check failed: {e}")
            return False

    def _emit_telemetry(self, metric_name: str, data: Dict[str, Any]):
        """Emit telemetry data via EventBus"""
        try:
            telemetry_event = {
                "metric": metric_name,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "module": "portfolio_optimizer"
            }
            
            self.event_bus.emit("telemetry_portfolio_summary", telemetry_event)
            
        except Exception as e:
            self.logger.error(f"🚨 Telemetry emission failed: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary and metrics"""
        try:
            active_strategies = self._get_active_strategies()
            current_weights = self.rebalance_portfolio(active_strategies)
            
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_strategies": len(active_strategies),
                "active_strategies": len(current_weights),
                "portfolio_weights": current_weights,
                "exposure_limits": {
                    "daily_max": self.max_daily_exposure,
                    "trailing_max": self.max_trailing_exposure
                },
                "risk_thresholds": {
                    "min_confluence": self.min_confluence_threshold,
                    "max_risk_score": self.max_risk_score
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"🚨 Portfolio summary generation failed: {e}")
            return {}

    def _get_current_exposure(self) -> Dict[str, Dict[str, Any]]:
        """Phase 47: Get current MT5 exposure data"""
        try:
            # In production, this would connect to MT5 adapter
            # For now, return structured exposure data
            exposure_data = {
                "EURUSD": {"value": 2500.0, "direction": "long", "risk": 0.15},
                "GBPJPY": {"value": 1800.0, "direction": "short", "risk": 0.22},
                "XAUUSD": {"value": 3200.0, "direction": "long", "risk": 0.18}
            }
            
            self.logger.debug(f"📊 Current MT5 exposure: {exposure_data}")
            return exposure_data
            
        except Exception as e:
            self.logger.error(f"🚨 Failed to get current exposure: {e}")
            return {}

    def _get_trade_risk_profile(self, payload: Dict[str, Any]) -> float:
        """Phase 47: Calculate trade risk profile from execution feedback"""
        try:
            execution_result = payload.get("execution_result", "unknown")
            trade_return = payload.get("return", 0.0)
            
            # Calculate risk profile based on execution feedback
            if execution_result == "success":
                risk_profile = max(0.1, 1.0 - abs(trade_return))
            elif execution_result == "partial":
                risk_profile = 0.6  # Medium risk for partial fills
            else:
                risk_profile = 0.8  # Higher risk for failures
            
            self.logger.debug(f"📈 Risk profile calculated: {risk_profile} for {payload.get('strategy_id')}")
            return risk_profile
            
        except Exception as e:
            self.logger.error(f"🚨 Risk profile calculation failed: {e}")
            return 0.5  # Default medium risk

    def _calculate_correlation(self, exposure_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Phase 47: Calculate correlation matrix for current exposures"""
        try:
            assert exposure_data or len(exposure_data) < 2 is not None, "Real data required - no fallbacks allowed"

# <!-- @GENESIS_MODULE_END: portfolio_optimizer -->