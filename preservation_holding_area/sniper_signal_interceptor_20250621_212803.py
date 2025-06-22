#!/usr/bin/env python3

# 🔗 GENESIS EventBus Integration - Auto-injected by Emergency Repair Engine
from datetime import datetime
import json

class SniperSignalInterceptorEventBusIntegration:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "sniper_signal_interceptor",
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
                print(f"Emergency stop error in sniper_signal_interceptor: {e}")
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
    """EventBus integration for sniper_signal_interceptor"""
    
    def __init__(self):
        self.module_id = "sniper_signal_interceptor"
        self.event_routes = []
        
    def emit_event(self, event_type, data):
        """Emit event to EventBus"""
        try:
            from event_bus import emit_event
            emit_event(event_type, data)
        except ImportError:
            print(f"🔗 EVENTBUS EMIT: {event_type}: {data}")
        
    def emit_telemetry(self, metric_name, value):
        """Emit telemetry data"""
        telemetry = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_id,
            "metric": metric_name,
            "value": value
        }
        try:
            from event_bus import emit_event
            emit_event("ModuleTelemetry", telemetry)
        except ImportError:
            print(f"📊 TELEMETRY: {telemetry}")

# Auto-instantiate EventBus integration
sniper_signal_interceptor_eventbus = SniperSignalInterceptorEventBusIntegration()

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║            🎯 GENESIS SNIPER SIGNAL INTERCEPTOR v3.0                         ║
║            PRECISION SIGNAL QUALITY FILTER & INTERCEPTOR                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🎯 OBJECTIVE:
Intercept and validate trading signals for quality and execution readiness
- Signal quality assessment
- Technical confluence validation
- Risk-reward ratio verification
- Market context analysis
- Execution timing optimization
- Performance tracking and telemetry
- EventBus integration for real-time updates
🎯 FEATURES:
- Multi-factor signal validation
- Context-aware filtering
- Adaptive risk management
- Real-time performance monitoring
- Telemetry and logging
- EventBus integration for modular architecture
- Calibration of thresholds based on performance data
- Alternative setup suggestions for intercepted signals
- Comprehensive performance statistics
🎯 INTERCEPTION CRITERIA:
1. Signal quality score < threshold
2. Poor risk-reward ratio
3. Low confluence score
4. Unfavorable market context
5. Timeframe conflict   
6. Excessive spread conditions
7. Weak setup strength
8. Overriding signals from higher authority modules
🎯 PERFORMANCE:
- Tracks total signals, intercepted, and passed
- Calculates interception rate
- Maintains recent signal cache for analysis
- Provides detailed performance statistics
🎯 TELEMETRY:
- All interceptions logged with quality scores and reasons
- EventBus events for real-time updates
- Performance reports generated periodically
🎯 USAGE:
- Integrate with trading systems via EventBus
- Subscribe to "SignalGenerated" events for interception
- Emit "SignalInterceptionResult" events for results
- Use "InterceptorPerformanceReport" for performance reviews
🎯 ARCHITECTURE:
- Modular design with EventBus for extensibility
- Compliant with ARCHITECT MODE v3.0 standards
- Supports real-time signal processing and filtering

🔗 EventBus Integration: Real-time signal filtering
📊 Telemetry: All interceptions logged
✅ ARCHITECT MODE v3.0 COMPLIANT
🎯 SNIPER SIGNAL INTERCEPTOR - High-Precision Trading Signal Filter
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics


# <!-- @GENESIS_MODULE_END: sniper_signal_interceptor -->


# <!-- @GENESIS_MODULE_START: sniper_signal_interceptor -->

class SignalQuality(Enum):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "sniper_signal_interceptor",
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
                print(f"Emergency stop error in sniper_signal_interceptor: {e}")
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
    """Signal quality levels"""
    EXCELLENT = "EXCELLENT"  # > 0.9
    GOOD = "GOOD"           # 0.7 - 0.9
    FAIR = "FAIR"           # 0.5 - 0.7
    POOR = "POOR"           # 0.3 - 0.5
    TERRIBLE = "TERRIBLE"   # < 0.3

class InterceptionReason(Enum):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "sniper_signal_interceptor",
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
                print(f"Emergency stop error in sniper_signal_interceptor: {e}")
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
    """Reasons for signal interception"""
    LOW_QUALITY_SCORE = "LOW_QUALITY_SCORE"
    POOR_RISK_REWARD = "POOR_RISK_REWARD"
    LOW_CONFLUENCE = "LOW_CONFLUENCE"
    UNFAVORABLE_CONTEXT = "UNFAVORABLE_CONTEXT"
    TIMEFRAME_CONFLICT = "TIMEFRAME_CONFLICT"
    HIGH_SPREAD = "HIGH_SPREAD"
    WEAK_SETUP = "WEAK_SETUP"
    OVERRIDING_SIGNAL = "OVERRIDING_SIGNAL"

@dataclass
class TradingSignal:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "sniper_signal_interceptor",
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
                print(f"Emergency stop error in sniper_signal_interceptor: {e}")
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
    """Trading signal structure"""
    signal_id: str
    symbol: str
    direction: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timeframe: str
    timestamp: datetime
    indicators: Dict[str, Any]
    context: Dict[str, Any]

@dataclass
class InterceptionResult:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "sniper_signal_interceptor",
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
                print(f"Emergency stop error in sniper_signal_interceptor: {e}")
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
    """Signal interception result"""
    signal_id: str
    intercepted: bool
    quality_score: float
    reasons: List[InterceptionReason]
    recommendations: List[str]
    alternative_setups: List[Dict]
    
class GenesisSniperSignalInterceptor:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "sniper_signal_interceptor",
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
                print(f"Emergency stop error in sniper_signal_interceptor: {e}")
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
    🎯 GENESIS Sniper Signal Interceptor
    
    High-precision signal filtering system that ensures only the highest
    quality trading opportunities pass through to execution
    - Multi-factor quality assessment
    - Market context validation
    - Risk management integration
    - Performance optimization
    """
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.is_active = True
        
        # Quality thresholds
        self.min_quality_score = 0.7
        self.min_risk_reward_ratio = 2.0
        self.min_confluence_score = 0.6
        self.max_spread_multiplier = 1.5
        
        # Market context weights
        self.context_weights = {
            'trend_alignment': 0.25,
            'support_resistance': 0.20,
            'volume_confirmation': 0.15,
            'momentum_indicators': 0.15,
            'market_structure': 0.15,
            'timeframe_confluence': 0.10
        }
        
        # Performance tracking
        self.intercepted_signals = []
        self.passed_signals = []
        self.performance_stats = {
            'total_signals': 0,
            'intercepted_count': 0,
            'passed_count': 0,
            'interception_rate': 0.0
        }
        
        # Signal cache for analysis
        self.recent_signals = []
        self.max_cache_size = 100
        
        self._initialize_eventbus_hooks()
        self._emit_telemetry("SNIPER_INTERCEPTOR_INITIALIZED", {
            "min_quality_score": self.min_quality_score,
            "min_risk_reward_ratio": self.min_risk_reward_ratio,
            "min_confluence_score": self.min_confluence_score,
            "context_weights": self.context_weights
        })
    
    def _initialize_eventbus_hooks(self):
        """Initialize EventBus subscriptions"""
        if self.event_bus:
            self.event_bus.subscribe("SignalGenerated", self._intercept_signal)
            self.event_bus.subscribe("SignalQualityUpdate", self._update_quality_thresholds)
            self.event_bus.subscribe("MarketContextUpdate", self._update_market_context)
            self.event_bus.subscribe("PerformanceReview", self._generate_performance_report)
            self.event_bus.subscribe("InterceptorCalibration", self._calibrate_thresholds)
    
    def intercept_signal(self, trading_signal: TradingSignal) -> InterceptionResult:
        """
        🎯 Intercept and evaluate trading signal
        
        Args:
            trading_signal: The trading signal to evaluate
            
        Returns:
            InterceptionResult with decision and analysis
        """
        self.performance_stats['total_signals'] += 1
        
        # Calculate quality scores
        quality_score = self._calculate_quality_score(trading_signal)
        risk_reward_ratio = self._calculate_risk_reward_ratio(trading_signal)
        confluence_score = self._calculate_confluence_score(trading_signal)
        market_context_score = self._calculate_market_context_score(trading_signal)
        
        # Check interception criteria
        interception_reasons = []
        
        # Quality score check
        if quality_score < self.min_quality_score:
            interception_reasons.append(InterceptionReason.LOW_QUALITY_SCORE)
        
        # Risk-reward ratio check
        if risk_reward_ratio < self.min_risk_reward_ratio:
            interception_reasons.append(InterceptionReason.POOR_RISK_REWARD)
        
        # Confluence check
        if confluence_score < self.min_confluence_score:
            interception_reasons.append(InterceptionReason.LOW_CONFLUENCE)
        
        # Market context check
        if market_context_score < 0.5:
            interception_reasons.append(InterceptionReason.UNFAVORABLE_CONTEXT)
        
        # Spread check
        if self._is_spread_too_high(trading_signal):
            interception_reasons.append(InterceptionReason.HIGH_SPREAD)
        
        # Timeframe conflict check
        if self._has_timeframe_conflicts(trading_signal):
            interception_reasons.append(InterceptionReason.TIMEFRAME_CONFLICT)
        
        # Overall setup strength
        setup_strength = self._calculate_setup_strength(trading_signal)
        if setup_strength < 0.6:
            interception_reasons.append(InterceptionReason.WEAK_SETUP)
        
        # Determine if signal should be intercepted
        intercepted = len(interception_reasons) > 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trading_signal, interception_reasons)
        
        # Find alternative setups if intercepted
        alternative_setups = []
        if intercepted:
            alternative_setups = self._find_alternative_setups(trading_signal)
        
        # Create result
        result = InterceptionResult(
            signal_id=trading_signal.signal_id,
            intercepted=intercepted,
            quality_score=quality_score,
            reasons=interception_reasons,
            recommendations=recommendations,
            alternative_setups=alternative_setups
        )
        
        # Update performance stats
        if intercepted:
            self.intercepted_signals.append(trading_signal)
            self.performance_stats['intercepted_count'] += 1
        else:
            self.passed_signals.append(trading_signal)
            self.performance_stats['passed_count'] += 1
        
        self.performance_stats['interception_rate'] = (
            self.performance_stats['intercepted_count'] / self.performance_stats['total_signals']
        )
        
        # Cache signal for analysis
        self._cache_signal(trading_signal, result)
        
        # Emit telemetry
        self._emit_telemetry("SIGNAL_INTERCEPTED" if intercepted else "SIGNAL_PASSED", {
            "signal_id": trading_signal.signal_id,
            "symbol": trading_signal.symbol,
            "quality_score": quality_score,
            "risk_reward_ratio": risk_reward_ratio,
            "confluence_score": confluence_score,
            "interception_reasons": [r.value for r in interception_reasons],
            "recommendations_count": len(recommendations)
        })
        
        # Emit to EventBus
        if self.event_bus:
            self.event_bus.emit("SignalInterceptionResult", {
                "signal_id": trading_signal.signal_id,
                "intercepted": intercepted,
                "quality_score": quality_score,
                "reasons": [r.value for r in interception_reasons]
            })
        
        return result
    
    def _calculate_quality_score(self, signal: TradingSignal) -> float:
        """Calculate overall signal quality score"""
        scores = []
        
        # Confidence score (normalized)
        scores.append(signal.confidence)
        
        # Technical indicator alignment
        indicator_alignment = self._calculate_indicator_alignment(signal.indicators)
        scores.append(indicator_alignment)
        
        # Signal strength based on context
        context_strength = self._calculate_context_strength(signal.context)
        scores.append(context_strength)
        
        # Time-based factors
        time_score = self._calculate_time_score(signal)
        scores.append(time_score)
        
        # Return weighted average
        return statistics.mean(scores)
    
    def _calculate_risk_reward_ratio(self, signal: TradingSignal) -> float:
        """Calculate risk-reward ratio"""
        if signal.direction.upper() == "BUY":
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
        else:  # SELL
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.take_profit
        
        if risk <= 0:
            return 0.0
        
        return reward / risk
    
    def _calculate_confluence_score(self, signal: TradingSignal) -> float:
        """Calculate technical confluence score"""
        confluence_factors = []
        
        indicators = signal.indicators
        
        # Trend alignment across timeframes
        if 'trend_alignment' in indicators:
            confluence_factors.append(indicators['trend_alignment'])
        
        # Support/Resistance confluence
        if 'sr_confluence' in indicators:
            confluence_factors.append(indicators['sr_confluence'])
        
        # Multiple indicator confirmation
        confirmed_indicators = 0
        total_indicators = 0
        
        for indicator_name, value in indicators.items():
            if indicator_name.endswith('_signal'):
                total_indicators += 1
                if isinstance(value, bool) and value:
                    confirmed_indicators += 1
                elif isinstance(value, (int, float)) and value > 0.5:
                    confirmed_indicators += 1
        
        if total_indicators > 0:
            confluence_factors.append(confirmed_indicators / total_indicators)
        
        return statistics.mean(confluence_factors) if confluence_factors else 0.5
    
    def _calculate_market_context_score(self, signal: TradingSignal) -> float:
        """Calculate market context favorability score"""
        context = signal.context
        weighted_score = 0.0
        
        for factor, weight in self.context_weights.items():
            if factor in context:
                factor_score = context[factor]
                if isinstance(factor_score, bool):
                    factor_score = 1.0 if factor_score else 0.0
                elif isinstance(factor_score, str):
                    # Convert string ratings to numeric
                    factor_score = {
                        'excellent': 1.0, 'good': 0.8, 'fair': 0.6,
                        'poor': 0.4, 'terrible': 0.2
                    }.get(factor_score.lower(), 0.5)
                
                weighted_score += factor_score * weight
        
        return weighted_score
    
    def _calculate_indicator_alignment(self, indicators: Dict[str, Any]) -> float:
        """Calculate how well indicators align with signal direction"""
        aligned_count = 0
        total_count = 0
        for indicator_name, value in indicators.items():
            if indicator_name.endswith('_signal') or indicator_name.endswith('_direction'):
                total_count += 1
                if isinstance(value, bool) and value:
                    aligned_count += 1
                elif isinstance(value, str) and value.lower() in ['bullish', 'bearish']:
                    aligned_count += 1
                elif isinstance(value, (int, float)) and abs(value) > 0.5:
                    aligned_count += 1
        return aligned_count / total_count if total_count > 0 else 0.5
    
    def _calculate_context_strength(self, context: Dict[str, Any]) -> float:
        """Calculate signal context strength"""
        strength_factors = []
        
        # Volume confirmation
        if 'volume_confirmation' in context:
            strength_factors.append(context['volume_confirmation'])
        
        # Market session
        if 'market_session' in context:
            session_scores = {
                'london': 0.9, 'new_york': 0.9, 'asian': 0.6,
                'london_new_york_overlap': 1.0
            }
            strength_factors.append(session_scores.get(context['market_session'], 0.5))
        
        # Volatility conditions
        if 'volatility' in context:
            vol = context['volatility']
            if isinstance(vol, str):
                vol_scores = {'low': 0.3, 'medium': 0.8, 'high': 0.6}
                strength_factors.append(vol_scores.get(vol, 0.5))
        
        return statistics.mean(strength_factors) if strength_factors else 0.5
    
    def _calculate_time_score(self, signal: TradingSignal) -> float:
        """Calculate time-based signal quality factors"""
        current_time = datetime.now()
        signal_age = (current_time - signal.timestamp).total_seconds() / 60  # minutes
        
        # Fresher signals score higher (decay over time)
        freshness_score = max(0.0, 1.0 - (signal_age / 30))  # 30-minute decay
        
        # Market session timing
        hour = current_time.hour
        session_score = 1.0
        
        # London session (7-16 GMT)
        if 7 <= hour <= 16:
            session_score = 0.9
        # New York session (12-21 GMT)
        elif 12 <= hour <= 21:
            session_score = 0.9
        # Overlap (12-16 GMT)
        elif 12 <= hour <= 16:
            session_score = 1.0
        # Asian session (21-6 GMT)
        else:
            session_score = 0.6
        
        return (freshness_score + session_score) / 2
    
    def _is_spread_too_high(self, signal: TradingSignal) -> bool:
        """Check if spread is too high for the signal"""
        if 'spread' in signal.context:
            spread = signal.context['spread']
            average_spread = signal.context.get('average_spread', spread)
            
            return spread > (average_spread * self.max_spread_multiplier)
        
        return False
    
    def _has_timeframe_conflicts(self, signal: TradingSignal) -> bool:
        """Check for conflicts across timeframes"""
        if 'timeframe_analysis' not in signal.context:
            return False
        
        timeframe_analysis = signal.context['timeframe_analysis']
        conflicts = 0
        total_timeframes = 0
        
        for tf, analysis in timeframe_analysis.items():
            if isinstance(analysis, dict) and 'direction' in analysis:
                total_timeframes += 1
                if analysis['direction'] != signal.direction:
                    conflicts += 1
        
        # Consider it a conflict if more than 30% of timeframes disagree
        return (conflicts / total_timeframes) > 0.3 if total_timeframes > 0 else False
    
    def _calculate_setup_strength(self, signal: TradingSignal) -> float:
        """Calculate overall setup strength"""
        setup_factors = []
        
        # Pattern strength
        if 'pattern_strength' in signal.indicators:
            setup_factors.append(signal.indicators['pattern_strength'])
        
        # Entry precision
        if 'entry_precision' in signal.context:
            setup_factors.append(signal.context['entry_precision'])
        
        # Market structure alignment
        if 'structure_alignment' in signal.context:
            setup_factors.append(signal.context['structure_alignment'])
        
        return statistics.mean(setup_factors) if setup_factors else 0.5
    
    def _generate_recommendations(self, signal: TradingSignal, 
                                 reasons: List[InterceptionReason]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for reason in reasons:
            if reason == InterceptionReason.LOW_QUALITY_SCORE:
                recommendations.append("Wait for higher confluence setup")
                recommendations.append("Check additional timeframes for confirmation")
            
            elif reason == InterceptionReason.POOR_RISK_REWARD:
                recommendations.append("Adjust stop loss or take profit for better RR")
                recommendations.append("Consider partial profit taking strategy")
            
            elif reason == InterceptionReason.LOW_CONFLUENCE:
                recommendations.append("Wait for additional technical confirmation")
                recommendations.append("Look for support/resistance confluence")
            
            elif reason == InterceptionReason.UNFAVORABLE_CONTEXT:
                recommendations.append("Wait for better market conditions")
                recommendations.append("Check market session and volatility")
            
            elif reason == InterceptionReason.HIGH_SPREAD:
                recommendations.append("Wait for spread to normalize")
                recommendations.append("Consider alternative broker or timing")
            
            elif reason == InterceptionReason.TIMEFRAME_CONFLICT:
                recommendations.append("Wait for timeframe alignment")
                recommendations.append("Focus on higher timeframe direction")
        
        return recommendations
    
    def _find_alternative_setups(self, signal: TradingSignal) -> List[Dict]:
        """Find alternative trading setups"""
        alternatives = []
        
        # Suggest waiting for better entry
        alternatives.append({
            'type': 'better_entry',
            'description': 'Wait for pullback to better entry level',
            'estimated_wait': '15-30 minutes'
        })
        
        # Suggest different timeframe
        if signal.timeframe == 'M5':
            alternatives.append({
                'type': 'higher_timeframe',
                'description': 'Look for M15 or M30 setup',
                'timeframe': 'M15'
            })
        
        # Suggest risk adjustment
        current_rr = self._calculate_risk_reward_ratio(signal)
        if current_rr < self.min_risk_reward_ratio:
            alternatives.append({
                'type': 'risk_adjustment',
                'description': 'Tighten stop loss or extend take profit',
                'target_rr': self.min_risk_reward_ratio
            })
        
        return alternatives
    
    def _cache_signal(self, signal: TradingSignal, result: InterceptionResult):
        """Cache signal for analysis"""
        cache_entry = {
            'signal': signal,
            'result': result,
            'timestamp': datetime.now()
        }
        
        self.recent_signals.append(cache_entry)
        
        # Maintain cache size
        if len(self.recent_signals) > self.max_cache_size:
            self.recent_signals.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get interceptor performance statistics"""
        return {
            **self.performance_stats,
            'recent_signals_count': len(self.recent_signals),
            'quality_threshold': self.min_quality_score,
            'rr_threshold': self.min_risk_reward_ratio,
            'confluence_threshold': self.min_confluence_score
        }
    
    def calibrate_thresholds(self, performance_data: Dict[str, Any]):
        """Calibrate interception thresholds based on performance"""
        # Adjust thresholds based on success rate of passed signals
        if 'passed_signal_success_rate' in performance_data:
            success_rate = performance_data['passed_signal_success_rate']
            
            if success_rate < 0.6:  # Too many failed signals passing through
                self.min_quality_score = min(0.9, self.min_quality_score + 0.05)
                self.min_confluence_score = min(0.8, self.min_confluence_score + 0.05)
            elif success_rate > 0.8:  # Maybe being too strict
                self.min_quality_score = max(0.5, self.min_quality_score - 0.05)
                self.min_confluence_score = max(0.4, self.min_confluence_score - 0.05)
        
        self._emit_telemetry("THRESHOLDS_CALIBRATED", {
            "new_quality_threshold": self.min_quality_score,
            "new_confluence_threshold": self.min_confluence_score,
            "performance_data": performance_data
        })
    
    def _intercept_signal(self, event_data: Dict[str, Any]):
        """EventBus handler for signal interception"""
        # Convert event data to TradingSignal
        signal = TradingSignal(
            signal_id=event_data.get('signal_id', 'unknown'),
            symbol=event_data.get('symbol', ''),
            direction=event_data.get('direction', ''),
            entry_price=event_data.get('entry_price', 0.0),
            stop_loss=event_data.get('stop_loss', 0.0),
            take_profit=event_data.get('take_profit', 0.0),
            confidence=event_data.get('confidence', 0.5),
            timeframe=event_data.get('timeframe', 'M5'),
            timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
            indicators=event_data.get('indicators', {}),
            context=event_data.get('context', {})
        )
        
        result = self.intercept_signal(signal)
        
        # Emit result back to EventBus
        if self.event_bus:
            self.event_bus.emit("SignalInterceptionComplete", {
                "signal_id": signal.signal_id,
                "intercepted": result.intercepted,
                "quality_score": result.quality_score,
                "reasons": [r.value for r in result.reasons],
                "recommendations": result.recommendations
            })
    
    def _update_quality_thresholds(self, event_data: Dict[str, Any]):
        """Update quality thresholds"""
        if 'min_quality_score' in event_data:
            self.min_quality_score = event_data['min_quality_score']
        if 'min_risk_reward_ratio' in event_data:
            self.min_risk_reward_ratio = event_data['min_risk_reward_ratio']
        if 'min_confluence_score' in event_data:
            self.min_confluence_score = event_data['min_confluence_score']
    
    def _update_market_context(self, event_data: Dict[str, Any]):
        """Update market context weights"""
        if 'context_weights' in event_data:
            self.context_weights.update(event_data['context_weights'])
    
    def _generate_performance_report(self, event_data: Dict[str, Any]):
        """Generate performance report"""
        report = {
            'performance_stats': self.get_performance_stats(),
            'recent_interceptions': len([s for s in self.recent_signals if s['result'].intercepted]),
            'quality_distribution': self._get_quality_distribution(),
            'top_interception_reasons': self._get_top_interception_reasons()
        }
        
        if self.event_bus:
            self.event_bus.emit("InterceptorPerformanceReport", report)
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of signal qualities"""
        distribution = {quality.value: 0 for quality in SignalQuality}
        
        for signal_entry in self.recent_signals:
            quality_score = signal_entry['result'].quality_score
            
            if quality_score >= 0.9:
                distribution[SignalQuality.EXCELLENT.value] += 1
            elif quality_score >= 0.7:
                distribution[SignalQuality.GOOD.value] += 1
            elif quality_score >= 0.5:
                distribution[SignalQuality.FAIR.value] += 1
            elif quality_score >= 0.3:
                distribution[SignalQuality.POOR.value] += 1
            else:
                distribution[SignalQuality.TERRIBLE.value] += 1
        
        return distribution
    
    def _get_top_interception_reasons(self) -> Dict[str, int]:
        """Get most common interception reasons"""
        reason_counts = {}
        
        for signal_entry in self.recent_signals:
            if signal_entry['result'].intercepted:
                for reason in signal_entry['result'].reasons:
                    reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1
        
        return dict(sorted(reason_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _calibrate_thresholds(self, event_data: Dict[str, Any]):
        """EventBus handler for threshold calibration"""
        self.calibrate_thresholds(event_data)
    
    def _emit_telemetry(self, event_type: str, data: Dict[str, Any]):
        """Emit telemetry event"""
        telemetry_data = {
            "module": "sniper_signal_interceptor",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        if self.event_bus:
            self.event_bus.emit("telemetry", telemetry_data)
        
        logging.info(f"🎯 SNIPER-INTERCEPTOR {event_type}: {data}")

if __name__ == "__main__":
    # Test sniper signal interceptor
    print("🎯 Testing GENESIS Sniper Signal Interceptor")
    
    interceptor = GenesisSniperSignalInterceptor()
    
    # Test high-quality signal
    good_signal = TradingSignal(
        signal_id="TEST_001",
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        confidence=0.85,
        timeframe="M15",
        timestamp=datetime.now(),
        indicators={
            'rsi_signal': True,
            'macd_signal': True,
            'trend_alignment': 0.8,
            'sr_confluence': 0.9
        },
        context={
            'volume_confirmation': 0.8,
            'market_session': 'london_new_york_overlap',
            'volatility': 'medium'
        }
    )
    
    result = interceptor.intercept_signal(good_signal)
    print(f"Good signal intercepted: {result.intercepted}")
    print(f"Quality score: {result.quality_score:.3f}")
    
    # Test poor-quality signal
    poor_signal = TradingSignal(
        signal_id="TEST_002",
        symbol="EURUSD", 
        direction="BUY",
        entry_price=1.1000,
        stop_loss=1.0980,  # Poor RR ratio
        take_profit=1.1010,
        confidence=0.3,
        timeframe="M5",
        timestamp=datetime.now(),
        indicators={
            'rsi_signal': False,
            'macd_signal': False,
            'trend_alignment': 0.2
        },
        context={
            'volume_confirmation': 0.2,
            'market_session': 'asian',
            'spread': 3.0,
            'average_spread': 1.5
        }
    )
    
    result = interceptor.intercept_signal(poor_signal)
    print(f"Poor signal intercepted: {result.intercepted}")
    print(f"Quality score: {result.quality_score:.3f}")
    print(f"Interception reasons: {[r.value for r in result.reasons]}")
    print(f"Recommendations: {result.recommendations}")
    
    print(f"Performance stats: {interceptor.get_performance_stats()}")


def detect_divergence(price_data: list, indicator_data: list, window: int = 10) -> Dict:
    """
    Detect regular and hidden divergences between price and indicator
    
    Args:
        price_data: List of price values (closing prices)
        indicator_data: List of indicator values (e.g., RSI, MACD)
        window: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence information
    """
    result = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "strength": 0.0
    }
    
    # Need at least window + 1 periods of data
    if len(price_data) < window + 1 or len(indicator_data) < window + 1:
        return result
        
    # Get the current and historical points
    current_price = price_data[-1]
    previous_price = min(price_data[-window:-1]) if price_data[-1] > price_data[-2] else max(price_data[-window:-1])
    previous_price_idx = price_data[-window:-1].index(previous_price) + len(price_data) - window
    
    current_indicator = indicator_data[-1]
    previous_indicator = indicator_data[previous_price_idx]
    
    # Check for regular divergences
    # Bullish - Lower price lows but higher indicator lows
    if current_price < previous_price and current_indicator > previous_indicator:
        result["regular_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Higher price highs but lower indicator highs
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["regular_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Check for hidden divergences
    # Bullish - Higher price lows but lower indicator lows
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["hidden_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Lower price highs but higher indicator highs
    elif current_price < previous_price and current_indicator > previous_indicator:
        result["hidden_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Emit divergence event if detected
    if any([result["regular_bullish"], result["regular_bearish"], 
            result["hidden_bullish"], result["hidden_bearish"]]):
        emit_event("divergence_detected", {
            "type": next(k for k, v in result.items() if v is True and k != "strength"),
            "strength": result["strength"],
            "symbol": price_data.symbol if hasattr(price_data, "symbol") else "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
    return result
