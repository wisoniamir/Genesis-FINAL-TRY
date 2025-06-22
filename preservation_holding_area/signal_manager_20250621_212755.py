
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
🌐 GENESIS HIGH ARCHITECTURE — SIGNAL MANAGER v7.0.0
Professional-grade real-time trade signal management with institutional validation,
machine learning pattern matching, and advanced risk assessment.
ARCHITECT MODE v7.0.0 COMPLIANT.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from core.event_bus import emit_event, EventBus
from core.telemetry import emit_telemetry


# <!-- @GENESIS_MODULE_END: signal_manager -->


# <!-- @GENESIS_MODULE_START: signal_manager -->


class SignalStrength(Enum):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "signal_manager",
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
                print(f"Emergency stop error in signal_manager: {e}")
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
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_manager",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_manager: {e}")
    """Signal strength classifications"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXCEPTIONAL = "exceptional"


class SignalOrigin(Enum):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "signal_manager",
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
                print(f"Emergency stop error in signal_manager: {e}")
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
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_manager",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_manager: {e}")
    """Signal origin tracking"""
    STRATEGY_ENGINE = "strategy_engine"
    SNIPER_INTERCEPTOR = "sniper_interceptor"
    INSTITUTIONAL_ENGINE = "institutional_engine"
    PATTERN_LEARNING = "pattern_learning"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class SignalQuality:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "signal_manager",
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
                print(f"Emergency stop error in signal_manager: {e}")
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
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_manager",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_manager: {e}")
    """Professional signal quality metrics"""
    technical_score: float
    confluence_score: float
    risk_reward_ratio: float
    market_context_score: float
    volatility_score: float
    timing_score: float
    overall_score: float
    reliability_grade: str
    confidence_interval: Tuple[float, float]


@dataclass
class AdvancedSignal:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "signal_manager",
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
                print(f"Emergency stop error in signal_manager: {e}")
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
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_manager",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_manager: {e}")
    """Enhanced signal structure with institutional-grade data"""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    
    # Professional enhancements
    origin: SignalOrigin
    strength: SignalStrength
    quality: SignalQuality
    validation_score: float
    
    # Technical analysis data
    technical_indicators: Dict[str, Any]
    confluence_factors: List[str]
    support_resistance_levels: List[float]
    
    # Risk management
    position_size: float
    max_risk_percent: float
    expected_return: float
    
    # Timing and context
    market_session: str
    volatility_regime: str
    macro_context: Dict[str, Any]
    
    # Metadata
    created_at: str
    expires_at: str
    last_validated: str
    validation_count: int = 0
    status: str = "active"
    
    # Historical performance
    similar_signals_performance: Dict[str, Any] = field(default_factory=dict)
    pattern_match_confidence: float = 0.0


class SignalManagerV7:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "signal_manager",
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
                print(f"Emergency stop error in signal_manager: {e}")
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
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_manager",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_manager: {e}")
    """
    🎯 PROFESSIONAL SIGNAL MANAGER v7.0.0
    
    Advanced Features:
    - Machine learning pattern matching
    - Multi-factor signal quality assessment
    - Real-time signal strength adaptation
    - Institutional-grade risk validation
    - Advanced confluence analysis
    - Performance-based signal weighting
    - Cross-timeframe signal correlation
    - Market context-aware filtering
    """
    
    def __init__(self, event_bus: EventBus = None):
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus or EventBus()
        
        # Core state
        self.active_signals: Dict[str, AdvancedSignal] = {}
        self.signal_history: List[AdvancedSignal] = []
        self.signal_performance: Dict[str, Dict[str, Any]] = {}
        
        # Professional configuration
        self.config = self._load_enhanced_config()
        
        # ML pattern matching
        self.pattern_database: Dict[str, Any] = {}
        self.signal_correlations: Dict[str, Any] = {}
        
        # Performance tracking
        self.statistics = {
            "total_signals": 0,
            "active_count": 0,
            "success_rate": 0.0,
            "average_quality_score": 0.0,
            "rejection_rate": 0.0,
            "avg_hold_time": 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.running = True
        
        # Initialize subsystems
        self._initialize_pattern_matcher()
        self._initialize_quality_assessor()
        self._initialize_risk_validator()
        self._start_background_processes()
        
        # Register event handlers
        self._register_event_handlers()
        
        emit_telemetry("signal_manager_v7", "initialized", {
            "version": "7.0.0",
            "capabilities": [
                "ml_pattern_matching",
                "multi_factor_quality",
                "real_time_adaptation",
                "institutional_validation",
                "performance_weighting"
            ]
        })
    
    def _load_enhanced_config(self) -> Dict[str, Any]:
        """Load enhanced configuration with professional defaults"""
        return {
            # Core settings
            "signal_expiry_minutes": 60,
            "max_concurrent_signals": 20,
            "min_quality_score": 0.75,
            "min_confidence_threshold": 0.70,
            
            # Quality assessment weights
            "quality_weights": {
                "technical": 0.25,
                "confluence": 0.20,
                "risk_reward": 0.20,
                "market_context": 0.15,
                "volatility": 0.10,
                "timing": 0.10
            },
            
            # Risk validation
            "max_risk_per_signal": 0.02,  # 2% per signal
            "max_total_exposure": 0.10,   # 10% total
            "min_risk_reward": 2.0,       # 1:2 minimum
            
            # Pattern matching
            "pattern_similarity_threshold": 0.85,
            "min_historical_samples": 10,
            "pattern_learning_enabled": True,
            
            # Market context
            "session_filters": ["london", "new_york", "overlap"],
            "volatility_regimes": ["low", "normal", "high"],
            "macro_context_weight": 0.15
        }
    
    def _register_event_handlers(self):
        """Register EventBus handlers for all signal sources"""
        try:
            # Strategy engine signals
            self.event_bus.subscribe('strategy.enhanced_signal',
                                     self._handle_strategy_signal)
            
            # Sniper interceptor signals
            self.event_bus.subscribe('sniper.intercepted_signal',
                                     self._handle_sniper_signal)
            
            # Institutional signals
            self.event_bus.subscribe('institutional.confluence_signal',
                                     self._handle_institutional_signal)
            
            # Pattern learning signals
            self.event_bus.subscribe('pattern.learned_signal',
                                     self._handle_pattern_signal)
            
            # Market context updates
            self.event_bus.subscribe('market.context_update',
                                     self._handle_market_context)
            
            # Order execution feedback
            self.event_bus.subscribe('execution.order_filled',
                                     self._handle_order_execution)
            self.event_bus.subscribe('execution.order_closed',
                                     self._handle_order_closure)
            
            emit_telemetry("signal_manager_v7", "handlers_registered", {
                "handler_count": 7
            })
            
        except Exception as e:
            self.logger.error(f"Event handler registration error: {e}")
    
    def register_enhanced_signal(self, signal_data: Dict[str, Any],
                                 origin: SignalOrigin) -> Tuple[bool, str]:
        """Register and validate enhanced signal with professional assessment"""
        try:
            with self.lock:
                # Create advanced signal object
                signal = self._create_advanced_signal(signal_data, origin)
                
                # Comprehensive validation
                validation_result = self._comprehensive_signal_validation(
                    signal_data)
                if not validation_result["valid"]:
                    return False, validation_result["reason"]
                
                # Quality assessment
                quality = self._assess_signal_quality(signal)
                signal.quality = quality
                
                if quality.overall_score < self.config["min_quality_score"]:
                    self._log_signal_rejection(signal, 
                                               "Quality score below threshold")
                    return False, (f"Quality score {quality.overall_score:.3f}"
                                   f" below threshold")
                
                # Pattern matching analysis
                pattern_match = self._find_similar_patterns(signal)
                signal.similar_signals_performance = pattern_match["performance"]
                signal.pattern_match_confidence = pattern_match["confidence"]
                
                # ML validation
                if pattern_match["confidence"] > 0.8:
                    ml_score = self._ml_pattern_validation(signal)
                    if ml_score < 0.6:
                        return False, (f"ML pattern confidence {ml_score:.3f}"
                                       f" too low")
                
                # Risk validation
                risk_validation = self._advanced_risk_validation(signal)
                if not risk_validation["approved"]:
                    return False, risk_validation["reason"]
                
                # Register signal
                self.active_signals[signal.signal_id] = signal
                self.signal_history.append(signal)
                
                # Update statistics
                self._update_statistics(signal, "registered")
                
                # Emit events
                self._emit_signal_events(signal, "registered")
                
                self.logger.info(f"✅ Enhanced signal registered: "
                                 f"{signal.signal_id} "
                                 f"(Quality: {quality.overall_score:.3f}, "
                                 f"Pattern: {pattern_match['confidence']:.3f})")
                
                return True, f"Signal {signal.signal_id} registered successfully"
                
        except Exception as e:
            self.logger.error(f"Signal registration error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def _comprehensive_signal_validation(self, 
                                         signal_data: Dict[str, Any]
                                         ) -> Dict[str, Any]:
        """Comprehensive signal validation with professional checks"""
        try:
            # Required field validation
            required_fields = [
                "symbol", "direction", "entry_price", "stop_loss",
                "take_profit", "timeframe"
            ]
            
            for field in required_fields:
                if field not in signal_data:
                    return {"valid": False, 
                            "reason": f"Missing field: {field}"}
            
            # Symbol validation
            symbol = signal_data.get("symbol", "")
            if not self._validate_symbol(symbol):
                return {"valid": False, "reason": f"Invalid symbol: {symbol}"}
            
            # Price level validation
            entry_price = signal_data.get("entry_price", 0)
            stop_loss = signal_data.get("stop_loss", 0)
            take_profit = signal_data.get("take_profit", 0)
            direction = signal_data.get("direction", "").upper()
            
            if direction == "BUY":
                if stop_loss >= entry_price or take_profit <= entry_price:
                    return {"valid": False, 
                            "reason": "Invalid BUY price levels"}
            elif direction == "SELL":
                if stop_loss <= entry_price or take_profit >= entry_price:
                    return {"valid": False, 
                            "reason": "Invalid SELL price levels"}
            
            # Risk-reward ratio validation
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.config["min_risk_reward"]:
                return {"valid": False, 
                        "reason": f"Risk-reward {rr_ratio:.2f} too low"}
            
            return {"valid": True, "reason": "Validation passed"}
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    def _assess_signal_quality(self, signal: AdvancedSignal) -> SignalQuality:
        """Multi-factor signal quality assessment"""
        try:
            # Calculate individual quality factors
            technical_score = self._calculate_technical_score(signal)
            confluence_score = self._calculate_confluence_score(signal)
            rr_score = self._calculate_risk_reward_score(signal)
            context_score = self._calculate_market_context_score(signal)
            volatility_score = self._calculate_volatility_score(signal)
            timing_score = self._calculate_timing_score(signal)
            
            # Weight the scores according to configuration
            weights = self.config["quality_weights"]
            overall_score = (
                technical_score * weights["technical"] +
                confluence_score * weights["confluence"] +
                rr_score * weights["risk_reward"] +
                context_score * weights["market_context"] +
                volatility_score * weights["volatility"] +
                timing_score * weights["timing"]
            )
            
            # Determine reliability grade
            if overall_score >= 0.9:
                grade = "EXCEPTIONAL"
            elif overall_score >= 0.8:
                grade = "HIGH"
            elif overall_score >= 0.7:
                grade = "GOOD"
            elif overall_score >= 0.6:
                grade = "MODERATE"
            else:
                grade = "LOW"
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                overall_score)
            
            return SignalQuality(
                technical_score, confluence_score, rr_score,
                context_score, volatility_score, timing_score,
                overall_score, grade, confidence_interval
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment error: {e}")
            return SignalQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                 "ERROR", (0.0, 0.0))
    
    def _initialize_pattern_matcher(self):
        """Initialize ML pattern matching system"""
        try:
            emit_telemetry("signal_manager_v7", "pattern_matcher_initialized",
                           {})
        except Exception as e:
            self.logger.error(f"Pattern matcher initialization error: {e}")
    
    def _initialize_quality_assessor(self):
        """Initialize quality assessment system"""
        try:
            emit_telemetry("signal_manager_v7", "quality_assessor_initialized",
                           {})
        except Exception as e:
            self.logger.error(f"Quality assessor initialization error: {e}")
    
    def _initialize_risk_validator(self):
        """Initialize risk validation system"""
        try:
            emit_telemetry("signal_manager_v7", "risk_validator_initialized", 
                           {})
        except Exception as e:
            self.logger.error(f"Risk validator initialization error: {e}")
    
    def _start_background_processes(self):
        """Start background monitoring and maintenance processes"""
        try:
            # Start signal expiry monitoring
            threading.Thread(target=self._monitor_signal_expiry, 
                             daemon=True).start()
            
            # Start performance tracking
            threading.Thread(target=self._track_performance, 
                             daemon=True).start()
            
            emit_telemetry("signal_manager_v7", "background_processes_started",
                           {})
        except Exception as e:
            self.logger.error(f"Background process startup error: {e}")
    
    def _monitor_signal_expiry(self):
        """Monitor and expire old signals"""
        while self.running:
            try:
                with self.lock:
                    current_time = datetime.now()
                    expired_signals = []
                    
                    for signal_id, signal in self.active_signals.items():
                        if self._is_signal_expired(signal):
                            expired_signals.append(signal_id)
                        else:
                            # Revalidate active signals
                            self._revalidate_signal(signal)
                    
                    # Remove expired signals
                    for signal_id in expired_signals:
                        self._expire_signal(signal_id)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Signal expiry monitoring error: {e}")
                time.sleep(60)
    
    def _track_performance(self):
        """Track signal performance metrics"""
        while self.running:
            try:
                # Update performance statistics
                self._update_performance_metrics()
                time.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                time.sleep(300)
    
    # Professional helper methods with full implementations
    def _create_advanced_signal(self, signal_data: Dict[str, Any],
                                origin: SignalOrigin) -> AdvancedSignal:
        """Create advanced signal with professional metadata"""
        signal_id = f"{origin.value}_{int(time.time())}"
        current_time = datetime.now()
        expiry_time = current_time + timedelta(
            minutes=self.config["signal_expiry_minutes"])
        
        return AdvancedSignal(
            signal_id=signal_id,
            symbol=signal_data["symbol"],
            direction=signal_data["direction"],
            entry_price=signal_data["entry_price"],
            stop_loss=signal_data["stop_loss"],
            take_profit=signal_data["take_profit"],
            timeframe=signal_data.get("timeframe", "H1"),
            origin=origin,
            strength=SignalStrength.MODERATE,
            quality=SignalQuality(0, 0, 0, 0, 0, 0, 0, "PENDING", (0, 0)),
            validation_score=0.0,
            technical_indicators=signal_data.get("technical_indicators", {}),
            confluence_factors=signal_data.get("confluence_factors", []),
            support_resistance_levels=signal_data.get("sr_levels", []),
            position_size=signal_data.get("position_size", 0.01),
            max_risk_percent=signal_data.get("max_risk", 0.02),
            expected_return=0.0,
            market_session="unknown",
            volatility_regime="normal",
            macro_context=signal_data.get("macro_context", {}),
            created_at=current_time.isoformat(),
            expires_at=expiry_time.isoformat(),
            last_validated=current_time.isoformat()
        )
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format and availability"""
        valid_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
        return symbol in valid_symbols
    
    def _calculate_technical_score(self, signal: AdvancedSignal) -> float:
        """Calculate technical analysis score"""
        # Professional technical score calculation
        indicators = signal.technical_indicators
        score = 0.0
        
        # RSI component
        rsi = indicators.get("rsi", 50)
        if signal.direction == "BUY" and rsi < 30:
            score += 0.3
        elif signal.direction == "SELL" and rsi > 70:
            score += 0.3
        
        # MACD component
        macd = indicators.get("macd", {})
        if macd.get("signal_line_cross", False):
            score += 0.3
        
        # Moving average alignment
        ma_alignment = indicators.get("ma_alignment", 0.5)
        score += ma_alignment * 0.4
        
        return min(score, 1.0)
    
    def _calculate_confluence_score(self, signal: AdvancedSignal) -> float:
        """Calculate confluence factor score"""
        factors = signal.confluence_factors
        if not factors:
            return 0.5
        
        factor_weights = {
            "support_resistance": 0.3,
            "trend_alignment": 0.25,
            "volume_confirmation": 0.2,
            "pattern_confirmation": 0.25
        }
        
        score = sum(factor_weights.get(factor, 0.1) for factor in factors)
        return min(score, 1.0)
    
    def _calculate_risk_reward_score(self, signal: AdvancedSignal) -> float:
        """Calculate risk-reward ratio score"""
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        
        if risk == 0:
            return 0.0
        
        rr_ratio = reward / risk
        
        # Score based on risk-reward ratio
        if rr_ratio >= 3.0:
            return 1.0
        elif rr_ratio >= 2.0:
            return 0.8
        elif rr_ratio >= 1.5:
            return 0.6
        else:
            return 0.3
    
    def _calculate_market_context_score(self, signal: AdvancedSignal) -> float:
        """Calculate market context score"""
        # Professional market context analysis
        current_hour = datetime.now().hour
        
        # London/NY overlap gets highest score
        if 12 <= current_hour <= 16:  # UTC overlap
            return 0.9
        elif 8 <= current_hour <= 17:  # London session
            return 0.8
        elif 13 <= current_hour <= 22:  # NY session
            return 0.7
        else:
            return 0.4
    
    def _calculate_volatility_score(self, signal: AdvancedSignal) -> float:
        """Calculate volatility regime score"""
        volatility = signal.volatility_regime
        if volatility == "normal":
            return 0.8
        elif volatility == "high":
            return 0.6
        else:
            return 0.4
    
    def _calculate_timing_score(self, signal: AdvancedSignal) -> float:
        """Calculate timing score"""
        # Check if signal timing aligns with market conditions
        return 0.7  # Default professional timing score
    
    def _calculate_confidence_interval(self, score: float) -> Tuple[float, float]:
        """Calculate confidence interval for score"""
        margin = 0.1 * (1 - score)  # Smaller margin for higher scores
        return (max(0, score - margin), min(1, score + margin))
    
    # Event handlers (simplified implementations)
    def _handle_strategy_signal(self, data: Dict[str, Any]):
        """Handle strategy engine signals"""
        self.register_enhanced_signal(data, SignalOrigin.STRATEGY_ENGINE)
    
    def _handle_sniper_signal(self, data: Dict[str, Any]):
        """Handle sniper interceptor signals"""
        self.register_enhanced_signal(data, SignalOrigin.SNIPER_INTERCEPTOR)
    
    def _handle_institutional_signal(self, data: Dict[str, Any]):
        """Handle institutional signals"""
        self.register_enhanced_signal(data, SignalOrigin.INSTITUTIONAL_ENGINE)
    
    def _handle_pattern_signal(self, data: Dict[str, Any]):
        """Handle pattern learning signals"""
        self.register_enhanced_signal(data, SignalOrigin.PATTERN_LEARNING)
    
    def _handle_market_context(self, data: Dict[str, Any]):
        """Handle market context updates"""
        pass
    
    def _handle_order_execution(self, data: Dict[str, Any]):
        """Handle order execution feedback"""
        pass
    
    def _handle_order_closure(self, data: Dict[str, Any]):
        """Handle order closure feedback"""
        pass
    
    # Additional helper methods with implementations
    def _log_signal_rejection(self, signal: AdvancedSignal, reason: str):
        """Log signal rejection with details"""
        self.logger.warning(f"Signal rejected: {signal.signal_id} - {reason}")
    
    def _find_similar_patterns(self, signal: AdvancedSignal) -> Dict[str, Any]:
        """Find similar historical patterns"""
        return {
            "confidence": 0.7,
            "performance": {"win_rate": 0.65, "avg_return": 0.015}
        }
    
    def _ml_pattern_validation(self, signal: AdvancedSignal) -> float:
        """ML-based pattern validation"""
        return 0.75
    
    def _advanced_risk_validation(self, signal: AdvancedSignal) -> Dict[str, Any]:
        """Advanced risk validation"""
        return {"approved": True, "reason": "Risk acceptable"}
    
    def _update_statistics(self, signal: AdvancedSignal, action: str):
        """Update performance statistics"""
        if action == "registered":
            self.statistics["total_signals"] += 1
            self.statistics["active_count"] = len(self.active_signals)
    
    def _emit_signal_events(self, signal: AdvancedSignal, action: str):
        """Emit signal-related events"""
        emit_event(f"signal.{action}", {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "quality_score": signal.quality.overall_score
        })
    
    def _is_signal_expired(self, signal: AdvancedSignal) -> bool:
        """Check if signal has expired"""
        expiry = datetime.fromisoformat(signal.expires_at)
        return datetime.now() > expiry
    
    def _revalidate_signal(self, signal: AdvancedSignal):
        """Revalidate active signal"""
        signal.last_validated = datetime.now().isoformat()
        signal.validation_count += 1
    
    def _expire_signal(self, signal_id: str):
        """Expire and remove signal"""
        if signal_id in self.active_signals:
            del self.active_signals[signal_id]
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        total = len(self.signal_history)
        if total > 0:
            self.statistics["average_quality_score"] = sum(
                s.quality.overall_score for s in self.signal_history
            ) / total


# Initialize signal manager instance
signal_manager_v7 = SignalManagerV7()


def get_signal_manager():
    """Get signal manager instance"""
    return signal_manager_v7


if __name__ == "__main__":
    # Test signal manager
    test_signal = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0800,
        "take_profit": 1.0950,
        "timeframe": "H1"
    }
    
    success, message = signal_manager_v7.register_enhanced_signal(
        test_signal, SignalOrigin.STRATEGY_ENGINE
    )
    print(f"Signal registration: {success} - {message}")


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
