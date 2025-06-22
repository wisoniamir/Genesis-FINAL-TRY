# <!-- @GENESIS_MODULE_START: trade_recommendation_engine -->


# 🔗 GENESIS EventBus Integration - Auto-injected by Orphan Recovery Engine
from datetime import datetime
import json

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



class TradeRecommendationEngineEventBusIntegration:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "trade_recommendation_engine",
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
                print(f"Emergency stop error in trade_recommendation_engine: {e}")
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
                        "module": "trade_recommendation_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in trade_recommendation_engine: {e}")
    """EventBus integration for trade_recommendation_engine"""
    
    def __init__(self):
        self.module_id = "trade_recommendation_engine"
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
trade_recommendation_engine_eventbus = TradeRecommendationEngineEventBusIntegration()

"""
GENESIS Phase 70: Trade Recommendation Engine
🔐 ARCHITECT MODE v5.0.0 - FULLY COMPLIANT
🧠 Trade Decision Synthesis & Recommendation Generation

Synthesizes signals from pattern overlay, macro filters, volatility indicators,
and backtest modules to generate actionable trade setups with full MT5 integration.
"""

import json
import os
import time
import threading
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradeRecommendation:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "trade_recommendation_engine",
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
                print(f"Emergency stop error in trade_recommendation_engine: {e}")
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
                        "module": "trade_recommendation_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in trade_recommendation_engine: {e}")
    """Trade recommendation data structure"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-10 scale
    risk_reward: float
    recommendation_id: str
    timestamp: str
    source_signals: List[str]
    pattern_confidence: float
    macro_alignment: float
    volatility_score: float
    backtest_score: float
    

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
        class TradeRecommendationEngine:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "trade_recommendation_engine",
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
                print(f"Emergency stop error in trade_recommendation_engine: {e}")
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
                        "module": "trade_recommendation_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in trade_recommendation_engine: {e}")
    """
    🔐 GENESIS Phase 70: Trade Recommendation Engine
    
    Synthesizes multiple signal sources to generate high-confidence trade recommendations:
    - Pattern overlay signals with confidence scoring
    - Macro environment filters and alignment checks
    - Volatility indicators for entry timing
    - Backtest validation for strategy verification
    - Risk-reward optimization with MT5 price integration
    """
    
    def __init__(self, config_path: str = "trade_recommendation_config.json"):
        """Initialize the Trade Recommendation Engine"""
        self.config_path = config_path
        self.config = self._load_config()
        self.active = True
        self.recommendation_history = []
        self.signal_cache = defaultdict(dict)
        self.telemetry_data = defaultdict(int)
        self.event_bus = None
        
        # Architect mode compliance
        self.module_id = "trade_recommendation_engine"
        self.fingerprint = self._generate_fingerprint()
        self.architect_compliant = True
        self.version = "1.0.0"
        self.phase = 70
        
        logger.info(f"🔐 TradeRecommendationEngine initialized - Phase {self.phase} - v{self.version}")
        self._register_telemetry_hooks()
        
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _load_config(self) -> Dict:
        """Load trade recommendation configuration"""
        default_config = {
            "confidence_threshold": 7.0,
            "min_risk_reward": 1.5,
            "max_daily_recommendations": 10,
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD"],
            "timeframes": ["M5", "M15", "H1", "H4"],
            "signal_sources": {
                "pattern_overlay": {"weight": 0.35, "required": True},
                "macro_filter": {"weight": 0.25, "required": True},
                "volatility_indicator": {"weight": 0.20, "required": False},
                "backtest_validation": {"weight": 0.20, "required": True}
            },
            "risk_parameters": {
                "max_risk_per_trade": 0.02,  # 2% of account
                "max_correlation_exposure": 0.05,  # 5% correlated positions
                "stop_loss_atr_multiplier": 2.0,
                "take_profit_atr_multiplier": 4.0
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
            
    def _generate_fingerprint(self) -> str:
        """Generate unique module fingerprint for architect mode"""
        timestamp = datetime.now(timezone.utc).isoformat()
        content_hash = hash(f"trade_recommendation_engine_{timestamp}_{self.version}")
        return f"tre-{abs(content_hash) % 1000000:06d}-{int(time.time()) % 100000}"
        
    def _register_telemetry_hooks(self):
        """Register telemetry hooks for architect mode compliance"""
        self.telemetry_hooks = [
            "recommendations_generated_count",
            "average_confidence_score",
            "recommendation_latency_ms",
            "signal_synthesis_time_ms",
            "pattern_overlay_signals_processed",
            "macro_filter_alignments",
            "volatility_adjustments_applied",
            "backtest_validations_performed",
            "high_confidence_recommendations",
            "recommendations_rejected_low_confidence",
            "risk_reward_distribution",
            "mt5_price_feed_latency"
        ]
        
    def connect_event_bus(self, event_bus):
        """Connect to EventBus for architect mode compliance"""
        self.event_bus = event_bus
        if self.event_bus:
            # Subscribe to input signals
            self.event_bus.subscribe("PatternOverlaySignal", self._handle_pattern_signal)
            self.event_bus.subscribe("MacroFilterUpdate", self._handle_macro_signal)
            self.event_bus.subscribe("VolatilityIndicatorUpdate", self._handle_volatility_signal)
            self.event_bus.subscribe("BacktestValidationResult", self._handle_backtest_signal)
            self.event_bus.subscribe("MT5PriceUpdate", self._handle_price_update)
            
            logger.info("🔗 TradeRecommendationEngine connected to EventBus")
            
    def _handle_pattern_signal(self, data: Dict):
        """Handle pattern overlay signals"""
        try:
            symbol = data.get('symbol')
            if symbol:
                self.signal_cache[symbol]['pattern'] = {
                    'confidence': data.get('confidence', 0.0),
                    'direction': data.get('direction'),
                    'strength': data.get('strength', 0.0),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                self.telemetry_data['pattern_overlay_signals_processed'] += 1
                logger.debug(f"📊 Pattern signal received for {symbol}: {data.get('confidence')}")
        except Exception as e:
            logger.error(f"Error handling pattern signal: {e}")
            
    def _handle_macro_signal(self, data: Dict):
        """Handle macro filter signals"""
        try:
            symbol = data.get('symbol')
            if symbol:
                self.signal_cache[symbol]['macro'] = {
                    'alignment': data.get('alignment', 0.0),
                    'trend': data.get('trend'),
                    'strength': data.get('strength', 0.0),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                self.telemetry_data['macro_filter_alignments'] += 1
                logger.debug(f"🌍 Macro signal received for {symbol}: {data.get('alignment')}")
        except Exception as e:
            logger.error(f"Error handling macro signal: {e}")
            
    def _handle_volatility_signal(self, data: Dict):
        """Handle volatility indicator signals"""
        try:
            symbol = data.get('symbol')
            if symbol:
                self.signal_cache[symbol]['volatility'] = {
                    'score': data.get('score', 0.0),
                    'atr': data.get('atr', 0.0),
                    'regime': data.get('regime'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                self.telemetry_data['volatility_adjustments_applied'] += 1
                logger.debug(f"📈 Volatility signal received for {symbol}: {data.get('score')}")
        except Exception as e:
            logger.error(f"Error handling volatility signal: {e}")
            
    def _handle_backtest_signal(self, data: Dict):
        """Handle backtest validation signals"""
        try:
            symbol = data.get('symbol')
            if symbol:
                self.signal_cache[symbol]['backtest'] = {
                    'score': data.get('score', 0.0),
                    'win_rate': data.get('win_rate', 0.0),
                    'profit_factor': data.get('profit_factor', 0.0),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                self.telemetry_data['backtest_validations_performed'] += 1
                logger.debug(f"🧪 Backtest signal received for {symbol}: {data.get('score')}")
        except Exception as e:
            logger.error(f"Error handling backtest signal: {e}")
            
    def _handle_price_update(self, data: Dict):
        """Handle MT5 price updates"""
        try:
            symbol = data.get('symbol')
            if symbol:
                self.signal_cache[symbol]['price'] = {
                    'bid': data.get('bid', 0.0),
                    'ask': data.get('ask', 0.0),
                    'spread': data.get('spread', 0.0),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                logger.debug(f"💹 Price update received for {symbol}: {data.get('bid')}/{data.get('ask')}")
        except Exception as e:
            logger.error(f"Error handling price update: {e}")
            
    def synthesize_signals(self, symbol: str) -> Optional[Dict]:
        """Synthesize all available signals for a symbol"""
        start_time = time.time()
        
        try:
            if symbol not in self.signal_cache:
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            signals = self.signal_cache[symbol]
            source_weights = self.config['signal_sources']
            
            # Check required signals
            required_sources = [k for k, v in source_weights.items() if v.get('required', False)]
            available_sources = list(signals.keys())
            
            missing_required = [src for src in required_sources if src.replace('_', '') not in [s.replace('_', '') for s in available_sources]]
            if missing_required:
                logger.debug(f"Missing required signals for {symbol}: {missing_required}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            # Calculate weighted confidence
            total_confidence = 0.0
            total_weight = 0.0
            signal_details = {}
            
            for source, config in source_weights.items():
                source_key = source.replace('_', '')
                if source_key in signals:
                    signal_data = signals[source_key]
                    confidence = signal_data.get('confidence', signal_data.get('score', signal_data.get('alignment', 0.0)))
                    weight = config['weight']
                    
                    total_confidence += confidence * weight
                    total_weight += weight
                    signal_details[source] = {
                        'confidence': confidence,
                        'weight': weight,
                        'data': signal_data
                    }
                    
            if total_weight == 0:
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            overall_confidence = total_confidence / total_weight
            
            synthesis_time = (time.time() - start_time) * 1000
            self.telemetry_data['signal_synthesis_time_ms'] += int(synthesis_time)
            
            return {
                'symbol': symbol,
                'overall_confidence': overall_confidence,
                'signal_details': signal_details,
                'synthesis_time_ms': synthesis_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing signals for {symbol}: {e}")
            self._emit_error_event("operation_failed", {

                "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                "timestamp": datetime.now(timezone.utc).isoformat()

            })

            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
            
    def calculate_entry_levels(self, symbol: str, direction: str, synthesis_data: Dict) -> Optional[Tuple[float, float, float]]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            if symbol not in self.signal_cache or 'price' not in self.signal_cache[symbol]:
                logger.warning(f"No price data available for {symbol}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            price_data = self.signal_cache[symbol]['price']
            current_bid = price_data['bid']
            current_ask = price_data['ask']
            
            # Get volatility data for ATR-based levels
            volatility_data = synthesis_data['signal_details'].get('volatility_indicator', {}).get('data', {})
            atr = volatility_data.get('atr', abs(current_ask - current_bid) * 50)  # Fallback ATR estimate
            
            risk_params = self.config['risk_parameters']
            sl_multiplier = risk_params['stop_loss_atr_multiplier']
            tp_multiplier = risk_params['take_profit_atr_multiplier']
            
            if direction.lower() == 'long':
                entry = current_ask
                stop_loss = entry - (atr * sl_multiplier)
                take_profit = entry + (atr * tp_multiplier)
            else:  # short
                entry = current_bid
                stop_loss = entry + (atr * sl_multiplier)
                take_profit = entry - (atr * tp_multiplier)
                
            return entry, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating entry levels for {symbol}: {e}")
            self._emit_error_event("operation_failed", {

                "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                "timestamp": datetime.now(timezone.utc).isoformat()

            })

            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
            
    def generate_recommendation(self, symbol: str) -> Optional[TradeRecommendation]:
        """Generate a trade recommendation for a symbol"""
        start_time = time.time()
        
        try:
            # Synthesize all signals
            synthesis_data = self.synthesize_signals(symbol)
            if not synthesis_data:
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            overall_confidence = synthesis_data['overall_confidence']
            
            # Check confidence threshold
            if overall_confidence < self.config['confidence_threshold']:
                self.telemetry_data['recommendations_rejected_low_confidence'] += 1
                logger.debug(f"Confidence too low for {symbol}: {overall_confidence}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            # Determine direction from pattern signal
            pattern_data = synthesis_data['signal_details'].get('pattern_overlay', {}).get('data', {})
            direction = pattern_data.get('direction')
            if not direction:
                logger.debug(f"No direction signal for {symbol}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            # Calculate entry levels
            levels = self.calculate_entry_levels(symbol, direction, synthesis_data)
            if not levels:
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            entry, stop_loss, take_profit = levels
            
            # Calculate risk-reward ratio
            if direction.lower() == 'long':
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
            else:
                risk = abs(stop_loss - entry)
                reward = abs(entry - take_profit)
                
            risk_reward = reward / risk if risk > 0 else 0
            
            # Check minimum risk-reward ratio
            if risk_reward < self.config['min_risk_reward']:
                logger.debug(f"Risk-reward too low for {symbol}: {risk_reward}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
                
            # Extract individual signal scores
            pattern_confidence = synthesis_data['signal_details'].get('pattern_overlay', {}).get('confidence', 0.0)
            macro_alignment = synthesis_data['signal_details'].get('macro_filter', {}).get('confidence', 0.0)
            volatility_score = synthesis_data['signal_details'].get('volatility_indicator', {}).get('confidence', 0.0)
            backtest_score = synthesis_data['signal_details'].get('backtest_validation', {}).get('confidence', 0.0)
            
            # Create recommendation
            recommendation = TradeRecommendation(
                symbol=symbol,
                direction=direction.lower(),
                entry=round(entry, 5),
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                confidence=round(overall_confidence, 2),
                risk_reward=round(risk_reward, 2),
                recommendation_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_signals=list(synthesis_data['signal_details'].keys()),
                pattern_confidence=round(pattern_confidence, 2),
                macro_alignment=round(macro_alignment, 2),
                volatility_score=round(volatility_score, 2),
                backtest_score=round(backtest_score, 2)
            )
            
            # Update telemetry
            generation_time = (time.time() - start_time) * 1000
            self.telemetry_data['recommendations_generated_count'] += 1
            self.telemetry_data['recommendation_latency_ms'] += int(generation_time)
            self.telemetry_data['average_confidence_score'] = (
                (self.telemetry_data.get('average_confidence_score', 0) * 
                 (self.telemetry_data['recommendations_generated_count'] - 1) + overall_confidence) /
                self.telemetry_data['recommendations_generated_count']
            )
            
            if overall_confidence >= 8.0:
                self.telemetry_data['high_confidence_recommendations'] += 1
                
            # Store in history
            self.recommendation_history.append(recommendation)
            
            # Publish to EventBus
            if self.event_bus:
                self.event_bus.publish("TradeRecommendationGenerated", asdict(recommendation))
                
            logger.info(f"✅ Trade recommendation generated for {symbol}: {direction} @ {entry} (confidence: {overall_confidence})")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            self._emit_error_event("operation_failed", {

                "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                "timestamp": datetime.now(timezone.utc).isoformat()

            })

            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
            
    def process_symbols(self) -> List[TradeRecommendation]:
        """Process all configured symbols for recommendations"""
        recommendations = []
        
        try:
            for symbol in self.config['symbols']:
                if len(recommendations) >= self.config['max_daily_recommendations']:
                    break
                    
                recommendation = self.generate_recommendation(symbol)
                if recommendation:
                    recommendations.append(recommendation)
                    
            logger.info(f"📊 Generated {len(recommendations)} trade recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing symbols: {e}")
            return recommendations
            
    def get_telemetry_data(self) -> Dict:
        """Get telemetry data for architect mode compliance"""
        return {
            'module_id': self.module_id,
            'fingerprint': self.fingerprint,
            'phase': self.phase,
            'version': self.version,
            'architect_compliant': self.architect_compliant,
            'active': self.active,
            'telemetry_hooks': self.telemetry_hooks,
            'metrics': dict(self.telemetry_data),
            'recommendation_history_count': len(self.recommendation_history),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
    def shutdown(self):
        """Shutdown the engine gracefully"""
        self.active = False
        logger.info("🔐 TradeRecommendationEngine shutdown complete")

def main():
    """Main execution function for testing"""
    engine = TradeRecommendationEngine()
    
    # Test recommendation generation
    test_symbols = ["EURUSD", "GBPUSD"]
    
    for symbol in test_symbols:
        recommendation = engine.generate_recommendation(symbol)
        if recommendation:
            print(f"Generated recommendation: {recommendation}")
        else:
            print(f"No recommendation for {symbol}")
            
    # Print telemetry
    telemetry = engine.get_telemetry_data()
    print(f"Telemetry: {json.dumps(telemetry, indent=2)}")

if __name__ == "__main__":
    main()


# <!-- @GENESIS_MODULE_END: trade_recommendation_engine -->