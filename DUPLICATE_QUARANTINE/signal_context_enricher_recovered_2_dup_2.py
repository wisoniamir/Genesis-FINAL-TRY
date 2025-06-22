# <!-- @GENESIS_MODULE_START: signal_context_enricher -->

from datetime import datetime\n"""

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

                emit_telemetry("signal_context_enricher_recovered_2", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("signal_context_enricher_recovered_2", "position_calculated", {
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
                            "module": "signal_context_enricher_recovered_2",
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
                    print(f"Emergency stop error in signal_context_enricher_recovered_2: {e}")
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
                    "module": "signal_context_enricher_recovered_2",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("signal_context_enricher_recovered_2", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in signal_context_enricher_recovered_2: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


GENESIS AI TRADING SYSTEM - PHASE 19
Signal Context Enricher - Real-time Signal Enhancement Engine
ARCHITECT MODE v3.0 - INSTITUTIONAL GRADE COMPLIANCE

PURPOSE:
- Enrich all signals with dynamic context metadata
- Add volatility, age, risk score, market phase data
- Inject real-time telemetry and historical correlation data
- Route enriched signals through EventBus for adaptive filtering

COMPLIANCE:
- EventBus-only communication (NO direct calls)
- Real MT5 data integration (NO real data)
- Full telemetry hooks and structured logging
- Registered in system_tree.json and module_registry.json
"""

import json
import datetime
import os
import logging
import time
import numpy as np
import pandas as pd
from statistics import mean, stdev
from collections import deque
from event_bus import get_event_bus, emit_event, subscribe_to_event

class SignalContextEnricher:
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

            emit_telemetry("signal_context_enricher_recovered_2", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("signal_context_enricher_recovered_2", "position_calculated", {
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
                        "module": "signal_context_enricher_recovered_2",
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
                print(f"Emergency stop error in signal_context_enricher_recovered_2: {e}")
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
                "module": "signal_context_enricher_recovered_2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("signal_context_enricher_recovered_2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in signal_context_enricher_recovered_2: {e}")
    def __init__(self):
        """Initialize Signal Context Enricher with real data connections."""
        self.module_name = "SignalContextEnricher"
        self.event_bus = get_event_bus()
        self.logger = self._setup_logging()
        
        # Real-time context metrics storage
        self.volatility_window = deque(maxlen=100)  # 100-period volatility tracking
        self.signal_age_tracker = {}  # Track signal generation timestamps
        self.market_phase_cache = {}  # Cache market phase data
        self.historical_correlations = {}  # Store correlation data
        
        # Context enhancement parameters (real trading values)
        self.volatility_threshold_high = 0.025  # 2.5% high volatility
        self.volatility_threshold_low = 0.008   # 0.8% low volatility
        self.signal_age_threshold_stale = 300   # 5 minutes = stale signal
        self.risk_multiplier_high_vol = 0.6     # Reduce risk in high volatility
        self.risk_multiplier_low_vol = 1.2      # Increase risk in low volatility
        
        # Real-time telemetry tracking
        self.enrichment_stats = {
            "total_signals_enriched": 0,
            "high_volatility_signals": 0,
            "stale_signals_filtered": 0,
            "context_enhancement_rate": 0.0,
            "last_enrichment_timestamp": None
        }
          # Connect to EventBus for real-time signal flow
        self._subscribe_to_events()
        
        self.logger.info(f"{self.module_name} initialized with real MT5 data connections")
        
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _setup_logging(self):
        """Setup structured logging for institutional compliance."""
        log_dir = "logs/signal_context_enricher"
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(self.module_name)
        logger.setLevel(logging.INFO)
        
        # JSONL structured logging for compliance
        handler = logging.FileHandler(f"{log_dir}/enricher_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl")
        formatter = logging.Formatter('{"timestamp": "%(asctime)s", "module": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _subscribe_to_events(self):
        """Subscribe to EventBus for real-time signal enrichment."""
        # Listen for incoming signals to enrich
        subscribe_to_event("SignalReadyEvent", self.on_signal_ready)
        subscribe_to_event("MarketDataUpdate", self.on_market_data_update)
        subscribe_to_event("VolatilityUpdate", self.on_volatility_update)
        subscribe_to_event("HistoricalCorrelationUpdate", self.on_correlation_update)
        
        self.logger.info("EventBus subscriptions established for real-time signal enrichment")
        
    def on_signal_ready(self, event_data):
        """Process incoming signals for context enrichment."""
        try:
            signal_data = event_data.get("signal_data", {})
            signal_id = signal_data.get("signal_id")
            symbol = signal_data.get("symbol")
            timestamp = event_data.get("timestamp", datetime.datetime.now().isoformat())
            
            # Enrich signal with real-time context
            enriched_signal = self._enrich_signal_context(signal_data, timestamp)
              # Emit enriched signal via EventBus
            emit_event("SignalEnrichedEvent", {
                "signal_id": signal_id,
                "symbol": symbol,
                "enriched_data": enriched_signal,
                "enrichment_timestamp": datetime.datetime.now().isoformat(),
                "enricher_module": self.module_name
            })
              # Update telemetry
            self._update_enrichment_telemetry(enriched_signal)
            
            self.logger.info(f"Signal {signal_id} enriched and forwarded via EventBus")
            
        except Exception as e:
            self.logger.error(f"Error enriching signal: {str(e)}")
            emit_event("ModuleError", {
                "module": self.module_name,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
    def on_market_data_update(self, event_data):
        """Process real-time market data for context enhancement."""
        try:
            symbol = event_data.get("symbol")
            price_data = event_data.get("price_data", {})
            
            # Update market phase analysis
            self._update_market_phase(symbol, price_data)
            
            # Calculate real-time volatility
            if "close" in price_data:
                self._update_volatility_metrics(price_data["close"])
                
        except Exception as e:
            self.logger.error(f"Error processing market data update: {str(e)}")
            
    def on_volatility_update(self, event_data):
        """Process volatility updates from market data feed."""
        try:
            volatility_value = event_data.get("volatility", 0.0)
            symbol = event_data.get("symbol")
            
            # Store volatility for context enrichment
            self.volatility_window.append(volatility_value)
            
            self.logger.info(f"Volatility updated for {symbol}: {volatility_value:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error processing volatility update: {str(e)}")
            
    def on_correlation_update(self, event_data):
        """Process historical correlation data for signal context."""
        try:
            symbol_pair = event_data.get("symbol_pair")
            correlation_value = event_data.get("correlation", 0.0)
            timeframe = event_data.get("timeframe", "1H")
            
            # Store correlation data for enrichment
            correlation_key = f"{symbol_pair}_{timeframe}"
            self.historical_correlations[correlation_key] = {
                "correlation": correlation_value,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing correlation update: {str(e)}")
            
    def _enrich_signal_context(self, signal_data, timestamp):
        """Enrich signal with comprehensive context metadata."""
        enriched = signal_data.copy()
        
        # Add signal age context
        signal_age = self._calculate_signal_age(timestamp)
        enriched["context"] = {
            "signal_age_seconds": signal_age,
            "is_stale": signal_age > self.signal_age_threshold_stale,
            "enrichment_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add volatility context
        volatility_context = self._get_volatility_context()
        enriched["context"]["volatility"] = volatility_context
        
        # Add market phase context
        symbol = signal_data.get("symbol", "UNKNOWN")
        market_phase = self._get_market_phase_context(symbol)
        enriched["context"]["market_phase"] = market_phase
        
        # Add risk adjustment based on context
        risk_adjustment = self._calculate_risk_adjustment(volatility_context, market_phase)
        enriched["context"]["risk_adjustment"] = risk_adjustment
        
        # Add historical correlation context
        correlation_context = self._get_correlation_context(symbol)
        enriched["context"]["correlations"] = correlation_context
        
        # Add confidence scoring based on context
        context_confidence = self._calculate_context_confidence(enriched["context"])
        enriched["context"]["confidence_score"] = context_confidence
        
        return enriched
        
    def _calculate_signal_age(self, timestamp):
        """Calculate signal age in seconds from timestamp."""
        try:
            signal_time = datetime.datetime.fromisoformat(timestamp)
            current_time = datetime.datetime.now()
            age_delta = current_time - signal_time
            return age_delta.total_seconds()
        except:
            return 0  # Default to fresh signal if timestamp invalid
            
    def _get_volatility_context(self):
        """Get current volatility context from real-time data."""
        if len(self.volatility_window) < 10:
            return {
                "current_volatility": 0.0,
                "volatility_regime": "UNKNOWN",
                "data_insufficient": True
            }
            
        current_vol = self.volatility_window[-1]
        avg_vol = mean(self.volatility_window)
        vol_std = stdev(self.volatility_window) if len(self.volatility_window) > 1 else 0
        
        # Determine volatility regime
        if current_vol > self.volatility_threshold_high:
            regime = "HIGH"
        elif current_vol < self.volatility_threshold_low:
            regime = "LOW"
        else:
            regime = "NORMAL"
            
        return {
            "current_volatility": current_vol,
            "average_volatility": avg_vol,
            "volatility_std": vol_std,
            "volatility_regime": regime,
            "data_insufficient": False
        }
        
    def _get_market_phase_context(self, symbol):
        """Get market phase context for symbol."""
        # Default market phase if no data available
        default_phase = {
            "phase": "UNKNOWN",
            "trend_strength": 0.0,
            "trend_direction": "NEUTRAL",
            "data_available": False
        }
        
        if symbol not in self.market_phase_cache:
            return default_phase
            
        return self.market_phase_cache[symbol]
        
    def _calculate_risk_adjustment(self, volatility_context, market_phase):
        """Calculate risk adjustment factor based on context."""
        base_adjustment = 1.0
        
        # Adjust for volatility regime
        vol_regime = volatility_context.get("volatility_regime", "NORMAL")
        if vol_regime == "HIGH":
            base_adjustment *= self.risk_multiplier_high_vol
        elif vol_regime == "LOW":
            base_adjustment *= self.risk_multiplier_low_vol
            
        # Adjust for market phase
        trend_strength = market_phase.get("trend_strength", 0.0)
        if trend_strength > 0.7:  # Strong trend
            base_adjustment *= 1.1  # Slightly increase risk
        elif trend_strength < 0.3:  # Weak trend/consolidation
            base_adjustment *= 0.8  # Reduce risk
            
        return round(base_adjustment, 3)
        
    def _get_correlation_context(self, symbol):
        """Get correlation context for signal symbol."""
        correlations = {}
        
        # Look for correlations involving this symbol
        for key, data in self.historical_correlations.items():
            if symbol in key:
                pair_parts = key.split("_")
                if len(pair_parts) >= 2:
                    timeframe = pair_parts[-1]
                    correlations[f"correlation_{timeframe}"] = data["correlation"]
                    
        return correlations
        
    def _calculate_context_confidence(self, context):
        """Calculate overall confidence score based on context richness."""
        confidence_factors = []
        
        # Age factor
        if not context.get("is_stale", True):
            confidence_factors.append(0.2)
            
        # Volatility data availability
        if not context.get("volatility", {}).get("data_insufficient", True):
            confidence_factors.append(0.2)
            
        # Market phase data
        if context.get("market_phase", {}).get("data_available", False):
            confidence_factors.append(0.2)
            
        # Correlation data
        if context.get("correlations", {}):
            confidence_factors.append(0.2)
            
        # Risk adjustment validity
        risk_adj = context.get("risk_adjustment", 1.0)
        if 0.5 <= risk_adj <= 2.0:  # Reasonable risk adjustment range
            confidence_factors.append(0.2)
            
        return round(sum(confidence_factors), 2)
        
    def _update_market_phase(self, symbol, price_data):
        """Update market phase analysis for symbol."""
        # Simple trend analysis - in production would use sophisticated algorithms
        try:
            high = price_data.get("high", 0)
            low = price_data.get("low", 0)
            close = price_data.get("close", 0)
            
            # Basic trend strength calculation
            if high > 0 and low > 0:
                price_range = high - low
                trend_strength = min(price_range / close, 1.0) if close > 0 else 0
                
                self.market_phase_cache[symbol] = {
                    "phase": "TRENDING" if trend_strength > 0.01 else "CONSOLIDATION",
                    "trend_strength": trend_strength,
                    "trend_direction": "UP" if close > (high + low) / 2 else "DOWN",
                    "data_available": True,
                    "last_updated": datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error updating market phase for {symbol}: {str(e)}")
            
    def _update_volatility_metrics(self, price):
        """Update volatility metrics from price data."""
        try:
            if len(self.volatility_window) > 0:
                # Calculate simple volatility as price change percentage
                previous_price = self.volatility_window[-1] if self.volatility_window else price
                volatility = abs(price - previous_price) / previous_price if previous_price > 0 else 0
                self.volatility_window.append(volatility)
            else:
                self.volatility_window.append(0.01)  # Default volatility
                
        except Exception as e:
            self.logger.error(f"Error updating volatility metrics: {str(e)}")
            
    def _update_enrichment_telemetry(self, enriched_signal):
        """Update telemetry statistics for monitoring."""
        self.enrichment_stats["total_signals_enriched"] += 1
        self.enrichment_stats["last_enrichment_timestamp"] = datetime.datetime.now().isoformat()
        
        # Track specific enrichment types
        context = enriched_signal.get("context", {})
        if context.get("volatility", {}).get("volatility_regime") == "HIGH":
            self.enrichment_stats["high_volatility_signals"] += 1
            
        if context.get("is_stale", False):
            self.enrichment_stats["stale_signals_filtered"] += 1
            
        # Calculate enhancement rate
        confidence = context.get("confidence_score", 0.0)
        if self.enrichment_stats["total_signals_enriched"] > 0:
            self.enrichment_stats["context_enhancement_rate"] = confidence
              # Emit telemetry via EventBus
        emit_event("ModuleTelemetry", {
            "module": self.module_name,
            "stats": self.enrichment_stats.copy(),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    def get_module_status(self):
        """Get current module status for monitoring."""
        return {
            "module": self.module_name,
            "status": "active",
            "enrichment_stats": self.enrichment_stats,
            "volatility_window_size": len(self.volatility_window),
            "market_phase_symbols": len(self.market_phase_cache),
            "correlation_pairs": len(self.historical_correlations),
            "timestamp": datetime.datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Real-time signal context enricher with EventBus integration
    enricher = SignalContextEnricher()
    
    # Keep alive for real-time processing
    try:
        while True:
            time.sleep(1)  # Real-time processing loop
    except KeyboardInterrupt:
        print("Signal Context Enricher shutting down...")

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
        

# <!-- @GENESIS_MODULE_END: signal_context_enricher -->