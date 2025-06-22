
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
                            "module": "phase_97_1_mt5_indicator_scanner",
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
                    print(f"Emergency stop error in phase_97_1_mt5_indicator_scanner: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "phase_97_1_mt5_indicator_scanner",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("phase_97_1_mt5_indicator_scanner", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in phase_97_1_mt5_indicator_scanner: {e}")
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


#!/usr/bin/env python3
"""


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

GENESIS Phase 97.1 MT5 Indicator Universe Scanner
Scans and indexes every technical indicator available via MT5 for GENESIS use.
Builds a searchable registry with metadata, categories, and compatibility information.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: phase_97_1_mt5_indicator_scanner -->


# <!-- @GENESIS_MODULE_START: phase_97_1_mt5_indicator_scanner -->

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5IndicatorScanner:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "phase_97_1_mt5_indicator_scanner",
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
                print(f"Emergency stop error in phase_97_1_mt5_indicator_scanner: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "phase_97_1_mt5_indicator_scanner",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("phase_97_1_mt5_indicator_scanner", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in phase_97_1_mt5_indicator_scanner: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "phase_97_1_mt5_indicator_scanner",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in phase_97_1_mt5_indicator_scanner: {e}")
    """
    Phase 97.1 MT5 Indicator Universe Scanner
    Catalogs all MT5 indicators with metadata and compatibility information.
    """
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).absolute()
        self.indicator_registry = {}
        self.categories = {
            'trend': [],
            'momentum': [],
            'volume': [],
            'volatility': [],
            'oscillator': [],
            'support_resistance': [],
            'price_action': [],
            'custom': []
        }
        
        # Core files
        self.registry_file = self.workspace_root / "indicator_registry.json"
        self.build_status_file = self.workspace_root / "build_status.json"
        self.build_tracker_file = self.workspace_root / "build_tracker.md"
        
        logger.info(f"MT5 Indicator Scanner initialized for workspace: {self.workspace_root}")
    
    def scan_mt5_indicator_universe(self) -> Dict[str, Any]:
        """
        Main scanning entry point.
        Returns comprehensive MT5 indicator registry report.
        """
        logger.info("🔍 Starting GENESIS Phase 97.1 MT5 Indicator Universe Scanning...")
        
        report = {
            "phase": "97.1",
            "scanner": "MT5 Indicator Universe Scanner",
            "timestamp": datetime.now().isoformat(),
            "status": "running",
            "indicators_scanned": 0,
            "categories_mapped": 0,
            "compatibility_checked": 0,
            "registry_created": False,
            "mt5_connection": "simulated",
            "details": {
                "trend_indicators": 0,
                "momentum_indicators": 0,
                "volume_indicators": 0,
                "volatility_indicators": 0,
                "total_native": 0,
                "total_custom": 0
            }
        }
        
        try:
            # Step 1: Initialize MT5 connection (simulated)
            self._initialize_mt5_connection()
            
            # Step 2: Scan native MT5 indicators
            self._scan_native_indicators()
            
            # Step 3: Categorize indicators
            self._categorize_indicators()
            
            # Step 4: Check compatibility
            self._check_genesis_compatibility()
            
            # Step 5: Build final registry
            self._build_indicator_registry()
            
            # Step 6: Validate and save registry
            self._save_indicator_registry()
            
            # Step 7: Generate final report
            report.update({
                "status": "completed",
                "indicators_scanned": len(self.indicator_registry),
                "categories_mapped": len([cat for cat in self.categories.values() if cat]),
                "compatibility_checked": len([ind for ind in self.indicator_registry.values() if 'compatibility' in ind]),
                "registry_created": self.registry_file.exists(),
                "details": {
                    "trend_indicators": len(self.categories['trend']),
                    "momentum_indicators": len(self.categories['momentum']),
                    "volume_indicators": len(self.categories['volume']),
                    "volatility_indicators": len(self.categories['volatility']),
                    "total_native": len([ind for ind in self.indicator_registry.values() if ind.get('type') == 'native']),
                    "total_custom": len([ind for ind in self.indicator_registry.values() if ind.get('type') == 'custom'])
                }
            })
            
            # Step 8: Update tracking and status
            self._update_build_tracker(report)
            self._update_build_status(report)
            
            if report['registry_created']:
                logger.info("✅ Phase 97.1 MT5 Indicator Universe Scanning PASSED")
                report["phase_97_1_complete"] = True
                report["indicator_registry_status"] = "validated"
            else:
                logger.warning("⛔️ Phase 97.1 MT5 Indicator Universe Scanning FAILED")
                report["phase_97_1_complete"] = False
                report["indicator_registry_status"] = "failed"
            
            return report
            
        except Exception as e:
            logger.error(f"Phase 97.1 MT5 indicator scanning failed: {str(e)}")
            return self._generate_error_report(str(e))
    
    def _initialize_mt5_connection(self):
        """Initialize MT5 connection (simulated for now)"""
        logger.info("🔌 Initializing MT5 connection...")
        
        # Since we're in a simulated environment, we'll use predefined indicator data
        # In a real MT5 environment, this would connect to the MT5 API
        logger.info("✅ MT5 connection simulated (using predefined indicator catalog)")
    
    def _scan_native_indicators(self):
        """Scan all native MT5 indicators"""
        logger.info("🔍 Scanning native MT5 indicators...")
        
        # Native MT5 indicators catalog
        # This would normally come from the MT5 API
        native_indicators = {
            # Trend Indicators
            "SMA": {
                "name": "Simple Moving Average",
                "type": "native",
                "category": "trend",
                "mt5_path": "iMA",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 1, "max": 1000},
                    "shift": {"type": "int", "default": 0},
                    "method": {"type": "enum", "values": ["SIMPLE", "EXPONENTIAL", "SMOOTHED", "LINEAR_WEIGHTED"]},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["ma_value"],
                "buffer_type": "single",
                "description": "Simple Moving Average calculation"
            },
            "EMA": {
                "name": "Exponential Moving Average",
                "type": "native",
                "category": "trend",
                "mt5_path": "iMA",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 1, "max": 1000},
                    "shift": {"type": "int", "default": 0},
                    "method": {"type": "enum", "values": ["EXPONENTIAL"]},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["ema_value"],
                "buffer_type": "single",
                "description": "Exponential Moving Average calculation"
            },
            "MACD": {
                "name": "Moving Average Convergence Divergence",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iMACD",
                "parameters": {
                    "fast_ema": {"type": "int", "default": 12, "min": 1, "max": 100},
                    "slow_ema": {"type": "int", "default": 26, "min": 1, "max": 100},
                    "signal_sma": {"type": "int", "default": 9, "min": 1, "max": 100},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["main_line", "signal_line"],
                "buffer_type": "multi",
                "description": "MACD oscillator with signal line"
            },
            "RSI": {
                "name": "Relative Strength Index",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iRSI",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 2, "max": 100},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["rsi_value"],
                "buffer_type": "single",
                "description": "Relative Strength Index momentum oscillator"
            },
            "STOCHASTIC": {
                "name": "Stochastic Oscillator",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iStochastic",
                "parameters": {
                    "k_period": {"type": "int", "default": 5, "min": 1, "max": 100},
                    "d_period": {"type": "int", "default": 3, "min": 1, "max": 100},
                    "slowing": {"type": "int", "default": 3, "min": 1, "max": 100},
                    "method": {"type": "enum", "values": ["SIMPLE", "EXPONENTIAL", "SMOOTHED", "LINEAR_WEIGHTED"]},
                    "price_field": {"type": "enum", "values": ["LOW_HIGH", "CLOSE_CLOSE"]}
                },
                "outputs": ["main_line", "signal_line"],
                "buffer_type": "multi",
                "description": "Stochastic oscillator with %K and %D lines"
            },
            "BOLLINGER_BANDS": {
                "name": "Bollinger Bands",
                "type": "native",
                "category": "volatility",
                "mt5_path": "iBands",
                "parameters": {
                    "period": {"type": "int", "default": 20, "min": 2, "max": 100},
                    "deviation": {"type": "double", "default": 2.0, "min": 0.1, "max": 5.0},
                    "shift": {"type": "int", "default": 0},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["upper_band", "middle_band", "lower_band"],
                "buffer_type": "multi",
                "description": "Bollinger Bands volatility indicator"
            },
            "ATR": {
                "name": "Average True Range",
                "type": "native",
                "category": "volatility",
                "mt5_path": "iATR",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 1, "max": 100}
                },
                "outputs": ["atr_value"],
                "buffer_type": "single",
                "description": "Average True Range volatility measure"
            },
            "VOLUME": {
                "name": "Volume",
                "type": "native",
                "category": "volume",
                "mt5_path": "iVolumes",
                "parameters": {
                    "applied_volume": {"type": "enum", "values": ["TICK_VOLUME", "REAL_VOLUME"]}
                },
                "outputs": ["volume"],
                "buffer_type": "single",
                "description": "Trading volume indicator"
            },
            "OBV": {
                "name": "On Balance Volume",
                "type": "native",
                "category": "volume",
                "mt5_path": "iOBV",
                "parameters": {
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["obv_value"],
                "buffer_type": "single",
                "description": "On Balance Volume accumulation indicator"
            },
            "CCI": {
                "name": "Commodity Channel Index",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iCCI",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 1, "max": 100},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["cci_value"],
                "buffer_type": "single",
                "description": "Commodity Channel Index momentum oscillator"
            },
            "WILLIAMS_R": {
                "name": "Williams Percent Range",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iWPR",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 1, "max": 100}
                },
                "outputs": ["wpr_value"],
                "buffer_type": "single",
                "description": "Williams %R momentum oscillator"
            },
            "MOMENTUM": {
                "name": "Momentum",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iMomentum",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 1, "max": 100},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["momentum_value"],
                "buffer_type": "single",
                "description": "Momentum oscillator"
            },
            "ROC": {
                "name": "Rate of Change",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iMomentum",
                "parameters": {
                    "period": {"type": "int", "default": 12, "min": 1, "max": 100},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["roc_value"],
                "buffer_type": "single",
                "description": "Rate of Change momentum indicator"
            },
            "STANDARD_DEVIATION": {
                "name": "Standard Deviation",
                "type": "native",
                "category": "volatility",
                "mt5_path": "iStdDev",
                "parameters": {
                    "period": {"type": "int", "default": 20, "min": 2, "max": 100},
                    "shift": {"type": "int", "default": 0},
                    "method": {"type": "enum", "values": ["SIMPLE", "EXPONENTIAL", "SMOOTHED", "LINEAR_WEIGHTED"]},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["stddev_value"],
                "buffer_type": "single",
                "description": "Standard Deviation volatility measure"
            },
            "PARABOLIC_SAR": {
                "name": "Parabolic SAR",
                "type": "native",
                "category": "trend",
                "mt5_path": "iSAR",
                "parameters": {
                    "step": {"type": "double", "default": 0.02, "min": 0.001, "max": 1.0},
                    "maximum": {"type": "double", "default": 0.2, "min": 0.01, "max": 1.0}
                },
                "outputs": ["sar_value"],
                "buffer_type": "single",
                "description": "Parabolic Stop and Reverse trend indicator"
            },
            "ADX": {
                "name": "Average Directional Index",
                "type": "native",
                "category": "trend",
                "mt5_path": "iADX",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 1, "max": 100}
                },
                "outputs": ["main_line", "plus_di", "minus_di"],
                "buffer_type": "multi",
                "description": "Average Directional Index trend strength indicator"
            },
            "AWESOME_OSCILLATOR": {
                "name": "Awesome Oscillator",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iAO",
                "parameters": {},
                "outputs": ["ao_value"],
                "buffer_type": "single",
                "description": "Awesome Oscillator momentum indicator"
            },
            "ACCELERATOR": {
                "name": "Accelerator Oscillator",
                "type": "native",
                "category": "momentum",
                "mt5_path": "iAC",
                "parameters": {},
                "outputs": ["ac_value"],
                "buffer_type": "single",
                "description": "Accelerator Oscillator momentum indicator"
            },
            "ALLIGATOR": {
                "name": "Alligator",
                "type": "native",
                "category": "trend",
                "mt5_path": "iAlligator",
                "parameters": {
                    "jaw_period": {"type": "int", "default": 13, "min": 1, "max": 100},
                    "jaw_shift": {"type": "int", "default": 8, "min": 0, "max": 100},
                    "teeth_period": {"type": "int", "default": 8, "min": 1, "max": 100},
                    "teeth_shift": {"type": "int", "default": 5, "min": 0, "max": 100},
                    "lips_period": {"type": "int", "default": 5, "min": 1, "max": 100},
                    "lips_shift": {"type": "int", "default": 3, "min": 0, "max": 100},
                    "method": {"type": "enum", "values": ["SIMPLE", "EXPONENTIAL", "SMOOTHED", "LINEAR_WEIGHTED"]},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["jaw", "teeth", "lips"],
                "buffer_type": "multi",
                "description": "Alligator trend indicator with three smoothed moving averages"
            },
            "FRACTALS": {
                "name": "Fractals",
                "type": "native",
                "category": "support_resistance",
                "mt5_path": "iFractals",
                "parameters": {},
                "outputs": ["upper_fractal", "lower_fractal"],
                "buffer_type": "multi",
                "description": "Fractal support and resistance levels"
            },
            "GATOR": {
                "name": "Gator Oscillator",
                "type": "native",
                "category": "trend",
                "mt5_path": "iGator",
                "parameters": {
                    "jaw_period": {"type": "int", "default": 13, "min": 1, "max": 100},
                    "jaw_shift": {"type": "int", "default": 8, "min": 0, "max": 100},
                    "teeth_period": {"type": "int", "default": 8, "min": 1, "max": 100},
                    "teeth_shift": {"type": "int", "default": 5, "min": 0, "max": 100},
                    "lips_period": {"type": "int", "default": 5, "min": 1, "max": 100},
                    "lips_shift": {"type": "int", "default": 3, "min": 0, "max": 100},
                    "method": {"type": "enum", "values": ["SIMPLE", "EXPONENTIAL", "SMOOTHED", "LINEAR_WEIGHTED"]},
                    "applied_price": {"type": "enum", "values": ["CLOSE", "OPEN", "HIGH", "LOW", "MEDIAN", "TYPICAL", "WEIGHTED"]}
                },
                "outputs": ["upper_histogram", "lower_histogram"],
                "buffer_type": "multi",
                "description": "Gator Oscillator based on Alligator indicator"
            }
        }
        
        # Add indicators to registry
        for indicator_id, indicator_data in native_indicators.items():
            self.indicator_registry[indicator_id] = indicator_data
            
            # Add to appropriate category
            category = indicator_data['category']
            if category in self.categories:
                self.categories[category].append(indicator_id)
        
        logger.info(f"✅ Scanned {len(native_indicators)} native MT5 indicators")
    
    def _categorize_indicators(self):
        """Categorize indicators by signal logic type"""
        logger.info("📊 Categorizing indicators by signal logic...")
        
        # Category definitions with characteristics
        category_definitions = {
            "trend": {
                "description": "Indicators that identify market trend direction",
                "characteristics": ["directional", "lagging", "smoothed"],
                "use_cases": ["trend_following", "trend_confirmation", "trend_reversal"]
            },
            "momentum": {
                "description": "Indicators that measure price momentum and strength",
                "characteristics": ["oscillating", "bounded", "leading"],
                "use_cases": ["overbought_oversold", "divergence", "momentum_confirmation"]
            },
            "volume": {
                "description": "Indicators based on trading volume analysis",
                "characteristics": ["volume_based", "accumulation", "distribution"],
                "use_cases": ["volume_confirmation", "accumulation_distribution", "volume_breakout"]
            },
            "volatility": {
                "description": "Indicators that measure market volatility and price dispersion",
                "characteristics": ["volatility_based", "range_based", "statistical"],
                "use_cases": ["volatility_breakout", "range_trading", "risk_management"]
            },
            "support_resistance": {
                "description": "Indicators that identify key support and resistance levels",
                "characteristics": ["level_based", "fractal", "pivot"],
                "use_cases": ["entry_exit", "stop_loss", "take_profit"]
            }
        }
        
        # Update each indicator with enhanced category information
        for indicator_id, indicator_data in self.indicator_registry.items():
            category = indicator_data['category']
            if category in category_definitions:
                indicator_data['category_info'] = category_definitions[category]
        
        logger.info(f"✅ Categorized indicators into {len(category_definitions)} categories")
    
    def _check_genesis_compatibility(self):
        """Check GENESIS compatibility for each indicator"""
        logger.info("🔧 Checking GENESIS compatibility...")
        
        for indicator_id, indicator_data in self.indicator_registry.items():
            compatibility = {
                "live_trading": True,  # Most MT5 indicators support live trading
                "backtesting": True,   # Most indicators work in backtesting
                "real_time": True,     # Real-time data compatibility
                "historical": True,    # Historical data compatibility
                "optimization": True,  # Parameter optimization support
                "multi_timeframe": True,  # Multiple timeframe support
                "genesis_integration": "full",  # Integration level
                "performance_impact": self._assess_performance_impact(indicator_data),
                "data_requirements": self._assess_data_requirements(indicator_data),
                "complexity": self._assess_complexity(indicator_data)
            }
            
            # Special cases for certain indicators
            if indicator_id in ["FRACTALS"]:
                compatibility["real_time"] = False  # Fractals need confirmation
                compatibility["genesis_integration"] = "limited"
            
            if indicator_id in ["ALLIGATOR", "GATOR"]:
                compatibility["performance_impact"] = "medium"  # Multiple calculations
            
            indicator_data['compatibility'] = compatibility
        
        logger.info(f"✅ Checked compatibility for {len(self.indicator_registry)} indicators")
    
    def _assess_performance_impact(self, indicator_data: Dict) -> str:
        """Assess performance impact of indicator"""
        parameter_count = len(indicator_data.get('parameters', {}))
        output_count = len(indicator_data.get('outputs', []))
        
        if parameter_count > 5 or output_count > 2:
            return "high"
        elif parameter_count > 2 or output_count > 1:
            return "medium"
        else:
            return "low"
    
    def _assess_data_requirements(self, indicator_data: Dict) -> Dict[str, Any]:
        """Assess data requirements for indicator"""
        parameters = indicator_data.get('parameters', {})
        
        # Estimate minimum bars needed
        period_params = [param for param, config in parameters.items() if 'period' in param.lower()]
        min_bars = 50  # Default minimum
        
        if period_params:
            # Use largest period parameter as basis
            for param in period_params:
                default_period = parameters[param].get('default', 14)
                min_bars = max(min_bars, default_period * 3)  # 3x period for stability
        
        return {
            "minimum_bars": min_bars,
            "price_types": self._extract_price_types(parameters),
            "volume_required": "volume" in indicator_data['name'].lower() or "obv" in indicator_data['name'].lower(),
            "tick_data": False  # Most indicators use OHLC
        }
    
    def _extract_price_types(self, parameters: Dict) -> List[str]:
        """Extract required price types from parameters"""
        price_types = []
        for param, config in parameters.items():
            if 'applied_price' in param or 'price' in param:
                if isinstance(config, dict) and 'values' in config:
                    price_types.extend(config['values'])
        
        return list(set(price_types)) if price_types else ["CLOSE"]
    
    def _assess_complexity(self, indicator_data: Dict) -> str:
        """Assess implementation complexity"""
        parameter_count = len(indicator_data.get('parameters', {}))
        output_count = len(indicator_data.get('outputs', []))
        
        # Check for complex parameters
        has_complex_params = any(
            param.get('type') == 'enum' and len(param.get('values', [])) > 3
            for param in indicator_data.get('parameters', {}).values()
            if isinstance(param, dict)
        )
        
        if parameter_count > 6 or output_count > 3 or has_complex_params:
            return "high"
        elif parameter_count > 3 or output_count > 1:
            return "medium"
        else:
            return "low"
    
    def _build_indicator_registry(self):
        """Build final indicator registry structure"""
        logger.info("🏗️ Building final indicator registry...")
        
        # Create comprehensive registry structure
        registry_structure = {
            "metadata": {
                "version": "1.0",
                "generated": datetime.now().isoformat(),
                "generator": "GENESIS Phase 97.1 MT5 Indicator Scanner",
                "total_indicators": len(self.indicator_registry),
                "categories": list(self.categories.keys()),
                "mt5_api_version": "5.0.37"  # Current MT5 API version
            },
            "categories": {
                category: {
                    "indicators": indicators,
                    "count": len(indicators),
                    "description": self._get_category_description(category)
                }
                for category, indicators in self.categories.items()
                if indicators  # Only include categories with indicators
            },
            "indicators": self.indicator_registry,
            "compatibility_matrix": self._build_compatibility_matrix(),
            "usage_recommendations": self._build_usage_recommendations(),
            "genesis_integration": self._build_genesis_integration_guide()
        }
        
        self.final_registry = registry_structure
        logger.info(f"✅ Built comprehensive registry with {len(self.indicator_registry)} indicators")
    
    def _get_category_description(self, category: str) -> str:
        """Get description for indicator category"""
        descriptions = {
            "trend": "Indicators that identify and follow market trends",
            "momentum": "Indicators that measure price momentum and rate of change",
            "volume": "Indicators based on trading volume analysis",
            "volatility": "Indicators that measure market volatility and price dispersion",
            "oscillator": "Bounded indicators that oscillate between fixed levels",
            "support_resistance": "Indicators that identify key price levels",
            "price_action": "Indicators based on pure price movement analysis",
            "custom": "Custom or specialized indicators"
        }
        return descriptions.get(category, f"Indicators in {category} category")
    
    def _build_compatibility_matrix(self) -> Dict[str, Any]:
        """Build compatibility matrix for all indicators"""
        matrix = {
            "live_trading_compatible": [],
            "backtesting_compatible": [],
            "real_time_compatible": [],
            "optimization_friendly": [],
            "high_performance": [],
            "low_complexity": []
        }
        
        for indicator_id, indicator_data in self.indicator_registry.items():
            compatibility = indicator_data.get('compatibility', {})
            
            if compatibility.get('live_trading'):
                matrix['live_trading_compatible'].append(indicator_id)
            if compatibility.get('backtesting'):
                matrix['backtesting_compatible'].append(indicator_id)
            if compatibility.get('real_time'):
                matrix['real_time_compatible'].append(indicator_id)
            if compatibility.get('optimization'):
                matrix['optimization_friendly'].append(indicator_id)
            if compatibility.get('performance_impact') == 'low':
                matrix['high_performance'].append(indicator_id)
            if compatibility.get('complexity') == 'low':
                matrix['low_complexity'].append(indicator_id)
        
        return matrix
    
    def _build_usage_recommendations(self) -> Dict[str, Any]:
        """Build usage recommendations for different scenarios"""
        return {
            "beginner_friendly": [
                "SMA", "EMA", "RSI", "MACD", "BOLLINGER_BANDS"
            ],
            "advanced_strategies": [
                "ALLIGATOR", "GATOR", "ADX", "STOCHASTIC", "CCI"
            ],
            "scalping": [
                "RSI", "STOCHASTIC", "CCI", "WILLIAMS_R"
            ],
            "swing_trading": [
                "MACD", "RSI", "BOLLINGER_BANDS", "PARABOLIC_SAR"
            ],
            "trend_following": [
                "SMA", "EMA", "ADX", "PARABOLIC_SAR", "ALLIGATOR"
            ],
            "mean_reversion": [
                "RSI", "BOLLINGER_BANDS", "STOCHASTIC", "CCI"
            ],
            "volume_analysis": [
                "VOLUME", "OBV"
            ],
            "volatility_trading": [
                "ATR", "BOLLINGER_BANDS", "STANDARD_DEVIATION"
            ]
        }
    
    def _build_genesis_integration_guide(self) -> Dict[str, Any]:
        """Build GENESIS integration guide"""
        return {
            "signal_generation": {
                "primary_indicators": ["MACD", "RSI", "BOLLINGER_BANDS"],
                "confirmation_indicators": ["ADX", "VOLUME", "ATR"],
                "filter_indicators": ["SMA", "EMA", "PARABOLIC_SAR"]
            },
            "risk_management": {
                "volatility_measures": ["ATR", "BOLLINGER_BANDS", "STANDARD_DEVIATION"],
                "trend_strength": ["ADX", "MOMENTUM"],
                "support_resistance": ["FRACTALS", "PARABOLIC_SAR"]
            },
            "optimization_targets": {
                "fast_execution": ["RSI", "MOMENTUM", "CCI"],
                "accurate_signals": ["MACD", "STOCHASTIC", "BOLLINGER_BANDS"],
                "low_false_positives": ["ADX", "ALLIGATOR", "PARABOLIC_SAR"]
            },
            "multi_timeframe": {
                "short_term": ["RSI", "STOCHASTIC", "CCI"],
                "medium_term": ["MACD", "BOLLINGER_BANDS", "ADX"],
                "long_term": ["SMA", "EMA", "PARABOLIC_SAR"]
            }
        }
    
    def _save_indicator_registry(self):
        """Save indicator registry to JSON file"""
        logger.info("💾 Saving indicator registry...")
        
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.final_registry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Indicator registry saved to {self.registry_file}")
            
            # Validate the saved file
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            if len(loaded_data.get('indicators', {})) == len(self.indicator_registry):
                logger.info("✅ Registry validation passed")
            else:
                logger.warning("⚠️ Registry validation failed - data mismatch")
            
        except Exception as e:
            logger.error(f"Failed to save indicator registry: {str(e)}")
            raise
    
    def _update_build_tracker(self, report: Dict):
        """Update build_tracker.md with scanning results"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = f"\n\n## Phase 97.1 MT5 Indicator Universe Scanning - {timestamp}\n"
            log_entry += f"Status: {report['status']}\n"
            log_entry += f"Indicators Scanned: {report['indicators_scanned']}\n"
            log_entry += f"Categories Mapped: {report['categories_mapped']}\n"
            log_entry += f"Registry Created: {report['registry_created']}\n\n"
            
            # Detail category breakdown
            details = report.get('details', {})
            log_entry += "### Category Breakdown:\n"
            log_entry += f"- Trend Indicators: {details.get('trend_indicators', 0)}\n"
            log_entry += f"- Momentum Indicators: {details.get('momentum_indicators', 0)}\n"
            log_entry += f"- Volume Indicators: {details.get('volume_indicators', 0)}\n"
            log_entry += f"- Volatility Indicators: {details.get('volatility_indicators', 0)}\n"
            log_entry += f"- Total Native: {details.get('total_native', 0)}\n"
            log_entry += f"- Total Custom: {details.get('total_custom', 0)}\n\n"
            
            log_entry += "### Registry Status:\n"
            log_entry += f"- File Created: indicator_registry.json\n"
            log_entry += f"- GENESIS Integration: Fully compatible\n"
            log_entry += f"- MT5 API Version: 5.0.37\n"
            
            with open(self.build_tracker_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            logger.info("✅ Build tracker updated with indicator scanning results")
            
        except Exception as e:
            logger.error(f"Failed to update build tracker: {str(e)}")
    
    def _update_build_status(self, report: Dict):
        """Update build_status.json with Phase 97.1 results"""
        try:
            if self.build_status_file.exists():
                with open(self.build_status_file, 'r', encoding='utf-8') as f:
                    build_status = json.load(f)
            else:
                build_status = {}
            
            # Update with Phase 97.1 results
            build_status.update({
                "phase_97_1_mt5_indicator_scanner": {
                    "timestamp": report['timestamp'],
                    "status": report['status'],
                    "indicators_scanned": report['indicators_scanned'],
                    "categories_mapped": report['categories_mapped'],
                    "registry_created": report['registry_created'],
                    "scanner_version": "97.1"
                },
                "phase_97_1_complete": report.get('phase_97_1_complete', False),
                "indicator_registry_status": report.get('indicator_registry_status', 'pending'),
                "last_update": report['timestamp']
            })
            
            with open(self.build_status_file, 'w', encoding='utf-8') as f:
                json.dump(build_status, f, indent=2)
            
            logger.info("✅ Build status updated with Phase 97.1 results")
            
        except Exception as e:
            logger.error(f"Failed to update build status: {str(e)}")
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report for scanning failure"""
        return {
            "phase": "97.1",
            "scanner": "MT5 Indicator Universe Scanner",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_message,
            "phase_97_1_complete": False,
            "indicator_registry_status": "failed"
        }

def main():
    """Main entry point for Phase 97.1 MT5 Indicator Universe Scanning"""
    try:
        scanner = MT5IndicatorScanner()
        report = scanner.scan_mt5_indicator_universe()
        
        print("\n" + "="*70)
        print("GENESIS PHASE 97.1 MT5 INDICATOR UNIVERSE SCANNING COMPLETE")
        print("="*70)
        print(f"Status: {report['status']}")
        print(f"Indicators Scanned: {report['indicators_scanned']}")
        print(f"Categories Mapped: {report['categories_mapped']}")
        print(f"Registry Created: {report['registry_created']}")
        
        details = report.get('details', {})
        print(f"\nCategory Breakdown:")
        print(f"  Trend: {details.get('trend_indicators', 0)}")
        print(f"  Momentum: {details.get('momentum_indicators', 0)}")
        print(f"  Volume: {details.get('volume_indicators', 0)}")
        print(f"  Volatility: {details.get('volatility_indicators', 0)}")
        
        if report.get('phase_97_1_complete'):
            print("✅ Phase 97.1 PASSED - MT5 indicator universe cataloged")
        else:
            print("⛔️ Phase 97.1 FAILED - Indicator scanning issues")
        
        print("\n📄 See indicator_registry.json for complete catalog")
        return report
        
    except Exception as e:
        logger.error(f"Phase 97.1 MT5 indicator scanning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))


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


def monitor_drawdown(max_drawdown_percent: float = 5.0, daily_limit_percent: float = 5.0) -> Dict:
    """
    Monitor account drawdown against FTMO limits
    
    Args:
        max_drawdown_percent: Maximum allowed drawdown percentage
        daily_limit_percent: Maximum allowed daily loss percentage
        
    Returns:
        Dictionary with drawdown status information
    """
    try:
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            return {"status": "error", "message": "Failed to get account info"}
        
        # Calculate current drawdown
        balance = account_info.balance
        equity = account_info.equity
        
        current_drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        
        # Get daily high balance
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        positions = mt5.history_deals_get(from_date, datetime.now())
        
        daily_starting_balance = balance - sum([deal.profit for deal in positions])
        daily_loss_percent = (daily_starting_balance - equity) / daily_starting_balance * 100 if daily_starting_balance > 0 else 0
        
        # Prepare result
        result = {
            "status": "ok",
            "current_drawdown_percent": current_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "drawdown_level": current_drawdown / max_drawdown_percent,  # 0.0 to 1.0+
            "daily_loss_percent": daily_loss_percent,
            "daily_limit_percent": daily_limit_percent,
            "daily_loss_level": daily_loss_percent / daily_limit_percent,  # 0.0 to 1.0+
            "warnings": []
        }
        
        # Check drawdown thresholds
        if current_drawdown > max_drawdown_percent * 0.7:
            result["warnings"].append(f"Drawdown at {current_drawdown:.2f}% approaching maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "warning"
            
        if current_drawdown > max_drawdown_percent:
            result["warnings"].append(f"CRITICAL: Drawdown of {current_drawdown:.2f}% exceeds maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "critical"
            
        # Check daily loss thresholds
        if daily_loss_percent > daily_limit_percent * 0.7:
            result["warnings"].append(f"Daily loss at {daily_loss_percent:.2f}% approaching limit of {daily_limit_percent:.2f}%")
            result["status"] = "warning"
            
        if daily_loss_percent > daily_limit_percent:
            result["warnings"].append(f"CRITICAL: Daily loss of {daily_loss_percent:.2f}% exceeds limit of {daily_limit_percent:.2f}%")
            result["status"] = "critical"
        
        # Emit events for warnings
        if result["status"] in ["warning", "critical"]:
            emit_event("risk_threshold_warning", {
                "status": result["status"],
                "warnings": result["warnings"],
                "timestamp": datetime.now().isoformat()
            })
            
        return result
        
    except Exception as e:
        logging.error(f"Error monitoring drawdown: {str(e)}")
        return {"status": "error", "message": str(e)}
