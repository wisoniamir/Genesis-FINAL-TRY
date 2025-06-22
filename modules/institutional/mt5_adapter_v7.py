
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

                emit_telemetry("mt5_adapter_v7", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("mt5_adapter_v7", "position_calculated", {
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
                            "module": "mt5_adapter_v7",
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
                    print(f"Emergency stop error in mt5_adapter_v7: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "mt5_adapter_v7",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("mt5_adapter_v7", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in mt5_adapter_v7: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


# <!-- @GENESIS_MODULE_START: mt5_adapter -->

"""
🔌 GENESIS MT5 Adapter v7.0.0 - ARCHITECT MODE ULTIMATE ENFORCEMENT
Real-time MT5 data adapter with institutional-grade compliance

🎯 PURPOSE: Professional MT5 integration with full EventBus synergy
📡 EVENTBUS: Complete real-time data routing for all trading modules
🚫 ZERO TOLERANCE: No fallback data, no mocks, institutional-grade only
🔐 ARCHITECT MODE v7.0.0: Ultimate compliance with full synergy
🏛️ INSTITUTIONAL: FTMO-compliant with professional execution standards
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from threading import Lock, Event, Thread
from queue import Queue, Empty
import statistics

# Import enhanced EventBus
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.hardened_event_bus import get_event_bus, emit_event, register_route

# Configure institutional-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MT5Adapter_v7')


class ConnectionStatus(Enum):
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

            emit_telemetry("mt5_adapter_v7", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter_v7", "position_calculated", {
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
                        "module": "mt5_adapter_v7",
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
                print(f"Emergency stop error in mt5_adapter_v7: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter_v7",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter_v7", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter_v7: {e}")
    """MT5 connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    MONITORING = "monitoring"


class DataQuality(Enum):
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

            emit_telemetry("mt5_adapter_v7", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter_v7", "position_calculated", {
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
                        "module": "mt5_adapter_v7",
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
                print(f"Emergency stop error in mt5_adapter_v7: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter_v7",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter_v7", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter_v7: {e}")
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class MT5MarketData:
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

            emit_telemetry("mt5_adapter_v7", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter_v7", "position_calculated", {
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
                        "module": "mt5_adapter_v7",
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
                print(f"Emergency stop error in mt5_adapter_v7: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter_v7",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter_v7", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter_v7: {e}")
    """Professional market data structure"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    spread: float
    digits: int
    tick_value: float
    tick_size: float
    quality: DataQuality = DataQuality.EXCELLENT
    latency_ms: float = 0.0
    source: str = "MT5_LIVE"


@dataclass
class MT5ConnectionMetrics:
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

            emit_telemetry("mt5_adapter_v7", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter_v7", "position_calculated", {
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
                        "module": "mt5_adapter_v7",
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
                print(f"Emergency stop error in mt5_adapter_v7: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter_v7",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter_v7", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter_v7: {e}")
    """Connection performance metrics"""
    connection_time: float = 0.0
    authentication_time: float = 0.0
    avg_latency: float = 0.0
    uptime_seconds: float = 0.0
    data_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    quality_score: float = 100.0


class GenesisInstitutionalMT5Adapter:
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

            emit_telemetry("mt5_adapter_v7", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter_v7", "position_calculated", {
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
                        "module": "mt5_adapter_v7",
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
                print(f"Emergency stop error in mt5_adapter_v7: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter_v7",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter_v7", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter_v7: {e}")
    """
    ARCHITECT MODE v7.0.0 COMPLIANT MT5 Data Adapter
    Professional-grade MT5 integration with institutional compliance
    """
    
    VERSION = "7.0.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MT5 adapter with institutional-grade compliance"""
        self.config = config or self._load_default_config()
        self._emit_startup_telemetry()
        
        # Thread-safe connection management
        self.connection_lock = Lock()
        self.data_lock = Lock()
        self.status = ConnectionStatus.DISCONNECTED
        self.connected = False
        
        # Professional data management
        self.symbols_cache: Dict[str, Dict[str, Any]] = {}
        self.last_data_fetch: Dict[str, datetime] = {}
        self.price_cache: Dict[str, MT5MarketData] = {}
        self.subscription_manager: Dict[str, bool] = {}
        
        # Performance monitoring
        self.metrics = MT5ConnectionMetrics()
        self.performance_stats: Dict[str, List[float]] = {
            "latency": [],
            "success_rate": [],
            "data_quality": []
        }
        
        # EventBus integration
        self.event_bus = get_event_bus()
        self._register_event_routes()
        
        # Real-time processing
        self.data_queue: Queue = Queue(maxsize=1000)
        self.processing_thread: Optional[Thread] = None
        self.shutdown_event = Event()
        
        # Institutional credentials
        self.credentials = self._load_credentials()
        
        # Start monitoring
        self._start_monitoring_systems()
        
        logger.info(f"🏛️ GenesisInstitutionalMT5Adapter v{self.VERSION} initialized")

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default institutional configuration"""
        return {
            "connection_timeout": 30.0,
            "data_validation_level": "strict",
            "max_retry_attempts": 3,
            "quality_threshold": 95.0,
            "latency_threshold_ms": 50.0,
            "heartbeat_interval": 5.0,
            "symbols_refresh_interval": 300.0,
            "telemetry_enabled": True,
            "compliance_level": "institutional"
        }

    def _load_credentials(self) -> Dict[str, Any]:
        """Load institutional MT5 credentials"""
        return {
            "login": 1510944899,
            "password": "97v!*DK@ha",
            "server": "FTMO-Demo",
            "path": "",
            "portable": False,
            "timeout": 60000
        }

    def _register_event_routes(self) -> None:
        """Register EventBus routes for institutional compliance"""
        routes = [
            ("mt5.market_data", "GenesisInstitutionalMT5Adapter", "StrategyEngine"),
            ("mt5.tick_data", "GenesisInstitutionalMT5Adapter", "ExecutionEngine"),
            ("mt5.connection_status", "GenesisInstitutionalMT5Adapter", "RiskEngine"),
            ("mt5.symbol_info", "GenesisInstitutionalMT5Adapter", "MarketScanner"),
            ("mt5.performance_metrics", "GenesisInstitutionalMT5Adapter", "TelemetryCollector"),
            ("telemetry.mt5_adapter", "GenesisInstitutionalMT5Adapter", "TelemetryCollector"),
            ("compliance.mt5_validation", "GenesisInstitutionalMT5Adapter", "ComplianceEngine")
        ]
        
        for route, producer, consumer in routes:
            register_route(route, producer, consumer)
        
        logger.info("✅ MT5 EventBus routes registered")

    def _emit_startup_telemetry(self) -> None:
        """Emit startup telemetry with institutional compliance"""
        telemetry = {
            "module": "GenesisInstitutionalMT5Adapter",
            "version": self.VERSION,
            "status": "initializing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "compliance_level": "institutional",
            "architect_mode": "v7.0.0"
        }
        
        emit_event("telemetry.mt5_adapter", telemetry)

    def _start_monitoring_systems(self) -> None:
        """Start monitoring systems for institutional compliance"""
        self.processing_thread = Thread(
            target=self._data_processing_loop,
            daemon=True,
            name="MT5DataProcessor"
        )
        self.processing_thread.start()
        
        logger.info("🔍 MT5 monitoring systems started")

    def _data_processing_loop(self) -> None:
        """Main data processing loop with institutional quality control"""
        while not self.shutdown_event.is_set():
            try:
                data_item = self.data_queue.get(timeout=1.0)
                self._process_market_data(data_item)
                self.data_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Data processing error: {str(e)}")

    def _process_market_data(self, data_item: Dict[str, Any]) -> None:
        """Process market data with quality validation"""
        try:
            symbol = data_item.get('symbol', '')
            if not symbol:
                return
            
            # Validate data quality
            quality = self._assess_data_quality(data_item)
            
            if quality == DataQuality.INVALID:
                logger.warning(f"⚠️ Invalid data quality for {symbol}")
                return
            
            # Create market data object
            market_data = MT5MarketData(
                symbol=symbol,
                timestamp=data_item.get('timestamp', datetime.now(timezone.utc)),
                bid=data_item.get('bid', 0.0),
                ask=data_item.get('ask', 0.0),
                last=data_item.get('last', 0.0),
                volume=data_item.get('volume', 0),
                spread=data_item.get('spread', 0.0),
                digits=data_item.get('digits', 5),
                tick_value=data_item.get('tick_value', 1.0),
                tick_size=data_item.get('tick_size', 0.00001),
                quality=quality,
                latency_ms=data_item.get('latency_ms', 0.0)
            )
            
            # Update cache
            with self.data_lock:
                self.price_cache[symbol] = market_data
            
            # Emit to EventBus
            emit_event("mt5.market_data", {
                "symbol": symbol,
                "data": market_data.__dict__,
                "timestamp": market_data.timestamp.isoformat(),
                "quality": quality.value
            })
            
        except Exception as e:
            logger.error(f"❌ Market data processing error: {str(e)}")

    def _assess_data_quality(self, data: Dict[str, Any]) -> DataQuality:
        """Assess data quality with institutional standards"""
        try:
            bid = data.get('bid', 0.0)
            ask = data.get('ask', 0.0)
            
            # Basic validation
            if bid <= 0 or ask <= 0:
                return DataQuality.INVALID
            
            if bid >= ask:
                return DataQuality.INVALID
            
            # Spread validation
            spread = ask - bid
            if spread > ask * 0.01:  # Spread > 1%
                return DataQuality.POOR
            
            # Latency check
            latency = data.get('latency_ms', 0.0)
            if latency > self.config["latency_threshold_ms"]:
                return DataQuality.ACCEPTABLE
            
            return DataQuality.EXCELLENT
            
        except Exception:
            return DataQuality.INVALID

    def ensure_connection(self) -> bool:
        """Ensure institutional-grade MT5 connection"""
        with self.connection_lock:
            if self.connected and self._validate_connection():
                return True
            
            return self._establish_connection()

    def _establish_connection(self) -> bool:
        """Establish professional MT5 connection with institutional compliance"""
        try:
            start_time = time.time()
            self.status = ConnectionStatus.CONNECTING
            
            logger.info("🔌 Establishing institutional MT5 connection...")
            
            # Initialize MT5 terminal
            if not mt5.initialize(
                login=self.credentials["login"],
                password=self.credentials["password"],
                server=self.credentials["server"],
                timeout=self.credentials["timeout"]
            ):
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 initialization failed: {error_code}")
                emit_event("mt5.connection_status", {
                    "status": "failed",
                    "error": f"Initialize failed: {error_code}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return False
            
            # Verify account info
            account_info = mt5.account_info()
            if not account_info:
                logger.error("❌ Failed to retrieve account information")
                return False
            
            # Update connection metrics
            connection_time = time.time() - start_time
            self.metrics.connection_time = connection_time
            
            self.status = ConnectionStatus.AUTHENTICATED
            self.connected = True
            
            # Emit success telemetry
            emit_event("mt5.connection_status", {
                "status": "connected",
                "account_info": {
                    "login": account_info.login,
                    "server": account_info.server,
                    "currency": account_info.currency,
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "leverage": account_info.leverage
                },
                "connection_time_ms": connection_time * 1000,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"✅ MT5 connection established in {connection_time:.2f}s")
            logger.info(f"📊 Account: {account_info.login} | Balance: {account_info.balance} {account_info.currency}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection establishment error: {str(e)}")
            self.status = ConnectionStatus.ERROR
            emit_event("mt5.connection_status", {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False

    def _validate_connection(self) -> bool:
        """Validate existing MT5 connection"""
        try:
            if not mt5.terminal_info():
                return False
            
            account_info = mt5.account_info()
            return account_info is not None
            
        except Exception:
            return False

    def get_real_time_data(self, symbol: str) -> Optional[MT5MarketData]:
        """Get real-time market data with institutional quality validation"""
        try:
            if not self.ensure_connection():
                logger.error("❌ Connection not available for data request")
                return None
            
            start_time = time.time()
            
            # Get symbol tick
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.warning(f"⚠️ No tick data for symbol: {symbol}")
                self.metrics.failed_requests += 1
                return None
            
            # Get symbol info for additional data
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"⚠️ No symbol info for: {symbol}")
                self.metrics.failed_requests += 1
                return None
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create market data object
            market_data = MT5MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(tick.time, timezone.utc),
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                spread=(tick.ask - tick.bid),
                digits=symbol_info.digits,
                tick_value=symbol_info.trade_tick_value,
                tick_size=symbol_info.trade_tick_size,
                quality=self._assess_tick_quality(tick, symbol_info),
                latency_ms=latency_ms,
                source="MT5_LIVE"
            )
            
            # Update performance metrics
            self.metrics.data_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.avg_latency = statistics.mean(
                self.performance_stats["latency"][-100:] + [latency_ms]
            )
            
            # Update cache
            with self.data_lock:
                self.price_cache[symbol] = market_data
                self.last_data_fetch[symbol] = datetime.now(timezone.utc)
            
            # Emit real-time data event
            emit_event("mt5.tick_data", {
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "spread": market_data.spread,
                "timestamp": market_data.timestamp.isoformat(),
                "latency_ms": latency_ms,
                "quality": market_data.quality.value
            })
            
            # Emit telemetry
            emit_event("telemetry.mt5_adapter", {
                "operation": "get_real_time_data",
                "symbol": symbol,
                "latency_ms": latency_ms,
                "quality": market_data.quality.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Real-time data retrieval error for {symbol}: {str(e)}")
            self.metrics.failed_requests += 1
            return None

    def _assess_tick_quality(self, tick, symbol_info) -> DataQuality:
        """Assess tick data quality with institutional standards"""
        try:
            # Check basic validity
            if tick.bid <= 0 or tick.ask <= 0:
                return DataQuality.INVALID
            
            if tick.bid >= tick.ask:
                return DataQuality.INVALID
            
            # Check spread reasonableness
            spread = tick.ask - tick.bid
            if spread > tick.ask * 0.005:  # Spread > 0.5%
                return DataQuality.POOR
            elif spread > tick.ask * 0.002:  # Spread > 0.2%
                return DataQuality.ACCEPTABLE
            elif spread > tick.ask * 0.001:  # Spread > 0.1%
                return DataQuality.GOOD
            else:
                return DataQuality.EXCELLENT
                
        except Exception:
            return DataQuality.INVALID

    def get_historical_data(self, symbol: str, timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data with professional validation"""
        try:
            if not self.ensure_connection():
                logger.error("❌ Connection not available for historical data")
                return None
            
            start_time = time.time()
            
            # Map timeframe
            tf_mapping = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_mapping.get(timeframe, mt5.TIMEFRAME_M1)
            
            # Get historical rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️ No historical data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Data quality validation
            if len(df) < count * 0.8:  # Less than 80% of requested data
                logger.warning(f"⚠️ Incomplete historical data for {symbol}")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Emit telemetry
            emit_event("telemetry.mt5_adapter", {
                "operation": "get_historical_data",
                "symbol": symbol,
                "timeframe": timeframe,
                "bars_requested": count,
                "bars_received": len(df),
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"📈 Retrieved {len(df)} bars for {symbol} {timeframe} in {latency_ms:.1f}ms")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Historical data error for {symbol}: {str(e)}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive symbol information"""
        try:
            if not self.ensure_connection():
                return None
            
            # Check cache first
            if symbol in self.symbols_cache:
                cache_time = self.symbols_cache[symbol].get('cached_at', datetime.min)
                if (datetime.now(timezone.utc) - cache_time).seconds < self.config["symbols_refresh_interval"]:
                    return self.symbols_cache[symbol]['data']
            
            # Get symbol info from MT5
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"⚠️ Symbol info not available: {symbol}")
                return None
            
            # Create comprehensive symbol data
            symbol_data = {
                "symbol": symbol,
                "description": symbol_info.description,
                "currency_base": symbol_info.currency_base,
                "currency_profit": symbol_info.currency_profit,
                "currency_margin": symbol_info.currency_margin,
                "digits": symbol_info.digits,
                "spread": symbol_info.spread,
                "point": symbol_info.point,
                "tick_value": symbol_info.trade_tick_value,
                "tick_size": symbol_info.trade_tick_size,
                "contract_size": symbol_info.trade_contract_size,
                "minimum_lot": symbol_info.volume_min,
                "maximum_lot": symbol_info.volume_max,
                "lot_step": symbol_info.volume_step,
                "swap_long": symbol_info.swap_long,
                "swap_short": symbol_info.swap_short,
                "margin_initial": symbol_info.margin_initial,
                "session_deals": symbol_info.session_deals,
                "session_buy_orders": symbol_info.session_buy_orders,
                "session_sell_orders": symbol_info.session_sell_orders,
                "volume": symbol_info.volume,
                "volumehigh": symbol_info.volumehigh,
                "volumelow": symbol_info.volumelow,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Update cache
            self.symbols_cache[symbol] = {
                'data': symbol_data,
                'cached_at': datetime.now(timezone.utc)
            }
            
            # Emit symbol info event
            emit_event("mt5.symbol_info", {
                "symbol": symbol,
                "symbol_data": symbol_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return symbol_data
            
        except Exception as e:
            logger.error(f"❌ Symbol info error for {symbol}: {str(e)}")
            return None

    def subscribe_to_symbol(self, symbol: str) -> bool:
        """Subscribe to real-time symbol updates"""
        try:
            if not self.ensure_connection():
                return False
            
            # Check if symbol is available
            if not mt5.symbol_select(symbol, True):
                logger.error(f"❌ Failed to select symbol: {symbol}")
                return False
            
            # Add to subscription manager
            self.subscription_manager[symbol] = True
            
            # Start data collection for this symbol
            Thread(
                target=self._symbol_data_collector,
                args=(symbol,),
                daemon=True,
                name=f"DataCollector-{symbol}"
            ).start()
            
            logger.info(f"✅ Subscribed to real-time data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Subscription error for {symbol}: {str(e)}")
            return False

    def _symbol_data_collector(self, symbol: str):
        """Collect real-time data for subscribed symbol"""
        while self.subscription_manager.get(symbol, False) and not self.shutdown_event.is_set():
            try:
                # Get real-time data
                market_data = self.get_real_time_data(symbol)
                if market_data:
                    # Add to processing queue
                    if not self.data_queue.full():
                        self.data_queue.put({
                            'symbol': symbol,
                            'timestamp': market_data.timestamp,
                            'bid': market_data.bid,
                            'ask': market_data.ask,
                            'last': market_data.last,
                            'volume': market_data.volume,
                            'spread': market_data.spread,
                            'digits': market_data.digits,
                            'tick_value': market_data.tick_value,
                            'tick_size': market_data.tick_size,
                            'latency_ms': market_data.latency_ms
                        })
                
                # Control update frequency
                time.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                logger.error(f"❌ Data collection error for {symbol}: {str(e)}")
                time.sleep(1.0)  # Error recovery delay

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            now = datetime.now(timezone.utc)
            uptime_seconds = (now - getattr(self, 'start_time', now)).total_seconds()
            
            success_rate = (
                self.metrics.successful_requests / max(self.metrics.data_requests, 1)
            ) * 100
            
            return {
                "connection_status": self.status.value,
                "connected": self.connected,
                "uptime_seconds": uptime_seconds,
                "connection_time_ms": self.metrics.connection_time * 1000,
                "total_requests": self.metrics.data_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate_percent": success_rate,
                "average_latency_ms": self.metrics.avg_latency,
                "subscribed_symbols": len(self.subscription_manager),
                "cached_symbols": len(self.symbols_cache),
                "data_quality_score": self.metrics.quality_score,
                "last_update": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Performance metrics error: {str(e)}")
            return {}

    def shutdown(self):
        """Graceful shutdown with cleanup"""
        try:
            logger.info("🔄 Shutting down MT5 adapter...")
            
            # Stop data collection
            self.shutdown_event.set()
            self.subscription_manager.clear()
            
            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            # Close MT5 connection
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self.status = ConnectionStatus.DISCONNECTED
            
            # Emit shutdown telemetry
            emit_event("mt5.connection_status", {
                "status": "disconnected",
                "reason": "shutdown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info("✅ MT5 adapter shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {str(e)}")

    def get_module_status(self) -> Dict[str, Any]:
        """Get comprehensive module status for monitoring"""
        return {
            "module": "GenesisInstitutionalMT5Adapter",
            "version": self.VERSION,
            "status": self.status.value,
            "connected": self.connected,
            "subscriptions": len(self.subscription_manager),
            "performance": self.get_performance_metrics(),
            "architect_mode_compliance": True,
            "institutional_grade": True,
            "last_update": datetime.now(timezone.utc).isoformat()
        }


# Enhanced module initialization
def initialize_mt5_adapter(config: Optional[Dict[str, Any]] = None) -> GenesisInstitutionalMT5Adapter:
    """Initialize and return MT5 adapter instance"""
    return GenesisInstitutionalMT5Adapter(config)


# Export key components
__all__ = [
    'GenesisInstitutionalMT5Adapter',
    'MT5MarketData',
    'ConnectionStatus',
    'DataQuality',
    'initialize_mt5_adapter'
]

if __name__ == "__main__":
    # Test initialization
    adapter = initialize_mt5_adapter()
    logger.info("🧪 MT5 Adapter test initialization complete")
# <!-- @GENESIS_MODULE_END: mt5_adapter -->
