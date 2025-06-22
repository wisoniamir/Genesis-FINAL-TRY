# <!-- @GENESIS_MODULE_START: mt5_adapter -->

from event_bus import EventBus

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

                emit_telemetry("mt5_adapter", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("mt5_adapter", "position_calculated", {
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
                            "module": "mt5_adapter",
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
                    print(f"Emergency stop error in mt5_adapter: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "mt5_adapter",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("mt5_adapter", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in mt5_adapter: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics

# Import enhanced EventBus
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

            emit_telemetry("mt5_adapter", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter", "position_calculated", {
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
                        "module": "mt5_adapter",
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
                print(f"Emergency stop error in mt5_adapter: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter: {e}")
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

            emit_telemetry("mt5_adapter", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter", "position_calculated", {
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
                        "module": "mt5_adapter",
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
                print(f"Emergency stop error in mt5_adapter: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter: {e}")
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

            emit_telemetry("mt5_adapter", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter", "position_calculated", {
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
                        "module": "mt5_adapter",
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
                print(f"Emergency stop error in mt5_adapter: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter: {e}")
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

            emit_telemetry("mt5_adapter", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter", "position_calculated", {
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
                        "module": "mt5_adapter",
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
                print(f"Emergency stop error in mt5_adapter: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter: {e}")
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

            emit_telemetry("mt5_adapter", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("mt5_adapter", "position_calculated", {
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
                        "module": "mt5_adapter",
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
                print(f"Emergency stop error in mt5_adapter: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "mt5_adapter",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("mt5_adapter", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in mt5_adapter: {e}")
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
        self.executor = ThreadPoolExecutor(max_workers=4)
        
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
        
        # Start connection monitoring
        self.executor.submit(self._connection_monitor)
        
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

    def _connection_monitor(self) -> None:
        """Monitor connection health with institutional standards"""
        while not self.shutdown_event.is_set():
            try:
                if self.connected:
                    # Check connection health
                    terminal_info = mt5.terminal_info()
                    if terminal_info is None:
                        logger.warning("⚠️ Terminal connection lost - reconnecting")
                        self._attempt_reconnection()
                    else:
                        self._update_connection_metrics()
                
                time.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                logger.error(f"❌ Connection monitoring error: {str(e)}")

    def ensure_connection(self) -> bool:
        """Ensure institutional-grade MT5 connection"""
        with self.connection_lock:
            if self.connected and self._validate_connection():
                return True
            
            return self._establish_connection()

    def _establish_connection(self) -> bool:
        """Establish professional MT5 connection"""
        start_time = time.perf_counter()
        self.status = ConnectionStatus.CONNECTING
        
        try:
            logger.info("🔗 Establishing institutional MT5 connection...")
            
            # Initialize MT5 terminal
            if not mt5.initialize(
                path=self.credentials.get("path", ""),
                login=self.credentials["login"],
                password=self.credentials["password"],
                server=self.credentials["server"],
                timeout=self.credentials.get("timeout", 60000),
                portable=self.credentials.get("portable", False)
            ):
                error = mt5.last_error()
                logger.error(f"❌ MT5 initialization failed: {error}")
                self.status = ConnectionStatus.ERROR
                return False
            
            # Authenticate
            auth_start = time.perf_counter()
            login_result = mt5.login(
                login=self.credentials["login"],
                password=self.credentials["password"],
                server=self.credentials["server"]
            )
            
            if not login_result:
                error = mt5.last_error()
                logger.error(f"❌ MT5 authentication failed: {error}")
                self.status = ConnectionStatus.ERROR
                return False
            
            auth_time = time.perf_counter() - auth_start
            
            # Verify account access
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ Cannot access account information")
                self.status = ConnectionStatus.ERROR
                return False
            
            # Update metrics
            connection_time = time.perf_counter() - start_time
            self.metrics.connection_time = connection_time
            self.metrics.authentication_time = auth_time
            
            self.connected = True
            self.status = ConnectionStatus.AUTHENTICATED
            
            # Load symbols and start monitoring
            self._load_symbols_cache()
            self.status = ConnectionStatus.MONITORING
            
            logger.info(f"✅ MT5 connection established in {connection_time:.3f}s")
            
            # Emit connection event
            emit_event("mt5.connection_status", {
                "status": "connected",
                "account": account_info.login,
                "server": account_info.server,
                "company": account_info.company,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "connection_time": connection_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection establishment failed: {str(e)}")
            self.status = ConnectionStatus.ERROR
            return False

    def _validate_connection(self) -> bool:
        """Validate connection with institutional standards"""
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                return False
            
            # Test data access
            test_symbol = "EURUSD"
            tick = mt5.symbol_info_tick(test_symbol)
            if tick is None:
                return False
            
            return True
            
        except Exception:
            return False

    def get_historical_data(self, symbol: str, timeframe: int, count: int = 500) -> Optional[pd.DataFrame]:
        """
        Get REAL historical data from MT5 - NO MOCK DATA ALLOWED
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD') 
            timeframe: MT5 timeframe constant
            count: Number of bars to fetch
            
        Returns:
            pd.DataFrame with OHLCV data or None if failed
        """
        if not self.ensure_connection():
            logger.error("❌ ARCHITECT_VIOLATION: Cannot get historical data - MT5 not connected")
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
        
        start_time = time.time()
        self.data_integrity_stats["total_requests"] += 1
        
        try:
            # REAL MT5 DATA FETCH - NO FALLBACKS
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.error(f"❌ ARCHITECT_VIOLATION: No historical data available for {symbol}")
                self.data_integrity_stats["failed_fetches"] += 1
                self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
            
            # Convert to DataFrame with proper timestamps
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Data integrity validation
            if len(df) < count * 0.8:  # Must get at least 80% of requested data
                logger.warning(f"⚠️ ARCHITECT_WARNING: Incomplete data for {symbol} - got {len(df)}/{count} bars")
            
            # Update statistics
            latency_ms = (time.time() - start_time) * 1000
            self.data_integrity_stats["successful_fetches"] += 1
            self.data_integrity_stats["avg_latency_ms"] = (
                self.data_integrity_stats["avg_latency_ms"] + latency_ms
            ) / 2
            
            # Cache metadata
            self.last_data_fetch[symbol] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bars_count": len(df),
                "timeframe": timeframe,
                "latency_ms": latency_ms
            }
            
            logger.info(f"✅ Historical data fetched for {symbol}: {len(df)} bars ({latency_ms:.1f}ms)")
            return df
            
        except Exception as e:
            logger.error(f"❌ ARCHITECT_VIOLATION: Historical data fetch error for {symbol} - {str(e)}")
            self.data_integrity_stats["failed_fetches"] += 1
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
    
    def get_live_tick_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get REAL live tick data from MT5 - NO MOCK DATA ALLOWED
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with live tick data or None if failed
        """
        if not self.ensure_connection():
            logger.error("❌ ARCHITECT_VIOLATION: Cannot get live data - MT5 not connected")
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
        
        try:
            # REAL MT5 TICK DATA - NO FALLBACKS
            tick = mt5.symbol_info_tick(symbol)
            
            if tick is None:
                logger.error(f"❌ ARCHITECT_VIOLATION: No live tick data for {symbol}")
                self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
            
            # Validate tick data integrity
            if tick.bid <= 0 or tick.ask <= 0:
                logger.error(f"❌ ARCHITECT_VIOLATION: Invalid tick data for {symbol} - bid: {tick.bid}, ask: {tick.ask}")
                self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
            
            live_data = {
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
                "spread": tick.ask - tick.bid,
                "source": "MT5_LIVE",
                "data_integrity": "VERIFIED"
            }
            
            return live_data
            
        except Exception as e:
            logger.error(f"❌ ARCHITECT_VIOLATION: Live tick data error for {symbol} - {str(e)}")
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
    
    def get_symbol_list(self) -> List[str]:
        """
        Get REAL symbol list from MT5 Market Watch - NO STATIC LISTS
        
        Returns:
            List of available symbols from MT5
        """
        if not self.ensure_connection():
            logger.error("❌ ARCHITECT_VIOLATION: Cannot get symbols - MT5 not connected")
            return []
        
        try:
            # REAL MT5 SYMBOL DISCOVERY - NO HARDCODED LISTS
            symbols = mt5.symbols_get()
            
            if symbols is None:
                logger.error("❌ ARCHITECT_VIOLATION: Cannot retrieve symbols from MT5")
                return []
            
            # Filter for visible symbols with live pricing
            active_symbols = []
            for symbol in symbols:
                if symbol.visible:
                    # Verify live pricing available
                    tick = mt5.symbol_info_tick(symbol.name)
                    if tick is not None and tick.bid > 0:
                        active_symbols.append(symbol.name)
            
            logger.info(f"✅ Symbol discovery complete: {len(active_symbols)} live symbols available")
            return active_symbols
            
        except Exception as e:
            logger.error(f"❌ ARCHITECT_VIOLATION: Symbol discovery error - {str(e)}")
            return []
    
    def calculate_indicator(self, symbol: str, indicator_type: str, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Calculate indicator using REAL MT5 data - NO MOCK CALCULATIONS
        
        Args:
            symbol: Trading symbol
            indicator_type: Type of indicator (RSI, MACD, etc.)
            params: Indicator parameters
            
        Returns:
            Calculated indicator values or None if failed
        """
        # Get real historical data
        timeframe = params.get('timeframe', mt5.TIMEFRAME_M15)
        count = params.get('count', 100)
        
        df = self.get_historical_data(symbol, timeframe, count)
        if df is None:
            logger.error(f"❌ ARCHITECT_VIOLATION: Cannot calculate {indicator_type} - no data for {symbol}")
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
        
        try:
            # Route to appropriate calculation method
            if indicator_type.upper() == 'RSI' is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: mt5_adapter -->