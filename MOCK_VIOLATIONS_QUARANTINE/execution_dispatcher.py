
# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "execution_dispatcher",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("execution_dispatcher", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in execution_dispatcher: {e}")
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


# <!-- @GENESIS_MODULE_START: execution_dispatcher -->

#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║     GENESIS PHASE 39 — TRADE EXECUTION DISPATCHER MODULE v1.0    ║
╚═══════════════════════════════════════════════════════════════════╝

🧠 MODULE ROLE:
Takes qualified signals from ExecutionSelector and executes live MT5 orders
with full broker context, FTMO compliance, and comprehensive error handling.

🔐 ARCHITECT MODE COMPLIANCE:
- MT5 live broker execution only
- Real-time telemetry hooks
- EventBus-driven architecture
- Full error handling and logging
- Performance metrics monitoring
- FTMO rule compliance enforcement
"""

import json
import time
import logging
import MetaTrader5 as mt5
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty
import traceback
import hashlib

# Import GENESIS core modules
from hardened_event_bus import get_event_bus, emit_event, subscribe_to_event, register_route
from telemetry_collector import TelemetryCollector

@dataclass
class ExecutionResult:
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "execution_dispatcher",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("execution_dispatcher", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in execution_dispatcher: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "execution_dispatcher",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in execution_dispatcher: {e}")
    """Data structure for execution results"""
    signal_id: str
    symbol: str
    order_type: str
    volume: float
    price: float
    ticket: Optional[int]
    status: str
    execution_time_ms: float
    broker: str
    magic_number: int
    comment: str
    error_message: Optional[str]
    timestamp: str

@dataclass
class ExecutionDispatcherMetrics:
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "execution_dispatcher",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("execution_dispatcher", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in execution_dispatcher: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "execution_dispatcher",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in execution_dispatcher: {e}")
    """Telemetry metrics for the Execution Dispatcher"""
    module_name: str = "ExecutionDispatcher"
    total_signals_received: int = 0
    total_orders_dispatched: int = 0
    total_orders_successful: int = 0
    total_orders_failed: int = 0
    average_execution_latency_ms: float = 0.0
    success_rate_percentage: float = 0.0
    last_execution_timestamp: str = ""
    total_execution_count: int = 0
    ftmo_violations: int = 0
    broker_connection_errors: int = 0

class ExecutionDispatcher:
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "execution_dispatcher",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("execution_dispatcher", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in execution_dispatcher: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "execution_dispatcher",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in execution_dispatcher: {e}")
    """
    🎯 GENESIS Trade Execution Dispatcher Engine
    
    Handles live MT5 order execution from qualified signals with
    comprehensive error handling and FTMO compliance.
    """
    
    def __init__(self):
        """Initialize the Execution Dispatcher with full GENESIS compliance"""
        self.module_name = "ExecutionDispatcher"
        self.version = "1.0.0"
        self.status = "initializing"
        
        # Core GENESIS components
        self.event_bus = get_event_bus()
        self.telemetry = TelemetryCollector()
        
        # MT5 connection
        self.mt5_initialized = False
        self.broker_context = {}
        
        # Internal state
        self.execution_queue = Queue()
        self.is_running = False
        self.ftmo_rules = {}
        self.broker_profiles = {}
        
        # Metrics tracking
        self.metrics = ExecutionDispatcherMetrics()
        
        # FTMO compliance settings
        self.max_daily_volume = 10.0  # FTMO daily volume limit
        self.max_simultaneous_trades = 5  # FTMO trade count limit
        self.min_trade_volume = 0.01  # Minimum trade size
        self.max_trade_volume = 2.0   # Maximum trade size for FTMO
        self.daily_volume_used = 0.0
        self.active_trades_count = 0
        
        # Magic number generation
        self.magic_base = 240617  # Date-based magic number base
        
        # Setup logging
        self.logger = logging.getLogger(f"genesis.{self.module_name}")
        self.logger.setLevel(logging.INFO)
        
        # Initialize MT5 connection
        self._initialize_mt5()
        self._load_broker_profiles()
        self._setup_eventbus_connections()
        self._start_execution_thread()
        
        self.status = "active"
        self.logger.info(f"✅ {self.module_name} v{self.version} initialized successfully")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _initialize_mt5(self):
        """Initialize MT5 connection with error handling"""
        try:
            assert mt5.initialize():
                self.logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
                self.mt5_initialized = False
                return
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("❌ Failed to get account info")
                self.mt5_initialized = False
                return
            
            self.broker_context = {
                "login": account_info.login,
                "server": account_info.server,
                "currency": account_info.currency,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "leverage": account_info.leverage,
                "trade_mode": account_info.trade_mode
            }
            
            self.mt5_initialized = True
            self.logger.info(f"✅ MT5 initialized - Account: {account_info.login}, Server: {account_info.server}")
            
        except Exception as e:
            self.logger.error(f"❌ MT5 initialization error: {e}")
            self.mt5_initialized = False
    
    def _load_broker_profiles(self):
        """Load broker-specific execution profiles"""
        try:
            # Default broker profiles with execution characteristics
            self.broker_profiles = {
                "MT5_DEMO": {
                    "execution_score": 0.85,
                    "latency_ms": 50,
                    "spread_markup": 0.0,
                    "slippage_tolerance": 2,
                    "max_volume": 1.0
                },
                "MT5_LIVE": {
                    "execution_score": 0.95,
                    "latency_ms": 30,
                    "spread_markup": 0.0,
                    "slippage_tolerance": 1,
                    "max_volume": 2.0
                },
                "FTMO": {
                    "execution_score": 0.90,
                    "latency_ms": 40,
                    "spread_markup": 0.0,
                    "slippage_tolerance": 1,
                    "max_volume": 2.0,
                    "ftmo_compliant": True
                }
            }
            
            # Load FTMO-specific rules
            self.ftmo_rules = {
                "max_daily_loss_pct": 5.0,
                "max_total_loss_pct": 10.0,
                "min_trading_days": 4,
                "max_lot_size": 2.0,
                "news_trading_allowed": False,
                "weekend_holding_allowed": True
            }
            
            self.logger.info("✅ Broker profiles loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load broker profiles: {e}")
    
    def _setup_eventbus_connections(self):
        """Register EventBus consumers and producers"""
        try:
            # Subscribe to qualified signals from ExecutionSelector
            subscribe_to_event("qualified_signals_for_execution", self._handle_qualified_signals, self.module_name)
            subscribe_to_event("broker_context_update", self._handle_broker_update, self.module_name)
            subscribe_to_event("ftmo_rules_update", self._handle_ftmo_update, self.module_name)
            subscribe_to_event("emergency_stop", self._handle_emergency_stop, self.module_name)
            
            # Register routes for output events
            register_route("execution_success", self.module_name, "TelemetryCollector")
            register_route("execution_error", self.module_name, "RiskEngine")
            register_route("order_placed", self.module_name, "ExecutionEngine")
            register_route("execution_metrics", self.module_name, "TelemetryCollector")
            
            self.logger.info("✅ EventBus connections established")
            
        except Exception as e:
            self.logger.error(f"❌ EventBus setup failed: {e}")
            raise
    
    def _start_execution_thread(self):
        """Start background execution thread"""
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        self.logger.info("✅ Execution thread started")
    
    def _execution_loop(self):
        """Main execution loop for processing orders"""
        while self.is_running:
            try:
                # Check for new orders to execute
                try:
                    task = self.execution_queue.get(timeout=1.0)
                    self._execute_order(task)
                    self.execution_queue.task_done()
                except Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"❌ Execution loop error: {e}")
                self.logger.error(f"Stack trace: {traceback.format_exc()}")
                time.sleep(1.0)
    
    def _handle_qualified_signals(self, event_data: Dict[str, Any]):
        """Handle incoming qualified signals for execution"""
        try:
            signals = event_data.get("signals", [])
            
            if not signals:
                self.logger.warning("⚠️ No qualified signals received")
                return
            
            self.metrics.total_signals_received += len(signals)
            
            # Queue each signal for execution
            for signal in signals:
                self.execution_queue.put({
                    "action": "execute_signal",
                    "signal": signal,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            self.logger.info(f"📥 Queued {len(signals)} qualified signals for execution")
            
        except Exception as e:
            self.logger.error(f"❌ Error handling qualified signals: {e}")
    
    def _handle_broker_update(self, event_data: Dict[str, Any]):
        """Handle broker context updates"""
        try:
            self.broker_context.update(event_data.get("context", {}))
            
        except Exception as e:
            self.logger.error(f"❌ Error handling broker update: {e}")
    
    def _handle_ftmo_update(self, event_data: Dict[str, Any]):
        """Handle FTMO rules updates"""
        try:
            self.ftmo_rules.update(event_data.get("rules", {}))
            
        except Exception as e:
            self.logger.error(f"❌ Error handling FTMO update: {e}")
    
    def _handle_emergency_stop(self, event_data: Dict[str, Any]):
        """Handle emergency stop command"""
        try:
            self.logger.warning("🚨 Emergency stop received - halting execution")
            self.is_running = False
            
            # Clear execution queue
            while not self.execution_queue.empty():
                try:
                    self.execution_queue.get_nowait()
                except Empty:
                    break
            
            emit_event("execution_error", {
                "module": self.module_name,
                "error": "Emergency stop activated",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, self.module_name)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling emergency stop: {e}")
    
    def _execute_order(self, task: Dict[str, Any]):
        """Execute a single order from qualified signal"""
        start_time = time.time()
        
        try:
            signal = task.get("signal", {})
            
            if not signal:
                self.logger.warning("⚠️ Empty signal received for execution")
                return
            
            # Validate MT5 connection
            if not self.mt5_initialized:
                self._handle_execution_error(signal, "MT5 not initialized")
                return
            
            # Validate FTMO compliance
            if not self._validate_ftmo_compliance(signal):
                self._handle_execution_error(signal, "FTMO compliance violation")
                return
            
            # Translate signal to MT5 order
            order_request = self._translate_signal_to_order(signal)
            
            if not order_request:
                self._handle_execution_error(signal, "Failed to translate signal to order")
                return
            
            # Execute the order
            result = mt5.order_send(order_request)
            execution_time_ms = (time.time() - start_time) * 1000
            
            if result is None:
                self._handle_execution_error(signal, f"Order send failed: {mt5.last_error()}")
                return
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self._handle_execution_error(signal, f"Order rejected: {result.comment}")
                return
            
            # Order successful
            self._handle_execution_success(signal, result, execution_time_ms)
            
        except Exception as e:
            self.logger.error(f"❌ Order execution failed: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            self._handle_execution_error(signal, str(e))
    
    def _translate_signal_to_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Translate GENESIS signal to MT5 order request"""
        try:
            symbol = signal.get("symbol", "")
            signal_type = signal.get("type", "").upper()
            
            # Determine order type
            if signal_type in ["BUY", "LONG"]:
                order_type = mt5.ORDER_TYPE_BUY
            elif signal_type in ["SELL", "SHORT"]:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                self.logger.error(f"❌ Invalid signal type: {signal_type}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
            
            # Calculate volume with FTMO compliance
            volume = self._calculate_volume(signal)
            
            if volume <= 0:
                self.logger.error(f"❌ Invalid volume calculated: {volume}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"❌ Failed to get tick data for {symbol}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
            
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            
            # Generate magic number
            magic_number = self._generate_magic_number(signal)
            
            # Create order request
            order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": signal.get("stop_loss", 0.0),
                "tp": signal.get("take_profit", 0.0),
                "deviation": self.broker_profiles.get(signal.get("broker", "MT5_DEMO"), {}).get("slippage_tolerance", 2),
                "magic": magic_number,
                "comment": f"GENESIS_EXEC_{signal.get('id', 'unknown')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            
            return order_request
            
        except Exception as e:
            self.logger.error(f"❌ Signal translation failed: {e}")
            self._emit_error_event("operation_failed", {

                "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                "timestamp": datetime.now(timezone.utc).isoformat()

            })

            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
    
    def _calculate_volume(self, signal: Dict[str, Any]) -> float:
        """Calculate trade volume with FTMO compliance"""
        try:
            # Base volume from signal
            base_volume = signal.get("volume", 0.1)
            
            # Apply FTMO limits
            volume = min(base_volume, self.max_trade_volume)
            volume = max(volume, self.min_trade_volume)
            
            # Check daily volume limit
            if self.daily_volume_used + volume > self.max_daily_volume:
                volume = self.max_daily_volume - self.daily_volume_used
            
            # Ensure minimum volume
            if volume < self.min_trade_volume is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: execution_dispatcher -->