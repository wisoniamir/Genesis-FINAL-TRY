# <!-- @GENESIS_MODULE_START: auto_execution_sync_engine -->

#!/usr/bin/env python3
"""
GENESIS Auto-Execution Sync Engine - Phase 89
Real-Time Autonomous Signal-to-Order Execution Loop

🎯 PURPOSE: Autonomous signal-to-order execution with FTMO compliance
🔁 EVENTBUS: Listens to signal:triggered, kill_switch:activated, trade:filled
📡 TELEMETRY: Real-time MT5 sync, order tracking, balance monitoring
🛡️ COMPLIANCE: FTMO constraints enforcement, drawdown protection
"""

import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import uuid
import MetaTrader5 as mt5
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AutoExecutionSyncEngine')

class ExecutionStatus(Enum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """Execution status enumeration"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class OrderType(Enum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """Order type enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"

@dataclass
class Signal:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """Signal data structure"""
    signal_id: str
    symbol: str
    action: str
    volume: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: str
    confidence: float
    strategy_id: str
    risk_score: float

@dataclass
class ExecutionOrder:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """Execution order data structure"""
    order_id: str
    signal_id: str
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    mt5_ticket: Optional[int]
    status: ExecutionStatus
    created_at: str
    submitted_at: Optional[str]
    confirmed_at: Optional[str]
    error_message: Optional[str]
    slippage_pips: float
    execution_latency_ms: float

@dataclass
class FTMOConstraints:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """FTMO trading constraints"""
    max_daily_loss: float = -10000.0  # $10,000 max daily loss
    max_total_loss: float = -20000.0  # $20,000 max total loss
    max_position_size: float = 0.10   # 10% of account balance max
    max_lot_size: float = 2.0         # 2 lot maximum per trade
    min_lot_size: float = 0.01        # 0.01 lot minimum
    max_daily_trades: int = 100       # 100 trades per day max
    trading_start_hour: int = 0       # 24/7 trading allowed
    trading_end_hour: int = 23
    weekend_trading: bool = False     # No weekend trading
    news_trading_pause_minutes: int = 2  # 2 min pause around news


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
        class AutoExecutionSyncEngine:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """
    GENESIS Auto-Execution Sync Engine
    Autonomous signal-to-order execution with real-time MT5 synchronization
    """
    
    def __init__(self):
        """Initialize Auto-Execution Sync Engine"""
        self.engine_id = f"auto_exec_sync_{int(time.time())}"
        self.start_time = datetime.now(timezone.utc)
        
        # MT5 connection
        self.mt5_connected = False
        self.account_info = {}
        
        # FTMO constraints
        self.ftmo_constraints = FTMOConstraints()
        
        # Execution tracking
        self.pending_orders: Dict[str, ExecutionOrder] = {}
        self.execution_history: List[ExecutionOrder] = []
        self.signal_queue = queue.Queue()
        self.kill_switch_active = False
        
        # Performance metrics
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trade_count = 0
        self.execution_latency_history = []
        self.order_success_rate = 0.0
        
        # EventBus callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            "signal:triggered": [],
            "kill_switch:activated": [],
            "trade:filled": []
        }
        
        # Telemetry
        self.telemetry_data = {
            "engine_status": "INITIALIZING",
            "orders_pending": 0,
            "orders_today": 0,
            "daily_pnl": 0.0,
            "account_balance": 0.0,
            "margin_level": 0.0,
            "last_execution_time": None
        }
        
        # File paths
        self.logs_dir = Path("logs")
        self.telemetry_dir = Path("telemetry")
        self.logs_dir.mkdir(exist_ok=True)
        self.telemetry_dir.mkdir(exist_ok=True)
        
        self.execution_log_path = self.logs_dir / "execution_log.json"
        self.telemetry_path = self.telemetry_dir / "execution_engine_telemetry.json"
        
        logger.info(f"Auto-Execution Sync Engine initialized: {self.engine_id}")
    
    def connect_to_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            logger.info("🔗 Connecting to MT5 terminal...")
            
            assert mt5.initialize():
                logger.error("❌ Failed to initialize MT5")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ Failed to get account info")
                mt5.shutdown()
                return False
            
            self.account_info = {
                "login": account_info.login,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.free_margin,
                "margin_level": account_info.margin_level,
                "currency": account_info.currency,
                "server": account_info.server
            }
            
            self.mt5_connected = True
            self.telemetry_data["engine_status"] = "CONNECTED"
            self.telemetry_data["account_balance"] = self.account_info["balance"]
            self.telemetry_data["margin_level"] = self.account_info["margin_level"]
            
            logger.info(f"✅ Connected to MT5: {self.account_info['login']} on {self.account_info['server']}")
            logger.info(f"✅ Account balance: {self.account_info['balance']} {self.account_info['currency']}")
            
            self.emit_event("execution:mt5_connected", {
                "account_info": self.account_info,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ MT5 connection error: {str(e)}")
            self.mt5_connected = False
            return False
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register EventBus callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
            logger.info(f"✅ Registered callback for {event_type}")
    
    def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit EventBus event"""
        try:
            event_data = {
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_id": self.engine_id,
                "data": data
            }
            
            # Write to EventBus log
            eventbus_log_path = self.logs_dir / "eventbus_execution.json"
            events = []
            if eventbus_log_path.exists():
                with open(eventbus_log_path, 'r') as f:
                    events = json.load(f)
            
            events.append(event_data)
            
            with open(eventbus_log_path, 'w') as f:
                json.dump(events, f, indent=2)
                
            logger.info(f"📡 EventBus emit: {event_type}")
            
        except Exception as e:
            logger.error(f"❌ EventBus emit error: {str(e)}")
    
    def validate_ftmo_constraints(self, signal: Signal) -> Dict[str, Any]:
        """Validate signal against FTMO constraints"""
        validation_result = {
            "valid": True,
            "violations": [],
            "risk_assessment": "LOW"
        }
        
        try:
            # Check daily loss limit
            if self.daily_pnl <= self.ftmo_constraints.max_daily_loss:
                validation_result["violations"].append("DAILY_LOSS_LIMIT_EXCEEDED")
                validation_result["valid"] = False
            
            # Check total loss limit
            if self.total_pnl <= self.ftmo_constraints.max_total_loss:
                validation_result["violations"].append("TOTAL_LOSS_LIMIT_EXCEEDED")
                validation_result["valid"] = False
            
            # Check position size
            account_balance = self.account_info.get("balance", 100000.0)
            max_position_value = account_balance * self.ftmo_constraints.max_position_size
            signal_value = signal.volume * 100000  # Assuming EURUSD-like pair
            
            if signal_value > max_position_value:
                validation_result["violations"].append("POSITION_SIZE_TOO_LARGE")
                validation_result["valid"] = False
            
            # Check lot size limits
            if signal.volume > self.ftmo_constraints.max_lot_size:
                validation_result["violations"].append("LOT_SIZE_TOO_LARGE")
                validation_result["valid"] = False
            
            if signal.volume < self.ftmo_constraints.min_lot_size:
                validation_result["violations"].append("LOT_SIZE_TOO_SMALL")
                validation_result["valid"] = False
            
            # Check daily trade limit
            if self.daily_trade_count >= self.ftmo_constraints.max_daily_trades:
                validation_result["violations"].append("DAILY_TRADE_LIMIT_EXCEEDED")
                validation_result["valid"] = False
            
            # Check trading hours (simplified)
            current_hour = datetime.now().hour
            if not (self.ftmo_constraints.trading_start_hour <= current_hour <= self.ftmo_constraints.trading_end_hour):
                validation_result["violations"].append("OUTSIDE_TRADING_HOURS")
                validation_result["valid"] = False
            
            # Check weekend trading
            if not self.ftmo_constraints.weekend_trading and datetime.now().weekday() >= 5:
                validation_result["violations"].append("WEEKEND_TRADING_NOT_ALLOWED")
                validation_result["valid"] = False
            
            # Risk assessment
            if len(validation_result["violations"]) == 0:
                validation_result["risk_assessment"] = "LOW"
            elif len(validation_result["violations"]) <= 2:
                validation_result["risk_assessment"] = "MEDIUM"
            else:
                validation_result["risk_assessment"] = "HIGH"
            
            logger.info(f"🛡️ FTMO validation: {validation_result['risk_assessment']} risk, {len(validation_result['violations'])} violations")
            
        except Exception as e:
            logger.error(f"❌ FTMO validation error: {str(e)}")
            validation_result["valid"] = False
            validation_result["violations"].append("VALIDATION_ERROR")
        
        return validation_result
    
    def process_signal(self, signal: Signal) -> Optional[ExecutionOrder]:
        """Process incoming signal and create execution order"""
        try:
            logger.info(f"📊 Processing signal: {signal.signal_id} ({signal.action} {signal.volume} {signal.symbol})")
            
            # Validate FTMO constraints
            validation = self.validate_ftmo_constraints(signal)
            if not validation["valid"]:                logger.warning(f"⚠️ Signal rejected: FTMO violations: {validation['violations']}")
                self.log_execution_event("SIGNAL_REJECTED", {
                    "signal_id": signal.signal_id,
                    "violations": validation["violations"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                raise ValueError(f"ARCHITECT_MODE_COMPLIANCE: FTMO validation failed - {validation['violations']}")
            
            # Create execution order
            order_id = f"order_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            order_type = OrderType.BUY if signal.action.upper() == "BUY" else OrderType.SELL
            
            # Get current market price
            current_price = self.get_current_price(signal.symbol, signal.action)
            if current_price is None:
                logger.error(f"❌ Failed to get current price for {signal.symbol}")
                self.log_execution_event("PRICE_FETCH_ERROR", {
                    "signal_id": signal.signal_id,
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                raise RuntimeError(f"ARCHITECT_MODE_COMPLIANCE: Failed to get current price for {signal.symbol}")
            
            execution_order = ExecutionOrder(
                order_id=order_id,
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                order_type=order_type,
                volume=signal.volume,
                price=signal.entry_price or current_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                mt5_ticket=None,
                status=ExecutionStatus.PENDING,
                created_at=datetime.now(timezone.utc).isoformat(),
                submitted_at=None,
                confirmed_at=None,
                error_message=None,
                slippage_pips=0.0,
                execution_latency_ms=0.0
            )
            
            self.pending_orders[order_id] = execution_order
            self.telemetry_data["orders_pending"] = len(self.pending_orders)
            
            logger.info(f"✅ Execution order created: {order_id}")
            
            self.emit_event("execution:pending", {
                "order_id": order_id,
                "signal_id": signal.signal_id,                "symbol": signal.symbol,
                "action": signal.action,
                "volume": signal.volume
            })
            
            return execution_order
            
        except Exception as e:
            logger.error(f"❌ Signal processing error: {str(e)}")
            self.log_execution_event("SIGNAL_PROCESSING_ERROR", {
                "error_message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise RuntimeError(f"ARCHITECT_MODE_COMPLIANCE: Signal processing failed - {e}")
    
    def get_current_price(self, symbol: str, action: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            if not self.mt5_connected:
                logger.error("❌ MT5 not connected")
                self.log_execution_event("MT5_CONNECTION_ERROR", {
                    "symbol": symbol,
                    "action": action,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                raise ConnectionError("ARCHITECT_MODE_COMPLIANCE: MT5 connection required")
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"❌ Failed to get tick for {symbol}")
                self.log_execution_event("TICK_FETCH_ERROR", {
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                raise RuntimeError(f"ARCHITECT_MODE_COMPLIANCE: Failed to get tick for {symbol}")
            
            # Use ask for buy orders, bid for sell orders
            price = tick.ask if action.upper() == "BUY" else tick.bid
            return float(price)
            
        except Exception as e:
            logger.error(f"❌ Price fetch error: {str(e)}")
            self.log_execution_event("PRICE_FETCH_EXCEPTION", {
                "symbol": symbol,
                "action": action,
                "error_message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise RuntimeError(f"ARCHITECT_MODE_COMPLIANCE: Price fetch failed - {e}")
    
    def submit_order_to_mt5(self, order: ExecutionOrder) -> bool:
        """Submit order to MT5 terminal"""
        try:
            start_time = time.time()
            logger.info(f"📈 Submitting order to MT5: {order.order_id}")
            
            if not self.mt5_connected:
                logger.error("❌ MT5 not connected")
                order.status = ExecutionStatus.FAILED
                order.error_message = "MT5_NOT_CONNECTED"
                return False
            
            # Get symbol info
            symbol_info = mt5.symbol_info(order.symbol)
            if symbol_info is None:
                logger.error(f"❌ Symbol {order.symbol} not found")
                order.status = ExecutionStatus.FAILED
                order.error_message = "SYMBOL_NOT_FOUND"
                return False
            
            # Prepare MT5 order request
            order_type = mt5.ORDER_TYPE_BUY if order.order_type == OrderType.BUY else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": order.volume,
                "type": order_type,
                "price": order.price,
                "deviation": 20,
                "magic": 89890,  # Phase 89 magic number
                "comment": f"GENESIS_AUTO_EXEC_{order.order_id[:8]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL/TP if specified
            if order.stop_loss:
                request["sl"] = order.stop_loss
            if order.take_profit:
                request["tp"] = order.take_profit
            
            # Submit order
            result = mt5.order_send(request)
            execution_time = (time.time() - start_time) * 1000
            
            order.submitted_at = datetime.now(timezone.utc).isoformat()
            order.execution_latency_ms = execution_time
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Order submission failed: {result.retcode} - {result.comment}")
                order.status = ExecutionStatus.REJECTED
                order.error_message = f"{result.retcode}: {result.comment}"
                
                self.emit_event("execution:rejected", {
                    "order_id": order.order_id,
                    "error_code": result.retcode,
                    "error_message": result.comment,
                    "execution_time_ms": execution_time
                })
                
                return False
            
            # Order successful
            order.mt5_ticket = result.order
            order.status = ExecutionStatus.SUBMITTED
            order.price = result.price  # Actual fill price
            order.slippage_pips = abs(order.price - result.price) * (10000 if "JPY" not in order.symbol else 100)
            
            self.execution_latency_history.append(execution_time)
            self.daily_trade_count += 1
            self.telemetry_data["orders_today"] = self.daily_trade_count
            
            logger.info(f"✅ Order submitted successfully: Ticket {result.order}, Price {result.price}")
            logger.info(f"✅ Execution time: {execution_time:.1f}ms, Slippage: {order.slippage_pips:.1f} pips")
            
            self.emit_event("execution:submitted", {
                "order_id": order.order_id,
                "mt5_ticket": result.order,
                "fill_price": result.price,
                "volume": result.volume,
                "execution_time_ms": execution_time,
                "slippage_pips": order.slippage_pips
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Order submission error: {str(e)}")
            order.status = ExecutionStatus.FAILED
            order.error_message = str(e)
            return False
    
    def monitor_order_status(self, order: ExecutionOrder):
        """Monitor order status and update when filled"""
        try:
            if not order.mt5_ticket:
                return
            
            # Check if order is filled
            positions = mt5.positions_get(ticket=order.mt5_ticket)
            if positions and len(positions) > 0:
                position = positions[0]
                
                order.status = ExecutionStatus.CONFIRMED
                order.confirmed_at = datetime.now(timezone.utc).isoformat()
                
                # Update PnL
                position_pnl = position.profit
                self.daily_pnl += position_pnl
                self.total_pnl += position_pnl
                
                self.telemetry_data["daily_pnl"] = self.daily_pnl
                self.telemetry_data["last_execution_time"] = order.confirmed_at
                
                logger.info(f"✅ Order confirmed: {order.order_id}, PnL: {position_pnl:.2f}")
                
                self.emit_event("execution:confirmed", {
                    "order_id": order.order_id,
                    "mt5_ticket": order.mt5_ticket,
                    "position_pnl": position_pnl,
                    "confirmed_at": order.confirmed_at
                })
                
                # Move to execution history
                self.execution_history.append(order)
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                
                self.telemetry_data["orders_pending"] = len(self.pending_orders)
            
        except Exception as e:
            logger.error(f"❌ Order monitoring error: {str(e)}")
    
    def sync_account_data(self):
        """Synchronize account data from MT5"""
        try:
            if not self.mt5_connected:
                return
            
            account_info = mt5.account_info()
            if account_info:
                self.account_info.update({
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.free_margin,
                    "margin_level": account_info.margin_level
                })
                
                self.telemetry_data["account_balance"] = account_info.balance
                self.telemetry_data["margin_level"] = account_info.margin_level
            
        except Exception as e:
            logger.error(f"❌ Account sync error: {str(e)}")
    
    def log_execution_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log execution event to execution_log.json"""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_id": self.engine_id,
                "event_type": event_type,
                "data": event_data
            }
            
            # Load existing logs
            execution_logs = []
            if self.execution_log_path.exists():
                with open(self.execution_log_path, 'r') as f:
                    execution_logs = json.load(f)
            
            # Add new log entry
            execution_logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(execution_logs) > 1000:
                execution_logs = execution_logs[-1000:]
            
            # Save logs
            with open(self.execution_log_path, 'w') as f:
                json.dump(execution_logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Execution log error: {str(e)}")
    
    def update_telemetry(self):
        """Update telemetry data"""
        try:
            self.telemetry_data.update({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_id": self.engine_id,
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                "mt5_connected": self.mt5_connected,
                "kill_switch_active": self.kill_switch_active,
                "avg_execution_latency_ms": sum(self.execution_latency_history[-100:]) / len(self.execution_latency_history[-100:]) if self.execution_latency_history else 0.0,
                "orders_executed_today": self.daily_trade_count,
                "total_orders_processed": len(self.execution_history),
                "account_info": self.account_info
            })
            
            # Save telemetry
            with open(self.telemetry_path, 'w') as f:
                json.dump(self.telemetry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Telemetry update error: {str(e)}")
    
    def handle_kill_switch(self, event_data: Dict[str, Any]):
        """Handle kill switch activation"""
        try:
            logger.warning("🚨 KILL SWITCH ACTIVATED - Stopping all executions")
            
            self.kill_switch_active = True
            self.telemetry_data["engine_status"] = "KILL_SWITCH_ACTIVE"
            
            # Cancel all pending orders
            for order_id, order in self.pending_orders.items():
                if order.mt5_ticket:
                    # Cancel order in MT5
                    self.cancel_mt5_order(order.mt5_ticket)
                
                order.status = ExecutionStatus.CANCELLED
                order.error_message = "KILL_SWITCH_ACTIVATED"
            
            self.log_execution_event("KILL_SWITCH_ACTIVATED", event_data)
            
            self.emit_event("execution:kill_switch_triggered", {
                "cancelled_orders": len(self.pending_orders),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Kill switch handling error: {str(e)}")
    
    def cancel_mt5_order(self, ticket: int) -> bool:
        """Cancel MT5 order"""
        try:
            if not self.mt5_connected is not None, "Real data required - no fallbacks allowed"

# <!-- @GENESIS_MODULE_END: auto_execution_sync_engine -->