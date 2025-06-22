
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔌 GENESIS MT5 CONNECTOR v4.0 - REAL-TIME MT5 BROKER INTERFACE
📊 ARCHITECT MODE v7.0.0 COMPLIANT | 🚫 NO MOCKS | 📡 MT5 DIRECT

🎯 PURPOSE:
Real-time MT5 broker interface with direct trading capabilities:
- Real-time tick data streaming
- Account monitoring and validation
- Connection health management
- Market session detection
- FTMO-compliant operations
"""

import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue, Empty
import math

# MT5 Integration - Architect Mode Compliant (LIVE ONLY)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.error("❌ CRITICAL: MT5 not available - LIVE TRADING DISABLED")
    raise ImportError("MetaTrader5 library required for live trading. Install with: pip install MetaTrader5")

# GENESIS Core Imports - Architect Mode Compliant
try:
    from modules.restored.event_bus import EventBus
    EVENTBUS_AVAILABLE = True
except ImportError:
    class EventBus:
        def subscribe(self, event, handler): pass
        def emit(self, event, data): pass
    EVENTBUS_AVAILABLE = False

try:
    from core.telemetry import TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    class TelemetryManager:
        def register_metric(self, name, type_): pass
        def set_gauge(self, name, value): pass
        def increment(self, name): pass
        def timer(self, name): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    TELEMETRY_AVAILABLE = False


class ConnectionStatus(Enum):
    """MT5 connection status"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class MarketSession(Enum):
    """Trading session types"""
    CLOSED = "CLOSED"
    PRE_MARKET = "PRE_MARKET"
    OPEN = "OPEN"
    POST_MARKET = "POST_MARKET"


@dataclass
class TickData:
    """Real-time tick data structure"""
    symbol: str
    bid: float
    ask: float
    spread: float
    time: datetime
    volume: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'time': self.time.isoformat(),
            'volume': self.volume
        }


@dataclass
class AccountInfo:
    """MT5 account information"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    leverage: int
    profit: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MT5Connector:
    """
    🔌 GENESIS MT5 Connector - Real-time MT5 broker interface
    
    Features:
    - Real-time tick data streaming
    - Account monitoring and validation
    - Connection health management
    - Market session detection
    - FTMO compliance verification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MT5 connector with configuration"""
        self.config = config or {}
        self.status = ConnectionStatus.DISCONNECTED
        self.event_bus = EventBus() if EVENTBUS_AVAILABLE else None
        self.telemetry = TelemetryManager() if TELEMETRY_AVAILABLE else None
        
        # Connection settings
        self.login = self.config.get('login', 0)
        self.password = self.config.get('password', '')
        self.server = self.config.get('server', '')
        self.timeout = self.config.get('timeout', 10000)
        
        # Monitoring settings
        self.symbols = set(self.config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY']))
        self.max_spread = self.config.get('max_spread', 3.0)
        self.min_balance = self.config.get('min_balance', 10000.0)
        
        # Threading
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._tick_thread = None
        
        # Telemetry setup
        if self.telemetry:
            self.telemetry.register_metric('mt5_connection_uptime', 'gauge')
            self.telemetry.register_metric('mt5_tick_latency', 'histogram')
            self.telemetry.register_metric('mt5_spread_alert_count', 'counter')
            self.telemetry.register_metric('mt5_connection_errors', 'counter')
        
        # EventBus subscriptions
        if self.event_bus:
            self.event_bus.subscribe('market_hours_check', self._handle_market_hours_check)
            self.event_bus.subscribe('connection_health_request', self._handle_health_request)
            self.event_bus.subscribe('emergency_disconnect', self._handle_emergency_disconnect)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("MT5Connector initialized")
    
    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if not MT5_AVAILABLE:
                self.logger.error("MT5 not available - cannot connect")
                if self.telemetry:
                    self.telemetry.increment('mt5_connection_errors')
                return False
            
            self.status = ConnectionStatus.CONNECTING
            self.logger.info(f"Connecting to MT5 server: {self.server}")
            
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                self.status = ConnectionStatus.ERROR
                if self.telemetry:
                    self.telemetry.increment('mt5_connection_errors')
                return False
            
            # Login to account
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    self.status = ConnectionStatus.ERROR
                    if self.telemetry:
                        self.telemetry.increment('mt5_connection_errors')
                    return False
            
            self.status = ConnectionStatus.CONNECTED
            self.logger.info("MT5 connection established successfully")
            
            # Emit connection event
            if self.event_bus:
                self.event_bus.emit('mt5_connected', {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'server': self.server,
                    'account': self.login
                })
            
            # Start monitoring threads
            self._start_monitoring()
            
            if self.telemetry:
                self.telemetry.set_gauge('mt5_connection_uptime', 1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            self.status = ConnectionStatus.ERROR
            if self.telemetry:
                self.telemetry.increment('mt5_connection_errors')
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from MT5 terminal
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            self.logger.info("Disconnecting from MT5")
            self._stop_event.set()
            
            # Wait for threads to stop
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)
            
            if self._tick_thread and self._tick_thread.is_alive():
                self._tick_thread.join(timeout=5)
            
            # Shutdown MT5
            if MT5_AVAILABLE:
                mt5.shutdown()
            
            self.status = ConnectionStatus.DISCONNECTED
            
            # Emit disconnection event
            if self.event_bus:
                self.event_bus.emit('mt5_disconnected', {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'reason': 'manual_disconnect'
                })
            
            if self.telemetry:
                self.telemetry.set_gauge('mt5_connection_uptime', 0)
            
            self.logger.info("MT5 disconnected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 disconnection error: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if MT5 is connected"""
        return self.status == ConnectionStatus.CONNECTED
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get current account information
        
        Returns:
            AccountInfo: Account details or None if error
        """
        try:
            if not MT5_AVAILABLE or not self.is_connected():
                return None
            
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None
            
            return AccountInfo(
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.margin_free,
                margin_level=account_info.margin_level,
                currency=account_info.currency,
                leverage=account_info.leverage,
                profit=account_info.profit
            )
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_tick_data(self, symbol: str) -> Optional[TickData]:
        """
        Get latest tick data for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TickData: Latest tick or None if error
        """
        try:
            if not MT5_AVAILABLE or not self.is_connected():
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            spread = round((tick.ask - tick.bid) / tick.ask * 100000, 1)
            
            tick_data = TickData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                spread=spread,
                time=datetime.fromtimestamp(tick.time, tz=timezone.utc),
                volume=tick.volume
            )
            
            # Check spread alert
            if spread > self.max_spread:
                if self.event_bus:
                    self.event_bus.emit('spread_alert', {
                        'symbol': symbol,
                        'spread': spread,
                        'max_spread': self.max_spread,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                
                if self.telemetry:
                    self.telemetry.increment('mt5_spread_alert_count')
            
            return tick_data
            
        except Exception as e:
            self.logger.error(f"Error getting tick data for {symbol}: {e}")
            return None
    
    def get_market_session(self) -> MarketSession:
        """
        Determine current market session
        
        Returns:
            MarketSession: Current market session
        """
        try:
            now = datetime.now(timezone.utc)
            hour = now.hour
            weekday = now.weekday()
            
            # Weekend - market closed
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                return MarketSession.CLOSED
            
            # Friday evening to Sunday evening
            if weekday == 4 and hour >= 21:  # Friday 9 PM UTC
                return MarketSession.CLOSED
            
            # Sunday evening - pre-market
            if weekday == 6 and hour >= 21:  # Sunday 9 PM UTC
                return MarketSession.PRE_MARKET
            
            # Main market hours (Monday 00:00 to Friday 21:00 UTC)
            if weekday < 4 or (weekday == 4 and hour < 21):
                return MarketSession.OPEN
            
            return MarketSession.CLOSED
            
        except Exception as e:
            self.logger.error(f"Error determining market session: {e}")
            return MarketSession.CLOSED
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        self._stop_event.clear()
        
        # Start account monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_connection, daemon=True)
        self._monitor_thread.start()
        
        # Start tick streaming thread
        self._tick_thread = threading.Thread(target=self._stream_ticks, daemon=True)
        self._tick_thread.start()
    
    def _monitor_connection(self):
        """Monitor connection health and account status"""
        while not self._stop_event.wait(5):
            try:
                if not self.is_connected():
                    continue
                
                # Check account health
                account_info = self.get_account_info()
                if account_info:
                    # FTMO compliance check
                    if account_info.balance < self.min_balance:
                        self.logger.warning(f"Low account balance: {account_info.balance}")
                        
                        if self.event_bus:
                            self.event_bus.emit('account_balance_warning', {
                                'balance': account_info.balance,
                                'min_balance': self.min_balance,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                    
                    # Update telemetry
                    if self.telemetry:
                        self.telemetry.set_gauge('mt5_account_balance', account_info.balance)
                        self.telemetry.set_gauge('mt5_account_equity', account_info.equity)
                        self.telemetry.set_gauge('mt5_margin_level', account_info.margin_level)
                
            except Exception as e:
                self.logger.error(f"Connection monitoring error: {e}")
    
    def _stream_ticks(self):
        """Stream real-time tick data"""
        while not self._stop_event.wait(0.1):
            try:
                if not self.is_connected():
                    continue
                
                start_time = time.time()
                
                for symbol in self.symbols:
                    tick_data = self.get_tick_data(symbol)
                    if tick_data:
                        # Emit tick data event
                        if self.event_bus:
                            self.event_bus.emit('tick_data', tick_data.to_dict())
                        
                        # Update latency telemetry
                        if self.telemetry:
                            latency = (time.time() - start_time) * 1000
                            self.telemetry.timer('mt5_tick_latency').observe(latency)
                
            except Exception as e:
                self.logger.error(f"Tick streaming error: {e}")
    
    def _handle_market_hours_check(self, data: Dict[str, Any]):
        """Handle market hours check request"""
        try:
            session = self.get_market_session()
            
            if self.event_bus:
                self.event_bus.emit('market_session_update', {
                    'session': session.value,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Market hours check error: {e}")
    
    def _handle_health_request(self, data: Dict[str, Any]):
        """Handle connection health request"""
        try:
            health_data = {
                'status': self.status.value,
                'connected': self.is_connected(),
                'server': self.server,
                'account': self.login,
                'symbols': list(self.symbols),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            account_info = self.get_account_info()
            if account_info:
                health_data['account_info'] = account_info.to_dict()
            
            if self.event_bus:
                self.event_bus.emit('connection_health_response', health_data)
                
        except Exception as e:
            self.logger.error(f"Health request error: {e}")
    
    def _handle_emergency_disconnect(self, data: Dict[str, Any]):
        """Handle emergency disconnect request"""
        try:
            self.logger.warning("Emergency disconnect requested")
            self.disconnect()
            
        except Exception as e:
            self.logger.error(f"Emergency disconnect error: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def main():
    """Test MT5 connector functionality"""
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'login': 12345678,  # Your MT5 login
        'password': 'your_password',  # Your MT5 password
        'server': 'YourBroker-Server',  # Your broker server
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'max_spread': 2.0,
        'min_balance': 10000.0
    }
    
    connector = MT5Connector(config)
    
    try:
        if connector.connect():
            print("✅ MT5 Connected successfully")
            
            # Test account info
            account_info = connector.get_account_info()
            if account_info:
                print(f"💰 Account Balance: {account_info.balance} {account_info.currency}")
                print(f"📊 Equity: {account_info.equity}")
                print(f"🎯 Margin Level: {account_info.margin_level}%")
            
            # Test tick data
            for symbol in ['EURUSD', 'GBPUSD']:
                tick = connector.get_tick_data(symbol)
                if tick:
                    print(f"📈 {symbol}: Bid={tick.bid}, Ask={tick.ask}, Spread={tick.spread}")
            
            # Test market session
            session = connector.get_market_session()
            print(f"🕐 Market Session: {session.value}")
            
            # Keep running for a bit to test monitoring
            time.sleep(10)
            
        else:
            print("❌ Failed to connect to MT5")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()
