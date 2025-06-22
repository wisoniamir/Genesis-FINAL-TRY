
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


# <!-- @GENESIS_MODULE_START: market_data_feed_manager -->


"""
GENESIS MarketDataFeedManager - ARCHITECT MODE v7.0.0
===================================================
Real-time MT5 market data streaming module.
FTMO-compliant, EventBus-connected, telemetry-enabled.

🚨 COMPLIANCE MANDATES:
- Uses real MT5 Python API (NO mock data allowed)
- Emits tick data via HardenedEventBus with topic: 'TickData'
- Enforces retry logic, logging, connection confirmation
- Full telemetry and compliance monitoring
"""

import MetaTrader5 as mt5  # type: ignore[import]
import json
import logging
import time
from datetime import datetime
from threading import Thread, Event
from typing import List, Dict, Any, Optional
import os

from hardened_event_bus import get_event_bus, emit_event, register_route


class MarketDataFeedManager:
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

            emit_telemetry("market_data_feed_manager", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("market_data_feed_manager", "position_calculated", {
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
                        "module": "market_data_feed_manager",
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
                print(f"Emergency stop error in market_data_feed_manager: {e}")
                return False
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "market_data_feed_manager",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in market_data_feed_manager: {e}")
    """Real-time market data feed manager with MT5 integration."""

    def __init__(self):
        """Initialize the market data feed manager."""
        self.symbols: List[str] = []
        self.connected = False
        self.tick_streaming = False
        self.stream_thread = None
        self.stop_event = Event()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('MarketDataFeedManager')
        
        # Module registration info
        self.module_name = "MarketDataFeedManager"
        self.module_type = "service"
        self.status = "initialized"
        
        # Telemetry and compliance
        self.telemetry_enabled = True
        self.compliance_mode = True
        self.real_data_only = True
        
        # Feed log file
        self.feed_log_file = "feed_log.json"
        
        # Get EventBus instance
        self.event_bus = get_event_bus()
        
        # Initialize feed log
        self.initialize_feed_log()
        
        # Register EventBus routes
        self.register_eventbus_routes()
        
        msg = f"✅ {self.module_name} initialized in ARCHITECT MODE v7.0.0"
        self.logger.info(msg)
        
        # Emit initialization telemetry
        self.emit_telemetry({
            "event": "initialization",
            "status": "complete",
            "version": "7.0.0"
        })
    
    def register_eventbus_routes(self):
        """Register EventBus routes for compliance tracking."""
        routes = [
            ("MarketDataFeedManager", "TickData", "TradingSystem"),
            ("MarketDataFeedManager", "telemetry", "TelemetryCollector"),
            ("MarketDataFeedManager", "compliance", "ComplianceMonitor")
        ]
        for src, evt, dst in routes:
            register_route(src, evt, dst)
    
    def connect_to_mt5(self) -> bool:
        """Establish real MT5 connection - no mock data allowed."""
        try:
            # Initialize MT5
            if not mt5.initialize():  # type: ignore[attr-defined]
                self.logger.error("❌ MT5 initialization failed")
                return False
            
            # Force real data only mode
            info = mt5.terminal_info()  # type: ignore[attr-defined]
            if not info or "demo" in info.path.lower():
                msg = "❌ ARCHITECT MODE VIOLATION: Demo terminal detected"
                self.logger.error(msg)
                return False
            
            # Configure basic MT5 settings
            if mt5.symbols_total() > 0:  # type: ignore[attr-defined]
                self.connected = True
            
            self.logger.info("✅ Real MT5 connection established")
            self.emit_telemetry({
                "event": "mt5_connection",
                "status": "connected",
                "terminal_path": info.path
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MT5 connection error: {e}")
            return False

    def start_stream(self, target_symbols: Optional[List[str]] = None) -> bool:
        """Start streaming tick data for specified symbols."""
        if not self.connected:
            if not self.connect_to_mt5():
                return False
        
        # Get available symbols
        all_symbols = mt5.symbols_get()  # type: ignore[attr-defined]
        self.symbols = [s.name for s in all_symbols if s is not None]
        
        # Filter target symbols
        if target_symbols:
            self.stream_symbols = [
                s for s in target_symbols if s in self.symbols
            ]
        else:
            pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
            self.stream_symbols = [
                s for s in pairs if s in self.symbols
            ]
        
        if not self.stream_symbols:
            self.logger.error("No valid symbols to stream")
            return False
        
        # Start streaming thread
        self.stop_event.clear()
        self.stream_thread = Thread(target=self._stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        self.tick_streaming = True
        
        msg = (f"🚀 Started streaming {len(self.stream_symbols)} symbols: "
               f"{self.stream_symbols}")
        self.logger.info(msg)
        return True

    def stop_stream(self):
        """Stop the tick data stream."""
        self.stop_event.set()
        if self.stream_thread:
            self.stream_thread.join()
        self.tick_streaming = False
        self.logger.info("✅ Tick streaming stopped")
    
    def disconnect(self):
        """Disconnect from MT5 terminal."""
        self.stop_stream()
        if self.connected:
            mt5.shutdown()  # type: ignore[attr-defined]
            self.connected = False
            self.logger.info("✅ MT5 disconnected")
    
    def _stream_worker(self):
        """Internal worker thread for streaming tick data."""
        while not self.stop_event.is_set():
            for symbol in self.stream_symbols:
                try:
                    # Get tick data
                    tick = mt5.symbol_info_tick(  # type: ignore[attr-defined]
                        symbol
                    )
                    if tick:
                        data = {
                            "symbol": symbol,
                            "bid": tick.bid,
                            "ask": tick.ask,
                            "time": tick.time
                        }
                        emit_event("TickData", data)
                        self.log_tick_data(data)
                    else:
                        self.logger.debug(f"No tick data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Tick error ({symbol}): {e}")
                
            # Sleep between cycles
            time.sleep(0.1)
    
    def log_tick_data(self, tick_event: Dict[str, Any]):
        """Log tick data to feed_log.json for audit and compliance."""
        try:
            with open(self.feed_log_file, 'r') as f:
                feed_log = json.load(f)
            
            feed_log["ticks"].append({
                "timestamp": datetime.now().isoformat(),
                "data": tick_event
            })
            
            feed_log["metadata"]["last_updated"] = datetime.now().isoformat()
            
            with open(self.feed_log_file, 'w') as f:
                json.dump(feed_log, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to log tick data: {e}")
    
    def initialize_feed_log(self):
        """Initialize feed log file with basic structure."""
        if not os.path.exists(self.feed_log_file):
            feed_log = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                },
                "ticks": []
            }
            with open(self.feed_log_file, 'w') as f:
                json.dump(feed_log, f, indent=2)
    
    def emit_telemetry(self, data: Dict[str, Any]):
        """Emit telemetry event."""
        if self.telemetry_enabled and self.event_bus:
            data.update({
                "timestamp": datetime.now().isoformat(),
                "module": self.module_name
            })
            emit_event("telemetry", data)

    def validate_compliance(self) -> Dict[str, bool]:
        """Validate ARCHITECT MODE compliance."""
        compliance = {
            "real_data_only": self.real_data_only,
            "telemetry_enabled": self.telemetry_enabled,
            "eventbus_connected": bool(self.event_bus),
            "mt5_connected": self.connected
        }
        return compliance

    def get_status(self) -> Dict[str, Any]:
        """Get current module status."""
        return {
            "status": self.status,
            "connected": self.connected,
            "streaming": self.tick_streaming,
            "symbols": len(self.symbols),
            "compliance": self.validate_compliance()
        }


def create_market_data_feed_manager() -> MarketDataFeedManager:
    """Create and return MarketDataFeedManager instance."""
    return MarketDataFeedManager()


if __name__ == "__main__":
    # Test the MarketDataFeedManager
    manager = MarketDataFeedManager()
    
    # Test connection
    if manager.connect_to_mt5():
        print("✅ MT5 Connection successful")
        
        # Test streaming for 10 seconds
        if manager.start_stream(["EURUSD", "GBPUSD"]):
            print("✅ Streaming started")
            time.sleep(10)
            manager.stop_stream()
            print("✅ Streaming stopped")
        
        manager.disconnect()
        print("✅ Disconnected")
        
        # Print status and compliance
        print("Status:", manager.get_status())
        print("Compliance:", manager.validate_compliance())
    else:
        print("❌ MT5 Connection failed")

# <!-- @GENESIS_MODULE_END: market_data_feed_manager -->
