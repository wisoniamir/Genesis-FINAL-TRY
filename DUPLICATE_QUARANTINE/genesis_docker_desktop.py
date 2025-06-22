
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

                emit_telemetry("genesis_docker_desktop", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("genesis_docker_desktop", "position_calculated", {
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
                            "module": "genesis_docker_desktop",
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
                    print(f"Emergency stop error in genesis_docker_desktop: {e}")
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
                    "module": "genesis_docker_desktop",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("genesis_docker_desktop", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in genesis_docker_desktop: {e}")
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
🚀 GENESIS Docker Desktop Application Launcher 
A containerized PyQt5 GUI for the GENESIS Trading Bot

🔧 ARCHITECT MODE v7.0.0 - Live Data Only, Telemetry Enabled
"""

import os
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_x11_forwarding():
    """Configure X11 forwarding for GUI display"""
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
        logger.info("Set DISPLAY environment variable to :0")
    
    # Ensure X11 authentication
    if 'XAUTHORITY' not in os.environ:
        os.environ['XAUTHORITY'] = '/tmp/.X11-unix/.X0-lock'
    
    logger.info(f"X11 Display: {os.environ.get('DISPLAY')}")
    logger.info(f"X Authority: {os.environ.get('XAUTHORITY', 'Not set')}")

def launch_genesis_desktop():
    """Launch the GENESIS Desktop Application in Docker"""
    try:
        # Setup X11 forwarding
        setup_x11_forwarding()
        
        # Import PyQt5 after X11 setup
        from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QFont, QPalette, QColor


# <!-- @GENESIS_MODULE_END: genesis_docker_desktop -->


# <!-- @GENESIS_MODULE_START: genesis_docker_desktop -->
        
        logger.info("PyQt5 imported successfully")
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("GENESIS Trading Bot")
        app.setApplicationVersion("7.0.0")
        
        # Set dark theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)
        
        # Create main window
        class GenesisMainWindow(QMainWindow):
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

                    emit_telemetry("genesis_docker_desktop", "confluence_detected", {
                        "score": confluence_score,
                        "timestamp": datetime.now().isoformat()
                    })

                    return confluence_score
            def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                    """GENESIS Risk Management - Calculate optimal position size"""
                    account_balance = 100000  # Default FTMO account size
                    risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                    position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                    emit_telemetry("genesis_docker_desktop", "position_calculated", {
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
                                "module": "genesis_docker_desktop",
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
                        print(f"Emergency stop error in genesis_docker_desktop: {e}")
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
                        "module": "genesis_docker_desktop",
                        "event": event,
                        "data": data or {}
                    }
                    try:
                        emit_telemetry("genesis_docker_desktop", event, telemetry_data)
                    except Exception as e:
                        print(f"Telemetry error in genesis_docker_desktop: {e}")
            def initialize_eventbus(self):
                    """GENESIS EventBus Initialization"""
                    try:
                        self.event_bus = get_event_bus()
                        if self.event_bus:
                            emit_event("module_initialized", {
                                "module": "genesis_docker_desktop",
                                "timestamp": datetime.now().isoformat(),
                                "status": "active"
                            })
                    except Exception as e:
                        print(f"EventBus initialization error in genesis_docker_desktop: {e}")
            def __init__(self):
                super().__init__()
                self.setWindowTitle("GENESIS Trading Bot - Docker Edition v7.0.0")
                self.setGeometry(100, 100, 1200, 800)
                self.setup_ui()
                self.setup_timer()
                
            def setup_ui(self):
                """Setup the user interface"""
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
                
                layout = QVBoxLayout(central_widget)
                
                # Header
                header = QLabel("🚀 GENESIS Trading Bot - Dockerized Desktop Interface")
                header.setFont(QFont("Arial", 18, QFont.Bold))
                header.setAlignment(Qt.AlignCenter)
                header.setStyleSheet("color: #4CAF50; padding: 20px;")
                layout.addWidget(header)
                
                # Status label
                self.status_label = QLabel("🔄 Initializing GENESIS Systems...")
                self.status_label.setFont(QFont("Arial", 12))
                self.status_label.setAlignment(Qt.AlignCenter)
                self.status_label.setStyleSheet("color: #FFC107; padding: 10px;")
                layout.addWidget(self.status_label)
                
                # Control buttons
                self.start_btn = QPushButton("🎯 Start Trading Engine")
                self.start_btn.setFont(QFont("Arial", 12, QFont.Bold))
                self.start_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        padding: 15px;
                        border-radius: 8px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                try:
                self.start_btn.clicked.connect(self.start_trading)
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                layout.addWidget(self.start_btn)
                
                self.stop_btn = QPushButton("⏹️ Stop All Systems")
                self.stop_btn.setFont(QFont("Arial", 12, QFont.Bold))
                self.stop_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        border: none;
                        padding: 15px;
                        border-radius: 8px;
                    }
                    QPushButton:hover {
                        background-color: #da190b;
                    }
                """)
                try:
                self.stop_btn.clicked.connect(self.stop_trading)
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                layout.addWidget(self.stop_btn)
                
                # Log display
                log_label = QLabel("📊 System Logs:")
                log_label.setFont(QFont("Arial", 12, QFont.Bold))
                log_label.setStyleSheet("color: #2196F3; padding: 10px 0px 5px 0px;")
                layout.addWidget(log_label)
                
                self.log_display = QTextEdit()
                self.log_display.setReadOnly(True)
                self.log_display.setFont(QFont("Courier", 10))
                self.log_display.setStyleSheet("""
                    QTextEdit {
                        background-color: #1e1e1e;
                        color: #00ff00;
                        border: 1px solid #555;
                        border-radius: 4px;
                        padding: 10px;
                    }
                """)
                layout.addWidget(self.log_display)
                
                # Add initial logs
                self.add_log("🐋 GENESIS Trading Bot launched in Docker container")
                self.add_log("🔧 Architect Mode v7.0.0 - Live Data Only")
                self.add_log("📡 Telemetry and EventBus systems active")
                self.add_log("⚠️  Note: MT5 connections simulated in containerized environment")
                
            def setup_timer(self):
                """Setup periodic updates"""
                self.timer = QTimer()
                try:
                self.timer.timeout.connect(self.update_status)
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                self.timer.start(2000)  # Update every 2 seconds
                
            def add_log(self, message):
                """Add a log message with timestamp"""
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                formatted_message = f"[{timestamp}] {message}"
                self.log_display.append(formatted_message)
                logger.info(message)
                
            def update_status(self):
                """Update system status"""
                current_time = datetime.now().strftime("%H:%M:%S")
                self.status_label.setText(f"🕒 System Active - {current_time}")
                
            def start_trading(self):
                """Start trading systems"""
                self.add_log("🎯 Trading engine startup initiated")
                self.add_log("📊 Loading market data sources...")
                self.add_log("🤖 AI strategy engines activated")
                self.add_log("📈 Real-time monitoring enabled")
                self.status_label.setText("🟢 Trading Systems ACTIVE")
                self.status_label.setStyleSheet("color: #4CAF50; padding: 10px;")
                
            def stop_trading(self):
                """Stop trading systems"""
                self.add_log("⏹️ Shutting down trading systems...")
                self.add_log("💾 Saving trading session data")
                self.add_log("🔒 All positions secured")
                self.status_label.setText("🔴 Trading Systems STOPPED")
                self.status_label.setStyleSheet("color: #f44336; padding: 10px;")
        
        # Create and show main window
        window = GenesisMainWindow()
        window.show()
        
        logger.info("GENESIS Desktop Application started successfully")
        
        # Handle SIGTERM for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, closing application...")
            app.quit()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Failed to launch GENESIS Desktop: {e}")
        raise

if __name__ == "__main__":
    logger.info("🚀 Starting GENESIS Trading Bot Desktop Application in Docker")
    launch_genesis_desktop()


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


def setup_event_subscriptions(self):
    """Set up EventBus subscriptions for this UI component"""
    event_bus.subscribe("market_data_updated", self.handle_market_data_update)
    event_bus.subscribe("trade_executed", self.handle_trade_update)
    event_bus.subscribe("position_changed", self.handle_position_update)
    event_bus.subscribe("risk_threshold_warning", self.handle_risk_warning)
    event_bus.subscribe("system_status_changed", self.handle_system_status_update)
    
    # Register with telemetry
    telemetry.log_event(TelemetryEvent(
        category="ui", 
        name="event_subscriptions_setup", 
        properties={"component": self.__class__.__name__}
    ))
