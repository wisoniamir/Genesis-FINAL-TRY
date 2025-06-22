
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


"""


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

Trade Execution Console Component
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QTableWidget, QTableWidgetItem, QPushButton,
                           QMessageBox)
from PyQt5.QtCore import Qt, QTimer

import MetaTrader5 as mt5
from core.telemetry import emit_telemetry
from core.event_bus import event_bus

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: execution_console -->


# <!-- @GENESIS_MODULE_START: execution_console -->

class ExecutionConsole(QWidget):
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

            emit_telemetry("execution_console", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("execution_console", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "execution_console",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in execution_console: {e}")
    """Real-time trade execution console"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI
        layout = QVBoxLayout(self)
        
        # Add controls
        self._add_controls(layout)
        
        # Add trade table
        self._add_trade_table(layout)
        
        # Subscribe to events
        self._subscribe_to_events()
        
    def _add_controls(self, layout: QVBoxLayout):
        """Add control buttons"""
        control_layout = QHBoxLayout()
        
        # Kill switch button
        self.kill_switch = QPushButton("🚨 KILL SWITCH")
        try:
        self.kill_switch.clicked.connect(self._handle_kill_switch)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        self.kill_switch.setStyleSheet("background-color: #ff4444; color: white;")
        control_layout.addWidget(self.kill_switch)
        
        # Close all button
        close_all = QPushButton("Close All Positions")
        try:
        close_all.clicked.connect(self._handle_close_all)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        control_layout.addWidget(close_all)
        
        # Add to main layout
        layout.addLayout(control_layout)
        
    def _add_trade_table(self, layout: QVBoxLayout):
        """Add trade execution table"""
        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(8)
        self.trade_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Type", "Volume", "Price", "SL", "TP", "Profit"
        ])
        
        # Auto-resize columns
        self.trade_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.trade_table)
        
    def _subscribe_to_events(self):
        """Subscribe to trading events"""
        event_bus.subscribe("trade_executed", self._handle_trade_executed)
        event_bus.subscribe("position_closed", self._handle_position_closed)
        event_bus.subscribe("trade_error", self._handle_trade_error)
        
    def _handle_kill_switch(self):
        """Handle kill switch activation"""
        try:
            reply = QMessageBox.question(
                self,
                "Kill Switch Confirmation",
                "Are you sure you want to activate the kill switch?\n"
                "This will close all positions and stop trading.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Close all positions
                self._close_all_positions()
                
                # Emit kill switch event
                emit_telemetry("execution", "kill_switch_activated", {
                    "timestamp": datetime.now().isoformat()
                })
                
                # Show confirmation
                QMessageBox.information(
                    self,
                    "Kill Switch Activated",
                    "All positions have been closed.\n"
                    "Trading has been stopped."
                )
                
        except Exception as e:
            self.logger.error(f"Kill switch error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to activate kill switch: {str(e)}"
            )
            
    def _handle_close_all(self):
        """Handle close all positions request"""
        try:
            reply = QMessageBox.question(
                self,
                "Close All Confirmation",
                "Are you sure you want to close all positions?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Close all positions
                self._close_all_positions()
                
                # Show confirmation
                QMessageBox.information(
                    self,
                    "Positions Closed",
                    "All positions have been closed."
                )
                
        except Exception as e:
            self.logger.error(f"Close all error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to close all positions: {str(e)}"
            )
            
    def _close_all_positions(self):
        """Close all open positions"""
        try:
            # Get open positions
            positions = mt5.positions_get()
            if positions is None:
                self.logger.error("Failed to get positions")
                return
                
            # Close each position
            for position in positions:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": position.price_current,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "GENESIS kill switch",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result is None:
                    self.logger.error(f"Failed to close position {position.ticket}")
                    continue
                    
                # Emit telemetry
                emit_telemetry("execution", "position_closed", {
                    "ticket": position.ticket,
                    "symbol": position.symbol,
                    "profit": position.profit,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Close positions error: {e}")
            raise
            
    def _handle_trade_executed(self, data: Dict[str, Any]):
        """Handle trade execution event"""
        try:
            # Add new row
            row = self.trade_table.rowCount()
            self.trade_table.insertRow(row)
            
            # Add trade data
            self.trade_table.setItem(row, 0, 
                                   QTableWidgetItem(data.get("time", "")))
            self.trade_table.setItem(row, 1, 
                                   QTableWidgetItem(data.get("symbol", "")))
            self.trade_table.setItem(row, 2, 
                                   QTableWidgetItem(data.get("type", "")))
            self.trade_table.setItem(row, 3, 
                                   QTableWidgetItem(str(data.get("volume", ""))))
            self.trade_table.setItem(row, 4, 
                                   QTableWidgetItem(str(data.get("price", ""))))
            self.trade_table.setItem(row, 5, 
                                   QTableWidgetItem(str(data.get("sl", ""))))
            self.trade_table.setItem(row, 6, 
                                   QTableWidgetItem(str(data.get("tp", ""))))
            self.trade_table.setItem(row, 7, 
                                   QTableWidgetItem(str(data.get("profit", "0.00"))))
                                   
        except Exception as e:
            self.logger.error(f"Failed to handle trade execution: {e}")
            
    def _handle_position_closed(self, data: Dict[str, Any]):
        """Handle position closed event"""
        try:
            ticket = data.get("ticket")
            if not ticket:
                return
                
            # Find and update row
            for row in range(self.trade_table.rowCount()):
                if self.trade_table.item(row, 0).text() == str(ticket):
                    self.trade_table.setItem(
                        row, 7,
                        QTableWidgetItem(str(data.get("profit", "0.00")))
                    )
                    break
                    
        except Exception as e:
            self.logger.error(f"Failed to handle position closed: {e}")
            
    def _handle_trade_error(self, data: Dict[str, Any]):
        """Handle trade error event"""
        try:
            error = data.get("error", "Unknown error")
            
            QMessageBox.warning(
                self,
                "Trade Error",
                f"Trade execution failed: {error}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle trade error: {e}")
            
    def closeEvent(self, event):
        """Handle console close"""
        try:
            # Emit telemetry
            emit_telemetry("execution", "console_closed", {
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error in close event: {e}")
            
        event.accept()



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))


def check_ftmo_limits(order_volume: float, symbol: str) -> bool:
    """Check order against FTMO trading limits"""
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to get account info")
        return False
    
    # Calculate position size as percentage of account
    equity = account_info.equity
    max_risk_percent = 0.05  # 5% max risk per trade (FTMO rule)
    
    # Calculate potential loss
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return False
    
    # Check if order volume exceeds max risk
    if (order_volume * symbol_info.trade_tick_value) > (equity * max_risk_percent):
        logging.warning(f"Order volume {order_volume} exceeds FTMO risk limit of {equity * max_risk_percent}")
        return False
    
    # Check daily loss limit
    daily_loss_limit = equity * 0.05  # 5% daily loss limit
    
    # Get today's closed positions
    from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    positions = mt5.history_deals_get(from_date, datetime.now())
    
    daily_pnl = sum([deal.profit for deal in positions if deal.profit < 0])
    
    if abs(daily_pnl) + (order_volume * symbol_info.trade_tick_value) > daily_loss_limit:
        logging.warning(f"Order would breach FTMO daily loss limit. Current loss: {abs(daily_pnl)}")
        return False
    
    return True
