
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

Trade Signal Feed Component
"""

from typing import Dict, Any, List
import json
import logging
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QTableWidget, QTableWidgetItem, QPushButton,
                           QComboBox)
from PyQt5.QtCore import Qt, QTimer

from core.telemetry import emit_telemetry
from core.event_bus import event_bus

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: signal_feed -->


# <!-- @GENESIS_MODULE_START: signal_feed -->

class SignalFeed(QWidget):
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

            emit_telemetry("signal_feed", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("signal_feed", "position_calculated", {
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
                        "module": "signal_feed",
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
                print(f"Emergency stop error in signal_feed: {e}")
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
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_feed",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_feed: {e}")
    """Real-time trade signal feed"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI
        layout = QVBoxLayout(self)
        
        # Add controls
        self._add_controls(layout)
        
        # Add signal table
        self._add_signal_table(layout)
        
        # Initialize update timer
        self._init_timer()
        
        # Subscribe to events
        self._subscribe_to_events()
        
    def _add_controls(self, layout: QVBoxLayout):
        """Add control elements"""
        control_layout = QHBoxLayout()
        
        # Add symbol filter
        symbol_label = QLabel("Symbol:")
        control_layout.addWidget(symbol_label)
        
        self.symbol_filter = QComboBox()
        try:
        self.symbol_filter.currentTextChanged.connect(self._apply_filters)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        control_layout.addWidget(self.symbol_filter)
        
        # Add type filter
        type_label = QLabel("Type:")
        control_layout.addWidget(type_label)
        
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", "OB", "Divergence", "S/R"])
        try:
        self.type_filter.currentTextChanged.connect(self._apply_filters)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        control_layout.addWidget(self.type_filter)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh")
        try:
        refresh_btn.clicked.connect(self._refresh_signals)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        control_layout.addWidget(refresh_btn)
        
        # Add to main layout
        layout.addLayout(control_layout)
        
    def _add_signal_table(self, layout: QVBoxLayout):
        """Add signal table widget"""
        self.signal_table = QTableWidget()
        self.signal_table.setColumnCount(7)
        self.signal_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Type", "Direction", "Price", "Score", "Status"
        ])
        
        # Auto-resize columns
        self.signal_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.signal_table)
        
        # Load initial symbols
        self._load_symbols()
        
    def _init_timer(self):
        """Initialize update timer"""
        self.update_timer = QTimer()
        try:
        self.update_timer.timeout.connect(self._refresh_signals)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        self.update_timer.start(5000)  # Refresh every 5 seconds
        
    def _subscribe_to_events(self):
        """Subscribe to signal events"""
        event_bus.subscribe("trade_signal", self._handle_signal)
        event_bus.subscribe("signal_update", self._handle_signal_update)
        
    def _load_symbols(self):
        """Load available symbols"""
        try:
            # Load symbol list
            symbols_path = Path("symbols.json")
            if symbols_path.exists():
                with open(symbols_path, "r") as f:
                    symbols = json.load(f).get("symbols", [])
                    
                # Add to filter
                self.symbol_filter.addItem("All")
                self.symbol_filter.addItems(symbols)
                
        except Exception as e:
            self.logger.error(f"Failed to load symbols: {e}")
            
    def _refresh_signals(self):
        """Refresh signal data"""
        try:
            # Load signals
            signals = self._load_signals()
            
            # Apply filters
            signals = self._filter_signals(signals)
            
            # Update table
            self._update_table(signals)
            
            # Emit telemetry
            emit_telemetry("signal_feed", "refreshed", {
                "signal_count": len(signals),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to refresh signals: {e}")
            
    def _load_signals(self) -> List[Dict[str, Any]]:
        """Load signals from storage"""
        try:
            signals_path = Path("signals.json")
            if not signals_path.exists():
                return []
                
            with open(signals_path, "r") as f:
                return json.load(f).get("signals", [])
                
        except Exception as e:
            self.logger.error(f"Failed to load signals: {e}")
            return []
            
    def _filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply filters to signals"""
        try:
            filtered = signals
            
            # Apply symbol filter
            symbol = self.symbol_filter.currentText()
            if symbol != "All":
                filtered = [s for s in filtered if s.get("symbol") == symbol]
                
            # Apply type filter
            signal_type = self.type_filter.currentText()
            if signal_type != "All":
                filtered = [s for s in filtered if s.get("type") == signal_type]
                
            return filtered
            
        except Exception as e:
            self.logger.error(f"Failed to filter signals: {e}")
            return signals
            
    def _update_table(self, signals: List[Dict[str, Any]]):
        """Update signal table"""
        try:
            # Clear table
            self.signal_table.setRowCount(0)
            
            # Add signals
            for signal in signals:
                row = self.signal_table.rowCount()
                self.signal_table.insertRow(row)
                
                # Add data
                self.signal_table.setItem(row, 0, 
                                        QTableWidgetItem(signal.get("time", "")))
                self.signal_table.setItem(row, 1, 
                                        QTableWidgetItem(signal.get("symbol", "")))
                self.signal_table.setItem(row, 2, 
                                        QTableWidgetItem(signal.get("type", "")))
                self.signal_table.setItem(row, 3, 
                                        QTableWidgetItem(signal.get("direction", "")))
                self.signal_table.setItem(row, 4, 
                                        QTableWidgetItem(str(signal.get("price", ""))))
                self.signal_table.setItem(row, 5, 
                                        QTableWidgetItem(str(signal.get("score", ""))))
                self.signal_table.setItem(row, 6, 
                                        QTableWidgetItem(signal.get("status", "")))
                                        
            # Auto-resize columns
            self.signal_table.resizeColumnsToContents()
            
        except Exception as e:
            self.logger.error(f"Failed to update table: {e}")
            
    def _apply_filters(self):
        """Apply current filters"""
        self._refresh_signals()
        
    def _handle_signal(self, signal: Dict[str, Any]):
        """Handle new signal event"""
        try:
            # Add signal to storage
            signals = self._load_signals()
            signals.append(signal)
            
            # Save signals
            signals_path = Path("signals.json")
            with open(signals_path, "w") as f:
                json.dump({"signals": signals}, f, indent=2)
                
            # Refresh display
            self._refresh_signals()
            
        except Exception as e:
            self.logger.error(f"Failed to handle signal: {e}")
            
    def _handle_signal_update(self, data: Dict[str, Any]):
        """Handle signal update event"""
        try:
            signal_id = data.get("id")
            if not signal_id:
                return
                
            # Update signal in storage
            signals = self._load_signals()
            for signal in signals:
                if signal.get("id") == signal_id:
                    signal.update(data)
                    break
                    
            # Save signals
            signals_path = Path("signals.json")
            with open(signals_path, "w") as f:
                json.dump({"signals": signals}, f, indent=2)
                
            # Refresh display
            self._refresh_signals()
            
        except Exception as e:
            self.logger.error(f"Failed to handle signal update: {e}")
            
    def closeEvent(self, event):
        """Handle feed close"""
        try:
            # Stop update timer
            self.update_timer.stop()
            
            # Emit telemetry
            emit_telemetry("signal_feed", "closed", {
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
