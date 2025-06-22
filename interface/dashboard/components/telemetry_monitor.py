
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

System Telemetry Monitor Component
"""

from typing import Dict, Any, List
import json
import logging
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QTreeWidget, QTreeWidgetItem, QPushButton)
from PyQt5.QtCore import Qt, QTimer

from core.telemetry import emit_telemetry
from core.event_bus import event_bus

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: telemetry_monitor -->


# <!-- @GENESIS_MODULE_START: telemetry_monitor -->

class TelemetryMonitor(QWidget):
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

            emit_telemetry("telemetry_monitor", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("telemetry_monitor", "position_calculated", {
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
                        "module": "telemetry_monitor",
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
                print(f"Emergency stop error in telemetry_monitor: {e}")
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
                        "module": "telemetry_monitor",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in telemetry_monitor: {e}")
    """Real-time system telemetry monitor"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI
        layout = QVBoxLayout(self)
        
        # Add controls
        self._add_controls(layout)
        
        # Add telemetry tree
        self._add_telemetry_tree(layout)
        
        # Initialize update timer
        self._init_timer()
        
        # Subscribe to events
        self._subscribe_to_events()
        
    def _add_controls(self, layout: QVBoxLayout):
        """Add control buttons"""
        control_layout = QHBoxLayout()
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        try:
        refresh_btn.clicked.connect(self._refresh_telemetry)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        control_layout.addWidget(refresh_btn)
        
        # Clear button
        clear_btn = QPushButton("Clear")
        try:
        clear_btn.clicked.connect(self._clear_telemetry)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        control_layout.addWidget(clear_btn)
        
        # Add to main layout
        layout.addLayout(control_layout)
        
    def _add_telemetry_tree(self, layout: QVBoxLayout):
        """Add telemetry tree widget"""
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([
            "Module", "Event", "Time", "Data"
        ])
        
        # Auto-resize columns
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(3)  # Data column stretches
        
        layout.addWidget(self.tree)
        
    def _init_timer(self):
        """Initialize update timer"""
        self.update_timer = QTimer()
        try:
        self.update_timer.timeout.connect(self._refresh_telemetry)
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        self.update_timer.start(5000)  # Refresh every 5 seconds
        
    def _subscribe_to_events(self):
        """Subscribe to telemetry events"""
        event_bus.subscribe("telemetry", self._handle_telemetry)
        
    def _refresh_telemetry(self):
        """Refresh telemetry data"""
        try:
            # Load telemetry file
            telemetry_path = Path("telemetry.json")
            if not telemetry_path.exists():
                return
                
            with open(telemetry_path, "r") as f:
                telemetry = json.load(f)
                
            # Clear tree
            self.tree.clear()
            
            # Add telemetry data
            for event in telemetry.get("events", []):
                self._add_telemetry_item(event)
                
            # Emit refresh event
            emit_telemetry("telemetry_monitor", "refreshed", {
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to refresh telemetry: {e}")
            
    def _clear_telemetry(self):
        """Clear telemetry data"""
        try:
            # Clear tree
            self.tree.clear()
            
            # Clear telemetry file
            telemetry_path = Path("telemetry.json")
            if telemetry_path.exists():
                with open(telemetry_path, "w") as f:
                    json.dump({"events": []}, f)
                    
            # Emit clear event
            emit_telemetry("telemetry_monitor", "cleared", {
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to clear telemetry: {e}")
            
    def _handle_telemetry(self, data: Dict[str, Any]):
        """Handle new telemetry event"""
        try:
            # Add to tree
            self._add_telemetry_item(data)
            
            # Keep only last 1000 items
            while self.tree.topLevelItemCount() > 1000:
                self.tree.takeTopLevelItem(0)
                
        except Exception as e:
            self.logger.error(f"Failed to handle telemetry: {e}")
            
    def _add_telemetry_item(self, data: Dict[str, Any]):
        """Add telemetry item to tree"""
        try:
            # Create tree item
            item = QTreeWidgetItem([
                data.get("module", ""),
                data.get("event", ""),
                data.get("timestamp", ""),
                json.dumps(data.get("data", {}), indent=2)
            ])
            
            # Add to tree
            self.tree.addTopLevelItem(item)
            
            # Auto-expand
            item.setExpanded(True)
            
        except Exception as e:
            self.logger.error(f"Failed to add telemetry item: {e}")
            
    def closeEvent(self, event):
        """Handle monitor close"""
        try:
            # Stop update timer
            self.update_timer.stop()
            
            # Emit telemetry
            emit_telemetry("telemetry_monitor", "closed", {
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
