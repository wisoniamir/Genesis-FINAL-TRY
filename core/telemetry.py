
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

🌐 GENESIS HIGH ARCHITECTURE — TELEMETRY SYSTEM v1.0.0

Real-time telemetry with heartbeat monitoring and performance metrics.
ARCHITECT MODE v7.0.0 COMPLIANT.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import threading
import time

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: telemetry -->


# <!-- @GENESIS_MODULE_START: telemetry -->

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelemetrySystem:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "telemetry",
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
                print(f"Emergency stop error in telemetry: {e}")
                return False
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "telemetry",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in telemetry: {e}")
    """GENESIS telemetry system for real-time monitoring"""
    
    def __init__(self):
        self.telemetry_file = Path("telemetry.json")
        self.heartbeats = {}
        self.lock = threading.Lock()
        self._start_heartbeat_monitor()
    
    def _start_heartbeat_monitor(self):
        """Start the heartbeat monitoring thread"""
        def monitor():
            while True:
                self._check_heartbeats()
                time.sleep(5)  # Check every 5 seconds
                
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _check_heartbeats(self):
        """Check module heartbeats"""
        now = datetime.now()
        with self.lock:
            for module_id, last_beat in list(self.heartbeats.items()):
                if (now - last_beat).total_seconds() > 30:  # 30 second timeout
                    logger.warning(f"⚠️ Module {module_id} missed heartbeat")
                    self.emit_telemetry(
                        module_id,
                        "missed_heartbeat",
                        {"last_seen": last_beat.isoformat()}
                    )
    
    def emit_telemetry(self, module_id: str, event: str, data: Optional[Dict[str, Any]] = None):
        """Emit a telemetry event"""
        if data is None:
            data = {}
            
        telemetry_data = {
            "timestamp": datetime.now().isoformat(),
            "module_id": module_id,
            "event": event,
            "data": data
        }
        
        try:
            with self.lock:
                if self.telemetry_file.exists():
                    with open(self.telemetry_file, 'r') as f:
                        telemetry = json.load(f)
                else:
                    telemetry = {"events": []}
                
                telemetry["events"].append(telemetry_data)
                
                with open(self.telemetry_file, 'w') as f:
                    json.dump(telemetry, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to emit telemetry: {e}")
    
    def check_heartbeat(self, module_id: str) -> bool:
        """Check if a module's heartbeat is active"""
        with self.lock:
            if module_id in self.heartbeats:
                last_beat = self.heartbeats[module_id]
                return (datetime.now() - last_beat).total_seconds() <= 30
            return False
    
    def update_heartbeat(self, module_id: str):
        """Update a module's heartbeat timestamp"""
        with self.lock:
            self.heartbeats[module_id] = datetime.now()

# Create global instance
telemetry = TelemetrySystem()
emit_telemetry = telemetry.emit_telemetry
check_heartbeat = telemetry.check_heartbeat

class TelemetryManager:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "telemetry",
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
                print(f"Emergency stop error in telemetry: {e}")
                return False
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "telemetry",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in telemetry: {e}")
    """
    📊 TELEMETRY MANAGER
    
    ARCHITECT MODE COMPLIANCE:
    - ✅ Real-time monitoring
    - ✅ Module registration
    - ✅ Heartbeat management
    - ✅ Performance tracking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.registered_modules = {}
        self.performance_data = {}
        self.heartbeat_intervals = {}
        self.heartbeat_threads = {}
        
    def register_module(self, module_name: str, metadata: Dict[str, Any]):
        """Register a module for telemetry monitoring"""
        try:
            self.registered_modules[module_name] = {
                **metadata,
                "registered_at": datetime.now().isoformat(),
                "last_heartbeat": None,
                "status": "registered"
            }
            
            emit_telemetry("telemetry_manager", "module_registered", {
                "module_name": module_name,
                "metadata": metadata            })
            
            self.logger.info(f"📊 Module registered: {module_name}")
            
        except Exception as e:
            self.logger.error(f"Module registration error for {module_name}: {e}")
            
    def start_heartbeat(self, module_name: str, interval: int = 30):
        """Start heartbeat monitoring for a module"""
        try:
            self.heartbeat_intervals[module_name] = interval
            
            def heartbeat_worker():
                while module_name in self.heartbeat_intervals:
                    try:
                        self.update_heartbeat(module_name)
                        emit_telemetry("heartbeat", module_name, {
                            "status": "alive",
                            "timestamp": datetime.now().isoformat()
                        })
                        time.sleep(interval)
                    except Exception as e:
                        self.logger.error(f"Heartbeat error for {module_name}: {e}")
                        break
                        
            if module_name in self.heartbeat_threads:
                self.stop_heartbeat(module_name)
                
            thread = threading.Thread(target=heartbeat_worker, daemon=True)
            thread.start()
            self.heartbeat_threads[module_name] = thread
            
            self.logger.info(f"💓 Heartbeat started for {module_name} (interval: {interval}s)")
            
        except Exception as e:
            self.logger.error(f"Heartbeat start error for {module_name}: {e}")
            
    def stop_heartbeat(self, module_name: str):
        """Stop heartbeat monitoring for a module"""
        try:
            if module_name in self.heartbeat_intervals:
                del self.heartbeat_intervals[module_name]
                
            if module_name in self.heartbeat_threads:
                # Thread will stop automatically when module_name is removed from intervals
                del self.heartbeat_threads[module_name]
                
            self.logger.info(f"💓 Heartbeat stopped for {module_name}")
            
        except Exception as e:
            self.logger.error(f"Heartbeat stop error for {module_name}: {e}")
            
    def update_heartbeat(self, module_name: str):
        """Update module heartbeat timestamp"""
        try:
            if module_name in self.registered_modules:
                self.registered_modules[module_name]["last_heartbeat"] = datetime.now().isoformat()
                self.registered_modules[module_name]["status"] = "active"
                
        except Exception as e:
            self.logger.error(f"Heartbeat update error for {module_name}: {e}")
            
    def record_performance(self, module_name: str, metric: str, value: float):
        """Record performance metric for a module"""
        try:
            if module_name not in self.performance_data:
                self.performance_data[module_name] = {}
                
            if metric not in self.performance_data[module_name]:
                self.performance_data[module_name][metric] = []
                
            self.performance_data[module_name][metric].append({
                "value": value,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 100 measurements
            if len(self.performance_data[module_name][metric]) > 100:
                self.performance_data[module_name][metric].pop(0)
                
        except Exception as e:
            self.logger.error(f"Performance recording error for {module_name}.{metric}: {e}")
            
    def get_module_status(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of one or all modules"""
        if module_name:
            return self.registered_modules.get(module_name, {})
        else:
            return self.registered_modules.copy()
            
    def get_performance_data(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance data for one or all modules"""
        if module_name:
            return self.performance_data.get(module_name, {})
        else:
            return self.performance_data.copy()
        
    def get_telemetry_data(self) -> Dict[str, Any]:
        """Get comprehensive telemetry data for dashboard display"""
        try:
            # Get recent telemetry events
            telemetry_file = Path("telemetry.json")
            recent_events = []
            
            if telemetry_file.exists():
                with open(telemetry_file, 'r') as f:
                    data = json.load(f)
                    recent_events = data.get('events', [])[-50:]  # Last 50 events
            
            # Get module status from telemetry manager
            manager = TelemetryManager()
            module_status = manager.get_module_status()
            performance_data = manager.get_performance_data()
            
            # Compile comprehensive data
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "modules": module_status,
                "performance": performance_data,
                "recent_events": recent_events,
                "system": {
                    "uptime": str(datetime.now() - datetime.fromtimestamp(time.time() - 3600)),  # Approximate
                    "total_events": len(recent_events),
                    "active_modules": len([m for m in module_status.values() if m.get('status') == 'active']),
                    "cpu_percent": 0,  # Would need psutil for real data
                    "memory_mb": 0     # Would need psutil for real data
                },
                "signals": [],  # Placeholder for signal data
                "compliance": {
                    "drawdown_percent": 0,
                    "daily_pnl": 0.0,
                    "risk_per_trade": 1.5,
                    "open_positions": 0
                }
            }
            
            return telemetry_data
            
        except Exception as e:
            logger.error(f"Failed to get telemetry data: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "modules": {},
                "system": {}
            }



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))


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
