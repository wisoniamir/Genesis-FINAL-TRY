#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔗 GENESIS EventBus Core Module v7.0.0
═══════════════════════════════════════════════════════════════════════════════════════

🧠 PURPOSE: Central event routing and communication system for GENESIS
📡 ARCHITECT MODE v7.0.0 COMPLIANT | 🔗 Real-time Event Processing | 📊 Telemetry Integrated

🚨 ARCHITECT MODE COMPLIANCE:
- No mock or simulated events
- Real-time event processing only
- Complete telemetry coverage
- Error handling without fallbacks
"""

import json
import uuid
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Callable, Optional
from collections import defaultdict
import logging

class EventBus:
    """
    🔗 GENESIS EventBus - Central event routing system
    Real-time event processing with full GENESIS compliance
    """
    
    def __init__(self):
        self.module_id = f"event_bus_{uuid.uuid4().hex[:8]}"
        self.version = "v7.0.0"
        self.architect_mode = True
        
        # Event routing
        self._routes: Dict[str, List[str]] = defaultdict(list)
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.events_emitted = 0
        self.events_consumed = 0
        self.routes_registered = 0
        
        # Logger
        self.logger = logging.getLogger("genesis_eventbus")
        
        # Initialize core routes
        self._initialize_core_routes()
    
    def _initialize_core_routes(self) -> None:
        """Initialize core EventBus routes"""
        core_routes = [
            "system_startup",
            "system_shutdown", 
            "module_registered",
            "event_route_added",
            "telemetry_event",
            "error_occurred",
            "critical_error"
        ]
        
        for route in core_routes:
            self.register_route(route, "event_bus_core")
    
    def emit(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Emit an event to all registered consumers
        Returns True if event was successfully emitted
        """
        try:
            with self._lock:
                event = {
                    "event_id": uuid.uuid4().hex,
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "emitter": "genesis_eventbus"
                }
                
                # Store in history
                self._event_history.append(event)
                
                # Keep only last 1000 events
                if len(self._event_history) > 1000:
                    self._event_history = self._event_history[-1000:]
                
                # Call registered handlers
                handlers_called = 0
                for handler in self._handlers.get(event_type, []):
                    try:
                        handler(event)
                        handlers_called += 1
                    except Exception as e:
                        self.logger.error(f"Handler error for {event_type}: {e}")
                
                self.events_emitted += 1
                
                self.logger.debug(f"Event emitted: {event_type} to {handlers_called} handlers")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to emit event {event_type}: {e}")
            return False
    
    def subscribe(self, event_type: str, handler: Callable) -> bool:
        """
        Subscribe a handler to an event type
        Returns True if subscription was successful
        """
        try:
            with self._lock:
                self._handlers[event_type].append(handler)
                self.logger.debug(f"Handler subscribed to {event_type}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {event_type}: {e}")
            return False
    
    def register_route(self, route: str, consumer: str) -> bool:
        """
        Register a route with a consumer
        Returns True if route was registered successfully
        """
        try:
            with self._lock:
                if consumer not in self._routes[route]:
                    self._routes[route].append(consumer)
                    self.routes_registered += 1
                    
                    self.logger.debug(f"Route registered: {route} -> {consumer}")
                    
                    # Emit route registration event
                    self.emit("event_route_added", {
                        "route": route,
                        "consumer": consumer,
                        "total_routes": self.routes_registered
                    })
                    
                    return True
                else:
                    self.logger.warning(f"Route {route} already registered for {consumer}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to register route {route}: {e}")
            return False
    
    def get_routes(self) -> Dict[str, List[str]]:
        """Get all registered routes"""
        with self._lock:
            return dict(self._routes)
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history"""
        with self._lock:
            return self._event_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get EventBus statistics"""
        with self._lock:
            return {
                "module_id": self.module_id,
                "version": self.version,
                "architect_mode": self.architect_mode,
                "events_emitted": self.events_emitted,
                "events_consumed": self.events_consumed,
                "routes_registered": self.routes_registered,
                "active_routes": len(self._routes),
                "active_handlers": sum(len(handlers) for handlers in self._handlers.values()),
                "event_history_size": len(self._event_history),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def validate_routes(self) -> Dict[str, Any]:
        """Validate all registered routes"""
        with self._lock:
            validation_results = {
                "valid_routes": [],
                "invalid_routes": [],
                "orphaned_routes": [],
                "total_routes": len(self._routes)
            }
            
            for route, consumers in self._routes.items():
                if consumers:
                    validation_results["valid_routes"].append({
                        "route": route,
                        "consumers": consumers,
                        "consumer_count": len(consumers)
                    })
                else:
                    validation_results["orphaned_routes"].append(route)
            
            return validation_results


# Global EventBus instance
_global_event_bus = None
_event_bus_lock = threading.Lock()

def get_event_bus() -> EventBus:
    """Get the global EventBus instance (singleton)"""
    global _global_event_bus
    
    if _global_event_bus is None:
        with _event_bus_lock:
            if _global_event_bus is None:
                _global_event_bus = EventBus()
    
    return _global_event_bus


# Convenience functions
def emit_event(event_type: str, data: Dict[str, Any]) -> bool:
    """Emit an event using the global EventBus"""
    return get_event_bus().emit(event_type, data)

def subscribe_to_event(event_type: str, handler: Callable) -> bool:
    """Subscribe to an event using the global EventBus"""
    return get_event_bus().subscribe(event_type, handler)

def register_event_route(route: str, consumer: str) -> bool:
    """Register a route using the global EventBus"""
    return get_event_bus().register_route(route, consumer)
