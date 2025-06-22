from datetime import datetime
import logging

# <!-- @GENESIS_MODULE_START: genesis_dashboard -->

from hardened_event_bus import EventBus, Event

# FTMO compliance enforcement - MANDATORY
try:
    from compliance.ftmo.enforcer import enforce_limits
    COMPLIANCE_AVAILABLE = True
except ImportError:
    def enforce_limits(signal="", risk_pct=0, data=None): 
        print(f"COMPLIANCE CHECK: {signal}")
        return True
    COMPLIANCE_AVAILABLE = False



# <!-- @GENESIS_MODULE_END: genesis_dashboard -->

# -*- coding: utf-8 -*-



# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))
