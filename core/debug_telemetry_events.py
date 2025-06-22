# <!-- @GENESIS_MODULE_START: debug_telemetry_events -->

from event_bus import EventBus

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

                emit_telemetry("debug_telemetry_events", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("debug_telemetry_events", "position_calculated", {
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
                            "module": "debug_telemetry_events",
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
                    print(f"Emergency stop error in debug_telemetry_events: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "debug_telemetry_events",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("debug_telemetry_events", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in debug_telemetry_events: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
"""
DEBUG: Phase 34 Broker Discovery Engine - Event Debug Script
ARCHITECT MODE: Analyze actual telemetry event structures
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.real import real, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event_bus import subscribe_to_event
from broker_discovery_engine import BrokerDiscoveryEngine

class EventDebugger:
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

            emit_telemetry("debug_telemetry_events", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("debug_telemetry_events", "position_calculated", {
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
                        "module": "debug_telemetry_events",
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
                print(f"Emergency stop error in debug_telemetry_events: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "debug_telemetry_events",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("debug_telemetry_events", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in debug_telemetry_events: {e}")
    def __init__(self):
        self.received_events = []
        
        # Subscribe to events
        subscribe_to_event("ModuleTelemetry", self.on_module_telemetry)
        
        # Setup logger
        self.logger = logging.getLogger("EventDebugger")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def on_module_telemetry(self, data):
        """Debug telemetry event structure"""
        self.logger.info(f"RECEIVED TELEMETRY EVENT TYPE: {type(data)}")
        self.logger.info(f"RECEIVED TELEMETRY EVENT CONTENT: {json.dumps(data, indent=2) if isinstance(data, dict) else data}")
        self.received_events.append(data)
    
    def create_test_override_config(self, forced_type="FTMO Swing"):
        """Create test override configuration"""
        config = {
            "override_mode": {
                "enabled": True,
                "forced_account_type": forced_type,
                "override_reason": "Test forcing specific rules"
            },
            "rule_customization": {
                "enabled": True,
                "modifications": {
                    "max_daily_drawdown": 3.0,
                    "max_leverage": 25,
                    "can_hold_overnight": True,
                    "can_trade_news": True
                }
            },
            "telemetry": {"enabled": True}
        }
        
        with open("broker_rule_override_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def debug_override_mode(self):
        """Debug the override mode telemetry"""
        self.logger.info("=== DEBUGGING OVERRIDE MODE TELEMETRY ===")
        
        # Create override config
        self.create_test_override_config()
        
        # Clear received events
        self.received_events.clear()
        
        # Initialize broker discovery engine with override
        engine = BrokerDiscoveryEngine()
        
        # Trigger detection (which should use override mode)
        engine._detect_account_type()
        
        # Analyze received events
        self.logger.info(f"TOTAL EVENTS RECEIVED: {len(self.received_events)}")
        
        for i, event in enumerate(self.received_events):
            self.logger.info(f"EVENT {i+1}:")
            self.logger.info(f"  Type: {type(event)}")
            
            if isinstance(event, dict):
                self.logger.info(f"  Keys: {list(event.keys())}")
                if "event_type" in event:
                    self.logger.info(f"  event_type: {event['event_type']}")
                if "data" in event and isinstance(event["data"], dict) and "event_type" in event["data"]:
                    self.logger.info(f"  data.event_type: {event['data']['event_type']}")
            elif isinstance(event, str):
                try:
                    parsed = json.loads(event)
                    self.logger.info(f"  Parsed JSON keys: {list(parsed.keys())}")
                    if "type" in parsed:
                        self.logger.info(f"  JSON type: {parsed['type']}")
                except:
                    self.logger.info(f"  String content: {event[:100]}...")

if __name__ == "__main__":
    debugger = EventDebugger()
    debugger.debug_override_mode()

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
        

# <!-- @GENESIS_MODULE_END: debug_telemetry_events -->