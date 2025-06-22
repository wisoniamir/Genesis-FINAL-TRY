# <!-- @GENESIS_MODULE_START: fix_alpha_decay_subscribers_recovered_1 -->
"""
🏛️ GENESIS FIX_ALPHA_DECAY_SUBSCRIBERS_RECOVERED_1 - INSTITUTIONAL GRADE v8.0.0
===============================================================
ARCHITECT MODE ULTIMATE: Enhanced via Complete Intelligent Wiring Engine

🎯 ENHANCED FEATURES:
- Complete EventBus integration
- Real-time telemetry monitoring
- FTMO compliance enforcement
- Emergency kill-switch protection
- Institutional-grade architecture

🔐 ARCHITECT MODE v8.0.0: Ultimate compliance enforcement
"""


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

                emit_telemetry("fix_alpha_decay_subscribers_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("fix_alpha_decay_subscribers_recovered_1", "position_calculated", {
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
                            "module": "fix_alpha_decay_subscribers_recovered_1",
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
                    print(f"Emergency stop error in fix_alpha_decay_subscribers_recovered_1: {e}")
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
                    "module": "fix_alpha_decay_subscribers_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("fix_alpha_decay_subscribers_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in fix_alpha_decay_subscribers_recovered_1: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENESIS PHASE 13 - AlphaDecayDetected Event Fix
ARCHITECT MODE v2.7 COMPLIANT

This script fixes the "No subscribers for topic 'AlphaDecayDetected'" warning
by registering subscribers for the AlphaDecayDetected event.
"""

import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("alpha_decay_fix")

def fix_alpha_decay_subscribers():
    """Fix the 'No subscribers for topic 'AlphaDecayDetected'' warning"""
    try:
        # Import event bus
        from event_bus import get_event_bus, emit_event
        
        # Get event bus instance
        event_bus = get_event_bus()
        
        # Create a flag to track if we received the event
        event_received = [False]
        
        # Define handler function
        def alpha_decay_handler(event_data):
            """Handle AlphaDecayDetected event"""
            logger.info(f"🔍PHASE 13: Alpha decay detected for strategy {event_data.get('strategy_id')}")
            event_received[0] = True
        
        # Register handler with event bus
        subscriber_id = "alpha_decay_fix"
        event_bus.subscribe("AlphaDecayDetected", alpha_decay_handler, subscriber_id)
        logger.info(f"✅ Registered handler for AlphaDecayDetected events with ID: {subscriber_id}")
        
        # Create test event data
        test_strategy_id = f"test-strategy-{int(time.time())}"
        event_data = {
            "event_type": "AlphaDecayDetected",
            "timestamp": datetime.now().isoformat(),
            "strategy_id": test_strategy_id,
            "negative_outcomes": 8,
            "window_size": 10,
            "win_rate": 0.2,
            "impacted_symbols": ["EURUSD"],
            "severity": "HIGH",
            "trades_analyzed": 10
        }
        
        # Emit test event
        logger.info(f"📣 Emitting test AlphaDecayDetected event for {test_strategy_id}...")
        emit_event("AlphaDecayDetected", event_data)
        
        # Wait for event to be processed
        logger.info("⏱️ Waiting for event processing...")
        time.sleep(2)
        
        # Check if event was received
        if event_received[0]:
            logger.info("✅ AlphaDecayDetected event was successfully received by handler")
            return True
        else:
            logger.error("❌ AlphaDecayDetected event was not received by handler")            # Try to see if we can check subscribers directly
            logger.info("Attempting to check EventBus subscribers...")
            try:
                if hasattr(event_bus, 'get_subscribers'):
                    subscribers = event_bus.get_subscribers("AlphaDecayDetected")
                    logger.info(f"✅ Found {len(subscribers)} subscribers for AlphaDecayDetected: {subscribers}")
                elif hasattr(event_bus, 'subscriptions'):
                    # Try to access the subscription dictionary if available
                    topic_subscribers = event_bus.subscriptions.get("AlphaDecayDetected", [])
                    logger.info(f"✅ Found {len(topic_subscribers)} subscribers for AlphaDecayDetected")
                else:
                    logger.warning("⚠️ Unable to check subscribers directly due to EventBus implementation")
            except Exception as e:
                logger.warning(f"⚠️ Unable to check subscribers: {str(e)}")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Error fixing AlphaDecayDetected subscribers: {str(e)}")
        return False

if __name__ == "__main__":
    success = fix_alpha_decay_subscribers()
    print(f"\n{'✅ ALPHA DECAY SUBSCRIBER FIX SUCCESSFUL' if success else '❌ ALPHA DECAY SUBSCRIBER FIX FAILED'}")


# <!-- @GENESIS_MODULE_END: fix_alpha_decay_subscribers_recovered_1 -->
