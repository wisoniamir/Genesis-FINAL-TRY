
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
╔═══════════════════════════════════════════════════════════════════════════════╗
║                🛡️ GENESIS KILL-SWITCH LOGIC v4.0                             ║
║            INSTITUTIONAL EXECUTION KILL-SWITCH INFRASTRUCTURE                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🔐 OBJECTIVE:
Master kill-switch control logic for live execution protection
- FTMO drawdown protection
- Risk threshold enforcement  
- Macro news disqualification
- Signal quality filtering
- Emergency halt capabilities

🛑 KILL-SWITCH TRIGGERS:
1. Drawdown breach (daily/max limits)
2. Per-trade risk exceeded
3. Session risk accumulated  
4. Macro disqualifier active
5. Signal quality below threshold
6. Manual emergency halt

🔗 EventBus Integration: MANDATORY
📊 Telemetry Hooks: ALL actions logged
✅ ARCHITECT MODE v3.0 COMPLIANT
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# <!-- @GENESIS_MODULE_END: kill_switch_logic -->


# <!-- @GENESIS_MODULE_START: kill_switch_logic -->

class KillSwitchTrigger(Enum):
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "kill_switch_logic",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in kill_switch_logic: {e}")
    """Kill-switch trigger types"""
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    RISK_THRESHOLD = "RISK_THRESHOLD"
    MACRO_CONFLICT = "MACRO_CONFLICT"
    SIGNAL_QUALITY = "SIGNAL_QUALITY"
    MANUAL_HALT = "MANUAL_HALT"
    SESSION_LIMIT = "SESSION_LIMIT"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"

@dataclass
class KillSwitchEvent:
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "kill_switch_logic",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in kill_switch_logic: {e}")
    """Kill-switch activation event"""
    trigger: KillSwitchTrigger
    timestamp: datetime
    severity: str  # CRITICAL, HIGH, MEDIUM
    message: str
    data: Dict[str, Any]
    suggested_action: str
    recovery_time: Optional[datetime] = None

class GenesisKillSwitchLogic:
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "kill_switch_logic",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in kill_switch_logic: {e}")
    """
    🛡️ GENESIS Kill-Switch Logic Controller
    
    Central authority for execution halt decisions
    - Monitors all risk parameters
    - Enforces FTMO compliance
    - Manages emergency halts
    - Coordinates with EventBus
    """
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.is_active = True
        self.kill_switch_engaged = False
        self.active_triggers = []
        
        # Load compliance configuration
        self.compliance_config = self._load_compliance_config()
        self.ftmo_limits = self.compliance_config.get('ftmo_limits', {})
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.trade_count = 0
        self.session_start = datetime.now()
        
        # Thresholds
        self.max_daily_loss = self.ftmo_limits.get('max_daily_loss', -500.0)
        self.max_drawdown = self.ftmo_limits.get('max_drawdown', -2000.0)
        self.max_risk_per_trade = self.ftmo_limits.get('max_risk_per_trade', 200.0)
        self.max_session_trades = self.ftmo_limits.get('max_session_trades', 50)
        
        # State tracking
        self.macro_disqualifiers = []
        self.signal_quality_threshold = 0.7
        self.manual_halt_active = False
        
        self._initialize_eventbus_hooks()
        self._emit_telemetry("KILL_SWITCH_INITIALIZED", {
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown": self.max_drawdown,
            "max_risk_per_trade": self.max_risk_per_trade,
            "session_start": self.session_start.isoformat()
        })
    
    def _load_compliance_config(self) -> Dict:
        """Load compliance configuration"""
        try:
            with open('compliance.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'ftmo_limits': {
                    'max_daily_loss': -500.0,
                    'max_drawdown': -2000.0,
                    'max_risk_per_trade': 200.0,
                    'max_session_trades': 50
                }
            }
    
    def _initialize_eventbus_hooks(self):
        """Initialize EventBus subscriptions"""
        if self.event_bus:
            # Monitor execution events
            self.event_bus.subscribe("TradeExecutionRequest", self._handle_trade_request)
            self.event_bus.subscribe("TradeCompleted", self._handle_trade_completed)
            self.event_bus.subscribe("MacroNewsUpdate", self._handle_macro_update)
            self.event_bus.subscribe("SignalGenerated", self._handle_signal_quality)
            self.event_bus.subscribe("DrawdownUpdate", self._handle_drawdown_update)
            self.event_bus.subscribe("ManualKillSwitch", self._handle_manual_halt)
            self.event_bus.subscribe("SessionReset", self._handle_session_reset)
    
    def evaluate_execution_request(self, trade_request: Dict[str, Any]) -> Tuple[bool, Optional[KillSwitchEvent]]:
        """
        🎯 Evaluate if execution request should be allowed
        
        Returns:
            (allow_execution: bool, kill_switch_event: Optional[KillSwitchEvent])
        """
        # Check if kill-switch is already engaged
        if self.kill_switch_engaged:
            return False, KillSwitchEvent(
                trigger=KillSwitchTrigger.MANUAL_HALT,
                timestamp=datetime.now(),
                severity="CRITICAL",
                message="Kill-switch engaged - all execution blocked",
                data={"active_triggers": [t.value for t in self.active_triggers]},
                suggested_action="Review and reset kill-switch triggers"
            )
        
        # Check drawdown limits
        projected_loss = trade_request.get('risk_amount', 0.0)
        if self.daily_pnl - projected_loss < self.max_daily_loss:
            return self._trigger_kill_switch(
                KillSwitchTrigger.DRAWDOWN_BREACH,
                f"Daily loss limit would be breached: {self.daily_pnl - projected_loss:.2f} < {self.max_daily_loss:.2f}",
                {
                    "current_daily_pnl": self.daily_pnl,
                    "projected_loss": projected_loss,
                    "daily_limit": self.max_daily_loss
                },
                "Stop trading for today - daily loss limit protection"
            )
        
        # Check per-trade risk
        if projected_loss > self.max_risk_per_trade:
            return self._trigger_kill_switch(
                KillSwitchTrigger.RISK_THRESHOLD,
                f"Per-trade risk exceeded: {projected_loss:.2f} > {self.max_risk_per_trade:.2f}",
                {
                    "trade_risk": projected_loss,
                    "max_allowed": self.max_risk_per_trade
                },
                "Reduce position size or skip trade"
            )
        
        # Check session trade count
        if self.trade_count >= self.max_session_trades:
            return self._trigger_kill_switch(
                KillSwitchTrigger.SESSION_LIMIT,
                f"Session trade limit reached: {self.trade_count} >= {self.max_session_trades}",
                {
                    "trades_today": self.trade_count,
                    "session_limit": self.max_session_trades
                },
                "Session limit reached - stop trading"
            )
        
        # Check macro disqualifiers
        if self.macro_disqualifiers:
            return self._trigger_kill_switch(
                KillSwitchTrigger.MACRO_CONFLICT,
                f"Macro disqualifiers active: {', '.join(self.macro_disqualifiers)}",
                {
                    "active_disqualifiers": self.macro_disqualifiers
                },
                "Wait for macro conditions to clear"
            )
        
        # Check signal quality
        signal_quality = trade_request.get('signal_quality', 1.0)
        if signal_quality < self.signal_quality_threshold:
            return self._trigger_kill_switch(
                KillSwitchTrigger.SIGNAL_QUALITY,
                f"Signal quality below threshold: {signal_quality:.3f} < {self.signal_quality_threshold:.3f}",
                {
                    "signal_quality": signal_quality,
                    "threshold": self.signal_quality_threshold
                },
                "Improve signal quality or skip trade"
            )
        
        # All checks passed
        self._emit_telemetry("EXECUTION_APPROVED", {
            "trade_request_id": trade_request.get('id', 'unknown'),
            "risk_amount": projected_loss,
            "daily_pnl": self.daily_pnl,
            "trade_count": self.trade_count,
            "signal_quality": signal_quality
        })
        
        return True, None
    
    def _trigger_kill_switch(self, trigger: KillSwitchTrigger, message: str, 
                           data: Dict[str, Any], suggested_action: str) -> Tuple[bool, KillSwitchEvent]:
        """Trigger kill-switch and create event"""
        self.kill_switch_engaged = True
        if trigger not in self.active_triggers:
            self.active_triggers.append(trigger)
        
        event = KillSwitchEvent(
            trigger=trigger,
            timestamp=datetime.now(),
            severity="CRITICAL",
            message=message,
            data=data,
            suggested_action=suggested_action
        )
        
        # Emit to EventBus
        if self.event_bus:
            self.event_bus.emit("KillSwitchTriggered", {
                "trigger": trigger.value,
                "message": message,
                "data": data,
                "timestamp": event.timestamp.isoformat()
            })
        
        self._emit_telemetry("KILL_SWITCH_TRIGGERED", {
            "trigger": trigger.value,
            "message": message,
            "active_triggers": [t.value for t in self.active_triggers],
            "kill_switch_engaged": self.kill_switch_engaged
        })
        
        return False, event
    
    def reset_kill_switch(self, override_reason: str = "") -> bool:
        """Reset kill-switch (manual intervention required)"""
        if not override_reason:
            self._emit_telemetry("KILL_SWITCH_RESET_DENIED", {
                "reason": "Override reason required"
            })
            return False
        
        self.kill_switch_engaged = False
        self.active_triggers = []
        
        if self.event_bus:
            self.event_bus.emit("KillSwitchReset", {
                "override_reason": override_reason,
                "timestamp": datetime.now().isoformat()
            })
        
        self._emit_telemetry("KILL_SWITCH_RESET", {
            "override_reason": override_reason,
            "reset_timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def _handle_trade_request(self, event_data: Dict[str, Any]):
        """Handle incoming trade execution request"""
        allowed, kill_event = self.evaluate_execution_request(event_data)
        
        if self.event_bus:
            self.event_bus.emit("TradeExecutionDecision", {
                "request_id": event_data.get('id', 'unknown'),
                "allowed": allowed,
                "kill_switch_event": kill_event.__dict__ if kill_event else None
            })
    
    def _handle_trade_completed(self, event_data: Dict[str, Any]):
        """Handle completed trade update"""
        pnl = event_data.get('pnl', 0.0)
        self.daily_pnl += pnl
        self.session_pnl += pnl
        self.trade_count += 1
        
        self._emit_telemetry("TRADE_COMPLETED_TRACKED", {
            "pnl": pnl,
            "daily_pnl": self.daily_pnl,
            "session_pnl": self.session_pnl,
            "trade_count": self.trade_count
        })
        
        # Check if we hit drawdown after trade completion
        if self.daily_pnl < self.max_daily_loss:
            self._trigger_kill_switch(
                KillSwitchTrigger.DRAWDOWN_BREACH,
                f"Daily loss limit breached after trade: {self.daily_pnl:.2f} < {self.max_daily_loss:.2f}",
                {
                    "daily_pnl": self.daily_pnl,
                    "daily_limit": self.max_daily_loss,
                    "breach_amount": self.daily_pnl - self.max_daily_loss
                },
                "STOP TRADING IMMEDIATELY - Daily loss limit breached"
            )
    
    def _handle_macro_update(self, event_data: Dict[str, Any]):
        """Handle macro news update"""
        disqualifier = event_data.get('disqualifier')
        if disqualifier and disqualifier not in self.macro_disqualifiers:
            self.macro_disqualifiers.append(disqualifier)
            
            self._emit_telemetry("MACRO_DISQUALIFIER_ADDED", {
                "disqualifier": disqualifier,
                "total_disqualifiers": len(self.macro_disqualifiers)
            })
        
        # Clear expired disqualifiers
        cleared = event_data.get('cleared', [])
        for cleared_item in cleared:
            if cleared_item in self.macro_disqualifiers:
                self.macro_disqualifiers.remove(cleared_item)
                
                self._emit_telemetry("MACRO_DISQUALIFIER_CLEARED", {
                    "cleared": cleared_item,
                    "remaining_disqualifiers": len(self.macro_disqualifiers)
                })
    
    def _handle_signal_quality(self, event_data: Dict[str, Any]):
        """Handle signal quality update"""
        quality = event_data.get('quality', 0.0)
        
        self._emit_telemetry("SIGNAL_QUALITY_MONITORED", {
            "quality": quality,
            "threshold": self.signal_quality_threshold,
            "passes_threshold": quality >= self.signal_quality_threshold
        })
    
    def _handle_drawdown_update(self, event_data: Dict[str, Any]):
        """Handle drawdown update"""
        current_drawdown = event_data.get('current_drawdown', 0.0)
        
        if current_drawdown < self.max_drawdown:
            self._trigger_kill_switch(
                KillSwitchTrigger.DRAWDOWN_BREACH,
                f"Maximum drawdown exceeded: {current_drawdown:.2f} < {self.max_drawdown:.2f}",
                {
                    "current_drawdown": current_drawdown,
                    "max_drawdown": self.max_drawdown
                },
                "EMERGENCY HALT - Maximum drawdown exceeded"
            )
    
    def _handle_manual_halt(self, event_data: Dict[str, Any]):
        """Handle manual kill-switch activation"""
        reason = event_data.get('reason', 'Manual intervention')
        
        self._trigger_kill_switch(
            KillSwitchTrigger.MANUAL_HALT,
            f"Manual kill-switch activated: {reason}",
            {"manual_reason": reason},
            "Manual intervention required to reset"
        )
    
    def _handle_session_reset(self, event_data: Dict[str, Any]):
        """Handle session reset"""
        self.session_pnl = 0.0
        self.trade_count = 0
        self.session_start = datetime.now()
        
        self._emit_telemetry("SESSION_RESET", {
            "new_session_start": self.session_start.isoformat(),
            "previous_session_pnl": event_data.get('previous_pnl', 0.0)
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kill-switch status"""
        return {
            "kill_switch_engaged": self.kill_switch_engaged,
            "active_triggers": [t.value for t in self.active_triggers],
            "daily_pnl": self.daily_pnl,
            "session_pnl": self.session_pnl,
            "trade_count": self.trade_count,
            "session_start": self.session_start.isoformat(),
            "macro_disqualifiers": self.macro_disqualifiers,
            "limits": {
                "max_daily_loss": self.max_daily_loss,
                "max_drawdown": self.max_drawdown,
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_session_trades": self.max_session_trades
            }
        }
    
    def _emit_telemetry(self, event_type: str, data: Dict[str, Any]):
        """Emit telemetry event"""
        telemetry_data = {
            "module": "kill_switch_logic",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        if self.event_bus:
            self.event_bus.emit("telemetry", telemetry_data)
        
        # Also log to console for visibility
        logging.info(f"🛡️ KILL-SWITCH {event_type}: {data}")

# Global kill-switch instance
_kill_switch_instance = None

def get_kill_switch(event_bus=None) -> GenesisKillSwitchLogic:
    """Get global kill-switch instance"""
    global _kill_switch_instance
    if _kill_switch_instance is None:
        _kill_switch_instance = GenesisKillSwitchLogic(event_bus)
    return _kill_switch_instance

def evaluate_trade_execution(trade_request: Dict[str, Any], event_bus=None) -> Tuple[bool, Optional[Dict]]:
    """
    🎯 Quick evaluation function for trade execution
    
    Args:
        trade_request: Dict containing trade details
        event_bus: EventBus instance
    
    Returns:
        (allowed: bool, kill_switch_event: Optional[Dict])
    """
    kill_switch = get_kill_switch(event_bus)
    allowed, event = kill_switch.evaluate_execution_request(trade_request)
    
    return allowed, event.__dict__ if event else None

if __name__ == "__main__":
    # Test kill-switch logic
    print("🛡️ Testing GENESIS Kill-Switch Logic")
    
    kill_switch = GenesisKillSwitchLogic()
    
    # Test normal trade
    test_trade = {
        'id': 'TEST_001',
        'risk_amount': 100.0,
        'signal_quality': 0.85
    }
    
    allowed, event = kill_switch.evaluate_execution_request(test_trade)
    print(f"Normal trade allowed: {allowed}")
    
    # Test high risk trade
    high_risk_trade = {
        'id': 'TEST_002', 
        'risk_amount': 300.0,
        'signal_quality': 0.75
    }
    
    allowed, event = kill_switch.evaluate_execution_request(high_risk_trade)
    print(f"High risk trade allowed: {allowed}")
    if event:
        print(f"Kill-switch triggered: {event.message}")
    
    print(f"Kill-switch status: {kill_switch.get_status()}")
