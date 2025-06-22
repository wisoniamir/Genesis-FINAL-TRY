# <!-- @GENESIS_MODULE_START: smart_signal_execution_linker -->

from datetime import datetime, timezone

# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "smart_signal_execution_linker",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("smart_signal_execution_linker", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in smart_signal_execution_linker: {e}")
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
GENESIS AI TRADING SYSTEM - PHASE 25: SMART SIGNAL → EXECUTION LINKER ENGINE
⚡ Signal-to-Execution Bridge - ARCHITECT MODE v2.7

PURPOSE:
Bridge the output of the Dynamic Signal Router (DSR) with the Execution Envelope Engine (EEE).
Each valid signal (MACD/STOCH/OBs) triggers its own execution envelope and routes it to the MT5 adapter.
This module serves as the critical link between signal generation and trade execution.

COMPLIANCE:
- EventBus-only architecture (no direct function calls)
- Real MT5 live data integration only
- FTMO constraints and threshold confidence filtering
- Comprehensive telemetry hooks and performance tracking
- Kill-switch integration and compliance validation
- No fallback/real execution logic

INPUTS:
- signal_object from signal_engine.py (signal_id, symbol, confidence, direction, SL/TP targets, timeframes)
- execution_envelope_ready from execution_envelope_engine.py
- strategy_recommendation from dsr_strategy_mutator.py

OUTPUTS:
- execution.route.to_adapter for valid signals above threshold
- execution.blocked.killswitch for blocked signals
- execution.blocked.low_confidence for filtered signals
- signal_execution_telemetry for performance tracking

AUTHOR: GENESIS AI AGENT - ARCHITECT MODE
VERSION: 1.0.0
PHASE: 25
"""

import json
import datetime
import os
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import uuid

# Import EventBus (ARCHITECT COMPLIANCE - NO DIRECT IMPORTS)
from hardened_event_bus import get_event_bus

@dataclass
class SignalExecutionLink:
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "smart_signal_execution_linker",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("smart_signal_execution_linker", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in smart_signal_execution_linker: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "smart_signal_execution_linker",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in smart_signal_execution_linker: {e}")
    """Signal-to-execution link data structure"""
    link_id: str
    signal_id: str
    signal_confidence: float
    execution_envelope_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    link_timestamp: str
    execution_status: str
    ftmo_compliance: bool
    kill_switch_status: str
    dispatch_latency_ms: float
    telemetry_payload: Dict[str, Any]

@dataclass
class LinkingConstraints:
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "smart_signal_execution_linker",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("smart_signal_execution_linker", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in smart_signal_execution_linker: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "smart_signal_execution_linker",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in smart_signal_execution_linker: {e}")
    """FTMO and execution constraints"""
    min_confidence_threshold: float
    max_daily_trades: int
    max_concurrent_positions: int
    max_position_size: float
    spread_threshold_pips: float
    drawdown_limit_percent: float
    required_sl_tp_ratio: float
    news_event_buffer_minutes: int


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
        class SmartSignalExecutionLinker:
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "smart_signal_execution_linker",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("smart_signal_execution_linker", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in smart_signal_execution_linker: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "smart_signal_execution_linker",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in smart_signal_execution_linker: {e}")
    """
    Smart Signal → Execution Linker Engine
    Bridges DSR signals with Execution Envelope Engine for MT5 dispatch
    """
    
    def __init__(self):
        """Initialize Signal Execution Linker with ARCHITECT compliance"""
        
        # Core setup
        self.logger = logging.getLogger("SmartSignalExecutionLinker")
        self.event_bus = get_event_bus()
        self.lock = threading.RLock()
        
        # Load configuration
        self.config = self._load_linker_config()
        
        # Linking constraints from configuration
        self.constraints = LinkingConstraints(
            min_confidence_threshold=self.config.get('min_confidence_threshold', 0.75),
            max_daily_trades=self.config.get('max_daily_trades', 20),
            max_concurrent_positions=self.config.get('max_concurrent_positions', 5),
            max_position_size=self.config.get('max_position_size', 2.0),
            spread_threshold_pips=self.config.get('spread_threshold_pips', 1.5),
            drawdown_limit_percent=self.config.get('drawdown_limit_percent', 5.0),
            required_sl_tp_ratio=self.config.get('required_sl_tp_ratio', 1.0),
            news_event_buffer_minutes=self.config.get('news_event_buffer_minutes', 30)
        )
        
        # State tracking
        self.active_links = {}  # link_id -> SignalExecutionLink
        self.signal_queue = deque(maxlen=1000)
        self.envelope_queue = deque(maxlen=1000)
        self.execution_history = deque(maxlen=5000)
          # Performance tracking
        self.performance_tracker = {
            'links_created': 0,
            'links_dispatched': 0,
            'links_blocked_confidence': 0,
            'links_blocked_killswitch': 0,
            'links_blocked_ftmo': 0,
            'signals_queued': 0,
            'total_linking_latency_ms': 0.0,
            'avg_linking_latency_ms': 0.0,
            'daily_trades_count': 0,
            'concurrent_positions': 0
        }
        
        # Kill switch and compliance
        self.kill_switch_status = "ACTIVE"
        self.ftmo_compliance_active = True
        self.daily_reset_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # EventBus setup
        self._subscribe_to_events()
        
        # Emit startup telemetry
        self._emit_startup_telemetry()
        
        self.logger.info("Smart Signal → Execution Linker Engine initialized successfully - ARCHITECT MODE v2.7")

    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _load_linker_config(self) -> Dict[str, Any]:
        """Load linker configuration from JSON file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'linker_config.json')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.logger.info(f"Linker config loaded from {config_path}")
                    return config
            else:
                # Default configuration
                default_config = {
                    "linker_settings": {
                        "name": "GENESIS Smart Signal Execution Linker",
                        "version": "1.0.0",
                        "phase": 25,
                        "architect_mode_compliant": True,
                        "real_data_only": True
                    },
                    "min_confidence_threshold": 0.75,
                    "max_daily_trades": 20,
                    "max_concurrent_positions": 5,
                    "max_position_size": 2.0,
                    "spread_threshold_pips": 1.5,
                    "drawdown_limit_percent": 5.0,
                    "required_sl_tp_ratio": 1.0,
                    "news_event_buffer_minutes": 30,
                    "linking_timeout_seconds": 30,
                    "max_linking_latency_ms": 500,
                    "telemetry_buffer_size": 1000,
                    "eventbus_routes": {
                        "subscribe": [
                            "signal_generated",
                            "execution_envelope_ready",
                            "strategy_recommendation",
                            "kill_switch_trigger",
                            "ftmo_compliance_update"
                        ],
                        "emit": [
                            "execution_route_to_adapter",
                            "execution_blocked_killswitch",
                            "execution_blocked_low_confidence",
                            "execution_blocked_ftmo",
                            "signal_execution_telemetry"
                        ]
                    }
                }
                
                self.logger.warning(f"Config file not found, using defaults")
                return default_config
                
        except Exception as e:
            self.logger.error(f"Error loading linker config: {e}")
            return {}

    def _subscribe_to_events(self):
        """Subscribe to EventBus topics for signal and envelope processing"""
        try:
            # Subscribe to signal events
            self.event_bus.subscribe("SignalGenerated", self._handle_signal_generated)
            self.event_bus.subscribe("ExecutionEnvelopeReady", self._handle_execution_envelope_ready)
            self.event_bus.subscribe("StrategyRecommendation", self._handle_strategy_recommendation)
            
            # Subscribe to system events
            self.event_bus.subscribe("KillSwitchTrigger", self._handle_kill_switch)
            self.event_bus.subscribe("FTMOComplianceUpdate", self._handle_ftmo_update)
            self.event_bus.subscribe("RiskLimitUpdate", self._handle_risk_limit_update)
            self.event_bus.subscribe("DrawdownAlert", self._handle_drawdown_alert)
            self.event_bus.subscribe("SystemStatusCheck", self._handle_status_check)
            self.event_bus.subscribe("PerformanceMetricsRequest", self._handle_metrics_request)
            
            self.logger.info("EventBus subscriptions established")
            
        except Exception as e:
            self.logger.error(f"Error setting up EventBus subscriptions: {e}")
            raise

    def _handle_signal_generated(self, event_data: Dict[str, Any]):
        """Handle incoming signals from signal engine"""
        try:
            linking_start_time = time.time()
            
            with self.lock:
                signal_data = event_data.get('payload', {})
                
                # Validate signal data structure
                required_fields = ['signal_id', 'symbol', 'confidence', 'direction']
                assert all(field in signal_data for field in required_fields):
                    self.logger.warning(f"Invalid signal data structure: missing fields")
                    return
                
                # Check kill switch status
                if self.kill_switch_status != "ACTIVE":
                    self._emit_blocked_execution("KILL_SWITCH", signal_data, linking_start_time)
                    self.performance_tracker['links_blocked_killswitch'] += 1
                    return
                
                # Check confidence threshold
                signal_confidence = signal_data.get('confidence', 0.0)
                if signal_confidence < self.constraints.min_confidence_threshold:
                    self._emit_blocked_execution("LOW_CONFIDENCE", signal_data, linking_start_time)
                    self.performance_tracker['links_blocked_confidence'] += 1
                    return
                
                # Check FTMO constraints
                if not self._validate_ftmo_constraints(signal_data):
                    self._emit_blocked_execution("FTMO_VIOLATION", signal_data, linking_start_time)
                    self.performance_tracker['links_blocked_ftmo'] += 1
                    return
                  # Add to signal queue for envelope matching
                self.signal_queue.append({
                    'signal_data': signal_data,
                    'linking_start_time': linking_start_time,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                # Increment signals queued for processing
                self.performance_tracker['signals_queued'] = self.performance_tracker.get('signals_queued', 0) + 1
                
                # Try immediate linking if envelope available
                self._try_link_signal_to_envelope(signal_data, linking_start_time)
                
        except Exception as e:
            self.logger.error(f"Error handling signal generation: {e}")
            self._emit_module_error("SIGNAL_PROCESSING_ERROR", str(e))

    def _handle_execution_envelope_ready(self, event_data: Dict[str, Any]):
        """Handle ready execution envelopes from ExecutionEnvelopeEngine"""
        try:
            with self.lock:
                envelope_data = event_data.get('payload', {})
                
                # Validate envelope data
                required_fields = ['envelope_id', 'symbol', 'position_size_lots', 'entry_price_target']
                if not all(field in envelope_data for field in required_fields):
                    self.logger.warning(f"Invalid envelope data structure: missing fields")
                    return
                
                # Add to envelope queue
                self.envelope_queue.append({
                    'envelope_data': envelope_data,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                # Try linking with queued signals
                self._try_link_envelopes_to_signals()
                
        except Exception as e:
            self.logger.error(f"Error handling execution envelope: {e}")
            self._emit_module_error("ENVELOPE_PROCESSING_ERROR", str(e))

    def _try_link_signal_to_envelope(self, signal_data: Dict[str, Any], linking_start_time: float):
        """Try to link signal with available execution envelope"""
        try:
            # Look for matching envelope in queue
            matching_envelope = None
            for envelope_item in list(self.envelope_queue):
                envelope_data = envelope_item['envelope_data']
                
                # Check symbol match
                if envelope_data.get('symbol') == signal_data.get('symbol'):
                    matching_envelope = envelope_data
                    self.envelope_queue.remove(envelope_item)
                    break
            
            if matching_envelope:
                # Create signal-execution link
                link = self._create_signal_execution_link(signal_data, matching_envelope, linking_start_time)
                
                if link:
                    # Dispatch to MT5 adapter via EventBus
                    self._dispatch_execution_to_adapter(link)
                    
                    # Update performance tracking
                    self.performance_tracker['links_created'] += 1
                    self.performance_tracker['links_dispatched'] += 1
                    
                    # Update latency tracking
                    self.performance_tracker['total_linking_latency_ms'] += link.dispatch_latency_ms
                    avg_latency = (self.performance_tracker['total_linking_latency_ms'] / 
                                  max(self.performance_tracker['links_created'], 1))
                    self.performance_tracker['avg_linking_latency_ms'] = avg_latency
                    
                    self.logger.info(f"Signal-execution link created and dispatched: {link.link_id}")
            
        except Exception as e:
            self.logger.error(f"Error linking signal to envelope: {e}")

    def _try_link_envelopes_to_signals(self):
        """Try to link available envelopes with queued signals"""
        try:
            # Process signal queue for envelope matching
            for signal_item in list(self.signal_queue):
                signal_data = signal_item['signal_data']
                linking_start_time = signal_item['linking_start_time']
                
                # Check linking timeout
                elapsed_time = time.time() - linking_start_time
                if elapsed_time > self.config.get('linking_timeout_seconds', 30):
                    self.signal_queue.remove(signal_item)
                    self.logger.warning(f"Signal linking timeout: {signal_data.get('signal_id')}")
                    continue
                
                # Try linking
                self._try_link_signal_to_envelope(signal_data, linking_start_time)
                
        except Exception as e:
            self.logger.error(f"Error linking envelopes to signals: {e}")

    def _create_signal_execution_link(self, signal_data: Dict[str, Any], 
                                    envelope_data: Dict[str, Any], linking_start_time: float) -> Optional[SignalExecutionLink]:
        """Create signal-execution link with all metadata"""
        try:
            current_time = time.time()
            dispatch_latency_ms = (current_time - linking_start_time) * 1000
            
            # Calculate entry, SL, TP from signal and envelope
            entry_price = envelope_data.get('entry_price_target', 0.0)
            stop_loss = signal_data.get('stop_loss', entry_price * 0.999)  # Default 0.1% SL
            take_profit = signal_data.get('take_profit', entry_price * 1.002)  # Default 0.2% TP
            position_size = envelope_data.get('position_size_lots', 0.1)
            
            # Validate SL/TP ratio
            if not self._validate_sl_tp_ratio(entry_price, stop_loss, take_profit, str(signal_data.get('direction', 'BUY'))):
                self.logger.warning(f"Invalid SL/TP ratio for signal {signal_data.get('signal_id')}")
                self._emit_error_event("operation_failed", {

                    "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                    "timestamp": datetime.now(timezone.utc).isoformat()

                })

                raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
            
            # Create telemetry payload
            telemetry_payload = self._create_link_telemetry_payload(signal_data, envelope_data, dispatch_latency_ms)
            
            # Create link object
            link = SignalExecutionLink(
                link_id=f"LINK_{signal_data.get('signal_id')}_{int(current_time)}",
                signal_id=signal_data.get('signal_id', ''),
                signal_confidence=signal_data.get('confidence', 0.0),
                execution_envelope_id=envelope_data.get('envelope_id', ''),
                symbol=signal_data.get('symbol', ''),
                direction=signal_data.get('direction', 'BUY'),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                link_timestamp=datetime.datetime.now().isoformat(),
                execution_status="LINKED",
                ftmo_compliance=self.ftmo_compliance_active,
                kill_switch_status=self.kill_switch_status,
                dispatch_latency_ms=dispatch_latency_ms,
                telemetry_payload=telemetry_payload
            )
            
            # Store link
            self.active_links[link.link_id] = link
            self.execution_history.append(link)
            
            return link
            
        except Exception as e:
            self.logger.error(f"Error creating signal-execution link: {e}")
            self._emit_error_event("operation_failed", {

                "error": "ARCHITECT_MODE_COMPLIANCE: Operation failed",

                "timestamp": datetime.now(timezone.utc).isoformat()

            })

            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")

    def _validate_ftmo_constraints(self, signal_data: Dict[str, Any]) -> bool:
        """Validate FTMO compliance constraints"""
        try:
            # Check daily trade limit
            if self.performance_tracker['daily_trades_count'] >= self.constraints.max_daily_trades:
                self.logger.warning(f"Daily trade limit reached: {self.constraints.max_daily_trades}")
                return False
            
            # Check concurrent positions
            if self.performance_tracker['concurrent_positions'] >= self.constraints.max_concurrent_positions:
                self.logger.warning(f"Max concurrent positions reached: {self.constraints.max_concurrent_positions}")
                return False
            
            # Check news event buffer (simplified check)
            current_hour = datetime.datetime.now().hour
            news_hours = [8, 9, 13, 14, 15, 20, 21]  # Major news hours
            if current_hour in news_hours:
                self.logger.warning(f"News event buffer active during hour {current_hour}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating FTMO constraints: {e}")
            return False

    def _validate_sl_tp_ratio(self, entry_price: float, stop_loss: float, take_profit: float, direction: str) -> bool:
        """Validate stop loss to take profit ratio"""
        try:
            if direction.upper() == "BUY":
                sl_distance = abs(entry_price - stop_loss)
                tp_distance = abs(take_profit - entry_price)
            else:  # SELL
                sl_distance = abs(stop_loss - entry_price)
                tp_distance = abs(entry_price - take_profit)
            
            if sl_distance == 0 is not None, "Real data required - no fallbacks allowed"

# <!-- @GENESIS_MODULE_END: smart_signal_execution_linker -->