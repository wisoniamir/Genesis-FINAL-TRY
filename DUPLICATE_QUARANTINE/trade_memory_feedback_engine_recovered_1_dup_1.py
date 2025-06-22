# <!-- @GENESIS_MODULE_START: trade_memory_feedback_engine -->

from datetime import datetime\n"""

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

                emit_telemetry("trade_memory_feedback_engine_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("trade_memory_feedback_engine_recovered_1", "position_calculated", {
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
                            "module": "trade_memory_feedback_engine_recovered_1",
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
                    print(f"Emergency stop error in trade_memory_feedback_engine_recovered_1: {e}")
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
                    "module": "trade_memory_feedback_engine_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("trade_memory_feedback_engine_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in trade_memory_feedback_engine_recovered_1: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


PHASE 28: Trade Memory + Feedback Learning Engine (TMFLE)
GENESIS AI Trading System - ARCHITECT MODE v2.8 COMPLIANT

Stores complete trade execution metadata and links to originating signals.
Maps all SL/TP/BE outcomes to signal confidence, queue tier, and execution path.
Provides real-time performance-adjusted weights back to EPL and SmartSignalEvaluator.

ARCHITECT COMPLIANCE:
- Event-driven only (EventBus)
- Real MT5 trade execution data only
- Full telemetry integration
- Persistent memory storage
- Cross-session learning capability
"""

import json
import time
import datetime
import threading
import logging
import sqlite3
import os
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

# EventBus integration - dynamic import
EVENTBUS_MODULE = "unknown"

try:
    from hardened_event_bus import get_event_bus, emit_event, subscribe_to_event
    EVENTBUS_MODULE = "hardened_event_bus"
except ImportError:
    try:
        from event_bus import get_event_bus, emit_event, subscribe_to_event
        EVENTBUS_MODULE = "event_bus"
    except ImportError:
        # Fallback for testing - basic event system
        EVENTBUS_MODULE = "fallback"
        def get_event_bus():
            return {}
        def emit_event(topic, data, producer="TradeMemoryFeedbackEngine"):
            print(f"[FALLBACK] Emit {topic}: {data}")
            return True
        def subscribe_to_event(topic, callback, module_name="TradeMemoryFeedbackEngine"):
            print(f"[FALLBACK] Subscribe {topic}: {callback}")
            return True

@dataclass
class TradeMemoryRecord:
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

            emit_telemetry("trade_memory_feedback_engine_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("trade_memory_feedback_engine_recovered_1", "position_calculated", {
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
                        "module": "trade_memory_feedback_engine_recovered_1",
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
                print(f"Emergency stop error in trade_memory_feedback_engine_recovered_1: {e}")
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
                "module": "trade_memory_feedback_engine_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("trade_memory_feedback_engine_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in trade_memory_feedback_engine_recovered_1: {e}")
    """Complete trade execution record with signal linkage"""
    trade_id: str
    signal_id: str
    symbol: str
    signal_confidence: float
    queue_tier: str  # From ExecutionPrioritizationEngine
    execution_timestamp: float
    outcome: str  # "SL", "TP", "BE", "PARTIAL", "RUNNING"
    entry_price: float
    exit_price: Optional[float]
    volume: float
    pnl: float
    execution_latency_ms: float
    market_condition: Dict[str, Any]
    signal_source: str
    execution_path: str
    created_at: datetime.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class FeedbackSignal:
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

            emit_telemetry("trade_memory_feedback_engine_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("trade_memory_feedback_engine_recovered_1", "position_calculated", {
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
                        "module": "trade_memory_feedback_engine_recovered_1",
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
                print(f"Emergency stop error in trade_memory_feedback_engine_recovered_1: {e}")
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
                "module": "trade_memory_feedback_engine_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("trade_memory_feedback_engine_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in trade_memory_feedback_engine_recovered_1: {e}")
    """Feedback signal for performance adjustment"""
    signal_id: str
    symbol: str
    original_confidence: float
    adjusted_confidence: float
    confidence_delta: float
    queue_tier: str
    outcome_quality: float  # 0.0-1.0 quality score
    execution_latency_factor: float
    market_condition_factor: float
    feedback_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class TradeMemoryFeedbackEngine:
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

            emit_telemetry("trade_memory_feedback_engine_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("trade_memory_feedback_engine_recovered_1", "position_calculated", {
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
                        "module": "trade_memory_feedback_engine_recovered_1",
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
                print(f"Emergency stop error in trade_memory_feedback_engine_recovered_1: {e}")
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
                "module": "trade_memory_feedback_engine_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("trade_memory_feedback_engine_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in trade_memory_feedback_engine_recovered_1: {e}")
    """
    PHASE 28: Trade Memory + Feedback Learning Engine (TMFLE)
    
    Manages comprehensive trade memory storage with signal linkage and
    provides real-time feedback learning for confidence adjustment.
    """
    
    def __init__(self, memory_db_path: str = "data/trade_memory.db"):
        """Initialize TradeMemoryFeedbackEngine"""
        self.processing_active = True
        self.trades_processed = 0
        self.feedback_signals_generated = 0
        self.engine_start_time = time.time()
        
        # Database setup for persistent memory
        self.memory_db_path = memory_db_path
        self._ensure_data_directory()
        self._initialize_database()
        
        # In-memory caches for performance
        self.recent_trades = deque(maxlen=1000)  # Last 1000 trades
        self.symbol_performance = defaultdict(lambda: {"success_rate": 0.5, "trade_count": 0})
        self.confidence_adjustments = defaultdict(list)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_adjustment_threshold = 0.1
        self.feedback_quality_weights = {
            "outcome": 0.6,      # 60% weight on trade outcome (SL/TP)
            "latency": 0.2,      # 20% weight on execution latency
            "market_condition": 0.2  # 20% weight on market conditions
        }
        
        # Telemetry tracking
        self.telemetry_data = {
            "trades_stored": 0,
            "feedback_signals_sent": 0,
            "confidence_adjustments": 0,
            "avg_feedback_quality": 0.0,
            "success_rate_improvements": 0,
            "learning_adaptations": 0
        }
        
        # Thread safety
        self.memory_lock = threading.RLock()
        
        # Initialize logging
        self.logger = logging.getLogger(f"TradeMemoryFeedbackEngine")
        self.logger.setLevel(logging.INFO)
        
        # Subscribe to events
        self._subscribe_to_events()
        
        self.logger.info(f"TradeMemoryFeedbackEngine initialized - PHASE 28 TMFLE ready")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _ensure_data_directory(self):
        """Ensure data directory exists"""
        data_dir = os.path.dirname(self.memory_db_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent trade memory"""
        try:
            try:
            with sqlite3.connect(self.memory_db_path) as conn:
            except Exception as e:
                logging.error(f"Operation failed: {e}")
                cursor = conn.cursor()
                
                # Create trade_memory table
                try:
                cursor.execute('''
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                CREATE TABLE IF NOT EXISTS trade_memory (
                    trade_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_confidence REAL NOT NULL,
                    queue_tier TEXT NOT NULL,
                    execution_timestamp REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    volume REAL NOT NULL,
                    pnl REAL NOT NULL,
                    execution_latency_ms REAL NOT NULL,
                    market_condition TEXT NOT NULL,
                    signal_source TEXT NOT NULL,
                    execution_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
                )
                ''')
                
                # Create feedback_history table
                try:
                cursor.execute('''
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                CREATE TABLE IF NOT EXISTS feedback_history (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    original_confidence REAL NOT NULL,
                    adjusted_confidence REAL NOT NULL,
                    confidence_delta REAL NOT NULL,
                    queue_tier TEXT NOT NULL,
                    outcome_quality REAL NOT NULL,
                    execution_latency_factor REAL NOT NULL,
                    market_condition_factor REAL NOT NULL,
                    feedback_timestamp REAL NOT NULL
                )
                ''')
                
                # Create indices for performance
                try:
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON trade_memory(symbol)')
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                try:
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_id ON trade_memory(signal_id)')
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                try:
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome ON trade_memory(outcome)')
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                try:
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON trade_memory(execution_timestamp)')
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                
                conn.commit()
                
            self.logger.info("Trade memory database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _subscribe_to_events(self):
        """Subscribe to required EventBus topics"""
        try:
            # Subscribe to trade execution events
            subscribe_to_event("LiveTradeExecuted", self._handle_live_trade_executed)
            subscribe_to_event("ExecutionResult", self._handle_execution_result)
            
            # Subscribe to signal confidence data
            subscribe_to_event("SignalConfidenceScore", self._handle_signal_confidence)
            
            # Subscribe to market condition snapshots
            subscribe_to_event("MarketConditionSnapshot", self._handle_market_condition)
            
            # Subscribe to system status requests
            subscribe_to_event("SystemStatusCheck", self._handle_system_status_check)
            
            self.logger.info(f"Subscribed to events using {EVENTBUS_MODULE}")
            
        except Exception as e:
            self.logger.error(f"Event subscription failed: {e}")
    
    def store_trade_memory(self, trade_record: TradeMemoryRecord) -> bool:
        """Store trade memory record in persistent database"""
        try:
            with self.memory_lock:
                # Store in database
                try:
                with sqlite3.connect(self.memory_db_path) as conn:
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                    cursor = conn.cursor()
                    
                    try:
                    cursor.execute('''
                    except Exception as e:
                        logging.error(f"Operation failed: {e}")
                    INSERT OR REPLACE INTO trade_memory VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_record.trade_id,
                        trade_record.signal_id,
                        trade_record.symbol,
                        trade_record.signal_confidence,
                        trade_record.queue_tier,
                        trade_record.execution_timestamp,
                        trade_record.outcome,
                        trade_record.entry_price,
                        trade_record.exit_price,
                        trade_record.volume,
                        trade_record.pnl,
                        trade_record.execution_latency_ms,
                        json.dumps(trade_record.market_condition),
                        trade_record.signal_source,
                        trade_record.execution_path,
                        trade_record.created_at.isoformat()
                    ))
                    
                    conn.commit()
                
                # Add to in-memory cache
                self.recent_trades.append(trade_record)
                
                # Update telemetry
                self.telemetry_data["trades_stored"] += 1
                self.trades_processed += 1
                
                # Update symbol performance tracking
                self._update_symbol_performance(trade_record)
                
                self.logger.info(f"Trade memory stored: {trade_record.trade_id} ({trade_record.symbol}, {trade_record.outcome})")
                
                # Generate feedback signal if conditions met
                self._generate_feedback_signal(trade_record)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store trade memory: {e}")
            return False
    
    def _update_symbol_performance(self, trade_record: TradeMemoryRecord):
        """Update symbol performance statistics"""
        try:
            symbol = trade_record.symbol
            is_successful = trade_record.outcome in ["TP", "PARTIAL"] and trade_record.pnl > 0
            
            perf = self.symbol_performance[symbol]
            current_count = perf["trade_count"]
            current_rate = perf["success_rate"]
            
            # Update with exponential moving average
            if current_count == 0:
                new_rate = 1.0 if is_successful else 0.0
            else:
                alpha = min(0.1, 1.0 / current_count)  # Adaptive learning rate
                new_rate = current_rate + alpha * (1.0 if is_successful else 0.0 - current_rate)
            
            perf["success_rate"] = new_rate
            perf["trade_count"] = current_count + 1
            
            self.logger.debug(f"Updated {symbol} performance: {current_rate:.3f} -> {new_rate:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update symbol performance: {e}")
    
    def _generate_feedback_signal(self, trade_record: TradeMemoryRecord):
        """Generate feedback signal for confidence adjustment"""
        try:
            # Calculate feedback quality score
            outcome_quality = self._calculate_outcome_quality(trade_record)
            latency_factor = self._calculate_latency_factor(trade_record.execution_latency_ms)
            market_factor = self._calculate_market_condition_factor(trade_record.market_condition)
            
            # Overall feedback quality
            feedback_quality = (
                outcome_quality * self.feedback_quality_weights["outcome"] +
                latency_factor * self.feedback_quality_weights["latency"] +
                market_factor * self.feedback_quality_weights["market_condition"]
            )
            
            # Calculate confidence adjustment
            confidence_delta = self._calculate_confidence_adjustment(
                trade_record.signal_confidence,
                feedback_quality,
                trade_record.symbol
            )
            
            # Only generate feedback if adjustment is significant
            if abs(confidence_delta) >= self.confidence_adjustment_threshold:
                adjusted_confidence = max(0.0, min(1.0, trade_record.signal_confidence + confidence_delta))
                
                feedback_signal = FeedbackSignal(
                    signal_id=trade_record.signal_id,
                    symbol=trade_record.symbol,
                    original_confidence=trade_record.signal_confidence,
                    adjusted_confidence=adjusted_confidence,
                    confidence_delta=confidence_delta,
                    queue_tier=trade_record.queue_tier,
                    outcome_quality=feedback_quality,
                    execution_latency_factor=latency_factor,
                    market_condition_factor=market_factor,
                    feedback_timestamp=time.time()
                )
                
                # Store feedback in database
                self._store_feedback_record(feedback_signal)
                
                # Emit feedback signals
                self._emit_feedback_signals(feedback_signal)
                
                self.feedback_signals_generated += 1
                self.telemetry_data["feedback_signals_sent"] += 1
                self.telemetry_data["confidence_adjustments"] += 1
                
                self.logger.info(f"Feedback signal generated: {trade_record.signal_id} confidence {trade_record.signal_confidence:.3f} -> {adjusted_confidence:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate feedback signal: {e}")
    
    def _calculate_outcome_quality(self, trade_record: TradeMemoryRecord) -> float:
        """Calculate outcome quality score (0.0-1.0)"""
        outcome = trade_record.outcome
        pnl = trade_record.pnl
        
        if outcome == "TP" and pnl > 0:
            return 1.0  # Perfect outcome
        elif outcome == "PARTIAL" and pnl > 0:
            return 0.8  # Good outcome
        elif outcome == "BE":
            return 0.5  # Neutral outcome
        elif outcome == "SL" and pnl < 0:
            return 0.1  # Poor outcome
        else:
            return 0.3  # Unknown/other outcomes
    
    def _calculate_latency_factor(self, latency_ms: float) -> float:
        """Calculate latency quality factor (0.0-1.0)"""
        # Lower latency = higher quality
        if latency_ms <= 50:
            return 1.0  # Excellent latency
        elif latency_ms <= 100:
            return 0.8  # Good latency
        elif latency_ms <= 200:
            return 0.6  # Acceptable latency
        elif latency_ms <= 500:
            return 0.4  # Poor latency
        else:
            return 0.2  # Very poor latency
    
    def _calculate_market_condition_factor(self, market_condition: Dict[str, Any]) -> float:
        """Calculate market condition quality factor (0.0-1.0)"""
        try:
            # Extract relevant market condition metrics
            volatility = market_condition.get("volatility", 0.5)
            spread = market_condition.get("spread_pips", 2.0)
            volume = market_condition.get("volume_ratio", 1.0)
            
            # Calculate factors
            volatility_factor = 1.0 - min(1.0, volatility / 0.02)  # Lower volatility = better
            spread_factor = max(0.0, 1.0 - spread / 5.0)  # Lower spread = better
            volume_factor = min(1.0, volume)  # Higher volume = better
            
            # Combined market condition factor
            return (volatility_factor + spread_factor + volume_factor) / 3.0
            
        except Exception:
            return 0.5  # Default neutral factor
    
    def _calculate_confidence_adjustment(self, original_confidence: float, feedback_quality: float, symbol: str) -> float:
        """Calculate confidence adjustment based on feedback quality"""
        try:
            # Get symbol performance
            symbol_perf = self.symbol_performance[symbol]["success_rate"]
            
            # Base adjustment calculation
            quality_delta = feedback_quality - 0.5  # Center around neutral
            performance_factor = symbol_perf - 0.5  # Center around neutral
            
            # Combined adjustment with learning rate
            adjustment = self.learning_rate * (quality_delta + performance_factor * 0.5)
            
            # Limit adjustment magnitude
            max_adjustment = 0.3  # Maximum 30% confidence change
            adjustment = max(-max_adjustment, min(max_adjustment, adjustment))
            
            return adjustment
            
        except Exception as e:
            self.logger.error(f"Confidence adjustment calculation failed: {e}")
            return 0.0
    
    def _store_feedback_record(self, feedback_signal: FeedbackSignal):
        """Store feedback record in database"""
        try:
            try:
            with sqlite3.connect(self.memory_db_path) as conn:
            except Exception as e:
                logging.error(f"Operation failed: {e}")
                cursor = conn.cursor()
                
                try:
                cursor.execute('''
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                INSERT INTO feedback_history VALUES 
                (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feedback_signal.signal_id,
                    feedback_signal.symbol,
                    feedback_signal.original_confidence,
                    feedback_signal.adjusted_confidence,
                    feedback_signal.confidence_delta,
                    feedback_signal.queue_tier,
                    feedback_signal.outcome_quality,
                    feedback_signal.execution_latency_factor,
                    feedback_signal.market_condition_factor,
                    feedback_signal.feedback_timestamp
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store feedback record: {e}")
    
    def _emit_feedback_signals(self, feedback_signal: FeedbackSignal):
        """Emit feedback signals to relevant modules"""
        try:
            # Emit to ExecutionPrioritizationEngine for performance adjustment
            emit_event("TradeFeedbackSignal", {
                "signal_id": feedback_signal.signal_id,
                "symbol": feedback_signal.symbol,
                "confidence_delta": feedback_signal.confidence_delta,
                "outcome_quality": feedback_signal.outcome_quality,
                "queue_tier": feedback_signal.queue_tier,
                "source": "TradeMemoryFeedbackEngine",
                "timestamp": feedback_signal.feedback_timestamp
            })
            
            # Emit to SignalConfidenceRatingEngine for confidence adjustment
            emit_event("AdjustedConfidenceScore", {
                "signal_id": feedback_signal.signal_id,
                "symbol": feedback_signal.symbol,
                "original_confidence": feedback_signal.original_confidence,
                "adjusted_confidence": feedback_signal.adjusted_confidence,
                "adjustment_reason": "trade_outcome_feedback",
                "quality_score": feedback_signal.outcome_quality,
                "source": "TradeMemoryFeedbackEngine",
                "timestamp": feedback_signal.feedback_timestamp
            })
            
            self.logger.info(f"Feedback signals emitted for {feedback_signal.signal_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit feedback signals: {e}")
    
    def _handle_live_trade_executed(self, trade_data: Dict[str, Any]):
        """Handle LiveTradeExecuted events"""
        try:
            # Create trade memory record
            trade_record = TradeMemoryRecord(
                trade_id=trade_data.get("trade_id", f"trade_{int(time.time())}"),
                signal_id=trade_data.get("signal_id", "unknown"),
                symbol=trade_data.get("symbol", "UNKNOWN"),
                signal_confidence=trade_data.get("signal_confidence", 0.5),
                queue_tier=trade_data.get("queue_tier", "medium"),
                execution_timestamp=trade_data.get("timestamp", time.time()),
                outcome=trade_data.get("outcome", "RUNNING"),
                entry_price=trade_data.get("entry_price", 0.0),
                exit_price=trade_data.get("exit_price"),
                volume=trade_data.get("volume", 0.01),
                pnl=trade_data.get("pnl", 0.0),
                execution_latency_ms=trade_data.get("execution_latency_ms", 100.0),
                market_condition=trade_data.get("market_condition", {}),
                signal_source=trade_data.get("signal_source", "unknown"),
                execution_path=trade_data.get("execution_path", "unknown"),
                created_at=datetime.datetime.now()
            )
            
            # Store trade memory
            self.store_trade_memory(trade_record)
            
        except Exception as e:
            self.logger.error(f"Live trade execution handling failed: {e}")
            # Emit error event
            emit_event("ModuleError", {
                "module": "TradeMemoryFeedbackEngine",
                "error": str(e),
                "event": "LiveTradeExecuted",
                "timestamp": time.time()
            })
    
    def _handle_execution_result(self, result_data: Dict[str, Any]):
        """Handle ExecutionResult events for trade outcome updates"""
        try:
            trade_id = result_data.get("trade_id")
            outcome = result_data.get("outcome")
            pnl = result_data.get("pnl", 0.0)
            exit_price = result_data.get("exit_price")
            
            if trade_id and outcome:
                # Update existing trade record
                self._update_trade_outcome(trade_id, outcome, pnl, exit_price)
            
        except Exception as e:
            self.logger.error(f"Execution result handling failed: {e}")
    
    def _update_trade_outcome(self, trade_id: str, outcome: str, pnl: float, exit_price: Optional[float]):
        """Update trade outcome in database"""
        try:
            try:
            with sqlite3.connect(self.memory_db_path) as conn:
            except Exception as e:
                logging.error(f"Operation failed: {e}")
                cursor = conn.cursor()
                
                try:
                cursor.execute('''
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                UPDATE trade_memory 
                SET outcome = ?, pnl = ?, exit_price = ?
                WHERE trade_id = ?
                ''', (outcome, pnl, exit_price, trade_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Trade outcome updated: {trade_id} -> {outcome} (PnL: {pnl})")
                    
                    # Regenerate feedback signal with updated outcome
                    trade_record = self._get_trade_record(trade_id)
                    if trade_record:
                        self._generate_feedback_signal(trade_record)
                
        except Exception as e:
            self.logger.error(f"Failed to update trade outcome: {e}")
    
    def _get_trade_record(self, trade_id: str) -> Optional[TradeMemoryRecord]:
        """Retrieve trade record from database"""
        try:
            try:
            with sqlite3.connect(self.memory_db_path) as conn:
            except Exception as e:
                logging.error(f"Operation failed: {e}")
                cursor = conn.cursor()
                
                try:
                cursor.execute('SELECT * FROM trade_memory WHERE trade_id = ?', (trade_id,))
                except Exception as e:
                    logging.error(f"Operation failed: {e}")
                row = cursor.fetchone()
                
                if row:
                    return TradeMemoryRecord(
                        trade_id=row[0],
                        signal_id=row[1],
                        symbol=row[2],
                        signal_confidence=row[3],
                        queue_tier=row[4],
                        execution_timestamp=row[5],
                        outcome=row[6],
                        entry_price=row[7],
                        exit_price=row[8],
                        volume=row[9],
                        pnl=row[10],
                        execution_latency_ms=row[11],
                        market_condition=json.loads(row[12]),
                        signal_source=row[13],
                        execution_path=row[14],
                        created_at=datetime.datetime.fromisoformat(row[15])
                    )
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve trade record: {e}")
        
        raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
    
    def _handle_signal_confidence(self, confidence_data: Dict[str, Any]):
        """Handle SignalConfidenceScore events for signal tracking"""
        try:
            # Store signal confidence for future trade linkage
            signal_id = confidence_data.get("signal_id")
            confidence = confidence_data.get("confidence", 0.5)
            
            # Add to confidence adjustments tracking
            if signal_id:
                self.confidence_adjustments[signal_id].append({
                    "confidence": confidence,
                    "timestamp": time.time(),
                    "source": "signal_confidence_rating"
                })
            
        except Exception as e:
            self.logger.error(f"Signal confidence handling failed: {e}")
    
    def _handle_market_condition(self, condition_data: Dict[str, Any]):
        """Handle MarketConditionSnapshot events"""
        try:
            # Store current market condition for trade context
            # This will be used when creating trade records
            self.current_market_condition = condition_data
            
        except Exception as e:
            self.logger.error(f"Market condition handling failed: {e}")
    
    def _handle_system_status_check(self, request_data: Dict[str, Any]):
        """Handle system status check requests"""
        try:
            status = self.get_status()
            
            # Emit status response
            emit_event("SystemStatusResponse", {
                "module": "TradeMemoryFeedbackEngine",
                "status": status,
                "request_id": request_data.get("request_id", "unknown"),
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"System status check failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status and metrics"""
        with self.memory_lock:
            return {
                "processing_active": self.processing_active,
                "trades_processed": self.trades_processed,
                "feedback_signals_generated": self.feedback_signals_generated,
                "recent_trades_count": len(self.recent_trades),
                "symbol_count": len(self.symbol_performance),
                "telemetry": self.telemetry_data.copy(),
                "uptime_seconds": time.time() - self.engine_start_time,
                "database_path": self.memory_db_path
            }
    
    def get_trade_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get trade statistics for analysis"""
        try:
            try:
            with sqlite3.connect(self.memory_db_path) as conn:
            except Exception as e:
                logging.error(f"Operation failed: {e}")
                cursor = conn.cursor()
                
                if symbol:
                    try:
                    cursor.execute('''
                    except Exception as e:
                        logging.error(f"Operation failed: {e}")
                    SELECT outcome, COUNT(*), AVG(pnl), AVG(execution_latency_ms)
                    FROM trade_memory 
                    WHERE symbol = ?
                    GROUP BY outcome
                    ''', (symbol,))
                else:
                    try:
                    cursor.execute('''
                    except Exception as e:
                        logging.error(f"Operation failed: {e}")
                    SELECT outcome, COUNT(*), AVG(pnl), AVG(execution_latency_ms)
                    FROM trade_memory 
                    GROUP BY outcome
                    ''')
                
                results = cursor.fetchall()
                
                stats = {}
                for row in results:
                    stats[row[0]] = {
                        "count": row[1],
                        "avg_pnl": row[2],
                        "avg_latency_ms": row[3]
                    }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get trade statistics: {e}")
            return {}
    
    def stop(self):
        """Stop the engine and cleanup"""
        self.processing_active = False
        self.logger.info("TradeMemoryFeedbackEngine stopped")

# Global instance
_trade_memory_feedback_engine = None

def initialize_trade_memory_feedback_engine() -> TradeMemoryFeedbackEngine:
    """Initialize and return TradeMemoryFeedbackEngine instance"""
    global _trade_memory_feedback_engine
    
    if _trade_memory_feedback_engine is None:
        _trade_memory_feedback_engine = TradeMemoryFeedbackEngine()
    
    return _trade_memory_feedback_engine

def get_trade_memory_feedback_engine() -> Optional[TradeMemoryFeedbackEngine]:
    """Get current TradeMemoryFeedbackEngine instance"""
    return _trade_memory_feedback_engine

if __name__ == "__main__":
    # Direct execution for testing
    engine = initialize_trade_memory_feedback_engine()
    print(f"PHASE 28 TradeMemoryFeedbackEngine initialized: {engine.get_status()}")

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
        

# <!-- @GENESIS_MODULE_END: trade_memory_feedback_engine -->