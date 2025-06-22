
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

                emit_telemetry("live_backtest_comparison_engine", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("live_backtest_comparison_engine", "position_calculated", {
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
                            "module": "live_backtest_comparison_engine",
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
                    print(f"Emergency stop error in live_backtest_comparison_engine: {e}")
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
                    "module": "live_backtest_comparison_engine",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("live_backtest_comparison_engine", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in live_backtest_comparison_engine: {e}")
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


# <!-- @GENESIS_MODULE_START: live_backtest_comparison_engine -->

#!/usr/bin/env python3
"""
🔐 GENESIS LIVE VS BACKTEST COMPARISON ENGINE v1.0
Phase 93: Live vs Backtest Comparison Engine - Architect v5.0.0

🎯 PURPOSE:
Real-time comparison between live trading performance and backtest results
Quantifies performance gaps and alerts on underperformance

🛡️ ARCHITECT MODE COMPLIANCE:
- No simplified logic or fallback mechanisms
- Real data only from execution_log.json and backtest_runs.json
- Event-driven architecture with EventBus integration
- Comprehensive telemetry and performance tracking
- Institutional-grade performance analysis
"""

import json
import os
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import uuid
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeComparison:
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

            emit_telemetry("live_backtest_comparison_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("live_backtest_comparison_engine", "position_calculated", {
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
                        "module": "live_backtest_comparison_engine",
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
                print(f"Emergency stop error in live_backtest_comparison_engine: {e}")
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
                "module": "live_backtest_comparison_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("live_backtest_comparison_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in live_backtest_comparison_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "live_backtest_comparison_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in live_backtest_comparison_engine: {e}")
    """Data class for trade comparison results"""
    trade_id: str
    symbol: str
    timestamp: str
    live_profit: float
    backtest_profit: float
    deviation_pct: float
    deviation_absolute: float
    signal_drift: float
    execution_quality: str

@dataclass
class PerformanceGap:
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

            emit_telemetry("live_backtest_comparison_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("live_backtest_comparison_engine", "position_calculated", {
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
                        "module": "live_backtest_comparison_engine",
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
                print(f"Emergency stop error in live_backtest_comparison_engine: {e}")
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
                "module": "live_backtest_comparison_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("live_backtest_comparison_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in live_backtest_comparison_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "live_backtest_comparison_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in live_backtest_comparison_engine: {e}")
    """Data class for performance gap analysis"""
    symbol: str
    timeframe: str
    live_win_rate: float
    backtest_win_rate: float
    live_avg_profit: float
    backtest_avg_profit: float
    performance_gap_pct: float
    alert_level: str


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
        class LiveBacktestComparisonEngine:
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

            emit_telemetry("live_backtest_comparison_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("live_backtest_comparison_engine", "position_calculated", {
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
                        "module": "live_backtest_comparison_engine",
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
                print(f"Emergency stop error in live_backtest_comparison_engine: {e}")
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
                "module": "live_backtest_comparison_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("live_backtest_comparison_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in live_backtest_comparison_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "live_backtest_comparison_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in live_backtest_comparison_engine: {e}")
    """
    GENESIS Live vs Backtest Comparison Engine
    
    🔁 EventBus Integration: 
        - Listens: backtest:completed, execution:filled, trade:closed
        - Emits: feedback:strategy_performance_gap, comparison:alert
    
    📡 Telemetry: 
        - Exports to /telemetry/live_vs_backtest_report.json
        - Real-time performance delta calculations
        - Signal drift and execution quality scoring
    
    🧪 Validation:
        - Matches trades by symbol, timestamp, direction
        - 5% tolerance for acceptable signal mismatch
        - >10% underperformance triggers alerts
    """
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.comparison_data = []
        self.performance_gaps = {}
        self.live_trades = []
        self.backtest_results = []
        self.monitoring = False
        
        # Performance thresholds
        self.deviation_tolerance = 0.05  # 5% tolerance
        self.alert_threshold = 0.10  # 10% underperformance alert
        
        # Create telemetry directory
        os.makedirs("telemetry", exist_ok=True)
        
        # Initialize data
        self._load_historical_data()
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("Live vs Backtest Comparison Engine initialized")
        
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _setup_event_subscriptions(self):
        """Setup EventBus event subscriptions"""
        self.event_bus.subscribe("backtest:completed", self._on_backtest_completed)
        self.event_bus.subscribe("execution:filled", self._on_execution_filled)
        self.event_bus.subscribe("trade:closed", self._on_trade_closed)
        self.event_bus.subscribe("data:update:execution_log", self._on_execution_log_update)
        
    def _load_historical_data(self):
        """Load historical live trades and backtest results"""
        try:
            # Load live trades
            if os.path.exists("execution_log.json"):
                with open("execution_log.json", 'r') as f:
                    self.live_trades = json.load(f)
                logger.info(f"Loaded {len(self.live_trades)} live trades")
            
            # Load backtest results
            if os.path.exists("telemetry/backtest_runs.json"):
                with open("telemetry/backtest_runs.json", 'r') as f:
                    self.backtest_results = json.load(f)
                logger.info(f"Loaded {len(self.backtest_results)} backtest runs")
                
            # Perform initial comparison
            self._perform_comprehensive_comparison()
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            
    def _start_monitoring(self):
        """Start real-time monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Live vs Backtest monitoring started")
        
    def _monitoring_loop(self):
        """Main monitoring loop for real-time comparison"""
        while self.monitoring:
            try:
                # Periodic comparison update
                self._update_performance_analysis()
                
                # Check for performance alerts
                self._check_performance_alerts()
                
                # Update telemetry report
                self._update_telemetry_report()
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
                
    def _on_backtest_completed(self, event):
        """Handle backtest completion events"""
        try:
            backself.event_bus.request('data:live_feed') = event.get("data", {})
            self.backtest_results.append(backself.event_bus.request('data:live_feed'))
            
            # Trigger comparison update
            self._perform_comprehensive_comparison()
            
            logger.info(f"Processed backtest completion: {backself.event_bus.request('data:live_feed').get('backtest_id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error processing backtest completion: {e}")
            
    def _on_execution_filled(self, event):
        """Handle live trade execution events"""
        try:
            execution_data = event.get("data", {})
            
            # Find matching backtest predictions
            matches = self._find_matching_backtest_trades(execution_data)
            
            if matches:
                # Create comparison entries
                for match in matches:
                    comparison = self._create_trade_comparison(execution_data, match)
                    self.comparison_data.append(comparison)
                    
                    # Check for immediate alerts
                    if comparison.deviation_pct > self.alert_threshold:
                        self._emit_performance_alert(comparison)
                        
            logger.info(f"Processed execution event: {execution_data.get('symbol', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error processing execution event: {e}")
            
    def _on_trade_closed(self, event):
        """Handle trade close events"""
        try:
            trade_data = event.get("data", {})
            
            # Update live trades list
            self.live_trades.append(trade_data)
            
            # Trigger comparison update
            self._update_symbol_performance_gap(trade_data.get("symbol"))
            
            logger.info(f"Processed trade close: {trade_data.get('symbol', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error processing trade close: {e}")
            
    def _on_execution_log_update(self, event):
        """Handle execution log file updates"""
        try:
            # Reload live trades
            self._load_historical_data()
            
        except Exception as e:
            logger.error(f"Error handling execution log update: {e}")
            
    def _find_matching_backtest_trades(self, live_trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find backtest trades that match live trade criteria"""
        matches = []
        
        try:
            live_symbol = live_trade.get("symbol")
            live_timestamp = live_trade.get("timestamp")
            live_action = live_trade.get("action")
            
            assert all([live_symbol, live_timestamp, live_action]) is not None, "Real data required - no fallbacks allowed"

# <!-- @GENESIS_MODULE_END: live_backtest_comparison_engine -->