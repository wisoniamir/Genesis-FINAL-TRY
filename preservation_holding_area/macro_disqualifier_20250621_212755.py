
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
║              🌍 GENESIS MACRO DISQUALIFIER v2.5                              ║
║               MACRO-ECONOMIC NEWS & EVENT DISQUALIFIER                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🎯 OBJECTIVE:
Monitor and filter trading signals based on macro-economic events
- High-impact news events (NFP, CPI, FOMC, etc.)
- Central bank announcements  
- Market-moving economic releases
- Geopolitical events
- Holiday periods and thin liquidity
- Weekend gaps
🔍 FEATURES:
- Real-time news calendar integration
- Dynamic disqualification based on event timing
- Market session awareness
- Geopolitical crisis monitoring
- Telemetry and logging for disqualifications
🌐 USAGE:
1. Initialize with an EventBus instance for real-time updates
2. Add news events using `add_news_event()`
3. Evaluate trading conditions with `evaluate_trading_conditions()`
4. Handle geopolitical events with `add_geopolitical_event()`
5. Clear expired disqualifiers with `clear_expired_disqualifiers()`
6. Subscribe to EventBus for real-time updates and disqualifier events


📅 MARKET SESSIONS:
- Asian: 00:00 - 09:00 GMT
- European: 09:00 - 17:00 GMT
- North American: 14:00 - 22:00 GMT
- Australian: 22:00 - 00:00 GMT
📅 HOLIDAY CALENDAR:
- Major holidays (New Year, Christmas, Independence Day)
- Pre-holiday thin liquidity periods
📈 DISQUALIFICATION LOGIC:

🌍 DISQUALIFIER TRIGGERS:
1. High-impact news within ±30 minutes
2. FOMC meetings and announcements
3. Central bank rate decisions
4. Major economic releases (GDP, CPI, NFP)
5. Market holidays and early closes
6. Geopolitical crisis events

🔗 EventBus Integration: Real-time news monitoring
📊 Telemetry: All disqualifications logged
✅ ARCHITECT MODE v3.0 COMPLIANT
"""

import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import calendar


# <!-- @GENESIS_MODULE_END: macro_disqualifier -->


# <!-- @GENESIS_MODULE_START: macro_disqualifier -->

class NewsImpact(Enum):
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

            emit_telemetry("macro_disqualifier", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("macro_disqualifier", "position_calculated", {
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
                        "module": "macro_disqualifier",
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
                print(f"Emergency stop error in macro_disqualifier: {e}")
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
                        "module": "macro_disqualifier",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in macro_disqualifier: {e}")
    """News impact levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class DisqualifierType(Enum):
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

            emit_telemetry("macro_disqualifier", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("macro_disqualifier", "position_calculated", {
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
                        "module": "macro_disqualifier",
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
                print(f"Emergency stop error in macro_disqualifier: {e}")
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
                        "module": "macro_disqualifier",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in macro_disqualifier: {e}")
    """Types of macro disqualifiers"""
    HIGH_IMPACT_NEWS = "HIGH_IMPACT_NEWS"
    CENTRAL_BANK_EVENT = "CENTRAL_BANK_EVENT"
    MAJOR_ECONOMIC_RELEASE = "MAJOR_ECONOMIC_RELEASE"
    MARKET_HOLIDAY = "MARKET_HOLIDAY"
    THIN_LIQUIDITY = "THIN_LIQUIDITY"
    GEOPOLITICAL_EVENT = "GEOPOLITICAL_EVENT"
    WEEKEND_GAP = "WEEKEND_GAP"

@dataclass
class NewsEvent:
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

            emit_telemetry("macro_disqualifier", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("macro_disqualifier", "position_calculated", {
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
                        "module": "macro_disqualifier",
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
                print(f"Emergency stop error in macro_disqualifier: {e}")
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
                        "module": "macro_disqualifier",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in macro_disqualifier: {e}")
    """Economic news event"""
    event_id: str
    title: str
    currency: str
    impact: NewsImpact
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None
    previous: Optional[str] = None

@dataclass
class DisqualifierRule:
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

            emit_telemetry("macro_disqualifier", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("macro_disqualifier", "position_calculated", {
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
                        "module": "macro_disqualifier",
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
                print(f"Emergency stop error in macro_disqualifier: {e}")
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
                        "module": "macro_disqualifier",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in macro_disqualifier: {e}")
    """Disqualifier rule configuration"""
    event_types: List[str]
    currencies: List[str]
    min_impact: NewsImpact
    before_minutes: int
    after_minutes: int
    
class GenesisMacroDisqualifier:
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

            emit_telemetry("macro_disqualifier", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("macro_disqualifier", "position_calculated", {
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
                        "module": "macro_disqualifier",
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
                print(f"Emergency stop error in macro_disqualifier: {e}")
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
                        "module": "macro_disqualifier",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in macro_disqualifier: {e}")
    """
    🌍 GENESIS Macro Disqualifier
    
    Intelligent macro-economic event monitoring and signal filtering
    - Real-time news calendar integration
    - Impact assessment and timing
    - Trading session awareness  
    - Holiday calendar management
    """
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.is_active = True
        
        # News and events tracking
        self.scheduled_news = []
        self.active_disqualifiers = []
        self.geopolitical_events = []
        
        # Disqualifier rules
        self.disqualifier_rules = self._initialize_disqualifier_rules()
        
        # Market sessions and holidays
        self.market_holidays = self._load_market_holidays()
        self.thin_liquidity_periods = self._initialize_thin_liquidity_periods()
        
        # Currency importance mapping
        self.major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
        self.high_impact_events = [
            'Non-Farm Payrolls', 'Federal Funds Rate', 'CPI', 'GDP',
            'Unemployment Rate', 'Retail Sales', 'FOMC Statement',
            'ECB Rate Decision', 'BoE Rate Decision', 'BoJ Rate Decision'
        ]
        
        self._initialize_eventbus_hooks()
        self._emit_telemetry("MACRO_DISQUALIFIER_INITIALIZED", {
            "rules_count": len(self.disqualifier_rules),
            "major_currencies": self.major_currencies,
            "high_impact_events": len(self.high_impact_events)
        })
    
    def _initialize_disqualifier_rules(self) -> List[DisqualifierRule]:
        """Initialize disqualifier rules"""
        return [
            # Critical events - Long blackout periods
            DisqualifierRule(
                event_types=['Non-Farm Payrolls', 'Federal Funds Rate', 'FOMC Statement'],
                currencies=['USD'],
                min_impact=NewsImpact.CRITICAL,
                before_minutes=30,
                after_minutes=60
            ),
            # High impact events - Medium blackout periods  
            DisqualifierRule(
                event_types=['CPI', 'GDP', 'Unemployment Rate', 'Retail Sales'],
                currencies=['USD', 'EUR', 'GBP'],
                min_impact=NewsImpact.HIGH,
                before_minutes=15,
                after_minutes=30
            ),
            # Central bank decisions
            DisqualifierRule(
                event_types=['ECB Rate Decision', 'BoE Rate Decision', 'BoJ Rate Decision'],
                currencies=['EUR', 'GBP', 'JPY'],
                min_impact=NewsImpact.HIGH,
                before_minutes=20,
                after_minutes=45
            ),
            # Medium impact events - Short blackout periods
            DisqualifierRule(
                event_types=['PMI', 'Industrial Production', 'Trade Balance'],
                currencies=self.major_currencies,
                min_impact=NewsImpact.MEDIUM,
                before_minutes=10,
                after_minutes=15
            )
        ]
    
    def _load_market_holidays(self) -> List[datetime]:
        """Load market holidays calendar"""
        # This would typically load from an external calendar API
        # For now, we'll define major holidays
        current_year = datetime.now().year
        holidays = [
            # New Year's Day
            datetime(current_year, 1, 1),
            # Good Friday (approximate - would calculate actual date)
            datetime(current_year, 4, 10),
            # Christmas Day
            datetime(current_year, 12, 25),
            # Independence Day (US)
            datetime(current_year, 7, 4),
            # Thanksgiving (US - 4th Thursday of November)
            datetime(current_year, 11, self._get_nth_weekday(current_year, 11, 3, 4)),
        ]
        return holidays
    
    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> int:
        """Get the nth occurrence of a weekday in a month"""
        first_weekday = calendar.monthrange(year, month)[0]
        first_occurrence = (weekday - first_weekday) % 7 + 1
        return first_occurrence + (n - 1) * 7
    
    def _initialize_thin_liquidity_periods(self) -> List[Tuple[time, time]]:
        """Initialize periods known for thin liquidity"""
        return [
            # Asian session overlap with NY close
            (time(21, 0), time(23, 0)),  # 9 PM - 11 PM GMT
            # Weekend gap periods already handled in weekend check
            # Holiday periods (would be dynamic based on calendar)
        ]
    
    def _initialize_eventbus_hooks(self):
        """Initialize EventBus subscriptions"""
        if self.event_bus:
            self.event_bus.subscribe("NewsCalendarUpdate", self._handle_news_calendar_update)
            self.event_bus.subscribe("BreakingNewsAlert", self._handle_breaking_news)
            self.event_bus.subscribe("GeopoliticalEvent", self._handle_geopolitical_event)
            self.event_bus.subscribe("SignalGenerationRequest", self._evaluate_signal_timing)
            self.event_bus.subscribe("MarketSessionChange", self._handle_session_change)
            self.event_bus.subscribe("NewsTradingCheck", self._handle_news_trading_check)
    
    def evaluate_trading_conditions(self, signal_time: datetime, currency_pairs: List[str]) -> Tuple[bool, List[Dict]]:
        """
        🎯 Evaluate if trading conditions are suitable
        
        Args:
            signal_time: Time of the trading signal
            currency_pairs: List of currency pairs involved
        
        Returns:
            (is_allowed: bool, active_disqualifiers: List[Dict])
        """
        disqualifiers = []
        
        # Extract currencies from pairs
        currencies = set()
        for pair in currency_pairs:
            if len(pair) >= 6:
                currencies.add(pair[:3])  # Base currency
                currencies.add(pair[3:6])  # Quote currency
        
        # Check scheduled news events
        news_disqualifiers = self._check_news_disqualifiers(signal_time, currencies)
        disqualifiers.extend(news_disqualifiers)
        
        # Check market holidays
        holiday_disqualifiers = self._check_holiday_disqualifiers(signal_time)
        disqualifiers.extend(holiday_disqualifiers)
        
        # Check thin liquidity periods
        liquidity_disqualifiers = self._check_liquidity_disqualifiers(signal_time)
        disqualifiers.extend(liquidity_disqualifiers)
        
        # Check weekend gap
        weekend_disqualifiers = self._check_weekend_gap(signal_time)
        disqualifiers.extend(weekend_disqualifiers)
        
        # Check geopolitical events
        geopolitical_disqualifiers = self._check_geopolitical_disqualifiers(signal_time)
        disqualifiers.extend(geopolitical_disqualifiers)
        
        is_allowed = len(disqualifiers) == 0
        
        # Emit telemetry
        self._emit_telemetry("TRADING_CONDITIONS_EVALUATED", {
            "signal_time": signal_time.isoformat(),
            "currency_pairs": currency_pairs,
            "is_allowed": is_allowed,
            "disqualifiers_count": len(disqualifiers),
            "disqualifier_types": [d['type'] for d in disqualifiers]
        })
        
        # Update active disqualifiers
        if disqualifiers:
            self.active_disqualifiers.extend(disqualifiers)
            
            # Emit disqualifier events
            if self.event_bus:
                self.event_bus.emit("MacroDisqualifierActive", {
                    "disqualifiers": disqualifiers,
                    "signal_time": signal_time.isoformat()
                })
        
        return is_allowed, disqualifiers
    
    def _check_news_disqualifiers(self, signal_time: datetime, currencies: set) -> List[Dict]:
        """Check for news-based disqualifiers"""
        disqualifiers = []
        
        for news_event in self.scheduled_news:
            # Check if news affects any of our currencies
            if news_event.currency not in currencies:
                continue
            
            # Find applicable rule
            applicable_rule = None
            for rule in self.disqualifier_rules:
                if (news_event.title in rule.event_types and 
                    news_event.currency in rule.currencies and
                    news_event.impact.value in [rule.min_impact.value, NewsImpact.HIGH.value, NewsImpact.CRITICAL.value]):
                    applicable_rule = rule
                    break
            
            if not applicable_rule:
                continue
            
            # Check timing
            event_time = news_event.actual_time or news_event.scheduled_time
            blackout_start = event_time - timedelta(minutes=applicable_rule.before_minutes)
            blackout_end = event_time + timedelta(minutes=applicable_rule.after_minutes)
            
            if blackout_start <= signal_time <= blackout_end:
                disqualifiers.append({
                    'type': DisqualifierType.HIGH_IMPACT_NEWS.value,
                    'event_title': news_event.title,
                    'currency': news_event.currency,
                    'impact': news_event.impact.value,
                    'event_time': event_time.isoformat(),
                    'blackout_start': blackout_start.isoformat(),
                    'blackout_end': blackout_end.isoformat(),
                    'message': f"High-impact news: {news_event.title} ({news_event.currency})"
                })
        
        return disqualifiers
    
    def _check_holiday_disqualifiers(self, signal_time: datetime) -> List[Dict]:
        """Check for market holiday disqualifiers"""
        disqualifiers = []
        
        signal_date = signal_time.date()
        
        for holiday in self.market_holidays:
            holiday_date = holiday.date()
            
            # Check if signal is on holiday
            if signal_date == holiday_date:
                disqualifiers.append({
                    'type': DisqualifierType.MARKET_HOLIDAY.value,
                    'holiday_date': holiday_date.isoformat(),
                    'message': f"Market holiday: {holiday_date}"
                })
            
            # Check day before major holidays (reduced liquidity)
            elif signal_date == holiday_date - timedelta(days=1):
                # Only for major holidays
                if holiday.month in [12, 1, 7]:  # Christmas, New Year, Independence Day
                    disqualifiers.append({
                        'type': DisqualifierType.THIN_LIQUIDITY.value,
                        'reason': 'pre_holiday',
                        'holiday_date': holiday_date.isoformat(),
                        'message': f"Pre-holiday thin liquidity before {holiday_date}"
                    })
        
        return disqualifiers
    
    def _check_liquidity_disqualifiers(self, signal_time: datetime) -> List[Dict]:
        """Check for thin liquidity period disqualifiers"""
        disqualifiers = []
        
        signal_time_only = signal_time.time()
        
        for start_time, end_time in self.thin_liquidity_periods:
            if start_time <= signal_time_only <= end_time:
                disqualifiers.append({
                    'type': DisqualifierType.THIN_LIQUIDITY.value,
                    'period_start': start_time.isoformat(),
                    'period_end': end_time.isoformat(),
                    'message': f"Thin liquidity period: {start_time} - {end_time}"
                })
        
        return disqualifiers
    
    def _check_weekend_gap(self, signal_time: datetime) -> List[Dict]:
        """Check for weekend gap trading restrictions"""
        disqualifiers = []
        
        weekday = signal_time.weekday()
        hour = signal_time.hour
        
        # Friday after 21:00 GMT
        if weekday == 4 and hour >= 21:
            disqualifiers.append({
                'type': DisqualifierType.WEEKEND_GAP.value,
                'message': "Weekend gap period: Friday after 21:00 GMT"
            })
        
        # Saturday (all day)
        elif weekday == 5:
            disqualifiers.append({
                'type': DisqualifierType.WEEKEND_GAP.value,
                'message': "Weekend gap period: Saturday"
            })
        
        # Sunday before 21:00 GMT
        elif weekday == 6 and hour < 21:
            disqualifiers.append({
                'type': DisqualifierType.WEEKEND_GAP.value,
                'message': "Weekend gap period: Sunday before 21:00 GMT"
            })
        
        return disqualifiers
    
    def _check_geopolitical_disqualifiers(self, signal_time: datetime) -> List[Dict]:
        """Check for active geopolitical event disqualifiers"""
        disqualifiers = []
        
        for event in self.geopolitical_events:
            event_start = datetime.fromisoformat(event['start_time'])
            event_end = datetime.fromisoformat(event.get('end_time', 
                (event_start + timedelta(hours=event.get('duration_hours', 24))).isoformat()))
            
            if event_start <= signal_time <= event_end:
                disqualifiers.append({
                    'type': DisqualifierType.GEOPOLITICAL_EVENT.value,
                    'event_description': event['description'],
                    'severity': event.get('severity', 'HIGH'),
                    'affected_currencies': event.get('affected_currencies', []),
                    'message': f"Geopolitical event: {event['description']}"
                })
        
        return disqualifiers
    
    def add_news_event(self, news_event: NewsEvent):
        """Add news event to monitoring"""
        self.scheduled_news.append(news_event)
        
        self._emit_telemetry("NEWS_EVENT_ADDED", {
            "event_id": news_event.event_id,
            "title": news_event.title,
            "currency": news_event.currency,
            "impact": news_event.impact.value,
            "scheduled_time": news_event.scheduled_time.isoformat()
        })
    
    def add_geopolitical_event(self, event: Dict[str, Any]):
        """Add geopolitical event"""
        self.geopolitical_events.append(event)
        
        self._emit_telemetry("GEOPOLITICAL_EVENT_ADDED", {
            "description": event['description'],
            "severity": event.get('severity', 'HIGH'),
            "duration_hours": event.get('duration_hours', 24)
        })
    
    def get_active_disqualifiers(self) -> List[Dict]:
        """Get currently active disqualifiers"""
        current_time = datetime.now()
        active = []
        
        for disqualifier in self.active_disqualifiers:
            # Check if disqualifier is still active based on end time
            if 'blackout_end' in disqualifier:
                end_time = datetime.fromisoformat(disqualifier['blackout_end'])
                if current_time <= end_time:
                    active.append(disqualifier)
        
        return active
    
    def clear_expired_disqualifiers(self):
        """Clear expired disqualifiers"""
        current_time = datetime.now()
        
        # Clear expired news events
        self.scheduled_news = [
            event for event in self.scheduled_news
            if (event.actual_time or event.scheduled_time) > current_time - timedelta(hours=2)
        ]
        
        # Clear expired geopolitical events
        self.geopolitical_events = [
            event for event in self.geopolitical_events
            if datetime.fromisoformat(event.get('end_time', 
                (datetime.fromisoformat(event['start_time']) + 
                 timedelta(hours=event.get('duration_hours', 24))).isoformat())) > current_time
        ]
        
        # Clear expired active disqualifiers
        self.active_disqualifiers = self.get_active_disqualifiers()
        
        self._emit_telemetry("EXPIRED_DISQUALIFIERS_CLEARED", {
            "news_events_remaining": len(self.scheduled_news),
            "geopolitical_events_remaining": len(self.geopolitical_events),
            "active_disqualifiers_remaining": len(self.active_disqualifiers)
        })
    
    def _handle_news_calendar_update(self, event_data: Dict[str, Any]):
        """Handle news calendar update"""
        for news_item in event_data.get('news_events', []):
            news_event = NewsEvent(
                event_id=news_item['id'],
                title=news_item['title'],
                currency=news_item['currency'],
                impact=NewsImpact(news_item['impact']),
                scheduled_time=datetime.fromisoformat(news_item['scheduled_time']),
                forecast=news_item.get('forecast'),
                previous=news_item.get('previous')
            )
            self.add_news_event(news_event)
    
    def _handle_breaking_news(self, event_data: Dict[str, Any]):
        """Handle breaking news alert"""
        # Create immediate disqualifier for breaking news
        disqualifier = {
            'type': DisqualifierType.HIGH_IMPACT_NEWS.value,
            'event_title': event_data['title'],
            'currency': event_data.get('currency', 'ALL'),
            'impact': 'CRITICAL',
            'event_time': datetime.now().isoformat(),
            'blackout_start': datetime.now().isoformat(),
            'blackout_end': (datetime.now() + timedelta(minutes=30)).isoformat(),
            'message': f"Breaking news: {event_data['title']}"
        }
        
        self.active_disqualifiers.append(disqualifier)
        
        if self.event_bus:
            self.event_bus.emit("MacroDisqualifierActive", {
                "disqualifiers": [disqualifier],
                "reason": "breaking_news"
            })
    
    def _handle_geopolitical_event(self, event_data: Dict[str, Any]):
        """Handle geopolitical event"""
        self.add_geopolitical_event(event_data)
    
    def _evaluate_signal_timing(self, event_data: Dict[str, Any]):
        """Evaluate signal timing"""
        signal_time = datetime.fromisoformat(event_data.get('signal_time', datetime.now().isoformat()))
        currency_pairs = event_data.get('currency_pairs', [])
        
        is_allowed, disqualifiers = self.evaluate_trading_conditions(signal_time, currency_pairs)
        
        if self.event_bus:
            self.event_bus.emit("MacroTimingEvaluation", {
                "signal_id": event_data.get('signal_id', 'unknown'),
                "is_allowed": is_allowed,
                "disqualifiers": disqualifiers
            })
    
    def _handle_session_change(self, event_data: Dict[str, Any]):
        """Handle market session change"""
        # Clear expired disqualifiers on session change
        self.clear_expired_disqualifiers()
    
    def _handle_news_trading_check(self, event_data: Dict[str, Any]):
        """Handle explicit news trading check request"""
        signal_time = datetime.fromisoformat(event_data.get('signal_time', datetime.now().isoformat()))
        currency_pairs = event_data.get('currency_pairs', [])
        
        is_allowed, disqualifiers = self.evaluate_trading_conditions(signal_time, currency_pairs)
        
        if self.event_bus:
            self.event_bus.emit("NewsTradingCheckResult", {
                "request_id": event_data.get('request_id', 'unknown'),
                "is_allowed": is_allowed,
                "disqualifiers": disqualifiers
            })
    
    def _emit_telemetry(self, event_type: str, data: Dict[str, Any]):
        """Emit telemetry event"""
        telemetry_data = {
            "module": "macro_disqualifier",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        if self.event_bus:
            self.event_bus.emit("telemetry", telemetry_data)
        
        logging.info(f"🌍 MACRO-DISQUALIFIER {event_type}: {data}")

if __name__ == "__main__":
    # Test macro disqualifier
    print("🌍 Testing GENESIS Macro Disqualifier")
    
    disqualifier = GenesisMacroDisqualifier()
    
    # Add test news event
    test_news = NewsEvent(
        event_id="TEST_001",
        title="Non-Farm Payrolls",
        currency="USD",
        impact=NewsImpact.CRITICAL,
        scheduled_time=datetime.now() + timedelta(minutes=10)
    )
    disqualifier.add_news_event(test_news)
    
    # Test trading conditions during news
    signal_time = datetime.now() + timedelta(minutes=5)
    currency_pairs = ["EURUSD", "GBPUSD"]
    
    is_allowed, disqualifiers = disqualifier.evaluate_trading_conditions(signal_time, currency_pairs)
    print(f"Trading allowed during NFP: {is_allowed}")
    print(f"Disqualifiers: {disqualifiers}")
    
    # Test trading conditions outside news
    signal_time = datetime.now() + timedelta(hours=2)
    is_allowed, disqualifiers = disqualifier.evaluate_trading_conditions(signal_time, currency_pairs)
    print(f"Trading allowed outside news: {is_allowed}")
    print(f"Disqualifiers: {disqualifiers}")
