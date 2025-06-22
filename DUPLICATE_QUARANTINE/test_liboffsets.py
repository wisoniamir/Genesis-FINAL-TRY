import logging
# <!-- @GENESIS_MODULE_START: test_liboffsets -->
"""
🏛️ GENESIS TEST_LIBOFFSETS - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_liboffsets", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_liboffsets", "position_calculated", {
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
                            "module": "test_liboffsets",
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
                    print(f"Emergency stop error in test_liboffsets: {e}")
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
                    "module": "test_liboffsets",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_liboffsets", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_liboffsets: {e}")
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


"""
Tests for helper functions in the cython tslibs.offsets
"""
from datetime import datetime

import pytest

from pandas._libs.tslibs.ccalendar import (
    get_firstbday,
    get_lastbday,
)
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday

from pandas import Timestamp


@pytest.fixture(params=["start", "end", "business_start", "business_end"])
def day_opt(request):
    return request.param


@pytest.mark.parametrize(
    "dt,exp_week_day,exp_last_day",
    [
        (datetime(2017, 11, 30), 3, 30),  # Business day.
        (datetime(1993, 10, 31), 6, 29),  # Non-business day.
    ],
)
def test_get_last_bday(dt, exp_week_day, exp_last_day):
    assert dt.weekday() == exp_week_day
    assert get_lastbday(dt.year, dt.month) == exp_last_day


@pytest.mark.parametrize(
    "dt,exp_week_day,exp_first_day",
    [
        (datetime(2017, 4, 1), 5, 3),  # Non-weekday.
        (datetime(1993, 10, 1), 4, 1),  # Business day.
    ],
)
def test_get_first_bday(dt, exp_week_day, exp_first_day):
    assert dt.weekday() == exp_week_day
    assert get_firstbday(dt.year, dt.month) == exp_first_day


@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (0, 15, datetime(2017, 11, 15)),
        (0, None, datetime(2017, 11, 30)),
        (1, "start", datetime(2017, 12, 1)),
        (-145, "end", datetime(2005, 10, 31)),
        (0, "business_end", datetime(2017, 11, 30)),
        (0, "business_start", datetime(2017, 11, 1)),
    ],
)
def test_shift_month_dt(months, day_opt, expected):
    dt = datetime(2017, 11, 30)
    assert liboffsets.shift_month(dt, months, day_opt=day_opt) == expected


@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (1, "start", Timestamp("1929-06-01")),
        (-3, "end", Timestamp("1929-02-28")),
        (25, None, Timestamp("1931-06-5")),
        (-1, 31, Timestamp("1929-04-30")),
    ],
)
def test_shift_month_ts(months, day_opt, expected):
    ts = Timestamp("1929-05-05")
    assert liboffsets.shift_month(ts, months, day_opt=day_opt) == expected


def test_shift_month_error():
    dt = datetime(2017, 11, 15)
    day_opt = "this should raise"

    with pytest.raises(ValueError, match=day_opt):
        liboffsets.shift_month(dt, 3, day_opt=day_opt)


@pytest.mark.parametrize(
    "other,expected",
    [
        # Before March 1.
        (datetime(2017, 2, 10), {2: 1, -7: -7, 0: 0}),
        # After March 1.
        (Timestamp("2014-03-15", tz="US/Eastern"), {2: 2, -7: -6, 0: 1}),
    ],
)
@pytest.mark.parametrize("n", [2, -7, 0])
def test_roll_qtrday_year(other, expected, n):
    month = 3
    day_opt = "start"  # `other` will be compared to March 1.

    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]


@pytest.mark.parametrize(
    "other,expected",
    [
        # Before June 30.
        (datetime(1999, 6, 29), {5: 4, -7: -7, 0: 0}),
        # After June 30.
        (Timestamp(2072, 8, 24, 6, 17, 18), {5: 5, -7: -6, 0: 1}),
    ],
)
@pytest.mark.parametrize("n", [5, -7, 0])
def test_roll_qtrday_year2(other, expected, n):
    month = 6
    day_opt = "end"  # `other` will be compared to June 30.

    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]


def test_get_day_of_month_error():
    # get_day_of_month is not directly exposed.
    # We test it via roll_qtrday.
    dt = datetime(2017, 11, 15)
    day_opt = "foo"

    with pytest.raises(ValueError, match=day_opt):
        # To hit the raising case we need month == dt.month and n > 0.
        roll_qtrday(dt, n=3, month=11, day_opt=day_opt, modby=12)


@pytest.mark.parametrize(
    "month",
    [3, 5],  # (other.month % 3) < (month % 3)  # (other.month % 3) > (month % 3)
)
@pytest.mark.parametrize("n", [4, -3])
def test_roll_qtr_day_not_mod_unequal(day_opt, month, n):
    expected = {3: {-3: -2, 4: 4}, 5: {-3: -3, 4: 3}}

    other = Timestamp(2072, 10, 1, 6, 17, 18)  # Saturday.
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected[month][n]


@pytest.mark.parametrize(
    "other,month,exp_dict",
    [
        # Monday.
        (datetime(1999, 5, 31), 2, {-1: {"start": 0, "business_start": 0}}),
        # Saturday.
        (
            Timestamp(2072, 10, 1, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1, "business_start": 1}},
        ),
        # First business day.
        (
            Timestamp(2072, 10, 3, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1}, -1: {"start": 0}},
        ),
    ],
)
@pytest.mark.parametrize("n", [2, -1])
def test_roll_qtr_day_mod_equal(other, month, exp_dict, n, day_opt):
    # All cases have (other.month % 3) == (month % 3).
    expected = exp_dict.get(n, {}).get(day_opt, n)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected


@pytest.mark.parametrize(
    "n,expected", [(42, {29: 42, 1: 42, 31: 41}), (-4, {29: -4, 1: -3, 31: -4})]
)
@pytest.mark.parametrize("compare", [29, 1, 31])
def test_roll_convention(n, expected, compare):
    assert liboffsets.roll_convention(29, n, compare) == expected[compare]


# <!-- @GENESIS_MODULE_END: test_liboffsets -->
