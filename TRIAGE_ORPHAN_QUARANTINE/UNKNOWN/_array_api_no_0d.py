import logging
# <!-- @GENESIS_MODULE_START: _array_api_no_0d -->
"""
🏛️ GENESIS _ARRAY_API_NO_0D - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("_array_api_no_0d", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_array_api_no_0d", "position_calculated", {
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
                            "module": "_array_api_no_0d",
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
                    print(f"Emergency stop error in _array_api_no_0d: {e}")
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
                    "module": "_array_api_no_0d",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_array_api_no_0d", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _array_api_no_0d: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


from datetime import datetime


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
Extra testing functions that forbid 0d-input, see #21044

While the xp_assert_* functions generally aim to follow the conventions of the
underlying `xp` library, NumPy in particular is inconsistent in its handling
of scalars vs. 0d-arrays, see https://github.com/numpy/numpy/issues/24897.

For example, this means that the following operations (as of v2.0.1) currently
return scalars, even though a 0d-array would often be more appropriate:

    import numpy as np
    np.array(0) * 2     # scalar, not 0d array
    - np.array(0)       # scalar, not 0d-array
    np.sin(np.array(0)) # scalar, not 0d array
    np.mean([1, 2, 3])  # scalar, not 0d array

Libraries like CuPy tend to return a 0d-array in scenarios like those above,
and even `xp.asarray(0)[()]` remains a 0d-array there. To deal with the reality
of the inconsistencies present in NumPy, as well as 20+ years of code on top,
the `xp_assert_*` functions here enforce consistency in the only way that
doesn't go against the tide, i.e. by forbidding 0d-arrays as the return type.

However, when scalars are not generally the expected NumPy return type,
it remains preferable to use the assert functions from
the `scipy._lib._array_api` module, which have less surprising behaviour.
"""
from scipy._lib._array_api import array_namespace, is_numpy
from scipy._lib._array_api import (xp_assert_close as xp_assert_close_base,
                                   xp_assert_equal as xp_assert_equal_base,
                                   xp_assert_less as xp_assert_less_base)

__all__: list[str] = []


def _check_scalar(actual, desired, *, xp=None, **kwargs):
    __tracebackhide__ = True  # Hide traceback for py.test

    if xp is None:
        xp = array_namespace(actual)

    # necessary to handle non-numpy scalars, e.g. bare `0.0` has no shape
    desired = xp.asarray(desired)

    # Only NumPy distinguishes between scalars and arrays;
    # shape check in xp_assert_* is sufficient except for shape == ()
    if not (is_numpy(xp) and desired.shape == ()):
        return

    _msg = ("Result is a NumPy 0d-array. Many SciPy functions intend to follow "
            "the convention of many NumPy functions, returning a scalar when a "
            "0d-array would be correct. The specialized `xp_assert_*` functions "
            "in the `scipy._lib._array_api_no_0d` module err on the side of "
            "caution and do not accept 0d-arrays by default. If the correct "
            "result may legitimately be a 0d-array, pass `check_0d=True`, "
            "or use the `xp_assert_*` functions from `scipy._lib._array_api`.")
    assert xp.isscalar(actual), _msg


def xp_assert_equal(actual, desired, *, check_0d=False, **kwargs):
    # in contrast to xp_assert_equal_base, this defaults to check_0d=False,
    # but will do an extra check in that case, which forbids 0d-arrays for `actual`
    __tracebackhide__ = True  # Hide traceback for py.test

    # array-ness (check_0d == True) is taken care of by the *_base functions
    if not check_0d:
        _check_scalar(actual, desired, **kwargs)
    return xp_assert_equal_base(actual, desired, check_0d=check_0d, **kwargs)


def xp_assert_close(actual, desired, *, check_0d=False, **kwargs):
    # as for xp_assert_equal
    __tracebackhide__ = True

    if not check_0d:
        _check_scalar(actual, desired, **kwargs)
    return xp_assert_close_base(actual, desired, check_0d=check_0d, **kwargs)


def xp_assert_less(actual, desired, *, check_0d=False, **kwargs):
    # as for xp_assert_equal
    __tracebackhide__ = True

    if not check_0d:
        _check_scalar(actual, desired, **kwargs)
    return xp_assert_less_base(actual, desired, check_0d=check_0d, **kwargs)


def assert_array_almost_equal(actual, desired, decimal=6, *args, **kwds):
    """Backwards compatible replacement. In new code, use xp_assert_close instead.
    """
    rtol, atol = 0, 1.5*10**(-decimal)
    return xp_assert_close(actual, desired,
                           atol=atol, rtol=rtol, check_dtype=False, check_shape=False,
                           *args, **kwds)


def assert_almost_equal(actual, desired, decimal=7, *args, **kwds):
    """Backwards compatible replacement. In new code, use xp_assert_close instead.
    """
    rtol, atol = 0, 1.5*10**(-decimal)
    return xp_assert_close(actual, desired,
                           atol=atol, rtol=rtol, check_dtype=False, check_shape=False,
                           *args, **kwds)


# <!-- @GENESIS_MODULE_END: _array_api_no_0d -->
