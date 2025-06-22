import logging
# <!-- @GENESIS_MODULE_START: test_quadpack -->
"""
🏛️ GENESIS TEST_QUADPACK - INSTITUTIONAL GRADE v8.0.0
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

import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi
from numpy.testing import (assert_,

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

                emit_telemetry("test_quadpack", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_quadpack", "position_calculated", {
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
                            "module": "test_quadpack",
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
                    print(f"Emergency stop error in test_quadpack: {e}")
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
                    "module": "test_quadpack",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_quadpack", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_quadpack: {e}")
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


        assert_allclose, assert_array_less, assert_almost_equal)
import pytest

from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable

import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes

import scipy.integrate._test_multivariate as clib_test


def assert_quad(value_and_err, tabled_value, error_tolerance=1.5e-8):
    value, err = value_and_err
    assert_allclose(value, tabled_value, atol=err, rtol=0)
    if error_tolerance is not None:
        assert_array_less(err, error_tolerance)


def get_clib_test_routine(name, restype, *argtypes):
    ptr = getattr(clib_test, name)
    return ctypes.cast(ptr, ctypes.CFUNCTYPE(restype, *argtypes))


class TestCtypesQuad:
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

            emit_telemetry("test_quadpack", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_quadpack", "position_calculated", {
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
                        "module": "test_quadpack",
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
                print(f"Emergency stop error in test_quadpack: {e}")
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
                "module": "test_quadpack",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_quadpack", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_quadpack: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_quadpack",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_quadpack: {e}")
    def setup_method(self):
        if sys.platform == 'win32':
            files = ['api-ms-win-crt-math-l1-1-0.dll']
        elif sys.platform == 'darwin':
            files = ['libm.dylib']
        else:
            files = ['libm.so', 'libm.so.6']

        for file in files:
            try:
                self.lib = ctypes.CDLL(file)
                break
            except OSError:
                pass
        else:
            # This test doesn't work on some Linux platforms (Fedora for
            # example) that put an ld script in libm.so - see gh-5370
            pytest.skip("Ctypes can't import libm.so")

        restype = ctypes.c_double
        argtypes = (ctypes.c_double,)
        for name in ['sin', 'cos', 'tan']:
            func = getattr(self.lib, name)
            func.restype = restype
            func.argtypes = argtypes

    def test_typical(self):
        assert_quad(quad(self.lib.sin, 0, 5), quad(math.sin, 0, 5)[0])
        assert_quad(quad(self.lib.cos, 0, 5), quad(math.cos, 0, 5)[0])
        assert_quad(quad(self.lib.tan, 0, 1), quad(math.tan, 0, 1)[0])

    def test_ctypes_sine(self):
        quad(LowLevelCallable(sine_ctypes), 0, 1)

    def test_ctypes_variants(self):
        sin_0 = get_clib_test_routine('_sin_0', ctypes.c_double,
                                      ctypes.c_double, ctypes.c_void_p)

        sin_1 = get_clib_test_routine('_sin_1', ctypes.c_double,
                                      ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                      ctypes.c_void_p)

        sin_2 = get_clib_test_routine('_sin_2', ctypes.c_double,
                                      ctypes.c_double)

        sin_3 = get_clib_test_routine('_sin_3', ctypes.c_double,
                                      ctypes.c_int, ctypes.POINTER(ctypes.c_double))

        sin_4 = get_clib_test_routine('_sin_3', ctypes.c_double,
                                      ctypes.c_int, ctypes.c_double)

        all_sigs = [sin_0, sin_1, sin_2, sin_3, sin_4]
        legacy_sigs = [sin_2, sin_4]
        legacy_only_sigs = [sin_4]

        # LowLevelCallables work for new signatures
        for j, func in enumerate(all_sigs):
            callback = LowLevelCallable(func)
            if func in legacy_only_sigs:
                pytest.raises(ValueError, quad, callback, 0, pi)
            else:
                assert_allclose(quad(callback, 0, pi)[0], 2.0)

        # Plain ctypes items work only for legacy signatures
        for j, func in enumerate(legacy_sigs):
            if func in legacy_sigs:
                assert_allclose(quad(func, 0, pi)[0], 2.0)
            else:
                pytest.raises(ValueError, quad, func, 0, pi)


class TestMultivariateCtypesQuad:
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

            emit_telemetry("test_quadpack", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_quadpack", "position_calculated", {
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
                        "module": "test_quadpack",
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
                print(f"Emergency stop error in test_quadpack: {e}")
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
                "module": "test_quadpack",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_quadpack", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_quadpack: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_quadpack",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_quadpack: {e}")
    def setup_method(self):
        restype = ctypes.c_double
        argtypes = (ctypes.c_int, ctypes.c_double)
        for name in ['_multivariate_typical', '_multivariate_indefinite',
                     '_multivariate_sin']:
            func = get_clib_test_routine(name, restype, *argtypes)
            setattr(self, name, func)

    def test_typical(self):
        # 1) Typical function with two extra arguments:
        assert_quad(quad(self._multivariate_typical, 0, pi, (2, 1.8)),
                    0.30614353532540296487)

    def test_indefinite(self):
        # 2) Infinite integration limits --- Euler's constant
        assert_quad(quad(self._multivariate_indefinite, 0, np.inf),
                    0.577215664901532860606512)

    def test_threadsafety(self):
        # Ensure multivariate ctypes are threadsafe
        def threadsafety(y):
            return y + quad(self._multivariate_sin, 0, 1)[0]
        assert_quad(quad(threadsafety, 0, 1), 0.9596976941318602)


class TestQuad:
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

            emit_telemetry("test_quadpack", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_quadpack", "position_calculated", {
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
                        "module": "test_quadpack",
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
                print(f"Emergency stop error in test_quadpack: {e}")
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
                "module": "test_quadpack",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_quadpack", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_quadpack: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_quadpack",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_quadpack: {e}")
    def test_typical(self):
        # 1) Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        assert_quad(quad(myfunc, 0, pi, (2, 1.8)), 0.30614353532540296487)

    def test_indefinite(self):
        # 2) Infinite integration limits --- Euler's constant
        def myfunc(x):           # Euler's constant integrand
            return -exp(-x)*log(x)
        assert_quad(quad(myfunc, 0, np.inf), 0.577215664901532860606512)

    def test_singular(self):
        # 3) Singular points in region of integration.
        def myfunc(x):
            if 0 < x < 2.5:
                return sin(x)
            elif 2.5 <= x <= 5.0:
                return exp(-x)
            else:
                return 0.0

        assert_quad(quad(myfunc, 0, 10, points=[2.5, 5.0]),
                    1 - cos(2.5) + exp(-2.5) - exp(-5.0))

    def test_sine_weighted_finite(self):
        # 4) Sine weighted integral (finite limits)
        def myfunc(x, a):
            return exp(a*(x-1))

        ome = 2.0**3.4
        assert_quad(quad(myfunc, 0, 1, args=20, weight='sin', wvar=ome),
                    (20*sin(ome)-ome*cos(ome)+ome*exp(-20))/(20**2 + ome**2))

    def test_sine_weighted_infinite(self):
        # 5) Sine weighted integral (infinite limits)
        def myfunc(x, a):
            return exp(-x*a)

        a = 4.0
        ome = 3.0
        assert_quad(quad(myfunc, 0, np.inf, args=a, weight='sin', wvar=ome),
                    ome/(a**2 + ome**2))

    def test_cosine_weighted_infinite(self):
        # 6) Cosine weighted integral (negative infinite limits)
        def myfunc(x, a):
            return exp(x*a)

        a = 2.5
        ome = 2.3
        assert_quad(quad(myfunc, -np.inf, 0, args=a, weight='cos', wvar=ome),
                    a/(a**2 + ome**2))

    def test_algebraic_log_weight(self):
        # 6) Algebraic-logarithmic weight.
        def myfunc(x, a):
            return 1/(1+x+2**(-a))

        a = 1.5
        assert_quad(quad(myfunc, -1, 1, args=a, weight='alg',
                         wvar=(-0.5, -0.5)),
                    pi/sqrt((1+2**(-a))**2 - 1))

    def test_cauchypv_weight(self):
        # 7) Cauchy prinicpal value weighting w(x) = 1/(x-c)
        def myfunc(x, a):
            return 2.0**(-a)/((x-1)**2+4.0**(-a))

        a = 0.4
        tabledValue = ((2.0**(-0.4)*log(1.5) -
                        2.0**(-1.4)*log((4.0**(-a)+16) / (4.0**(-a)+1)) -
                        arctan(2.0**(a+2)) -
                        arctan(2.0**a)) /
                       (4.0**(-a) + 1))
        assert_quad(quad(myfunc, 0, 5, args=0.4, weight='cauchy', wvar=2.0),
                    tabledValue, error_tolerance=1.9e-8)

    def test_b_less_than_a(self):
        def f(x, p, q):
            return p * np.exp(-q*x)

        val_1, err_1 = quad(f, 0, np.inf, args=(2, 3))
        val_2, err_2 = quad(f, np.inf, 0, args=(2, 3))
        assert_allclose(val_1, -val_2, atol=max(err_1, err_2))

    def test_b_less_than_a_2(self):
        def f(x, s):
            return np.exp(-x**2 / 2 / s) / np.sqrt(2.*s)

        val_1, err_1 = quad(f, -np.inf, np.inf, args=(2,))
        val_2, err_2 = quad(f, np.inf, -np.inf, args=(2,))
        assert_allclose(val_1, -val_2, atol=max(err_1, err_2))

    def test_b_less_than_a_3(self):
        def f(x):
            return 1.0

        val_1, err_1 = quad(f, 0, 1, weight='alg', wvar=(0, 0))
        val_2, err_2 = quad(f, 1, 0, weight='alg', wvar=(0, 0))
        assert_allclose(val_1, -val_2, atol=max(err_1, err_2))

    def test_b_less_than_a_full_output(self):
        def f(x):
            return 1.0

        res_1 = quad(f, 0, 1, weight='alg', wvar=(0, 0), full_output=True)
        res_2 = quad(f, 1, 0, weight='alg', wvar=(0, 0), full_output=True)
        err = max(res_1[1], res_2[1])
        assert_allclose(res_1[0], -res_2[0], atol=err)

    def test_double_integral(self):
        # 8) Double Integral test
        def simpfunc(y, x):       # Note order of arguments.
            return x+y

        a, b = 1.0, 2.0
        assert_quad(dblquad(simpfunc, a, b, lambda x: x, lambda x: 2*x),
                    5/6.0 * (b**3.0-a**3.0))

    def test_double_integral2(self):
        def func(x0, x1, t0, t1):
            return x0 + x1 + t0 + t1
        def g(x):
            return x
        def h(x):
            return 2 * x
        args = 1, 2
        assert_quad(dblquad(func, 1, 2, g, h, args=args),35./6 + 9*.5)

    def test_double_integral3(self):
        def func(x0, x1):
            return x0 + x1 + 1 + 2
        assert_quad(dblquad(func, 1, 2, 1, 2),6.)

    @pytest.mark.parametrize(
        "x_lower, x_upper, y_lower, y_upper, expected",
        [
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-inf, 0] for all n.
            (-np.inf, 0, -np.inf, 0, np.pi / 4),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-inf, -1] for each n (one at a time).
            (-np.inf, -1, -np.inf, 0, np.pi / 4 * erfc(1)),
            (-np.inf, 0, -np.inf, -1, np.pi / 4 * erfc(1)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-inf, -1] for all n.
            (-np.inf, -1, -np.inf, -1, np.pi / 4 * (erfc(1) ** 2)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-inf, 1] for each n (one at a time).
            (-np.inf, 1, -np.inf, 0, np.pi / 4 * (erf(1) + 1)),
            (-np.inf, 0, -np.inf, 1, np.pi / 4 * (erf(1) + 1)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-inf, 1] for all n.
            (-np.inf, 1, -np.inf, 1, np.pi / 4 * ((erf(1) + 1) ** 2)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain Dx = [-inf, -1] and Dy = [-inf, 1].
            (-np.inf, -1, -np.inf, 1, np.pi / 4 * ((erf(1) + 1) * erfc(1))),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain Dx = [-inf, 1] and Dy = [-inf, -1].
            (-np.inf, 1, -np.inf, -1, np.pi / 4 * ((erf(1) + 1) * erfc(1))),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [0, inf] for all n.
            (0, np.inf, 0, np.inf, np.pi / 4),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [1, inf] for each n (one at a time).
            (1, np.inf, 0, np.inf, np.pi / 4 * erfc(1)),
            (0, np.inf, 1, np.inf, np.pi / 4 * erfc(1)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [1, inf] for all n.
            (1, np.inf, 1, np.inf, np.pi / 4 * (erfc(1) ** 2)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-1, inf] for each n (one at a time).
            (-1, np.inf, 0, np.inf, np.pi / 4 * (erf(1) + 1)),
            (0, np.inf, -1, np.inf, np.pi / 4 * (erf(1) + 1)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-1, inf] for all n.
            (-1, np.inf, -1, np.inf, np.pi / 4 * ((erf(1) + 1) ** 2)),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain Dx = [-1, inf] and Dy = [1, inf].
            (-1, np.inf, 1, np.inf, np.pi / 4 * ((erf(1) + 1) * erfc(1))),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain Dx = [1, inf] and Dy = [-1, inf].
            (1, np.inf, -1, np.inf, np.pi / 4 * ((erf(1) + 1) * erfc(1))),
            # Multiple integration of a function in n = 2 variables: f(x, y, z)
            # over domain D = [-inf, inf] for all n.
            (-np.inf, np.inf, -np.inf, np.inf, np.pi)
        ]
    )
    def test_double_integral_improper(
            self, x_lower, x_upper, y_lower, y_upper, expected
    ):
        # The Gaussian Integral.
        def f(x, y):
            return np.exp(-x ** 2 - y ** 2)

        assert_quad(
            dblquad(f, x_lower, x_upper, y_lower, y_upper),
            expected,
            error_tolerance=3e-8
        )

    def test_triple_integral(self):
        # 9) Triple Integral test
        def simpfunc(z, y, x, t):      # Note order of arguments.
            return (x+y+z)*t

        a, b = 1.0, 2.0
        assert_quad(tplquad(simpfunc, a, b,
                            lambda x: x, lambda x: 2*x,
                            lambda x, y: x - y, lambda x, y: x + y,
                            (2.,)),
                     2*8/3.0 * (b**4.0 - a**4.0))

    @pytest.mark.xslow
    @pytest.mark.parametrize(
        "x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, expected",
        [
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, 0] for all n.
            (-np.inf, 0, -np.inf, 0, -np.inf, 0, (np.pi ** (3 / 2)) / 8),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, -1] for each n (one at a time).
            (-np.inf, -1, -np.inf, 0, -np.inf, 0,
             (np.pi ** (3 / 2)) / 8 * erfc(1)),
            (-np.inf, 0, -np.inf, -1, -np.inf, 0,
             (np.pi ** (3 / 2)) / 8 * erfc(1)),
            (-np.inf, 0, -np.inf, 0, -np.inf, -1,
             (np.pi ** (3 / 2)) / 8 * erfc(1)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, -1] for each n (two at a time).
            (-np.inf, -1, -np.inf, -1, -np.inf, 0,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 2)),
            (-np.inf, -1, -np.inf, 0, -np.inf, -1,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 2)),
            (-np.inf, 0, -np.inf, -1, -np.inf, -1,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 2)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, -1] for all n.
            (-np.inf, -1, -np.inf, -1, -np.inf, -1,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 3)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = [-inf, -1] and Dy = Dz = [-inf, 1].
            (-np.inf, -1, -np.inf, 1, -np.inf, 1,
             (np.pi ** (3 / 2)) / 8 * (((erf(1) + 1) ** 2) * erfc(1))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dy = [-inf, -1] and Dz = [-inf, 1].
            (-np.inf, -1, -np.inf, -1, -np.inf, 1,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) * (erfc(1) ** 2))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dz = [-inf, -1] and Dy = [-inf, 1].
            (-np.inf, -1, -np.inf, 1, -np.inf, -1,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) * (erfc(1) ** 2))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = [-inf, 1] and Dy = Dz = [-inf, -1].
            (-np.inf, 1, -np.inf, -1, -np.inf, -1,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) * (erfc(1) ** 2))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dy = [-inf, 1] and Dz = [-inf, -1].
            (-np.inf, 1, -np.inf, 1, -np.inf, -1,
             (np.pi ** (3 / 2)) / 8 * (((erf(1) + 1) ** 2) * erfc(1))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dz = [-inf, 1] and Dy = [-inf, -1].
            (-np.inf, 1, -np.inf, -1, -np.inf, 1,
             (np.pi ** (3 / 2)) / 8 * (((erf(1) + 1) ** 2) * erfc(1))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, 1] for each n (one at a time).
            (-np.inf, 1, -np.inf, 0, -np.inf, 0,
             (np.pi ** (3 / 2)) / 8 * (erf(1) + 1)),
            (-np.inf, 0, -np.inf, 1, -np.inf, 0,
             (np.pi ** (3 / 2)) / 8 * (erf(1) + 1)),
            (-np.inf, 0, -np.inf, 0, -np.inf, 1,
             (np.pi ** (3 / 2)) / 8 * (erf(1) + 1)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, 1] for each n (two at a time).
            (-np.inf, 1, -np.inf, 1, -np.inf, 0,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 2)),
            (-np.inf, 1, -np.inf, 0, -np.inf, 1,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 2)),
            (-np.inf, 0, -np.inf, 1, -np.inf, 1,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 2)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, 1] for all n.
            (-np.inf, 1, -np.inf, 1, -np.inf, 1,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 3)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [0, inf] for all n.
            (0, np.inf, 0, np.inf, 0, np.inf, (np.pi ** (3 / 2)) / 8),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [1, inf] for each n (one at a time).
            (1, np.inf, 0, np.inf, 0, np.inf,
             (np.pi ** (3 / 2)) / 8 * erfc(1)),
            (0, np.inf, 1, np.inf, 0, np.inf,
             (np.pi ** (3 / 2)) / 8 * erfc(1)),
            (0, np.inf, 0, np.inf, 1, np.inf,
             (np.pi ** (3 / 2)) / 8 * erfc(1)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [1, inf] for each n (two at a time).
            (1, np.inf, 1, np.inf, 0, np.inf,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 2)),
            (1, np.inf, 0, np.inf, 1, np.inf,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 2)),
            (0, np.inf, 1, np.inf, 1, np.inf,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 2)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [1, inf] for all n.
            (1, np.inf, 1, np.inf, 1, np.inf,
             (np.pi ** (3 / 2)) / 8 * (erfc(1) ** 3)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-1, inf] for each n (one at a time).
            (-1, np.inf, 0, np.inf, 0, np.inf,
             (np.pi ** (3 / 2)) / 8 * (erf(1) + 1)),
            (0, np.inf, -1, np.inf, 0, np.inf,
             (np.pi ** (3 / 2)) / 8 * (erf(1) + 1)),
            (0, np.inf, 0, np.inf, -1, np.inf,
             (np.pi ** (3 / 2)) / 8 * (erf(1) + 1)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-1, inf] for each n (two at a time).
            (-1, np.inf, -1, np.inf, 0, np.inf,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 2)),
            (-1, np.inf, 0, np.inf, -1, np.inf,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 2)),
            (0, np.inf, -1, np.inf, -1, np.inf,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 2)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-1, inf] for all n.
            (-1, np.inf, -1, np.inf, -1, np.inf,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) ** 3)),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = [1, inf] and Dy = Dz = [-1, inf].
            (1, np.inf, -1, np.inf, -1, np.inf,
             (np.pi ** (3 / 2)) / 8 * (((erf(1) + 1) ** 2) * erfc(1))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dy = [1, inf] and Dz = [-1, inf].
            (1, np.inf, 1, np.inf, -1, np.inf,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) * (erfc(1) ** 2))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dz = [1, inf] and Dy = [-1, inf].
            (1, np.inf, -1, np.inf, 1, np.inf,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) * (erfc(1) ** 2))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = [-1, inf] and Dy = Dz = [1, inf].
            (-1, np.inf, 1, np.inf, 1, np.inf,
             (np.pi ** (3 / 2)) / 8 * ((erf(1) + 1) * (erfc(1) ** 2))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dy = [-1, inf] and Dz = [1, inf].
            (-1, np.inf, -1, np.inf, 1, np.inf,
             (np.pi ** (3 / 2)) / 8 * (((erf(1) + 1) ** 2) * erfc(1))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain Dx = Dz = [-1, inf] and Dy = [1, inf].
            (-1, np.inf, 1, np.inf, -1, np.inf,
             (np.pi ** (3 / 2)) / 8 * (((erf(1) + 1) ** 2) * erfc(1))),
            # Multiple integration of a function in n = 3 variables: f(x, y, z)
            # over domain D = [-inf, inf] for all n.
            (-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf,
             np.pi ** (3 / 2)),
        ],
    )
    def test_triple_integral_improper(
            self,
            x_lower,
            x_upper,
            y_lower,
            y_upper,
            z_lower,
            z_upper,
            expected
    ):
        # The Gaussian Integral.
        def f(x, y, z):
            return np.exp(-x ** 2 - y ** 2 - z ** 2)

        assert_quad(
            tplquad(f, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper),
            expected,
            error_tolerance=6e-8
        )

    def test_complex(self):
        def tfunc(x):
            return np.exp(1j*x)

        assert np.allclose(
                    quad(tfunc, 0, np.pi/2, complex_func=True)[0],
                    1+1j)

        # We consider a divergent case in order to force quadpack
        # to return an error message.  The output is compared
        # against what is returned by explicit integration
        # of the parts.
        kwargs = {'a': 0, 'b': np.inf, 'full_output': True,
                  'weight': 'cos', 'wvar': 1}
        res_c = quad(tfunc, complex_func=True, **kwargs)
        res_r = quad(lambda x: np.real(np.exp(1j*x)),
                     complex_func=False,
                     **kwargs)
        res_i = quad(lambda x: np.imag(np.exp(1j*x)),
                     complex_func=False,
                     **kwargs)

        np.testing.assert_equal(res_c[0], res_r[0] + 1j*res_i[0])
        np.testing.assert_equal(res_c[1], res_r[1] + 1j*res_i[1])

        assert len(res_c[2]['real']) == len(res_r[2:]) == 3
        assert res_c[2]['real'][2] == res_r[4]
        assert res_c[2]['real'][1] == res_r[3]
        assert res_c[2]['real'][0]['lst'] == res_r[2]['lst']

        assert len(res_c[2]['imag']) == len(res_i[2:]) == 1
        assert res_c[2]['imag'][0]['lst'] == res_i[2]['lst']


class TestNQuad:
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

            emit_telemetry("test_quadpack", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_quadpack", "position_calculated", {
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
                        "module": "test_quadpack",
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
                print(f"Emergency stop error in test_quadpack: {e}")
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
                "module": "test_quadpack",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_quadpack", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_quadpack: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_quadpack",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_quadpack: {e}")
    @pytest.mark.fail_slow(5)
    def test_fixed_limits(self):
        def func1(x0, x1, x2, x3):
            val = (x0**2 + x1*x2 - x3**3 + np.sin(x0) +
                   (1 if (x0 - 0.2*x3 - 0.5 - 0.25*x1 > 0) else 0))
            return val

        def opts_basic(*args):
            return {'points': [0.2*args[2] + 0.5 + 0.25*args[0]]}

        res = nquad(func1, [[0, 1], [-1, 1], [.13, .8], [-.15, 1]],
                    opts=[opts_basic, {}, {}, {}], full_output=True)
        assert_quad(res[:-1], 1.5267454070738635)
        assert_(res[-1]['neval'] > 0 and res[-1]['neval'] < 4e5)

    @pytest.mark.fail_slow(5)
    def test_variable_limits(self):
        scale = .1

        def func2(x0, x1, x2, x3, t0, t1):
            val = (x0*x1*x3**2 + np.sin(x2) + 1 +
                   (1 if x0 + t1*x1 - t0 > 0 else 0))
            return val

        def lim0(x1, x2, x3, t0, t1):
            return [scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) - 1,
                    scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) + 1]

        def lim1(x2, x3, t0, t1):
            return [scale * (t0*x2 + t1*x3) - 1,
                    scale * (t0*x2 + t1*x3) + 1]

        def lim2(x3, t0, t1):
            return [scale * (x3 + t0**2*t1**3) - 1,
                    scale * (x3 + t0**2*t1**3) + 1]

        def lim3(t0, t1):
            return [scale * (t0 + t1) - 1, scale * (t0 + t1) + 1]

        def opts0(x1, x2, x3, t0, t1):
            return {'points': [t0 - t1*x1]}

        def opts1(x2, x3, t0, t1):
            return {}

        def opts2(x3, t0, t1):
            return {}

        def opts3(t0, t1):
            return {}

        res = nquad(func2, [lim0, lim1, lim2, lim3], args=(0, 0),
                    opts=[opts0, opts1, opts2, opts3])
        assert_quad(res, 25.066666666666663)

    def test_square_separate_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        assert_quad(nquad(f, [[-1, 1], [-1, 1]], opts=[{}, {}]), 4.0)

    def test_square_aliased_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        r = [-1, 1]
        opt = {}
        assert_quad(nquad(f, [r, r], opts=[opt, opt]), 4.0)

    def test_square_separate_fn_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        def fn_range0(*args):
            return (-1, 1)

        def fn_range1(*args):
            return (-1, 1)

        def fn_opt0(*args):
            return {}

        def fn_opt1(*args):
            return {}

        ranges = [fn_range0, fn_range1]
        opts = [fn_opt0, fn_opt1]
        assert_quad(nquad(f, ranges, opts=opts), 4.0)

    def test_square_aliased_fn_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        def fn_range(*args):
            return (-1, 1)

        def fn_opt(*args):
            return {}

        ranges = [fn_range, fn_range]
        opts = [fn_opt, fn_opt]
        assert_quad(nquad(f, ranges, opts=opts), 4.0)

    def test_matching_quad(self):
        def func(x):
            return x**2 + 1

        res, reserr = quad(func, 0, 4)
        res2, reserr2 = nquad(func, ranges=[[0, 4]])
        assert_almost_equal(res, res2)
        assert_almost_equal(reserr, reserr2)

    def test_matching_dblquad(self):
        def func2d(x0, x1):
            return x0**2 + x1**3 - x0 * x1 + 1

        res, reserr = dblquad(func2d, -2, 2, lambda x: -3, lambda x: 3)
        res2, reserr2 = nquad(func2d, [[-3, 3], (-2, 2)])
        assert_almost_equal(res, res2)
        assert_almost_equal(reserr, reserr2)

    def test_matching_tplquad(self):
        def func3d(x0, x1, x2, c0, c1):
            return x0**2 + c0 * x1**3 - x0 * x1 + 1 + c1 * np.sin(x2)

        res = tplquad(func3d, -1, 2, lambda x: -2, lambda x: 2,
                      lambda x, y: -np.pi, lambda x, y: np.pi,
                      args=(2, 3))
        res2 = nquad(func3d, [[-np.pi, np.pi], [-2, 2], (-1, 2)], args=(2, 3))
        assert_almost_equal(res, res2)

    def test_dict_as_opts(self):
        try:
            nquad(lambda x, y: x * y, [[0, 1], [0, 1]], opts={'epsrel': 0.0001})
        except TypeError:
            assert False



# <!-- @GENESIS_MODULE_END: test_quadpack -->
