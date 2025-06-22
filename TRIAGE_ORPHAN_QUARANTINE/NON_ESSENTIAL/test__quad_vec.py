import logging
# <!-- @GENESIS_MODULE_START: test__quad_vec -->
"""
🏛️ GENESIS TEST__QUAD_VEC - INSTITUTIONAL GRADE v8.0.0
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

import pytest

import numpy as np
from numpy.testing import assert_allclose

from scipy.integrate import quad_vec

from multiprocessing.dummy import Pool

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

                emit_telemetry("test__quad_vec", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test__quad_vec", "position_calculated", {
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
                            "module": "test__quad_vec",
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
                    print(f"Emergency stop error in test__quad_vec: {e}")
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
                    "module": "test__quad_vec",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test__quad_vec", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test__quad_vec: {e}")
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




quadrature_params = pytest.mark.parametrize(
    'quadrature', [None, "gk15", "gk21", "trapezoid"])


@quadrature_params
def test_quad_vec_simple(quadrature):
    n = np.arange(10)
    def f(x):
        return x ** n
    for epsabs in [0.1, 1e-3, 1e-6]:
        if quadrature == 'trapezoid' and epsabs < 1e-4:
            # slow: skip
            continue

        kwargs = dict(epsabs=epsabs, quadrature=quadrature)

        exact = 2**(n+1)/(n + 1)

        res, err = quad_vec(f, 0, 2, norm='max', **kwargs)
        assert_allclose(res, exact, rtol=0, atol=epsabs)

        res, err = quad_vec(f, 0, 2, norm='2', **kwargs)
        assert np.linalg.norm(res - exact) < epsabs

        res, err = quad_vec(f, 0, 2, norm='max', points=(0.5, 1.0), **kwargs)
        assert_allclose(res, exact, rtol=0, atol=epsabs)

        res, err, *rest = quad_vec(f, 0, 2, norm='max',
                                   epsrel=1e-8,
                                   full_output=True,
                                   limit=10000,
                                   **kwargs)
        assert_allclose(res, exact, rtol=0, atol=epsabs)


@quadrature_params
def test_quad_vec_simple_inf(quadrature):
    def f(x):
        return 1 / (1 + np.float64(x) ** 2)

    for epsabs in [0.1, 1e-3, 1e-6]:
        if quadrature == 'trapezoid' and epsabs < 1e-4:
            # slow: skip
            continue

        kwargs = dict(norm='max', epsabs=epsabs, quadrature=quadrature)

        res, err = quad_vec(f, 0, np.inf, **kwargs)
        assert_allclose(res, np.pi/2, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, 0, -np.inf, **kwargs)
        assert_allclose(res, -np.pi/2, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, -np.inf, 0, **kwargs)
        assert_allclose(res, np.pi/2, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, np.inf, 0, **kwargs)
        assert_allclose(res, -np.pi/2, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, -np.inf, np.inf, **kwargs)
        assert_allclose(res, np.pi, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, np.inf, -np.inf, **kwargs)
        assert_allclose(res, -np.pi, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, np.inf, np.inf, **kwargs)
        assert_allclose(res, 0, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, -np.inf, -np.inf, **kwargs)
        assert_allclose(res, 0, rtol=0, atol=max(epsabs, err))

        res, err = quad_vec(f, 0, np.inf, points=(1.0, 2.0), **kwargs)
        assert_allclose(res, np.pi/2, rtol=0, atol=max(epsabs, err))

    def f(x):
        return np.sin(x + 2) / (1 + x ** 2)
    exact = np.pi / np.e * np.sin(2)
    epsabs = 1e-5

    res, err, info = quad_vec(f, -np.inf, np.inf, limit=1000, norm='max', epsabs=epsabs,
                              quadrature=quadrature, full_output=True)
    assert info.status == 1
    assert_allclose(res, exact, rtol=0, atol=max(epsabs, 1.5 * err))


def test_quad_vec_args():
    def f(x, a):
        return x * (x + a) * np.arange(3)
    a = 2
    exact = np.array([0, 4/3, 8/3])

    res, err = quad_vec(f, 0, 1, args=(a,))
    assert_allclose(res, exact, rtol=0, atol=1e-4)


def _lorenzian(x):
    return 1 / (1 + x**2)


@pytest.mark.fail_slow(10)
def test_quad_vec_pool():
    f = _lorenzian
    res, err = quad_vec(f, -np.inf, np.inf, norm='max', epsabs=1e-4, workers=4)
    assert_allclose(res, np.pi, rtol=0, atol=1e-4)

    with Pool(10) as pool:
        def f(x):
            return 1 / (1 + x ** 2)
        res, _ = quad_vec(f, -np.inf, np.inf, norm='max', epsabs=1e-4, workers=pool.map)
        assert_allclose(res, np.pi, rtol=0, atol=1e-4)


def _func_with_args(x, a):
    return x * (x + a) * np.arange(3)


@pytest.mark.fail_slow(10)
@pytest.mark.parametrize('extra_args', [2, (2,)])
@pytest.mark.parametrize('workers', [1, 10])
def test_quad_vec_pool_args(extra_args, workers):
    f = _func_with_args
    exact = np.array([0, 4/3, 8/3])

    res, err = quad_vec(f, 0, 1, args=extra_args, workers=workers)
    assert_allclose(res, exact, rtol=0, atol=1e-4)

    with Pool(workers) as pool:
        res, err = quad_vec(f, 0, 1, args=extra_args, workers=pool.map)
        assert_allclose(res, exact, rtol=0, atol=1e-4)


@quadrature_params
def test_num_eval(quadrature):
    def f(x):
        count[0] += 1
        return x**5

    count = [0]
    res = quad_vec(f, 0, 1, norm='max', full_output=True, quadrature=quadrature)
    assert res[2].neval == count[0]


def test_info():
    def f(x):
        return np.ones((3, 2, 1))

    res, err, info = quad_vec(f, 0, 1, norm='max', full_output=True)

    assert info.success is True
    assert info.status == 0
    assert info.message == 'Target precision reached.'
    assert info.neval > 0
    assert info.intervals.shape[1] == 2
    assert info.integrals.shape == (info.intervals.shape[0], 3, 2, 1)
    assert info.errors.shape == (info.intervals.shape[0],)


def test_nan_inf():
    def f_nan(x):
        return np.nan

    def f_inf(x):
        return np.inf if x < 0.1 else 1/x

    res, err, info = quad_vec(f_nan, 0, 1, full_output=True)
    assert info.status == 3

    res, err, info = quad_vec(f_inf, 0, 1, full_output=True)
    assert info.status == 3


@pytest.mark.parametrize('a,b', [(0, 1), (0, np.inf), (np.inf, 0),
                                 (-np.inf, np.inf), (np.inf, -np.inf)])
def test_points(a, b):
    # Check that initial interval splitting is done according to
    # `points`, by checking that consecutive sets of 15 point (for
    # gk15) function evaluations lie between `points`

    points = (0, 0.25, 0.5, 0.75, 1.0)
    points += tuple(-x for x in points)

    quadrature_points = 15
    interval_sets = []
    count = 0

    def f(x):
        nonlocal count

        if count % quadrature_points == 0:
            interval_sets.append(set())

        count += 1
        interval_sets[-1].add(float(x))
        return 0.0

    quad_vec(f, a, b, points=points, quadrature='gk15', limit=0)

    # Check that all point sets lie in a single `points` interval
    for p in interval_sets:
        j = np.searchsorted(sorted(points), tuple(p))
        assert np.all(j == j[0])


@pytest.mark.thread_unsafe
def test_trapz_deprecation():
    with pytest.deprecated_call(match="`quadrature='trapz'`"):
        quad_vec(lambda x: x, 0, 1, quadrature="trapz")


# <!-- @GENESIS_MODULE_END: test__quad_vec -->
