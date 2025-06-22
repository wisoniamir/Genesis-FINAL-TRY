import logging
# <!-- @GENESIS_MODULE_START: test_milp -->
"""
🏛️ GENESIS TEST_MILP - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_milp", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_milp", "position_calculated", {
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
                            "module": "test_milp",
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
                    print(f"Emergency stop error in test_milp: {e}")
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
                    "module": "test_milp",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_milp", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_milp: {e}")
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
Unit test for Mixed Integer Linear Programming
"""
import re
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse


_IS_32BIT = (sys.maxsize < 2**32)

def test_milp_iv():

    message = "`c` must be a dense array"
    with pytest.raises(ValueError, match=message):
        milp(sparse.coo_array([0, 0]))

    message = "`c` must be a one-dimensional array of finite numbers with"
    with pytest.raises(ValueError, match=message):
        milp(np.zeros((3, 4)))
    with pytest.raises(ValueError, match=message):
        milp([])
    with pytest.raises(ValueError, match=message):
        milp(None)

    message = "`bounds` must be convertible into an instance of..."
    with pytest.raises(ValueError, match=message):
        milp(1, bounds=10)

    message = "`constraints` (or each element within `constraints`) must be"
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(1, constraints=10)
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(3), constraints=([[1, 2, 3]], [2, 3], [2, 3]))
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(2), constraints=([[1, 2]], [2], sparse.coo_array([2])))

    message = "The shape of `A` must be (len(b_l), len(c))."
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(3), constraints=([[1, 2]], [2], [2]))

    message = "`integrality` must be a dense array"
    with pytest.raises(ValueError, match=message):
        milp([1, 2], integrality=sparse.coo_array([1, 2]))

    message = ("`integrality` must contain integers 0-3 and be broadcastable "
               "to `c.shape`.")
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], integrality=[1, 2])
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], integrality=[1, 5, 3])

    message = "Lower and upper bounds must be dense arrays."
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2], sparse.coo_array([3, 4])))

    message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2], [3, 4, 5]))
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2, 3], [4, 5]))

    message = "`bounds.lb` and `bounds.ub` must contain reals and..."
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2], [3, 4]))
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2, 3], ["3+4", 4, 5]))
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2, 3], [set(), 4, 5]))


@pytest.mark.xfail(run=False,
                   reason="Needs to be fixed in `_highs_wrapper`")
def test_milp_options(capsys):
    # run=False now because of gh-16347
    message = "Unrecognized options detected: {'ekki'}..."
    options = {'ekki': True}
    with pytest.warns(RuntimeWarning, match=message):
        milp(1, options=options)

    A, b, c, numbers, M = magic_square(3)
    options = {"disp": True, "presolve": False, "time_limit": 0.05}
    res = milp(c=c, constraints=(A, b, b), bounds=(0, 1), integrality=1,
               options=options)

    captured = capsys.readouterr()
    assert "Presolve is switched off" in captured.out
    assert "Time Limit Reached" in captured.out
    assert not res.success


def test_result():
    A, b, c, numbers, M = magic_square(3)
    res = milp(c=c, constraints=(A, b, b), bounds=(0, 1), integrality=1)
    assert res.status == 0
    assert res.success
    msg = "Optimization terminated successfully. (HiGHS Status 7:"
    assert res.message.startswith(msg)
    assert isinstance(res.x, np.ndarray)
    assert isinstance(res.fun, float)
    assert isinstance(res.mip_node_count, int)
    assert isinstance(res.mip_dual_bound, float)
    assert isinstance(res.mip_gap, float)

    A, b, c, numbers, M = magic_square(6)
    res = milp(c=c*0, constraints=(A, b, b), bounds=(0, 1), integrality=1,
               options={'time_limit': 0.05})
    assert res.status == 1
    assert not res.success
    msg = "Time limit reached. (HiGHS Status 13:"
    assert res.message.startswith(msg)
    assert (res.fun is res.mip_dual_bound is res.mip_gap
            is res.mip_node_count is res.x is None)

    res = milp(1, bounds=(1, -1))
    assert res.status == 2
    assert not res.success
    msg = "The problem is infeasible. (HiGHS Status 8:"
    assert res.message.startswith(msg)
    assert (res.fun is res.mip_dual_bound is res.mip_gap
            is res.mip_node_count is res.x is None)

    res = milp(-1)
    assert res.status == 3
    assert not res.success
    msg = "The problem is unbounded. (HiGHS Status 10:"
    assert res.message.startswith(msg)
    assert (res.fun is res.mip_dual_bound is res.mip_gap
            is res.mip_node_count is res.x is None)


def test_milp_optional_args():
    # check that arguments other than `c` are indeed optional
    res = milp(1)
    assert res.fun == 0
    assert_array_equal(res.x, [0])


def test_milp_1():
    # solve magic square problem
    n = 3
    A, b, c, numbers, M = magic_square(n)
    A = sparse.csc_array(A)  # confirm that sparse arrays are accepted
    res = milp(c=c*0, constraints=(A, b, b), bounds=(0, 1), integrality=1)

    # check that solution is a magic square
    x = np.round(res.x)
    s = (numbers.flatten() * x).reshape(n**2, n, n)
    square = np.sum(s, axis=0)
    np.testing.assert_allclose(square.sum(axis=0), M)
    np.testing.assert_allclose(square.sum(axis=1), M)
    np.testing.assert_allclose(np.diag(square).sum(), M)
    np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)


def test_milp_2():
    # solve MIP with inequality constraints and all integer constraints
    # source: slide 5,
    # https://www.cs.upc.edu/~erodri/webpage/cps/theory/lp/milp/slides.pdf
    # also check that `milp` accepts all valid ways of specifying constraints
    c = -np.ones(2)
    A = [[-2, 2], [-8, 10]]
    b_l = [1, -np.inf]
    b_u = [np.inf, 13]
    linear_constraint = LinearConstraint(A, b_l, b_u)

    # solve original problem
    res1 = milp(c=c, constraints=(A, b_l, b_u), integrality=True)
    res2 = milp(c=c, constraints=linear_constraint, integrality=True)
    res3 = milp(c=c, constraints=[(A, b_l, b_u)], integrality=True)
    res4 = milp(c=c, constraints=[linear_constraint], integrality=True)
    res5 = milp(c=c, integrality=True,
                constraints=[(A[:1], b_l[:1], b_u[:1]),
                             (A[1:], b_l[1:], b_u[1:])])
    res6 = milp(c=c, integrality=True,
                constraints=[LinearConstraint(A[:1], b_l[:1], b_u[:1]),
                             LinearConstraint(A[1:], b_l[1:], b_u[1:])])
    res7 = milp(c=c, integrality=True,
                constraints=[(A[:1], b_l[:1], b_u[:1]),
                             LinearConstraint(A[1:], b_l[1:], b_u[1:])])
    xs = np.array([res1.x, res2.x, res3.x, res4.x, res5.x, res6.x, res7.x])
    funs = np.array([res1.fun, res2.fun, res3.fun,
                     res4.fun, res5.fun, res6.fun, res7.fun])
    np.testing.assert_allclose(xs, np.broadcast_to([1, 2], xs.shape))
    np.testing.assert_allclose(funs, -3)

    # solve relaxed problem
    res = milp(c=c, constraints=(A, b_l, b_u))
    np.testing.assert_allclose(res.x, [4, 4.5])
    np.testing.assert_allclose(res.fun, -8.5)


def test_milp_3():
    # solve MIP with inequality constraints and all integer constraints
    # source: https://en.wikipedia.org/wiki/Integer_programming#Example
    c = [0, -1]
    A = [[-1, 1], [3, 2], [2, 3]]
    b_u = [1, 12, 12]
    b_l = np.full_like(b_u, -np.inf, dtype=np.float64)
    constraints = LinearConstraint(A, b_l, b_u)

    integrality = np.ones_like(c)

    # solve original problem
    res = milp(c=c, constraints=constraints, integrality=integrality)
    assert_allclose(res.fun, -2)
    # two optimal solutions possible, just need one of them
    assert np.allclose(res.x, [1, 2]) or np.allclose(res.x, [2, 2])

    # solve relaxed problem
    res = milp(c=c, constraints=constraints)
    assert_allclose(res.fun, -2.8)
    assert_allclose(res.x, [1.8, 2.8])


def test_milp_4():
    # solve MIP with inequality constraints and only one integer constraint
    # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
    c = [8, 1]
    integrality = [0, 1]
    A = [[1, 2], [-4, -1], [2, 1]]
    b_l = [-14, -np.inf, -np.inf]
    b_u = [np.inf, -33, 20]
    constraints = LinearConstraint(A, b_l, b_u)
    bounds = Bounds(-np.inf, np.inf)

    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)
    assert_allclose(res.fun, 59)
    assert_allclose(res.x, [6.5, 7])


def test_milp_5():
    # solve MIP with inequality and equality constraints
    # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
    c = [-3, -2, -1]
    integrality = [0, 0, 1]
    lb = [0, 0, 0]
    ub = [np.inf, np.inf, 1]
    bounds = Bounds(lb, ub)
    A = [[1, 1, 1], [4, 2, 1]]
    b_l = [-np.inf, 12]
    b_u = [7, 12]
    constraints = LinearConstraint(A, b_l, b_u)

    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)
    # there are multiple solutions
    assert_allclose(res.fun, -12)


@pytest.mark.xslow
def test_milp_6():
    # solve a larger MIP with only equality constraints
    # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
    integrality = 1
    A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26],
                     [39, 16, 22, 28, 26, 30, 23, 24],
                     [18, 14, 29, 27, 30, 38, 26, 26],
                     [41, 26, 28, 36, 18, 38, 16, 26]])
    b_eq = np.array([7872, 10466, 11322, 12058])
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])

    res = milp(c=c, constraints=(A_eq, b_eq, b_eq), integrality=integrality)

    np.testing.assert_allclose(res.fun, 1854)


def test_infeasible_prob_16609():
    # Ensure presolve does not mark trivially infeasible problems
    # as Optimal -- see gh-16609
    c = [1.0, 0.0]
    integrality = [0, 1]

    lb = [0, -np.inf]
    ub = [np.inf, np.inf]
    bounds = Bounds(lb, ub)

    A_eq = [[0.0, 1.0]]
    b_eq = [0.5]
    constraints = LinearConstraint(A_eq, b_eq, b_eq)

    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)
    np.testing.assert_equal(res.status, 2)


_msg_time = "Time limit reached. (HiGHS Status 13:"
_msg_iter = "Iteration limit reached. (HiGHS Status 14:"

@pytest.mark.thread_unsafe
# See https://github.com/scipy/scipy/pull/19255#issuecomment-1778438888
@pytest.mark.xfail(reason="Often buggy, revisit with callbacks, gh-19255")
@pytest.mark.skipif(np.intp(0).itemsize < 8,
                    reason="Unhandled 32-bit GCC FP bug")
@pytest.mark.slow
@pytest.mark.parametrize(["options", "msg"], [({"time_limit": 0.1}, _msg_time),
                                              ({"node_limit": 1}, _msg_iter)])
def test_milp_timeout_16545(options, msg):
    # Ensure solution is not thrown away if MILP solver times out
    # -- see gh-16545
    rng = np.random.default_rng(5123833489170494244)
    A = rng.integers(0, 5, size=(100, 100))
    b_lb = np.full(100, fill_value=-np.inf)
    b_ub = np.full(100, fill_value=25)
    constraints = LinearConstraint(A, b_lb, b_ub)
    variable_lb = np.zeros(100)
    variable_ub = np.ones(100)
    variable_bounds = Bounds(variable_lb, variable_ub)
    integrality = np.ones(100)
    c_vector = -np.ones(100)
    res = milp(
        c_vector,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=constraints,
        options=options,
    )

    assert res.message.startswith(msg)
    assert res["x"] is not None

    # ensure solution is feasible
    x = res["x"]
    tol = 1e-8  # sometimes needed due to finite numerical precision
    assert np.all(b_lb - tol <= A @ x) and np.all(A @ x <= b_ub + tol)
    assert np.all(variable_lb - tol <= x) and np.all(x <= variable_ub + tol)
    assert np.allclose(x, np.round(x))


def test_three_constraints_16878():
    # `milp` failed when exactly three constraints were passed
    # Ensure that this is no longer the case.
    rng = np.random.default_rng(5123833489170494244)
    A = rng.integers(0, 5, size=(6, 6))
    bl = np.full(6, fill_value=-np.inf)
    bu = np.full(6, fill_value=10)
    constraints = [LinearConstraint(A[:2], bl[:2], bu[:2]),
                   LinearConstraint(A[2:4], bl[2:4], bu[2:4]),
                   LinearConstraint(A[4:], bl[4:], bu[4:])]
    constraints2 = [(A[:2], bl[:2], bu[:2]),
                    (A[2:4], bl[2:4], bu[2:4]),
                    (A[4:], bl[4:], bu[4:])]
    lb = np.zeros(6)
    ub = np.ones(6)
    variable_bounds = Bounds(lb, ub)
    c = -np.ones(6)
    res1 = milp(c, bounds=variable_bounds, constraints=constraints)
    res2 = milp(c, bounds=variable_bounds, constraints=constraints2)
    ref = milp(c, bounds=variable_bounds, constraints=(A, bl, bu))
    assert res1.success and res2.success
    assert_allclose(res1.x, ref.x)
    assert_allclose(res2.x, ref.x)


@pytest.mark.xslow
def test_mip_rel_gap_passdown():
    # Solve problem with decreasing mip_gap to make sure mip_rel_gap decreases
    # Adapted from test_linprog::TestLinprogHiGHSMIP::test_mip_rel_gap_passdown
    # MIP taken from test_mip_6 above
    A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26],
                     [39, 16, 22, 28, 26, 30, 23, 24],
                     [18, 14, 29, 27, 30, 38, 26, 26],
                     [41, 26, 28, 36, 18, 38, 16, 26]])
    b_eq = np.array([7872, 10466, 11322, 12058])
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])

    mip_rel_gaps = [0.25, 0.01, 0.001]
    sol_mip_gaps = []
    for mip_rel_gap in mip_rel_gaps:
        res = milp(c=c, bounds=(0, np.inf), constraints=(A_eq, b_eq, b_eq),
                   integrality=True, options={"mip_rel_gap": mip_rel_gap})
        # assert that the solution actually has mip_gap lower than the
        # required mip_rel_gap supplied
        assert res.mip_gap <= mip_rel_gap
        # check that `res.mip_gap` is as defined in the documentation
        assert res.mip_gap == (res.fun - res.mip_dual_bound)/res.fun
        sol_mip_gaps.append(res.mip_gap)

    # make sure that the mip_rel_gap parameter is actually doing something
    # check that differences between solution gaps are declining
    # monotonically with the mip_rel_gap parameter.
    assert np.all(np.diff(sol_mip_gaps) < 0)

@pytest.mark.xfail(reason='Upstream / Wrapper issue, see gh-20116')
def test_large_numbers_gh20116():
    h = 10 ** 12
    A = np.array([[100.4534, h], [100.4534, -h]])
    b = np.array([h, 0])
    constraints = LinearConstraint(A=A, ub=b)
    bounds = Bounds([0, 0], [1, 1])
    c = np.array([0, 0])
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=1)
    assert res.status == 0
    assert np.all(A @ res.x < b)


def test_presolve_gh18907():
    from scipy.optimize import milp
    import numpy as np
    inf = np.inf

    # set up problem
    c = np.array([-0.85850509, -0.82892676, -0.80026454, -0.63015535, -0.5099006,
                  -0.50077193, -0.4894404, -0.47285865,  -0.39867774, -0.38069646,
                  -0.36733012, -0.36733012, -0.35820411, -0.31576141, -0.20626091,
                  -0.12466144, -0.10679516, -0.1061887, -0.1061887, -0.1061887,
                  -0., -0., -0., -0., 0., 0., 0., 0.])

    A = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                   1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   1., 0., 0., 0., 0., 0., 1., 0., 0., 0., -25., -0., -0., -0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   -1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 2., 0., 0., 0.],
                  [0., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., -0., -25., -0., -0.],
                  [0., 0., 0., 0., -1., -1., -1., -1., 0., -1., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 1., 1., 1., 0., 0., 0., 0., -0., -0., -25., -0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., -1., -1., -1., 0., 0., 0., 0., 0., 0., 2., 0.],
                  [1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0.,
                   1., 1., 0., 0., 0., 0., 1., 1., 1., -0., -0., -0., -25.],
                  [-1., -1., -1., -1., 0., 0., 0., 0., -1., 0., -1., -1., -1., -1.,
                   0., -1., -1., 0., 0., 0., 0., -1., -1., -1., 0., 0., 0., 2.]])
    bl = np.array([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
    bu = np.array([100., 0., 0., 0., 0., 0., 0., 0., 0.])
    constraints = LinearConstraint(A, bl, bu)
    integrality = 1
    bounds = (0, 1)
    r1 = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
              options={'presolve': True})
    r2 = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
              options={'presolve': False})
    assert r1.status == r2.status
    assert_allclose(r1.x, r2.x)

    # another example from the same issue
    bounds = Bounds(lb=0, ub=1)
    integrality = [1, 1, 0, 0]
    c = [10, 9.52380952, -1000, -952.38095238]
    A = [[1, 1, 0, 0], [0, 0, 1, 1], [200, 0, 0, 0], [0, 200, 0, 0],
         [0, 0, 2000, 0], [0, 0, 0, 2000], [-1, 0, 1, 0], [-1, -1, 0, 1]]
    ub = [1, 1, 200, 200, 1000, 1000, 0, 0]
    constraints = LinearConstraint(A, ub=ub)
    r1 = milp(c=c, constraints=constraints,  bounds=bounds,
              integrality=integrality, options={"presolve": False})
    r2 = milp(c=c, constraints=constraints,  bounds=bounds,
              integrality=integrality, options={"presolve": False})
    assert r1.status == r2.status
    assert_allclose(r1.x, r2.x)


# <!-- @GENESIS_MODULE_END: test_milp -->
