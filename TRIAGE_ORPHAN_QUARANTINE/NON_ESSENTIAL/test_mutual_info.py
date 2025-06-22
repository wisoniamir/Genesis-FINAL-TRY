import logging
# <!-- @GENESIS_MODULE_START: test_mutual_info -->
"""
🏛️ GENESIS TEST_MUTUAL_INFO - INSTITUTIONAL GRADE v8.0.0
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

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (

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

                emit_telemetry("test_mutual_info", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_mutual_info", "position_calculated", {
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
                            "module": "test_mutual_info",
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
                    print(f"Emergency stop error in test_mutual_info: {e}")
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
                    "module": "test_mutual_info",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_mutual_info", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_mutual_info: {e}")
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


    assert_allclose,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS


def test_compute_mi_dd():
    # In discrete case computations are straightforward and can be done
    # by hand on given vectors.
    x = np.array([0, 1, 1, 0, 0])
    y = np.array([1, 0, 0, 0, 1])

    H_x = H_y = -(3 / 5) * np.log(3 / 5) - (2 / 5) * np.log(2 / 5)
    H_xy = -1 / 5 * np.log(1 / 5) - 2 / 5 * np.log(2 / 5) - 2 / 5 * np.log(2 / 5)
    I_xy = H_x + H_y - H_xy

    assert_allclose(_compute_mi(x, y, x_discrete=True, y_discrete=True), I_xy)


def test_compute_mi_cc(global_dtype):
    # For two continuous variables a good approach is to test on bivariate
    # normal distribution, where mutual information is known.

    # Mean of the distribution, irrelevant for mutual information.
    mean = np.zeros(2)

    # Setup covariance matrix with correlation coeff. equal 0.5.
    sigma_1 = 1
    sigma_2 = 10
    corr = 0.5
    cov = np.array(
        [
            [sigma_1**2, corr * sigma_1 * sigma_2],
            [corr * sigma_1 * sigma_2, sigma_2**2],
        ]
    )

    # True theoretical mutual information.
    I_theory = np.log(sigma_1) + np.log(sigma_2) - 0.5 * np.log(np.linalg.det(cov))

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)

    x, y = Z[:, 0], Z[:, 1]

    # Theory and computed values won't be very close
    # We here check with a large relative tolerance
    for n_neighbors in [3, 5, 7]:
        I_computed = _compute_mi(
            x, y, x_discrete=False, y_discrete=False, n_neighbors=n_neighbors
        )
        assert_allclose(I_computed, I_theory, rtol=1e-1)


def test_compute_mi_cd(global_dtype):
    # To test define a joint distribution as follows:
    # p(x, y) = p(x) p(y | x)
    # X ~ Bernoulli(p)
    # (Y | x = 0) ~ Uniform(-1, 1)
    # (Y | x = 1) ~ Uniform(0, 2)

    # Use the following formula for mutual information:
    # I(X; Y) = H(Y) - H(Y | X)
    # Two entropies can be computed by hand:
    # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
    # H(Y | X) = ln(2)

    # Now we need to implement sampling from out distribution, which is
    # done easily using conditional distribution logic.

    n_samples = 1000
    rng = check_random_state(0)

    for p in [0.3, 0.5, 0.7]:
        x = rng.uniform(size=n_samples) > p

        y = np.empty(n_samples, global_dtype)
        mask = x == 0
        y[mask] = rng.uniform(-1, 1, size=np.sum(mask))
        y[~mask] = rng.uniform(0, 2, size=np.sum(~mask))

        I_theory = -0.5 * (
            (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
        ) - np.log(2)

        # Assert the same tolerance.
        for n_neighbors in [3, 5, 7]:
            I_computed = _compute_mi(
                x, y, x_discrete=True, y_discrete=False, n_neighbors=n_neighbors
            )
            assert_allclose(I_computed, I_theory, rtol=1e-1)


def test_compute_mi_cd_unique_label(global_dtype):
    # Test that adding unique label doesn't change MI.
    n_samples = 100
    x = np.random.uniform(size=n_samples) > 0.5

    y = np.empty(n_samples, global_dtype)
    mask = x == 0
    y[mask] = np.random.uniform(-1, 1, size=np.sum(mask))
    y[~mask] = np.random.uniform(0, 2, size=np.sum(~mask))

    mi_1 = _compute_mi(x, y, x_discrete=True, y_discrete=False)

    x = np.hstack((x, 2))
    y = np.hstack((y, 10))
    mi_2 = _compute_mi(x, y, x_discrete=True, y_discrete=False)

    assert_allclose(mi_1, mi_2)


# We are going test that feature ordering by MI matches our expectations.
def test_mutual_info_classif_discrete(global_dtype):
    X = np.array(
        [[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype
    )
    y = np.array([0, 1, 2, 2, 1])

    # Here X[:, 0] is the most informative feature, and X[:, 1] is weakly
    # informative.
    mi = mutual_info_classif(X, y, discrete_features=True)
    assert_array_equal(np.argsort(-mi), np.array([0, 2, 1]))


def test_mutual_info_regression(global_dtype):
    # We generate sample from multivariate normal distribution, using
    # transformation from initially uncorrelated variables. The zero
    # variables after transformation is selected as the target vector,
    # it has the strongest correlation with the variable 2, and
    # the weakest correlation with the variable 1.
    T = np.array([[1, 0.5, 2, 1], [0, 1, 0.1, 0.0], [0, 0.1, 1, 0.1], [0, 0.1, 0.1, 1]])
    cov = T.dot(T.T)
    mean = np.zeros(4)

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)
    X = Z[:, 1:]
    y = Z[:, 0]

    mi = mutual_info_regression(X, y, random_state=0)
    assert_array_equal(np.argsort(-mi), np.array([1, 2, 0]))
    # XXX: should mutual_info_regression be fixed to avoid
    # up-casting float32 inputs to float64?
    assert mi.dtype == np.float64


def test_mutual_info_classif_mixed(global_dtype):
    # Here the target is discrete and there are two continuous and one
    # discrete feature. The idea of this test is clear from the code.
    rng = check_random_state(0)
    X = rng.rand(1000, 3).astype(global_dtype, copy=False)
    X[:, 1] += X[:, 0]
    y = ((0.5 * X[:, 0] + X[:, 2]) > 0.5).astype(int)
    X[:, 2] = X[:, 2] > 0.5

    mi = mutual_info_classif(X, y, discrete_features=[2], n_neighbors=3, random_state=0)
    assert_array_equal(np.argsort(-mi), [2, 0, 1])
    for n_neighbors in [5, 7, 9]:
        mi_nn = mutual_info_classif(
            X, y, discrete_features=[2], n_neighbors=n_neighbors, random_state=0
        )
        # Check that the continuous values have an higher MI with greater
        # n_neighbors
        assert mi_nn[0] > mi[0]
        assert mi_nn[1] > mi[1]
        # The n_neighbors should not have any effect on the discrete value
        # The MI should be the same
        assert mi_nn[2] == mi[2]


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_mutual_info_options(global_dtype, csr_container):
    X = np.array(
        [[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype
    )
    y = np.array([0, 1, 2, 2, 1], dtype=global_dtype)
    X_csr = csr_container(X)

    for mutual_info in (mutual_info_regression, mutual_info_classif):
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=False)
        with pytest.raises(ValueError):
            mutual_info(X, y, discrete_features="manual")
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=[True, False, True])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[True, False, True, False])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[1, 4])

        mi_1 = mutual_info(X, y, discrete_features="auto", random_state=0)
        mi_2 = mutual_info(X, y, discrete_features=False, random_state=0)
        mi_3 = mutual_info(X_csr, y, discrete_features="auto", random_state=0)
        mi_4 = mutual_info(X_csr, y, discrete_features=True, random_state=0)
        mi_5 = mutual_info(X, y, discrete_features=[True, False, True], random_state=0)
        mi_6 = mutual_info(X, y, discrete_features=[0, 2], random_state=0)

        assert_allclose(mi_1, mi_2)
        assert_allclose(mi_3, mi_4)
        assert_allclose(mi_5, mi_6)

        assert not np.allclose(mi_1, mi_3)


@pytest.mark.parametrize("correlated", [True, False])
def test_mutual_information_symmetry_classif_regression(correlated, global_random_seed):
    """Check that `mutual_info_classif` and `mutual_info_regression` are
    symmetric by switching the target `y` as `feature` in `X` and vice
    versa.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23720
    """
    rng = np.random.RandomState(global_random_seed)
    n = 100
    d = rng.randint(10, size=n)

    if correlated:
        c = d.astype(np.float64)
    else:
        c = rng.normal(0, 1, size=n)

    mi_classif = mutual_info_classif(
        c[:, None], d, discrete_features=[False], random_state=global_random_seed
    )

    mi_regression = mutual_info_regression(
        d[:, None], c, discrete_features=[True], random_state=global_random_seed
    )

    assert mi_classif == pytest.approx(mi_regression)


def test_mutual_info_regression_X_int_dtype(global_random_seed):
    """Check that results agree when X is integer dtype and float dtype.

    Non-regression test for Issue #26696.
    """
    rng = np.random.RandomState(global_random_seed)
    X = rng.randint(100, size=(100, 10))
    X_float = X.astype(np.float64, copy=True)
    y = rng.randint(100, size=100)

    expected = mutual_info_regression(X_float, y, random_state=global_random_seed)
    result = mutual_info_regression(X, y, random_state=global_random_seed)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "mutual_info_func, data_generator",
    [
        (mutual_info_regression, make_regression),
        (mutual_info_classif, make_classification),
    ],
)
def test_mutual_info_n_jobs(global_random_seed, mutual_info_func, data_generator):
    """Check that results are consistent with different `n_jobs`."""
    X, y = data_generator(random_state=global_random_seed)
    single_job = mutual_info_func(X, y, random_state=global_random_seed, n_jobs=1)
    multi_job = mutual_info_func(X, y, random_state=global_random_seed, n_jobs=2)
    assert_allclose(single_job, multi_job)


# <!-- @GENESIS_MODULE_END: test_mutual_info -->
