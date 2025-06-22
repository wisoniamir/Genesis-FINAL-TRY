import logging
# <!-- @GENESIS_MODULE_START: test_trustregion_exact -->
"""
🏛️ GENESIS TEST_TRUSTREGION_EXACT - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_trustregion_exact", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_trustregion_exact", "position_calculated", {
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
                            "module": "test_trustregion_exact",
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
                    print(f"Emergency stop error in test_trustregion_exact: {e}")
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
                    "module": "test_trustregion_exact",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_trustregion_exact", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_trustregion_exact: {e}")
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
Unit tests for trust-region iterative subproblem.

"""
import pytest
import numpy as np
from scipy.optimize._trustregion_exact import (
    estimate_smallest_singular_value,
    singular_leading_submatrix,
    IterativeSubproblem)
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
                           assert_equal, assert_array_almost_equal)


def random_entry(n, min_eig, max_eig, case):

    # Generate random matrix
    rand = np.random.uniform(-1, 1, (n, n))

    # QR decomposition
    Q, _, _ = qr(rand, pivoting='True')

    # Generate random eigenvalues
    eigvalues = np.random.uniform(min_eig, max_eig, n)
    eigvalues = np.sort(eigvalues)[::-1]

    # Generate matrix
    Qaux = np.multiply(eigvalues, Q)
    A = np.dot(Qaux, Q.T)

    # Generate gradient vector accordingly
    # to the case is being tested.
    if case == 'hard':
        g = np.zeros(n)
        g[:-1] = np.random.uniform(-1, 1, n-1)
        g = np.dot(Q, g)
    elif case == 'jac_equal_zero':
        g = np.zeros(n)
    else:
        g = np.random.uniform(-1, 1, n)

    return A, g


class TestEstimateSmallestSingularValue:
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

            emit_telemetry("test_trustregion_exact", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_trustregion_exact", "position_calculated", {
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
                        "module": "test_trustregion_exact",
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
                print(f"Emergency stop error in test_trustregion_exact: {e}")
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
                "module": "test_trustregion_exact",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_trustregion_exact", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_trustregion_exact: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_trustregion_exact",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_trustregion_exact: {e}")

    def test_for_ill_condiotioned_matrix(self):

        # Ill-conditioned triangular matrix
        C = np.array([[1, 2, 3, 4],
                      [0, 0.05, 60, 7],
                      [0, 0, 0.8, 9],
                      [0, 0, 0, 10]])

        # Get svd decomposition
        U, s, Vt = svd(C)

        # Get smallest singular value and correspondent right singular vector.
        smin_svd = s[-1]
        zmin_svd = Vt[-1, :]

        # Estimate smallest singular value
        smin, zmin = estimate_smallest_singular_value(C)

        # Check the estimation
        assert_array_almost_equal(smin, smin_svd, decimal=8)
        assert_array_almost_equal(abs(zmin), abs(zmin_svd), decimal=8)


class TestSingularLeadingSubmatrix:
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

            emit_telemetry("test_trustregion_exact", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_trustregion_exact", "position_calculated", {
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
                        "module": "test_trustregion_exact",
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
                print(f"Emergency stop error in test_trustregion_exact: {e}")
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
                "module": "test_trustregion_exact",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_trustregion_exact", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_trustregion_exact: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_trustregion_exact",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_trustregion_exact: {e}")

    def test_for_already_singular_leading_submatrix(self):

        # Define test matrix A.
        # Note that the leading 2x2 submatrix is singular.
        A = np.array([[1, 2, 3],
                      [2, 4, 5],
                      [3, 5, 6]])

        # Get Cholesky from lapack functions
        cholesky, = get_lapack_funcs(('potrf',), (A,))

        # Compute Cholesky Decomposition
        c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)

        delta, v = singular_leading_submatrix(A, c, k)

        A[k-1, k-1] += delta

        # Check if the leading submatrix is singular.
        assert_array_almost_equal(det(A[:k, :k]), 0)

        # Check if `v` fulfil the specified properties
        quadratic_term = np.dot(v, np.dot(A, v))
        assert_array_almost_equal(quadratic_term, 0)

    def test_for_simetric_indefinite_matrix(self):

        # Define test matrix A.
        # Note that the leading 5x5 submatrix is indefinite.
        A = np.asarray([[1, 2, 3, 7, 8],
                        [2, 5, 5, 9, 0],
                        [3, 5, 11, 1, 2],
                        [7, 9, 1, 7, 5],
                        [8, 0, 2, 5, 8]])

        # Get Cholesky from lapack functions
        cholesky, = get_lapack_funcs(('potrf',), (A,))

        # Compute Cholesky Decomposition
        c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)

        delta, v = singular_leading_submatrix(A, c, k)

        A[k-1, k-1] += delta

        # Check if the leading submatrix is singular.
        assert_array_almost_equal(det(A[:k, :k]), 0)

        # Check if `v` fulfil the specified properties
        quadratic_term = np.dot(v, np.dot(A, v))
        assert_array_almost_equal(quadratic_term, 0)

    def test_for_first_element_equal_to_zero(self):

        # Define test matrix A.
        # Note that the leading 2x2 submatrix is singular.
        A = np.array([[0, 3, 11],
                      [3, 12, 5],
                      [11, 5, 6]])

        # Get Cholesky from lapack functions
        cholesky, = get_lapack_funcs(('potrf',), (A,))

        # Compute Cholesky Decomposition
        c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)

        delta, v = singular_leading_submatrix(A, c, k)

        A[k-1, k-1] += delta

        # Check if the leading submatrix is singular
        assert_array_almost_equal(det(A[:k, :k]), 0)

        # Check if `v` fulfil the specified properties
        quadratic_term = np.dot(v, np.dot(A, v))
        assert_array_almost_equal(quadratic_term, 0)


class TestIterativeSubproblem:
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

            emit_telemetry("test_trustregion_exact", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_trustregion_exact", "position_calculated", {
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
                        "module": "test_trustregion_exact",
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
                print(f"Emergency stop error in test_trustregion_exact: {e}")
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
                "module": "test_trustregion_exact",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_trustregion_exact", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_trustregion_exact: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_trustregion_exact",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_trustregion_exact: {e}")

    def test_for_the_easy_case(self):

        # `H` is chosen such that `g` is not orthogonal to the
        # eigenvector associated with the smallest eigenvalue `s`.
        H = [[10, 2, 3, 4],
             [2, 1, 7, 1],
             [3, 7, 1, 7],
             [4, 1, 7, 2]]
        g = [1, 1, 1, 1]

        # Trust Radius
        trust_radius = 1

        # Solve Subproblem
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H),
                                      k_easy=1e-10,
                                      k_hard=1e-10)
        p, hits_boundary = subprob.solve(trust_radius)

        assert_array_almost_equal(p, [0.00393332, -0.55260862,
                                      0.67065477, -0.49480341])
        assert_array_almost_equal(hits_boundary, True)

    def test_for_the_hard_case(self):

        # `H` is chosen such that `g` is orthogonal to the
        # eigenvector associated with the smallest eigenvalue `s`.
        H = [[10, 2, 3, 4],
             [2, 1, 7, 1],
             [3, 7, 1, 7],
             [4, 1, 7, 2]]
        g = [6.4852641521327437, 1, 1, 1]
        s = -8.2151519874416614

        # Trust Radius
        trust_radius = 1

        # Solve Subproblem
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H),
                                      k_easy=1e-10,
                                      k_hard=1e-10)
        p, hits_boundary = subprob.solve(trust_radius)

        assert_array_almost_equal(-s, subprob.lambda_current)

    def test_for_interior_convergence(self):

        H = [[1.812159, 0.82687265, 0.21838879, -0.52487006, 0.25436988],
             [0.82687265, 2.66380283, 0.31508988, -0.40144163, 0.08811588],
             [0.21838879, 0.31508988, 2.38020726, -0.3166346, 0.27363867],
             [-0.52487006, -0.40144163, -0.3166346, 1.61927182, -0.42140166],
             [0.25436988, 0.08811588, 0.27363867, -0.42140166, 1.33243101]]

        g = [0.75798952, 0.01421945, 0.33847612, 0.83725004, -0.47909534]

        # Solve Subproblem
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H))
        p, hits_boundary = subprob.solve(1.1)

        assert_array_almost_equal(p, [-0.68585435, 0.1222621, -0.22090999,
                                      -0.67005053, 0.31586769])
        assert_array_almost_equal(hits_boundary, False)
        assert_array_almost_equal(subprob.lambda_current, 0)
        assert_array_almost_equal(subprob.niter, 1)

    def test_for_jac_equal_zero(self):

        H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
             [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
             [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
             [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
             [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]

        g = [0, 0, 0, 0, 0]

        # Solve Subproblem
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H),
                                      k_easy=1e-10,
                                      k_hard=1e-10)
        p, hits_boundary = subprob.solve(1.1)

        assert_array_almost_equal(p, [0.06910534, -0.01432721,
                                      -0.65311947, -0.23815972,
                                      -0.84954934])
        assert_array_almost_equal(hits_boundary, True)

    def test_for_jac_very_close_to_zero(self):

        H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
             [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
             [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
             [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
             [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]

        g = [0, 0, 0, 0, 1e-15]

        # Solve Subproblem
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H),
                                      k_easy=1e-10,
                                      k_hard=1e-10)
        p, hits_boundary = subprob.solve(1.1)

        assert_array_almost_equal(p, [0.06910534, -0.01432721,
                                      -0.65311947, -0.23815972,
                                      -0.84954934])
        assert_array_almost_equal(hits_boundary, True)

    @pytest.mark.fail_slow(10)
    def test_for_random_entries(self):
        # Seed
        np.random.seed(1)

        # Dimension
        n = 5

        for case in ('easy', 'hard', 'jac_equal_zero'):

            eig_limits = [(-20, -15),
                          (-10, -5),
                          (-10, 0),
                          (-5, 5),
                          (-10, 10),
                          (0, 10),
                          (5, 10),
                          (15, 20)]

            for min_eig, max_eig in eig_limits:
                # Generate random symmetric matrix H with
                # eigenvalues between min_eig and max_eig.
                H, g = random_entry(n, min_eig, max_eig, case)

                # Trust radius
                trust_radius_list = [0.1, 0.3, 0.6, 0.8, 1, 1.2, 3.3, 5.5, 10]

                for trust_radius in trust_radius_list:
                    # Solve subproblem with very high accuracy
                    subprob_ac = IterativeSubproblem(0,
                                                     lambda x: 0,
                                                     lambda x: g,
                                                     lambda x: H,
                                                     k_easy=1e-10,
                                                     k_hard=1e-10)

                    p_ac, hits_boundary_ac = subprob_ac.solve(trust_radius)

                    # Compute objective function value
                    J_ac = 1/2*np.dot(p_ac, np.dot(H, p_ac))+np.dot(g, p_ac)

                    stop_criteria = [(0.1, 2),
                                     (0.5, 1.1),
                                     (0.9, 1.01)]

                    for k_opt, k_trf in stop_criteria:

                        # k_easy and k_hard computed in function
                        # of k_opt and k_trf accordingly to
                        # Conn, A. R., Gould, N. I., & Toint, P. L. (2000).
                        # "Trust region methods". Siam. p. 197.
                        k_easy = min(k_trf-1,
                                     1-np.sqrt(k_opt))
                        k_hard = 1-k_opt

                        # Solve subproblem
                        subprob = IterativeSubproblem(0,
                                                      lambda x: 0,
                                                      lambda x: g,
                                                      lambda x: H,
                                                      k_easy=k_easy,
                                                      k_hard=k_hard)
                        p, hits_boundary = subprob.solve(trust_radius)

                        # Compute objective function value
                        J = 1/2*np.dot(p, np.dot(H, p))+np.dot(g, p)

                        # Check if it respect k_trf
                        if hits_boundary:
                            assert_array_equal(np.abs(norm(p)-trust_radius) <=
                                               (k_trf-1)*trust_radius, True)
                        else:
                            assert_equal(norm(p) <= trust_radius, True)

                        # Check if it respect k_opt
                        assert_equal(J <= k_opt*J_ac, True)



# <!-- @GENESIS_MODULE_END: test_trustregion_exact -->
