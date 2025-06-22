import logging
import sys
from pathlib import Path


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

                emit_telemetry("_remove_redundancy", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_remove_redundancy", "position_calculated", {
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
                            "module": "_remove_redundancy",
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
                    print(f"Emergency stop error in _remove_redundancy: {e}")
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
                    "module": "_remove_redundancy",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_remove_redundancy", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _remove_redundancy: {e}")
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


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

Routines for removing redundant (linearly dependent) equations from linear
programming equality constraints.
"""
# Author: Matt Haberland

import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: _remove_redundancy -->


# <!-- @GENESIS_MODULE_START: _remove_redundancy -->


def _row_count(A):
    """
    Counts the number of nonzeros in each row of input array A.
    Nonzeros are defined as any element with absolute value greater than
    tol = 1e-13. This value should probably be an input to the function.

    Parameters
    ----------
    A : 2-D array
        An array representing a matrix

    Returns
    -------
    rowcount : 1-D array
        Number of nonzeros in each row of A

    """
    tol = 1e-13
    return np.array((abs(A) > tol).sum(axis=1)).flatten()


def _get_densest(A, eligibleRows):
    """
    Returns the index of the densest row of A. Ignores rows that are not
    eligible for consideration.

    Parameters
    ----------
    A : 2-D array
        An array representing a matrix
    eligibleRows : 1-D logical array
        Values indicate whether the corresponding row of A is eligible
        to be considered

    Returns
    -------
    i_densest : int
        Index of the densest row in A eligible for consideration

    """
    rowCounts = _row_count(A)
    return np.argmax(rowCounts * eligibleRows)


def _remove_zero_rows(A, b):
    """
    Eliminates trivial equations from system of equations defined by Ax = b
   and identifies trivial infeasibilities

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the removal operation
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    """
    status = 0
    message = ""
    i_zero = _row_count(A) == 0
    A = A[np.logical_not(i_zero), :]
    if not np.allclose(b[i_zero], 0):
        status = 2
        message = "There is a zero row in A_eq with a nonzero corresponding " \
                  "entry in b_eq. The problem is infeasible."
    b = b[np.logical_not(i_zero)]
    return A, b, status, message


def bg_update_dense(plu, perm_r, v, j):
    LU, p = plu

    vperm = v[perm_r]
    u = dtrsm(1, LU, vperm, lower=1, diag=1)
    LU[:j+1, j] = u[:j+1]
    l = u[j+1:]
    piv = LU[j, j]
    LU[j+1:, j] += (l/piv)
    return LU, p


def _remove_redundancy_pivot_dense(A, rhs, true_rank=None):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D sparse matrix
        An matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D sparse matrix
        A matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.

    """
    tolapiv = 1e-8
    tolprimal = 1e-8
    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    A, rhs, status, message = _remove_zero_rows(A, rhs)

    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    v = list(range(m))      # Artificial column indices.
    b = list(v)             # Basis column indices.
    # This is better as a list than a set because column order of basis matrix
    # needs to be consistent.
    d = []                  # Indices of dependent rows
    perm_r = None

    A_orig = A
    A = np.zeros((m, m + n), order='F')
    np.fill_diagonal(A, 1)
    A[:, m:] = A_orig
    e = np.zeros(m)

    js_candidates = np.arange(m, m+n, dtype=int)  # candidate columns for basis
    # manual masking was faster than masked array
    js_mask = np.ones(js_candidates.shape, dtype=bool)

    # Implements basic algorithm from [2]
    # Uses some of the suggested improvements (removing zero rows and
    # Bartels-Golub update idea).
    # Removing column singletons would be easy, but it is not as important
    # because the procedure is performed only on the equality constraint
    # matrix from the original problem - not on the canonical form matrix,
    # which would have many more column singletons due to slack variables
    # from the inequality constraints.
    # The thoughts on "crashing" the initial basis are only really useful if
    # the matrix is sparse.

    lu = np.eye(m, order='F'), np.arange(m)  # initial LU is trivial
    perm_r = lu[1]
    for i in v:

        e[i] = 1
        if i > 0:
            e[i-1] = 0

        try:  # fails for i==0 and any time it gets ill-conditioned
            j = b[i-1]
            lu = bg_update_dense(lu, perm_r, A[:, j], i-1)
        except Exception:
            lu = scipy.linalg.lu_factor(A[:, b])
            LU, p = lu
            perm_r = list(range(m))
            for i1, i2 in enumerate(p):
                perm_r[i1], perm_r[i2] = perm_r[i2], perm_r[i1]

        pi = scipy.linalg.lu_solve(lu, e, trans=1)

        js = js_candidates[js_mask]
        batch = 50

        # This is a tiny bit faster than looping over columns individually,
        # like for j in js: if abs(A[:,j].transpose().dot(pi)) > tolapiv:
        for j_index in range(0, len(js), batch):
            j_indices = js[j_index: min(j_index+batch, len(js))]

            c = abs(A[:, j_indices].transpose().dot(pi))
            if (c > tolapiv).any():
                j = js[j_index + np.argmax(c)]  # very independent column
                b[i] = j
                js_mask[j-m] = False
                break
        else:
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar)/(1+bnorm) > tolprimal:  # inconsistent
                status = 2
                message = inconsistent
                return A_orig, rhs, status, message
            else:  # dependent
                d.append(i)
                if true_rank is not None and len(d) == m - true_rank:
                    break   # found all redundancies

    keep = set(range(m))
    keep = list(keep - set(d))
    return A_orig[keep, :], rhs[keep], status, message


def _remove_redundancy_pivot_sparse(A, rhs):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D sparse matrix
        An matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D sparse matrix
        A matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.

    """

    tolapiv = 1e-8
    tolprimal = 1e-8
    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    A, rhs, status, message = _remove_zero_rows(A, rhs)

    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    v = list(range(m))      # Artificial column indices.
    b = list(v)             # Basis column indices.
    # This is better as a list than a set because column order of basis matrix
    # needs to be consistent.
    k = set(range(m, m+n))  # Structural column indices.
    d = []                  # Indices of dependent rows

    A_orig = A
    A = scipy.sparse.hstack((scipy.sparse.eye(m), A)).tocsc()
    e = np.zeros(m)

    # Implements basic algorithm from [2]
    # Uses only one of the suggested improvements (removing zero rows).
    # Removing column singletons would be easy, but it is not as important
    # because the procedure is performed only on the equality constraint
    # matrix from the original problem - not on the canonical form matrix,
    # which would have many more column singletons due to slack variables
    # from the inequality constraints.
    # The thoughts on "crashing" the initial basis sound useful, but the
    # description of the procedure seems to assume a lot of familiarity with
    # the subject; it is not very explicit. I already went through enough
    # trouble getting the basic algorithm working, so I was not interested in
    # trying to decipher this, too. (Overall, the paper is fraught with
    # mistakes and ambiguities - which is strange, because the rest of
    # Andersen's papers are quite good.)
    # I tried and tried and tried to improve performance using the
    # Bartels-Golub update. It works, but it's only practical if the LU
    # factorization can be specialized as described, and that is not possible
    # until the SciPy SuperLU interface permits control over column
    # permutation - see issue #7700.

    for i in v:
        B = A[:, b]

        e[i] = 1
        if i > 0:
            e[i-1] = 0

        pi = scipy.sparse.linalg.spsolve(B.transpose(), e).reshape(-1, 1)

        js = list(k-set(b))  # not efficient, but this is not the time sink...

        # Due to overhead, it tends to be faster (for problems tested) to
        # compute the full matrix-vector product rather than individual
        # vector-vector products (with the chance of terminating as soon
        # as any are nonzero). For very large matrices, it might be worth
        # it to compute, say, 100 or 1000 at a time and stop when a nonzero
        # is found.

        c = (np.abs(A[:, js].transpose().dot(pi)) > tolapiv).nonzero()[0]
        if len(c) > 0:  # independent
            j = js[c[0]]
            # in a previous commit, the previous line was changed to choose
            # index j corresponding with the maximum dot product.
            # While this avoided issues with almost
            # singular matrices, it slowed the routine in most NETLIB tests.
            # I think this is because these columns were denser than the
            # first column with nonzero dot product (c[0]).
            # It would be nice to have a heuristic that balances sparsity with
            # high dot product, but I don't think it's worth the time to
            # develop one right now. Bartels-Golub update is a much higher
            # priority.
            b[i] = j  # replace artificial column
        else:
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar)/(1 + bnorm) > tolprimal:
                status = 2
                message = inconsistent
                return A_orig, rhs, status, message
            else:  # dependent
                d.append(i)

    keep = set(range(m))
    keep = list(keep - set(d))
    return A_orig[keep, :], rhs[keep], status, message


def _remove_redundancy_svd(A, b):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.

    """

    A, b, status, message = _remove_zero_rows(A, b)

    if status != 0:
        return A, b, status, message

    U, s, Vh = svd(A)
    eps = np.finfo(float).eps
    tol = s.max() * max(A.shape) * eps

    m, n = A.shape
    s_min = s[-1] if m <= n else 0

    # this algorithm is faster than that of [2] when the nullspace is small
    # but it could probably be improvement by randomized algorithms and with
    # a sparse implementation.
    # it relies on repeated singular value decomposition to find linearly
    # dependent rows (as identified by columns of U that correspond with zero
    # singular values). Unfortunately, only one row can be removed per
    # decomposition (I tried otherwise; doing so can cause problems.)
    # It would be nice if we could do truncated SVD like sp.sparse.linalg.svds
    # but that function is unreliable at finding singular values near zero.
    # Finding max eigenvalue L of A A^T, then largest eigenvalue (and
    # associated eigenvector) of -A A^T + L I (I is identity) via power
    # iteration would also work in theory, but is only efficient if the
    # smallest nonzero eigenvalue of A A^T is close to the largest nonzero
    # eigenvalue.

    while abs(s_min) < tol:
        v = U[:, -1]  # IMPLEMENTED: return these so user can eliminate from problem?
        # rows need to be represented in significant amount
        eligibleRows = np.abs(v) > tol * 10e6
        if not np.any(eligibleRows) or np.any(np.abs(v.dot(A)) > tol):
            status = 4
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            break
        if np.any(np.abs(v.dot(b)) > tol * 100):  # factor of 100 to fix 10038 and 10349
            status = 2
            message = ("There is a linear combination of rows of A_eq that "
                       "results in zero, suggesting a redundant constraint. "
                       "However the same linear combination of b_eq is "
                       "nonzero, suggesting that the constraints conflict "
                       "and the problem is infeasible.")
            break

        i_remove = _get_densest(A, eligibleRows)
        A = np.delete(A, i_remove, axis=0)
        b = np.delete(b, i_remove)
        U, s, Vh = svd(A)
        m, n = A.shape
        s_min = s[-1] if m <= n else 0

    return A, b, status, message


def _remove_redundancy_id(A, rhs, rank=None, randomized=True):
    """Eliminates redundant equations from a system of equations.

    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    rank : int, optional
        The rank of A
    randomized: bool, optional
        True for randomized interpolative decomposition

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    """

    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")

    A, rhs, status, message = _remove_zero_rows(A, rhs)

    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    k = rank
    if rank is None:
        k = np.linalg.matrix_rank(A)

    idx, proj = interp_decomp(A.T, k, rand=randomized)

    # first k entries in idx are indices of the independent rows
    # remaining entries are the indices of the m-k dependent rows
    # proj provides a linear combinations of rows of A2 that form the
    # remaining m-k (dependent) rows. The same linear combination of entries
    # in rhs2 must give the remaining m-k entries. If not, the system is
    # inconsistent, and the problem is infeasible.
    if not np.allclose(rhs[idx[:k]] @ proj, rhs[idx[k:]]):
        status = 2
        message = inconsistent

    # sort indices because the other redundancy removal routines leave rows
    # in original order and tests were written with that in mind
    idx = sorted(idx[:k])
    A2 = A[idx, :]
    rhs2 = rhs[idx]
    return A2, rhs2, status, message



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))


def detect_divergence(price_data: list, indicator_data: list, window: int = 10) -> Dict:
    """
    Detect regular and hidden divergences between price and indicator
    
    Args:
        price_data: List of price values (closing prices)
        indicator_data: List of indicator values (e.g., RSI, MACD)
        window: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence information
    """
    result = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "strength": 0.0
    }
    
    # Need at least window + 1 periods of data
    if len(price_data) < window + 1 or len(indicator_data) < window + 1:
        return result
        
    # Get the current and historical points
    current_price = price_data[-1]
    previous_price = min(price_data[-window:-1]) if price_data[-1] > price_data[-2] else max(price_data[-window:-1])
    previous_price_idx = price_data[-window:-1].index(previous_price) + len(price_data) - window
    
    current_indicator = indicator_data[-1]
    previous_indicator = indicator_data[previous_price_idx]
    
    # Check for regular divergences
    # Bullish - Lower price lows but higher indicator lows
    if current_price < previous_price and current_indicator > previous_indicator:
        result["regular_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Higher price highs but lower indicator highs
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["regular_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Check for hidden divergences
    # Bullish - Higher price lows but lower indicator lows
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["hidden_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Lower price highs but higher indicator highs
    elif current_price < previous_price and current_indicator > previous_indicator:
        result["hidden_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Emit divergence event if detected
    if any([result["regular_bullish"], result["regular_bearish"], 
            result["hidden_bullish"], result["hidden_bearish"]]):
        emit_event("divergence_detected", {
            "type": next(k for k, v in result.items() if v is True and k != "strength"),
            "strength": result["strength"],
            "symbol": price_data.symbol if hasattr(price_data, "symbol") else "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
    return result
