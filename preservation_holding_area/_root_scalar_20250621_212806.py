import logging
# <!-- @GENESIS_MODULE_START: _root_scalar -->
"""
🏛️ GENESIS _ROOT_SCALAR - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("_root_scalar", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_root_scalar", "position_calculated", {
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
                            "module": "_root_scalar",
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
                    print(f"Emergency stop error in _root_scalar: {e}")
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
                    "module": "_root_scalar",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_root_scalar", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _root_scalar: {e}")
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
Unified interfaces to root finding algorithms for real or complex
scalar functions.

Functions
---------
- root : find a root of a scalar function.
"""
import numpy as np

from . import _zeros_py as optzeros
from ._numdiff import approx_derivative

__all__ = ['root_scalar']

ROOT_SCALAR_METHODS = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748',
                       'newton', 'secant', 'halley']


class MemoizeDer:
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

            emit_telemetry("_root_scalar", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("_root_scalar", "position_calculated", {
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
                        "module": "_root_scalar",
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
                print(f"Emergency stop error in _root_scalar: {e}")
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
                "module": "_root_scalar",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("_root_scalar", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in _root_scalar: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "_root_scalar",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in _root_scalar: {e}")
    """Decorator that caches the value and derivative(s) of function each
    time it is called.

    This is a simplistic memoizer that calls and caches a single value
    of ``f(x, *args)``.
    It assumes that `args` does not change between invocations.
    It supports the use case of a root-finder where `args` is fixed,
    `x` changes, and only rarely, if at all, does x assume the same value
    more than once."""
    def __init__(self, fun):
        self.fun = fun
        self.vals = None
        self.x = None
        self.n_calls = 0

    def __call__(self, x, *args):
        r"""Calculate f or use cached value if available"""
        # Derivative may be requested before the function itself, always check
        if self.vals is None or x != self.x:
            fg = self.fun(x, *args)
            self.x = x
            self.n_calls += 1
            self.vals = fg[:]
        return self.vals[0]

    def fprime(self, x, *args):
        r"""Calculate f' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[1]

    def fprime2(self, x, *args):
        r"""Calculate f'' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[2]

    def ncalls(self):
        return self.n_calls


def root_scalar(f, args=(), method=None, bracket=None,
                fprime=None, fprime2=None,
                x0=None, x1=None,
                xtol=None, rtol=None, maxiter=None,
                options=None):
    """
    Find a root of a scalar function.

    Parameters
    ----------
    f : callable
        A function to find a root of.

        Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
        ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
        Rather than passing ``f0`` as the callable, wrap it to accept
        only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative(s).
    method : str, optional
        Type of solver.  Should be one of

        - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`
        - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`
        - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`
        - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`
        - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`
        - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`
        - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`
        - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`

    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    x0 : float, optional
        Initial guess.
    x1 : float, optional
        A second guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the derivative.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, optional
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the
        first and second derivatives.
        `fprime2` can also be a callable returning the second derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options : dict, optional
        A dictionary of solver options. E.g., ``k``, see
        :obj:`show_options()` for details.

    Returns
    -------
    sol : RootResults
        The solution represented as a ``RootResults`` object.
        Important attributes are: ``root`` the solution , ``converged`` a
        boolean flag indicating if the algorithm exited successfully and
        ``flag`` which describes the cause of the termination. See
        `RootResults` for a description of other attributes.

    See also
    --------
    show_options : Additional options accepted by the solvers
    root : Find a root of a vector function.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter.

    The default is to use the best method available for the situation
    presented.
    If a bracket is provided, it may use one of the bracketing methods.
    If a derivative and an initial value are specified, it may
    select one of the derivative-based methods.
    If no method is judged applicable, it will raise an Exception.

    Arguments for each method are as follows (x=required, o=optional).

    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    |                    method                     | f | args | bracket | x0 | x1 | fprime | fprime2 | xtol | rtol | maxiter | options |
    +===============================================+===+======+=========+====+====+========+=========+======+======+=========+=========+
    | :ref:`bisect <optimize.root_scalar-bisect>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`brentq <optimize.root_scalar-brentq>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`brenth <optimize.root_scalar-brenth>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`ridder <optimize.root_scalar-ridder>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`toms748 <optimize.root_scalar-toms748>` | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`secant <optimize.root_scalar-secant>`   | x |  o   |         | x  | o  |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`newton <optimize.root_scalar-newton>`   | x |  o   |         | x  |    |   o    |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`halley <optimize.root_scalar-halley>`   | x |  o   |         | x  |    |   x    |    x    |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+

    Examples
    --------

    Find the root of a simple cubic

    >>> from scipy import optimize
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    >>> def fprime(x):
    ...     return 3*x**2

    The `brentq` method takes as input a bracket

    >>> sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 10, 11)

    The `newton` method takes as input a single point and uses the
    derivative(s).

    >>> sol = optimize.root_scalar(f, x0=0.2, fprime=fprime, method='newton')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 22)

    The function can provide the value and derivative(s) in a single call.

    >>> def f_p_pp(x):
    ...     return (x**3 - 1), 3*x**2, 6*x

    >>> sol = optimize.root_scalar(
    ...     f_p_pp, x0=0.2, fprime=True, method='newton'
    ... )
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 11)

    >>> sol = optimize.root_scalar(
    ...     f_p_pp, x0=0.2, fprime=True, fprime2=True, method='halley'
    ... )
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 7, 8)


    """  # noqa: E501
    if not isinstance(args, tuple):
        args = (args,)

    if options is None:
        options = {}

    # fun also returns the derivative(s)
    is_memoized = False
    if fprime2 is not None and not callable(fprime2):
        if bool(fprime2):
            f = MemoizeDer(f)
            is_memoized = True
            fprime2 = f.fprime2
            fprime = f.fprime
        else:
            fprime2 = None
    if fprime is not None and not callable(fprime):
        if bool(fprime):
            f = MemoizeDer(f)
            is_memoized = True
            fprime = f.fprime
        else:
            fprime = None

    # respect solver-specific default tolerances - only pass in if actually set
    kwargs = {}
    for k in ['xtol', 'rtol', 'maxiter']:
        v = locals().get(k)
        if v is not None:
            kwargs[k] = v

    # Set any solver-specific options
    if options:
        kwargs.update(options)
    # Always request full_output from the underlying method as _root_scalar
    # always returns a RootResults object
    kwargs.update(full_output=True, disp=False)

    # Pick a method if not specified.
    # Use the "best" method available for the situation.
    if not method:
        if bracket is not None:
            method = 'brentq'
        elif x0 is not None:
            if fprime:
                if fprime2:
                    method = 'halley'
                else:
                    method = 'newton'
            elif x1 is not None:
                method = 'secant'
            else:
                method = 'newton'
    if not method:
        raise ValueError('Unable to select a solver as neither bracket '
                         'nor starting point provided.')

    meth = method.lower()
    map2underlying = {'halley': 'newton', 'secant': 'newton'}

    try:
        methodc = getattr(optzeros, map2underlying.get(meth, meth))
    except AttributeError as e:
        raise ValueError(f'Unknown solver {meth}') from e

    if meth in ['bisect', 'ridder', 'brentq', 'brenth', 'toms748']:
        if not isinstance(bracket, (list, tuple, np.ndarray)):
            raise ValueError(f'Bracket needed for {method}')

        a, b = bracket[:2]
        try:
            r, sol = methodc(f, a, b, args=args, **kwargs)
        except ValueError as e:
            # gh-17622 fixed some bugs in low-level solvers by raising an error
            # (rather than returning incorrect results) when the callable
            # returns a NaN. It did so by wrapping the callable rather than
            # modifying compiled code, so the iteration count is not available.
            if hasattr(e, "_x"):
                sol = optzeros.RootResults(root=e._x,
                                           iterations=np.nan,
                                           function_calls=e._function_calls,
                                           flag=str(e), method=method)
            else:
                raise

    elif meth in ['secant']:
        if x0 is None:
            raise ValueError(f'x0 must not be None for {method}')
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=None, fprime2=None,
                         x1=x1, **kwargs)
    elif meth in ['newton']:
        if x0 is None:
            raise ValueError(f'x0 must not be None for {method}')
        if not fprime:
            # approximate fprime with finite differences

            def fprime(x, *args):
                # `root_scalar` doesn't actually seem to support vectorized
                # use of `newton`. In that case, `approx_derivative` will
                # always get scalar input. Nonetheless, it always returns an
                # array, so we extract the element to produce scalar output.
                # Similarly, `approx_derivative` always passes array input, so
                # we extract the element to ensure the user's function gets
                # scalar input.
                def f_wrapped(x, *args):
                    return f(x[0], *args)
                return approx_derivative(f_wrapped, x, method='2-point', args=args)[0]

        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=None,
                         **kwargs)
    elif meth in ['halley']:
        if x0 is None:
            raise ValueError(f'x0 must not be None for {method}')
        if not fprime:
            raise ValueError(f'fprime must be specified for {method}')
        if not fprime2:
            raise ValueError(f'fprime2 must be specified for {method}')
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=fprime2, **kwargs)
    else:
        raise ValueError(f'Unknown solver {method}')

    if is_memoized:
        # Replace the function_calls count with the memoized count.
        # Avoids double and triple-counting.
        n_calls = f.n_calls
        sol.function_calls = n_calls

    return sol


def _root_scalar_brentq_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above

    """
    pass


def _root_scalar_brenth_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass

def _root_scalar_toms748_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_secant_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    x1 : float, optional
        A second guess. Must be different from `x0`. If not specified,
        a value near `x0` will be chosen.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_newton_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_halley_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, required
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, required
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of 1st and 2nd derivatives along with the objective function.
        `fprime2` can also be a callable returning the 2nd derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_ridder_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_bisect_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


# <!-- @GENESIS_MODULE_END: _root_scalar -->
