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

                emit_telemetry("_at_recovered_2", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_at_recovered_2", "position_calculated", {
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
                            "module": "_at_recovered_2",
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
                    print(f"Emergency stop error in _at_recovered_2: {e}")
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
                    "module": "_at_recovered_2",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_at_recovered_2", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _at_recovered_2: {e}")
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


"""Update operations for read-only arrays."""

from __future__ import annotations

import operator
from collections.abc import Callable
from enum import Enum
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, cast

from ._utils._compat import (


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

    array_namespace,
    is_dask_array,
    is_jax_array,
    is_writeable_array,
)
from ._utils._helpers import meta_namespace
from ._utils._typing import Array, SetIndex

if TYPE_CHECKING:  # pragma: no cover
    # TODO import from typing (requires Python >=3.11)
    from typing_extensions import Self


class _AtOp(Enum):
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

            emit_telemetry("_at_recovered_2", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("_at_recovered_2", "position_calculated", {
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
                        "module": "_at_recovered_2",
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
                print(f"Emergency stop error in _at_recovered_2: {e}")
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
                "module": "_at_recovered_2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("_at_recovered_2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in _at_recovered_2: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "_at_recovered_2",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in _at_recovered_2: {e}")
    """Operations for use in `xpx.at`."""

    SET = "set"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    MIN = "min"
    MAX = "max"

    # @override from Python 3.12
    def __str__(self) -> str:  # type: ignore[explicit-override]  # pyright: ignore[reportImplicitOverride]
        """
        Return string representation (useful for pytest logs).

        Returns
        -------
        str
            The operation's name.
        """
        return self.value


class Undef(Enum):
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

            emit_telemetry("_at_recovered_2", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("_at_recovered_2", "position_calculated", {
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
                        "module": "_at_recovered_2",
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
                print(f"Emergency stop error in _at_recovered_2: {e}")
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
                "module": "_at_recovered_2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("_at_recovered_2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in _at_recovered_2: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "_at_recovered_2",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in _at_recovered_2: {e}")
    """Sentinel for undefined values."""

    UNDEF = 0


_undef = Undef.UNDEF


class at:  # pylint: disable=invalid-name  # numpydoc ignore=PR02
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

            emit_telemetry("_at_recovered_2", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("_at_recovered_2", "position_calculated", {
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
                        "module": "_at_recovered_2",
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
                print(f"Emergency stop error in _at_recovered_2: {e}")
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
                "module": "_at_recovered_2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("_at_recovered_2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in _at_recovered_2: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "_at_recovered_2",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in _at_recovered_2: {e}")
    """
    Update operations for read-only arrays.

    This implements ``jax.numpy.ndarray.at`` for all writeable
    backends (those that support ``__setitem__``) and routes
    to the ``.at[]`` method for JAX arrays.

    Parameters
    ----------
    x : array
        Input array.
    idx : index, optional
        Only `array API standard compliant indices
        <https://data-apis.org/array-api/latest/API_specification/indexing.html>`_
        are supported.

        You may use two alternate syntaxes::

          >>> import array_api_extra as xpx
          >>> xpx.at(x, idx).set(value)  # or add(value), etc.
          >>> xpx.at(x)[idx].set(value)

    copy : bool, optional
        None (default)
            The array parameter *may* be modified in place if it is
            possible and beneficial for performance.
            You should not reuse it after calling this function.
        True
            Ensure that the inputs are not modified.
        False
            Ensure that the update operation writes back to the input.
            Raise ``ValueError`` if a copy cannot be avoided.

    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    Updated input array.

    Warnings
    --------
    (a) When you omit the ``copy`` parameter, you should never reuse the parameter
    array later on; ideally, you should reassign it immediately::

        >>> import array_api_extra as xpx
        >>> x = xpx.at(x, 0).set(2)

    The above best practice pattern ensures that the behaviour won't change depending
    on whether ``x`` is writeable or not, as the original ``x`` object is dereferenced
    as soon as ``xpx.at`` returns; this way there is no risk to accidentally update it
    twice.

    On the reverse, the anti-pattern below must be avoided, as it will result in
    different behaviour on read-only versus writeable arrays::

        >>> x = xp.asarray([0, 0, 0])
        >>> y = xpx.at(x, 0).set(2)
        >>> z = xpx.at(x, 1).set(3)

    In the above example, both calls to ``xpx.at`` update ``x`` in place *if possible*.
    This causes the behaviour to diverge depending on whether ``x`` is writeable or not:

    - If ``x`` is writeable, then after the snippet above you'll have
      ``x == y == z == [2, 3, 0]``
    - If ``x`` is read-only, then you'll end up with
      ``x == [0, 0, 0]``, ``y == [2, 0, 0]`` and ``z == [0, 3, 0]``.

    The correct pattern to use if you want diverging outputs from the same input is
    to enforce copies::

        >>> x = xp.asarray([0, 0, 0])
        >>> y = xpx.at(x, 0).set(2, copy=True)  # Never updates x
        >>> z = xpx.at(x, 1).set(3)  # May or may not update x in place
        >>> del x  # avoid accidental reuse of x as we don't know its state anymore

    (b) The array API standard does not support integer array indices.
    The behaviour of update methods when the index is an array of integers is
    undefined and will vary between backends; this is particularly true when the
    index contains multiple occurrences of the same index, e.g.::

        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> import array_api_extra as xpx
        >>> xpx.at(np.asarray([123]), np.asarray([0, 0])).add(1)
        array([124])
        >>> xpx.at(jnp.asarray([123]), jnp.asarray([0, 0])).add(1)
        Array([125], dtype=int32)

    See Also
    --------
    jax.numpy.ndarray.at : Equivalent array method in JAX.

    Notes
    -----
    `sparse <https://sparse.pydata.org/>`_, as well as read-only arrays from libraries
    not explicitly covered by ``array-api-compat``, are not supported by update
    methods.

    Boolean masks are supported on Dask and jitted JAX arrays exclusively
    when `idx` has the same shape as `x` and `y` is 0-dimensional.
    Note that this support is not available in JAX's native
    ``x.at[mask].set(y)``.

    This pattern::

        >>> mask = m(x)
        >>> x[mask] = f(x[mask])

    Can't be replaced by `at`, as it won't work on Dask and JAX inside jax.jit::

        >>> mask = m(x)
        >>> x = xpx.at(x, mask).set(f(x[mask])  # Crash on Dask and jax.jit

    You should instead use::

        >>> x = xp.where(m(x), f(x), x)

    Examples
    --------
    Given either of these equivalent expressions::

      >>> import array_api_extra as xpx
      >>> x = xpx.at(x)[1].add(2)
      >>> x = xpx.at(x, 1).add(2)

    If x is a JAX array, they are the same as::

      >>> x = x.at[1].add(2)

    If x is a read-only NumPy array, they are the same as::

      >>> x = x.copy()
      >>> x[1] += 2

    For other known backends, they are the same as::

      >>> x[1] += 2
    """

    _x: Array
    _idx: SetIndex | Undef
    __slots__: ClassVar[tuple[str, ...]] = ("_idx", "_x")

    def __init__(
        self, x: Array, idx: SetIndex | Undef = _undef, /
    ) -> None:  # numpydoc ignore=GL08
        self._x = x
        self._idx = idx

    def __getitem__(self, idx: SetIndex, /) -> Self:  # numpydoc ignore=PR01,RT01
        """
        Allow for the alternate syntax ``at(x)[start:stop:step]``.

        It looks prettier than ``at(x, slice(start, stop, step))``
        and feels more intuitive coming from the JAX documentation.
        """
        if self._idx is not _undef:
            msg = "Index has already been set"
            raise ValueError(msg)
        return type(self)(self._x, idx)

    def _op(
        self,
        at_op: _AtOp,
        in_place_op: Callable[[Array, Array | complex], Array] | None,
        out_of_place_op: Callable[[Array, Array], Array] | None,
        y: Array | complex,
        /,
        copy: bool | None,
        xp: ModuleType | None,
    ) -> Array:
        """
        Implement all update operations.

        Parameters
        ----------
        at_op : _AtOp
            Method of JAX's Array.at[].
        in_place_op : Callable[[Array, Array | complex], Array] | None
            In-place operation to apply on mutable backends::

                x[idx] = in_place_op(x[idx], y)

            If None::

                x[idx] = y

        out_of_place_op : Callable[[Array, Array], Array] | None
            Out-of-place operation to apply when idx is a boolean mask and the backend
            doesn't support in-place updates::

                x = xp.where(idx, out_of_place_op(x, y), x)

            If None::

                x = xp.where(idx, y, x)

        y : array or complex
            Right-hand side of the operation.
        copy : bool or None
            Whether to copy the input array. See the class docstring for details.
        xp : array_namespace, optional
            The array namespace for the input array. Default: infer.

        Returns
        -------
        Array
            Updated `x`.
        """
        from ._funcs import apply_where  # pylint: disable=cyclic-import

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: _at_recovered_2 -->


# <!-- @GENESIS_MODULE_START: _at_recovered_2 -->

        x, idx = self._x, self._idx
        xp = array_namespace(x, y) if xp is None else xp

        if isinstance(idx, Undef):
            msg = (
                "Index has not been set.\n"
                "Usage: either\n"
                "    at(x, idx).set(value)\n"
                "or\n"
                "    at(x)[idx].set(value)\n"
                "(same for all other methods)."
            )
            raise ValueError(msg)

        if copy not in (True, False, None):
            msg = f"copy must be True, False, or None; got {copy!r}"
            raise ValueError(msg)

        writeable = None if copy else is_writeable_array(x)

        # JAX inside jax.jit doesn't support in-place updates with boolean
        # masks; Dask exclusively supports __setitem__ but not iops.
        # We can handle the common special case of 0-dimensional y
        # with where(idx, y, x) instead.
        if (
            (is_dask_array(idx) or is_jax_array(idx))
            and idx.dtype == xp.bool
            and idx.shape == x.shape
        ):
            y_xp = xp.asarray(y, dtype=x.dtype)
            if y_xp.ndim == 0:
                if out_of_place_op:  # add(), subtract(), ...
                    # suppress inf warnings on Dask
                    out = apply_where(
                        idx, (x, y_xp), out_of_place_op, fill_value=x, xp=xp
                    )
                    # Undo int->float promotion on JAX after _AtOp.DIVIDE
                    out = xp.astype(out, x.dtype, copy=False)
                else:  # set()
                    out = xp.where(idx, y_xp, x)

                if copy is False:
                    x[()] = out
                    return x
                return out

            # else: this will work on eager JAX and crash on jax.jit and Dask

        if copy or (copy is None and not writeable):
            if is_jax_array(x):
                # Use JAX's at[]
                func = cast(
                    Callable[[Array | complex], Array],
                    getattr(x.at[idx], at_op.value),  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue,reportUnknownArgumentType]
                )
                out = func(y)
                # Undo int->float promotion on JAX after _AtOp.DIVIDE
                return xp.astype(out, x.dtype, copy=False)

            # Emulate at[] behaviour for non-JAX arrays
            # with a copy followed by an update

            x = xp.asarray(x, copy=True)
            # A copy of a read-only numpy array is writeable
            # Note: this assumes that a copy of a writeable array is writeable
            assert not writeable
            writeable = None

        if writeable is None:
            writeable = is_writeable_array(x)
        if not writeable:
            # sparse crashes here
            msg = f"Can't update read-only array {x}"
            raise ValueError(msg)

        if in_place_op:  # add(), subtract(), ...
            x[idx] = in_place_op(x[idx], y)
        else:  # set()
            x[idx] = y
        return x

    def set(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] = y`` and return the update array."""
        return self._op(_AtOp.SET, None, None, y, copy=copy, xp=xp)

    def add(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] += y`` and return the updated array."""

        # Note for this and all other methods based on _iop:
        # operator.iadd and operator.add subtly differ in behaviour, as
        # only iadd will trigger exceptions when y has an incompatible dtype.
        return self._op(_AtOp.ADD, operator.iadd, operator.add, y, copy=copy, xp=xp)

    def subtract(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] -= y`` and return the updated array."""
        return self._op(
            _AtOp.SUBTRACT, operator.isub, operator.sub, y, copy=copy, xp=xp
        )

    def multiply(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] *= y`` and return the updated array."""
        return self._op(
            _AtOp.MULTIPLY, operator.imul, operator.mul, y, copy=copy, xp=xp
        )

    def divide(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] /= y`` and return the updated array."""
        return self._op(
            _AtOp.DIVIDE, operator.itruediv, operator.truediv, y, copy=copy, xp=xp
        )

    def power(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] **= y`` and return the updated array."""
        return self._op(_AtOp.POWER, operator.ipow, operator.pow, y, copy=copy, xp=xp)

    def min(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] = minimum(x[idx], y)`` and return the updated array."""
        # On Dask, this function runs on the chunks, so we need to determine the
        # namespace that Dask is wrapping.
        # Note that da.minimum _incidentally_ works on NumPy, CuPy, and sparse
        # thanks to all these meta-namespaces implementing the __array_ufunc__
        # interface, but there's no guarantee that it will work for other
        # wrapped libraries in the future.
        xp = array_namespace(self._x) if xp is None else xp
        mxp = meta_namespace(self._x, xp=xp)
        y = xp.asarray(y)
        return self._op(_AtOp.MIN, mxp.minimum, mxp.minimum, y, copy=copy, xp=xp)

    def max(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] = maximum(x[idx], y)`` and return the updated array."""
        # See note on min()
        xp = array_namespace(self._x) if xp is None else xp
        mxp = meta_namespace(self._x, xp=xp)
        y = xp.asarray(y)
        return self._op(_AtOp.MAX, mxp.maximum, mxp.maximum, y, copy=copy, xp=xp)



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))
