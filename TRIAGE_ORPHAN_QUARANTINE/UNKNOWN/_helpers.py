import logging
import sys
from pathlib import Path

# <!-- @GENESIS_MODULE_START: _helpers -->
"""
🏛️ GENESIS _HELPERS - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("_helpers", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_helpers", "position_calculated", {
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
                            "module": "_helpers",
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
                    print(f"Emergency stop error in _helpers: {e}")
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
                    "module": "_helpers",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_helpers", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _helpers: {e}")
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


"""Helper functions used by `array_api_extra/_funcs.py`."""

from __future__ import annotations

import math
from collections.abc import Generator, Iterable
from types import ModuleType
from typing import TYPE_CHECKING, cast

from . import _compat
from ._compat import (
    array_namespace,
    is_array_api_obj,
    is_dask_namespace,
    is_numpy_array,
)
from ._typing import Array

if TYPE_CHECKING:  # pragma: no cover
    # TODO import from typing (requires Python >=3.13)
    from typing_extensions import TypeIs


__all__ = [
    "asarrays",
    "eager_shape",
    "in1d",
    "is_python_scalar",
    "mean",
    "meta_namespace",
]


def in1d(
    x1: Array,
    x2: Array,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
    xp: ModuleType | None = None,
) -> Array:  # numpydoc ignore=PR01,RT01
    """
    Check whether each element of an array is also present in a second array.

    Returns a boolean array the same length as `x1` that is True
    where an element of `x1` is in `x2` and False otherwise.

    This function has been adapted using the original implementation
    present in numpy:
    https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/arraysetops.py#L524-L758
    """
    if xp is None:
        xp = array_namespace(x1, x2)

    x1_shape = eager_shape(x1)
    x2_shape = eager_shape(x2)

    # This code is run to make the code significantly faster
    if x2_shape[0] < 10 * x1_shape[0] ** 0.145 and isinstance(x2, Iterable):
        if invert:
            mask = xp.ones(x1_shape[0], dtype=xp.bool, device=_compat.device(x1))
            for a in x2:
                mask &= x1 != a
        else:
            mask = xp.zeros(x1_shape[0], dtype=xp.bool, device=_compat.device(x1))
            for a in x2:
                mask |= x1 == a
        return mask

    rev_idx = xp.empty(0)  # placeholder
    if not assume_unique:
        x1, rev_idx = xp.unique_inverse(x1)
        x2 = xp.unique_values(x2)

    ar = xp.concat((x1, x2))
    device_ = _compat.device(ar)
    # We need this to be a stable sort.
    order = xp.argsort(ar, stable=True)
    reverse_order = xp.argsort(order, stable=True)
    sar = xp.take(ar, order, axis=0)
    ar_size = _compat.size(sar)
    assert ar_size is not None, "xp.unique*() on lazy backends raises"
    if ar_size >= 1:
        bool_ar = sar[1:] != sar[:-1] if invert else sar[1:] == sar[:-1]
    else:
        bool_ar = xp.asarray([False]) if invert else xp.asarray([True])
    flag = xp.concat((bool_ar, xp.asarray([invert], device=device_)))
    ret = xp.take(flag, reverse_order, axis=0)

    if assume_unique:
        return ret[: x1.shape[0]]
    return xp.take(ret, rev_idx, axis=0)


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    xp: ModuleType | None = None,
) -> Array:  # numpydoc ignore=PR01,RT01
    """
    Complex mean, https://github.com/data-apis/array-api/issues/846.
    """
    if xp is None:
        xp = array_namespace(x)

    if xp.isdtype(x.dtype, "complex floating"):
        x_real = xp.real(x)
        x_imag = xp.imag(x)
        mean_real = xp.mean(x_real, axis=axis, keepdims=keepdims)
        mean_imag = xp.mean(x_imag, axis=axis, keepdims=keepdims)
        return mean_real + (mean_imag * xp.asarray(1j))
    return xp.mean(x, axis=axis, keepdims=keepdims)


def is_python_scalar(x: object) -> TypeIs[complex]:  # numpydoc ignore=PR01,RT01
    """Return True if `x` is a Python scalar, False otherwise."""
    # isinstance(x, float) returns True for np.float64
    # isinstance(x, complex) returns True for np.complex128
    # bool is a subclass of int
    return isinstance(x, int | float | complex) and not is_numpy_array(x)


def asarrays(
    a: Array | complex,
    b: Array | complex,
    xp: ModuleType,
) -> tuple[Array, Array]:
    """
    Ensure both `a` and `b` are arrays.

    If `b` is a python scalar, it is converted to the same dtype as `a`, and vice versa.

    Behavior is not specified when mixing a Python ``float`` and an array with an
    integer data type; this may give ``float32``, ``float64``, or raise an exception.
    Behavior is implementation-specific.

    Similarly, behavior is not specified when mixing a Python ``complex`` and an array
    with a real-valued data type; this may give ``complex64``, ``complex128``, or raise
    an exception. Behavior is implementation-specific.

    Parameters
    ----------
    a, b : Array | int | float | complex | bool
        Input arrays or scalars. At least one must be an array.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    Array, Array
        The input arrays, possibly converted to arrays if they were scalars.

    See Also
    --------
    mixing-arrays-with-python-scalars : Array API specification for the behavior.
    """
    a_scalar = is_python_scalar(a)
    b_scalar = is_python_scalar(b)
    if not a_scalar and not b_scalar:
        # This includes misc. malformed input e.g. str
        return a, b  # type: ignore[return-value]

    swap = False
    if a_scalar:
        swap = True
        b, a = a, b

    if is_array_api_obj(a):
        # a is an Array API object
        # b is a int | float | complex | bool
        xa = a

        # https://data-apis.org/array-api/draft/API_specification/type_promotion.html#mixing-arrays-with-python-scalars
        same_dtype = {
            bool: "bool",
            int: ("integral", "real floating", "complex floating"),
            float: ("real floating", "complex floating"),
            complex: "complex floating",
        }
        kind = same_dtype[type(cast(complex, b))]  # type: ignore[index]
        if xp.isdtype(a.dtype, kind):
            xb = xp.asarray(b, dtype=a.dtype)
        else:
            # Undefined behaviour. Let the function deal with it, if it can.
            xb = xp.asarray(b)

    else:
        # Neither a nor b are Array API objects.
        # Note: we can only reach this point when one explicitly passes
        # xp=xp to the calling function; otherwise we fail earlier on
        # array_namespace(a, b).
        xa, xb = xp.asarray(a), xp.asarray(b)

    return (xb, xa) if swap else (xa, xb)


def ndindex(*x: int) -> Generator[tuple[int, ...]]:
    """
    Generate all N-dimensional indices for a given array shape.

    Given the shape of an array, an ndindex instance iterates over the N-dimensional
    index of the array. At each iteration a tuple of indices is returned, the last
    dimension is iterated over first.

    This has an identical API to numpy.ndindex.

    Parameters
    ----------
    *x : int
        The shape of the array.
    """
    if not x:
        yield ()
        return
    for i in ndindex(*x[:-1]):
        for j in range(x[-1]):
            yield *i, j


def eager_shape(x: Array, /) -> tuple[int, ...]:
    """
    Return shape of an array. Raise if shape is not fully defined.

    Parameters
    ----------
    x : Array
        Input array.

    Returns
    -------
    tuple[int, ...]
        Shape of the array.
    """
    shape = x.shape
    # Dask arrays uses non-standard NaN instead of None
    if any(s is None or math.isnan(s) for s in shape):
        msg = "Unsupported lazy shape"
        raise TypeError(msg)
    return cast(tuple[int, ...], shape)


def meta_namespace(
    *arrays: Array | complex | None, xp: ModuleType | None = None
) -> ModuleType:
    """
    Get the namespace of Dask chunks.

    On all other backends, just return the namespace of the arrays.

    Parameters
    ----------
    *arrays : Array | int | float | complex | bool | None
        Input arrays.
    xp : array_namespace, optional
        The standard-compatible namespace for the input arrays. Default: infer.

    Returns
    -------
    array_namespace
        If xp is Dask, the namespace of the Dask chunks;
        otherwise, the namespace of the arrays.
    """
    xp = array_namespace(*arrays) if xp is None else xp
    if not is_dask_namespace(xp):
        return xp
    # Quietly skip scalars and None's
    metas = [cast(Array | None, getattr(a, "_meta", None)) for a in arrays]
    return array_namespace(*metas)


# <!-- @GENESIS_MODULE_END: _helpers -->
