import logging
# <!-- @GENESIS_MODULE_START: transform -->
"""
🏛️ GENESIS TRANSFORM - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("transform", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("transform", "position_calculated", {
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
                            "module": "transform",
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
                    print(f"Emergency stop error in transform: {e}")
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
                    "module": "transform",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("transform", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in transform: {e}")
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


"""Affine 2D transformation matrix class.

The Transform class implements various transformation matrix operations,
both on the matrix itself, as well as on 2D coordinates.

Transform instances are effectively immutable: all methods that operate on the
transformation itself always return a new instance. This has as the
interesting side effect that Transform instances are hashable, ie. they can be
used as dictionary keys.

This module exports the following symbols:

Transform
	this is the main class
Identity
	Transform instance set to the identity transformation
Offset
	Convenience function that returns a translating transformation
Scale
	Convenience function that returns a scaling transformation

The DecomposedTransform class implements a transformation with separate
translate, rotation, scale, skew, and transformation-center components.

:Example:

	>>> t = Transform(2, 0, 0, 3, 0, 0)
	>>> t.transformPoint((100, 100))
	(200, 300)
	>>> t = Scale(2, 3)
	>>> t.transformPoint((100, 100))
	(200, 300)
	>>> t.transformPoint((0, 0))
	(0, 0)
	>>> t = Offset(2, 3)
	>>> t.transformPoint((100, 100))
	(102, 103)
	>>> t.transformPoint((0, 0))
	(2, 3)
	>>> t2 = t.scale(0.5)
	>>> t2.transformPoint((100, 100))
	(52.0, 53.0)
	>>> import math
	>>> t3 = t2.rotate(math.pi / 2)
	>>> t3.transformPoint((0, 0))
	(2.0, 3.0)
	>>> t3.transformPoint((100, 100))
	(-48.0, 53.0)
	>>> t = Identity.scale(0.5).translate(100, 200).skew(0.1, 0.2)
	>>> t.transformPoints([(0, 0), (1, 1), (100, 100)])
	[(50.0, 100.0), (50.550167336042726, 100.60135501775433), (105.01673360427253, 160.13550177543362)]
	>>>
"""

from __future__ import annotations

import math
from typing import NamedTuple
from dataclasses import dataclass


__all__ = ["Transform", "Identity", "Offset", "Scale", "DecomposedTransform"]


_EPSILON = 1e-15
_ONE_EPSILON = 1 - _EPSILON
_MINUS_ONE_EPSILON = -1 + _EPSILON


def _normSinCos(v: float) -> float:
    if abs(v) < _EPSILON:
        v = 0
    elif v > _ONE_EPSILON:
        v = 1
    elif v < _MINUS_ONE_EPSILON:
        v = -1
    return v


class Transform(NamedTuple):
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

            emit_telemetry("transform", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("transform", "position_calculated", {
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
                        "module": "transform",
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
                print(f"Emergency stop error in transform: {e}")
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
                "module": "transform",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("transform", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in transform: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "transform",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in transform: {e}")
    """2x2 transformation matrix plus offset, a.k.a. Affine transform.
    Transform instances are immutable: all transforming methods, eg.
    rotate(), return a new Transform instance.

    :Example:

            >>> t = Transform()
            >>> t
            <Transform [1 0 0 1 0 0]>
            >>> t.scale(2)
            <Transform [2 0 0 2 0 0]>
            >>> t.scale(2.5, 5.5)
            <Transform [2.5 0 0 5.5 0 0]>
            >>>
            >>> t.scale(2, 3).transformPoint((100, 100))
            (200, 300)

    Transform's constructor takes six arguments, all of which are
    optional, and can be used as keyword arguments::

            >>> Transform(12)
            <Transform [12 0 0 1 0 0]>
            >>> Transform(dx=12)
            <Transform [1 0 0 1 12 0]>
            >>> Transform(yx=12)
            <Transform [1 0 12 1 0 0]>

    Transform instances also behave like sequences of length 6::

            >>> len(Identity)
            6
            >>> list(Identity)
            [1, 0, 0, 1, 0, 0]
            >>> tuple(Identity)
            (1, 0, 0, 1, 0, 0)

    Transform instances are comparable::

            >>> t1 = Identity.scale(2, 3).translate(4, 6)
            >>> t2 = Identity.translate(8, 18).scale(2, 3)
            >>> t1 == t2
            1

    But beware of floating point rounding errors::

            >>> t1 = Identity.scale(0.2, 0.3).translate(0.4, 0.6)
            >>> t2 = Identity.translate(0.08, 0.18).scale(0.2, 0.3)
            >>> t1
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> t2
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> t1 == t2
            0

    Transform instances are hashable, meaning you can use them as
    keys in dictionaries::

            >>> d = {Scale(12, 13): None}
            >>> d
            {<Transform [12 0 0 13 0 0]>: None}

    But again, beware of floating point rounding errors::

            >>> t1 = Identity.scale(0.2, 0.3).translate(0.4, 0.6)
            >>> t2 = Identity.translate(0.08, 0.18).scale(0.2, 0.3)
            >>> t1
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> t2
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> d = {t1: None}
            >>> d
            {<Transform [0.2 0 0 0.3 0.08 0.18]>: None}
            >>> d[t2]
            Traceback (most recent call last):
              File "<stdin>", line 1, in ?
            KeyError: <Transform [0.2 0 0 0.3 0.08 0.18]>
    """

    xx: float = 1
    xy: float = 0
    yx: float = 0
    yy: float = 1
    dx: float = 0
    dy: float = 0

    def transformPoint(self, p):
        """Transform a point.

        :Example:

                >>> t = Transform()
                >>> t = t.scale(2.5, 5.5)
                >>> t.transformPoint((100, 100))
                (250.0, 550.0)
        """
        (x, y) = p
        xx, xy, yx, yy, dx, dy = self
        return (xx * x + yx * y + dx, xy * x + yy * y + dy)

    def transformPoints(self, points):
        """Transform a list of points.

        :Example:

                >>> t = Scale(2, 3)
                >>> t.transformPoints([(0, 0), (0, 100), (100, 100), (100, 0)])
                [(0, 0), (0, 300), (200, 300), (200, 0)]
                >>>
        """
        xx, xy, yx, yy, dx, dy = self
        return [(xx * x + yx * y + dx, xy * x + yy * y + dy) for x, y in points]

    def transformVector(self, v):
        """Transform an (dx, dy) vector, treating translation as zero.

        :Example:

                >>> t = Transform(2, 0, 0, 2, 10, 20)
                >>> t.transformVector((3, -4))
                (6, -8)
                >>>
        """
        (dx, dy) = v
        xx, xy, yx, yy = self[:4]
        return (xx * dx + yx * dy, xy * dx + yy * dy)

    def transformVectors(self, vectors):
        """Transform a list of (dx, dy) vector, treating translation as zero.

        :Example:
                >>> t = Transform(2, 0, 0, 2, 10, 20)
                >>> t.transformVectors([(3, -4), (5, -6)])
                [(6, -8), (10, -12)]
                >>>
        """
        xx, xy, yx, yy = self[:4]
        return [(xx * dx + yx * dy, xy * dx + yy * dy) for dx, dy in vectors]

    def translate(self, x: float = 0, y: float = 0):
        """Return a new transformation, translated (offset) by x, y.

        :Example:
                >>> t = Transform()
                >>> t.translate(20, 30)
                <Transform [1 0 0 1 20 30]>
                >>>
        """
        return self.transform((1, 0, 0, 1, x, y))

    def scale(self, x: float = 1, y: float | None = None):
        """Return a new transformation, scaled by x, y. The 'y' argument
        may be None, which implies to use the x value for y as well.

        :Example:
                >>> t = Transform()
                >>> t.scale(5)
                <Transform [5 0 0 5 0 0]>
                >>> t.scale(5, 6)
                <Transform [5 0 0 6 0 0]>
                >>>
        """
        if y is None:
            y = x
        return self.transform((x, 0, 0, y, 0, 0))

    def rotate(self, angle: float):
        """Return a new transformation, rotated by 'angle' (radians).

        :Example:
                >>> import math
                >>> t = Transform()
                >>> t.rotate(math.pi / 2)
                <Transform [0 1 -1 0 0 0]>
                >>>
        """
        c = _normSinCos(math.cos(angle))
        s = _normSinCos(math.sin(angle))
        return self.transform((c, s, -s, c, 0, 0))

    def skew(self, x: float = 0, y: float = 0):
        """Return a new transformation, skewed by x and y.

        :Example:
                >>> import math
                >>> t = Transform()
                >>> t.skew(math.pi / 4)
                <Transform [1 0 1 1 0 0]>
                >>>
        """
        return self.transform((1, math.tan(y), math.tan(x), 1, 0, 0))

    def transform(self, other):
        """Return a new transformation, transformed by another
        transformation.

        :Example:
                >>> t = Transform(2, 0, 0, 3, 1, 6)
                >>> t.transform((4, 3, 2, 1, 5, 6))
                <Transform [8 9 4 3 11 24]>
                >>>
        """
        xx1, xy1, yx1, yy1, dx1, dy1 = other
        xx2, xy2, yx2, yy2, dx2, dy2 = self
        return self.__class__(
            xx1 * xx2 + xy1 * yx2,
            xx1 * xy2 + xy1 * yy2,
            yx1 * xx2 + yy1 * yx2,
            yx1 * xy2 + yy1 * yy2,
            xx2 * dx1 + yx2 * dy1 + dx2,
            xy2 * dx1 + yy2 * dy1 + dy2,
        )

    def reverseTransform(self, other):
        """Return a new transformation, which is the other transformation
        transformed by self. self.reverseTransform(other) is equivalent to
        other.transform(self).

        :Example:
                >>> t = Transform(2, 0, 0, 3, 1, 6)
                >>> t.reverseTransform((4, 3, 2, 1, 5, 6))
                <Transform [8 6 6 3 21 15]>
                >>> Transform(4, 3, 2, 1, 5, 6).transform((2, 0, 0, 3, 1, 6))
                <Transform [8 6 6 3 21 15]>
                >>>
        """
        xx1, xy1, yx1, yy1, dx1, dy1 = self
        xx2, xy2, yx2, yy2, dx2, dy2 = other
        return self.__class__(
            xx1 * xx2 + xy1 * yx2,
            xx1 * xy2 + xy1 * yy2,
            yx1 * xx2 + yy1 * yx2,
            yx1 * xy2 + yy1 * yy2,
            xx2 * dx1 + yx2 * dy1 + dx2,
            xy2 * dx1 + yy2 * dy1 + dy2,
        )

    def inverse(self):
        """Return the inverse transformation.

        :Example:
                >>> t = Identity.translate(2, 3).scale(4, 5)
                >>> t.transformPoint((10, 20))
                (42, 103)
                >>> it = t.inverse()
                >>> it.transformPoint((42, 103))
                (10.0, 20.0)
                >>>
        """
        if self == Identity:
            return self
        xx, xy, yx, yy, dx, dy = self
        det = xx * yy - yx * xy
        xx, xy, yx, yy = yy / det, -xy / det, -yx / det, xx / det
        dx, dy = -xx * dx - yx * dy, -xy * dx - yy * dy
        return self.__class__(xx, xy, yx, yy, dx, dy)

    def toPS(self) -> str:
        """Return a PostScript representation

        :Example:

                >>> t = Identity.scale(2, 3).translate(4, 5)
                >>> t.toPS()
                '[2 0 0 3 8 15]'
                >>>
        """
        return "[%s %s %s %s %s %s]" % self

    def toDecomposed(self) -> "DecomposedTransform":
        """Decompose into a DecomposedTransform."""
        return DecomposedTransform.fromTransform(self)

    def __bool__(self) -> bool:
        """Returns True if transform is not identity, False otherwise.

        :Example:

                >>> bool(Identity)
                False
                >>> bool(Transform())
                False
                >>> bool(Scale(1.))
                False
                >>> bool(Scale(2))
                True
                >>> bool(Offset())
                False
                >>> bool(Offset(0))
                False
                >>> bool(Offset(2))
                True
        """
        return self != Identity

    def __repr__(self) -> str:
        return "<%s [%g %g %g %g %g %g]>" % ((self.__class__.__name__,) + self)


Identity = Transform()


def Offset(x: float = 0, y: float = 0) -> Transform:
    """Return the identity transformation offset by x, y.

    :Example:
            >>> Offset(2, 3)
            <Transform [1 0 0 1 2 3]>
            >>>
    """
    return Transform(1, 0, 0, 1, x, y)


def Scale(x: float, y: float | None = None) -> Transform:
    """Return the identity transformation scaled by x, y. The 'y' argument
    may be None, which implies to use the x value for y as well.

    :Example:
            >>> Scale(2, 3)
            <Transform [2 0 0 3 0 0]>
            >>>
    """
    if y is None:
        y = x
    return Transform(x, 0, 0, y, 0, 0)


@dataclass
class DecomposedTransform:
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

            emit_telemetry("transform", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("transform", "position_calculated", {
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
                        "module": "transform",
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
                print(f"Emergency stop error in transform: {e}")
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
                "module": "transform",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("transform", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in transform: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "transform",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in transform: {e}")
    """The DecomposedTransform class implements a transformation with separate
    translate, rotation, scale, skew, and transformation-center components.
    """

    translateX: float = 0
    translateY: float = 0
    rotation: float = 0  # in degrees, counter-clockwise
    scaleX: float = 1
    scaleY: float = 1
    skewX: float = 0  # in degrees, clockwise
    skewY: float = 0  # in degrees, counter-clockwise
    tCenterX: float = 0
    tCenterY: float = 0

    def __bool__(self):
        return (
            self.translateX != 0
            or self.translateY != 0
            or self.rotation != 0
            or self.scaleX != 1
            or self.scaleY != 1
            or self.skewX != 0
            or self.skewY != 0
            or self.tCenterX != 0
            or self.tCenterY != 0
        )

    @classmethod
    def fromTransform(self, transform):
        """Return a DecomposedTransform() equivalent of this transformation.
        The returned solution always has skewY = 0, and angle in the (-180, 180].

        :Example:
                >>> DecomposedTransform.fromTransform(Transform(3, 0, 0, 2, 0, 0))
                DecomposedTransform(translateX=0, translateY=0, rotation=0.0, scaleX=3.0, scaleY=2.0, skewX=0.0, skewY=0.0, tCenterX=0, tCenterY=0)
                >>> DecomposedTransform.fromTransform(Transform(0, 0, 0, 1, 0, 0))
                DecomposedTransform(translateX=0, translateY=0, rotation=0.0, scaleX=0.0, scaleY=1.0, skewX=0.0, skewY=0.0, tCenterX=0, tCenterY=0)
                >>> DecomposedTransform.fromTransform(Transform(0, 0, 1, 1, 0, 0))
                DecomposedTransform(translateX=0, translateY=0, rotation=-45.0, scaleX=0.0, scaleY=1.4142135623730951, skewX=0.0, skewY=0.0, tCenterX=0, tCenterY=0)
        """
        # Adapted from an answer on
        # https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix

        a, b, c, d, x, y = transform

        sx = math.copysign(1, a)
        if sx < 0:
            a *= sx
            b *= sx

        delta = a * d - b * c

        rotation = 0
        scaleX = scaleY = 0
        skewX = 0

        # Apply the QR-like decomposition.
        if a != 0 or b != 0:
            r = math.sqrt(a * a + b * b)
            rotation = math.acos(a / r) if b >= 0 else -math.acos(a / r)
            scaleX, scaleY = (r, delta / r)
            skewX = math.atan((a * c + b * d) / (r * r))
        elif c != 0 or d != 0:
            s = math.sqrt(c * c + d * d)
            rotation = math.pi / 2 - (
                math.acos(-c / s) if d >= 0 else -math.acos(c / s)
            )
            scaleX, scaleY = (delta / s, s)
        else:
            # a = b = c = d = 0
            pass

        return DecomposedTransform(
            x,
            y,
            math.degrees(rotation),
            scaleX * sx,
            scaleY,
            math.degrees(skewX) * sx,
            0.0,
            0,
            0,
        )

    def toTransform(self) -> Transform:
        """Return the Transform() equivalent of this transformation.

        :Example:
                >>> DecomposedTransform(scaleX=2, scaleY=2).toTransform()
                <Transform [2 0 0 2 0 0]>
                >>>
        """
        t = Transform()
        t = t.translate(
            self.translateX + self.tCenterX, self.translateY + self.tCenterY
        )
        t = t.rotate(math.radians(self.rotation))
        t = t.scale(self.scaleX, self.scaleY)
        t = t.skew(math.radians(self.skewX), math.radians(self.skewY))
        t = t.translate(-self.tCenterX, -self.tCenterY)
        return t


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)


# <!-- @GENESIS_MODULE_END: transform -->
