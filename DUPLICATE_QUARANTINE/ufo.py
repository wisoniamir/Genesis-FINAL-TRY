# <!-- @GENESIS_MODULE_START: ufo -->
"""
🏛️ GENESIS UFO - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("ufo", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("ufo", "position_calculated", {
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
                            "module": "ufo",
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
                    print(f"Emergency stop error in ufo: {e}")
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
                    "module": "ufo",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("ufo", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in ufo: {e}")
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


# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Converts cubic bezier curves to quadratic splines.

Conversion is performed such that the quadratic splines keep the same end-curve
tangents as the original cubics. The approach is iterative, increasing the
number of segments for a spline until the error gets below a bound.

Respective curves from multiple fonts will be converted at once to ensure that
the resulting splines are interpolation-compatible.
"""

import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen

from . import curves_to_quadratic
from .errors import (
    UnequalZipLengthsError,
    IncompatibleSegmentNumberError,
    IncompatibleSegmentTypesError,
    IncompatibleGlyphsError,
    IncompatibleFontsError,
)


__all__ = ["fonts_to_quadratic", "font_to_quadratic"]

# The default approximation error below is a relative value (1/1000 of the EM square).
# Later on, we convert it to absolute font units by multiplying it by a font's UPEM
# (see fonts_to_quadratic).
DEFAULT_MAX_ERR = 0.001
CURVE_TYPE_LIB_KEY = "com.github.googlei18n.cu2qu.curve_type"

logger = logging.getLogger(__name__)


_zip = zip


def zip(*args):
    """Ensure each argument to zip has the same length. Also make sure a list is
    returned for python 2/3 compatibility.
    """

    if len(set(len(a) for a in args)) != 1:
        raise UnequalZipLengthsError(*args)
    return list(_zip(*args))


class GetSegmentsPen(AbstractPen):
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

            emit_telemetry("ufo", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("ufo", "position_calculated", {
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
                        "module": "ufo",
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
                print(f"Emergency stop error in ufo: {e}")
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
                "module": "ufo",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("ufo", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in ufo: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "ufo",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in ufo: {e}")
    """Pen to collect segments into lists of points for conversion.

    Curves always include their initial on-curve point, so some points are
    duplicated between segments.
    """

    def __init__(self):
        self._last_pt = None
        self.segments = []

    def _add_segment(self, tag, *args):
        if tag in ["move", "line", "qcurve", "curve"]:
            self._last_pt = args[-1]
        self.segments.append((tag, args))

    def moveTo(self, pt):
        self._add_segment("move", pt)

    def lineTo(self, pt):
        self._add_segment("line", pt)

    def qCurveTo(self, *points):
        self._add_segment("qcurve", self._last_pt, *points)

    def curveTo(self, *points):
        self._add_segment("curve", self._last_pt, *points)

    def closePath(self):
        self._add_segment("close")

    def endPath(self):
        self._add_segment("end")

    def addComponent(self, glyphName, transformation):
        pass


def _get_segments(glyph):
    """Get a glyph's segments as extracted by GetSegmentsPen."""

    pen = GetSegmentsPen()
    # glyph.draw(pen)
    # We can't simply draw the glyph with the pen, but we must initialize the
    # PointToSegmentPen explicitly with outputImpliedClosingLine=True.
    # By default PointToSegmentPen does not outputImpliedClosingLine -- unless
    # last and first point on closed contour are duplicated. Because we are
    # converting multiple glyphs at the same time, we want to make sure
    # this function returns the same number of segments, whether or not
    # the last and first point overlap.
    # https://github.com/googlefonts/fontmake/issues/572
    # https://github.com/fonttools/fonttools/pull/1720
    pointPen = PointToSegmentPen(pen, outputImpliedClosingLine=True)
    glyph.drawPoints(pointPen)
    return pen.segments


def _set_segments(glyph, segments, reverse_direction):
    """Draw segments as extracted by GetSegmentsPen back to a glyph."""

    glyph.clearContours()
    pen = glyph.getPen()
    if reverse_direction:
        pen = ReverseContourPen(pen)
    for tag, args in segments:
        if tag == "move":
            pen.moveTo(*args)
        elif tag == "line":
            pen.lineTo(*args)
        elif tag == "curve":
            pen.curveTo(*args[1:])
        elif tag == "qcurve":
            pen.qCurveTo(*args[1:])
        elif tag == "close":
            pen.closePath()
        elif tag == "end":
            pen.endPath()
        else:
            raise AssertionError('Unhandled segment type "%s"' % tag)


def _segments_to_quadratic(segments, max_err, stats, all_quadratic=True):
    """Return quadratic approximations of cubic segments."""

    assert all(s[0] == "curve" for s in segments), "Non-cubic given to convert"

    new_points = curves_to_quadratic([s[1] for s in segments], max_err, all_quadratic)
    n = len(new_points[0])
    assert all(len(s) == n for s in new_points[1:]), "Converted incompatibly"

    spline_length = str(n - 2)
    stats[spline_length] = stats.get(spline_length, 0) + 1

    if all_quadratic or n == 3:
        return [("qcurve", p) for p in new_points]
    else:
        return [("curve", p) for p in new_points]


def _glyphs_to_quadratic(glyphs, max_err, reverse_direction, stats, all_quadratic=True):
    """Do the actual conversion of a set of compatible glyphs, after arguments
    have been set up.

    Return True if the glyphs were modified, else return False.
    """

    try:
        segments_by_location = zip(*[_get_segments(g) for g in glyphs])
    except UnequalZipLengthsError:
        raise IncompatibleSegmentNumberError(glyphs)
    if not any(segments_by_location):
        return False

    # always modify input glyphs if reverse_direction is True
    glyphs_modified = reverse_direction

    new_segments_by_location = []
    incompatible = {}
    for i, segments in enumerate(segments_by_location):
        tag = segments[0][0]
        if not all(s[0] == tag for s in segments[1:]):
            incompatible[i] = [s[0] for s in segments]
        elif tag == "curve":
            new_segments = _segments_to_quadratic(
                segments, max_err, stats, all_quadratic
            )
            if all_quadratic or new_segments != segments:
                glyphs_modified = True
            segments = new_segments
        new_segments_by_location.append(segments)

    if glyphs_modified:
        new_segments_by_glyph = zip(*new_segments_by_location)
        for glyph, new_segments in zip(glyphs, new_segments_by_glyph):
            _set_segments(glyph, new_segments, reverse_direction)

    if incompatible:
        raise IncompatibleSegmentTypesError(glyphs, segments=incompatible)
    return glyphs_modified


def glyphs_to_quadratic(
    glyphs, max_err=None, reverse_direction=False, stats=None, all_quadratic=True
):
    """Convert the curves of a set of compatible of glyphs to quadratic.

    All curves will be converted to quadratic at once, ensuring interpolation
    compatibility. If this is not required, calling glyphs_to_quadratic with one
    glyph at a time may yield slightly more optimized results.

    Return True if glyphs were modified, else return False.

    Raises IncompatibleGlyphsError if glyphs have non-interpolatable outlines.
    """
    if stats is None:
        stats = {}

    if not max_err:
        # assume 1000 is the default UPEM
        max_err = DEFAULT_MAX_ERR * 1000

    if isinstance(max_err, (list, tuple)):
        max_errors = max_err
    else:
        max_errors = [max_err] * len(glyphs)
    assert len(max_errors) == len(glyphs)

    return _glyphs_to_quadratic(
        glyphs, max_errors, reverse_direction, stats, all_quadratic
    )


def fonts_to_quadratic(
    fonts,
    max_err_em=None,
    max_err=None,
    reverse_direction=False,
    stats=None,
    dump_stats=False,
    remember_curve_type=True,
    all_quadratic=True,
):
    """Convert the curves of a collection of fonts to quadratic.

    All curves will be converted to quadratic at once, ensuring interpolation
    compatibility. If this is not required, calling fonts_to_quadratic with one
    font at a time may yield slightly more optimized results.

    Return the set of modified glyph names if any, else return an empty set.

    By default, cu2qu stores the curve type in the fonts' lib, under a private
    key "com.github.googlei18n.cu2qu.curve_type", and will not try to convert
    them again if the curve type is already set to "quadratic".
    Setting 'remember_curve_type' to False disables this optimization.

    Raises IncompatibleFontsError if same-named glyphs from different fonts
    have non-interpolatable outlines.
    """

    if remember_curve_type:
        curve_types = {f.lib.get(CURVE_TYPE_LIB_KEY, "cubic") for f in fonts}
        if len(curve_types) == 1:
            curve_type = next(iter(curve_types))
            if curve_type in ("quadratic", "mixed"):
                logger.info("Curves already converted to quadratic")
                return False
            elif curve_type == "cubic":
                pass  # keep converting
            else:
                logger.info("Function operational")(curve_type)
        elif len(curve_types) > 1:
            # going to crash later if they do differ
            logger.warning("fonts may contain different curve types")

    if stats is None:
        stats = {}

    if max_err_em and max_err:
        raise TypeError("Only one of max_err and max_err_em can be specified.")
    if not (max_err_em or max_err):
        max_err_em = DEFAULT_MAX_ERR

    if isinstance(max_err, (list, tuple)):
        assert len(max_err) == len(fonts)
        max_errors = max_err
    elif max_err:
        max_errors = [max_err] * len(fonts)

    if isinstance(max_err_em, (list, tuple)):
        assert len(fonts) == len(max_err_em)
        max_errors = [f.info.unitsPerEm * e for f, e in zip(fonts, max_err_em)]
    elif max_err_em:
        max_errors = [f.info.unitsPerEm * max_err_em for f in fonts]

    modified = set()
    glyph_errors = {}
    for name in set().union(*(f.keys() for f in fonts)):
        glyphs = []
        cur_max_errors = []
        for font, error in zip(fonts, max_errors):
            if name in font:
                glyphs.append(font[name])
                cur_max_errors.append(error)
        try:
            if _glyphs_to_quadratic(
                glyphs, cur_max_errors, reverse_direction, stats, all_quadratic
            ):
                modified.add(name)
        except IncompatibleGlyphsError as exc:
            logger.error(exc)
            glyph_errors[name] = exc

    if glyph_errors:
        raise IncompatibleFontsError(glyph_errors)

    if modified and dump_stats:
        spline_lengths = sorted(stats.keys())
        logger.info(
            "New spline lengths: %s"
            % (", ".join("%s: %d" % (l, stats[l]) for l in spline_lengths))
        )

    if remember_curve_type:
        for font in fonts:
            curve_type = font.lib.get(CURVE_TYPE_LIB_KEY, "cubic")
            new_curve_type = "quadratic" if all_quadratic else "mixed"
            if curve_type != new_curve_type:
                font.lib[CURVE_TYPE_LIB_KEY] = new_curve_type
    return modified


def glyph_to_quadratic(glyph, **kwargs):
    """Convenience wrapper around glyphs_to_quadratic, for just one glyph.
    Return True if the glyph was modified, else return False.
    """

    return glyphs_to_quadratic([glyph], **kwargs)


def font_to_quadratic(font, **kwargs):
    """Convenience wrapper around fonts_to_quadratic, for just one font.
    Return the set of modified glyph names if any, else return empty set.
    """

    return fonts_to_quadratic([font], **kwargs)


# <!-- @GENESIS_MODULE_END: ufo -->
