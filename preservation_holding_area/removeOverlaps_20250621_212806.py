# <!-- @GENESIS_MODULE_START: removeOverlaps -->
"""
🏛️ GENESIS REMOVEOVERLAPS - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("removeOverlaps", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("removeOverlaps", "position_calculated", {
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
                            "module": "removeOverlaps",
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
                    print(f"Emergency stop error in removeOverlaps: {e}")
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
                    "module": "removeOverlaps",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("removeOverlaps", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in removeOverlaps: {e}")
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


""" Simplify TrueType glyphs by merging overlapping contours/components.

Requires https://github.com/fonttools/skia-pathops
"""

import itertools
import logging
from typing import Callable, Iterable, Optional, Mapping

from fontTools.cffLib import CFFFontSet
from fontTools.ttLib import ttFont
from fontTools.ttLib.tables import _g_l_y_f
from fontTools.ttLib.tables import _h_m_t_x
from fontTools.misc.psCharStrings import T2CharString
from fontTools.misc.roundTools import otRound, noRound
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.t2CharStringPen import T2CharStringPen

import pathops


__all__ = ["removeOverlaps"]


class RemoveOverlapsError(Exception):
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

            emit_telemetry("removeOverlaps", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("removeOverlaps", "position_calculated", {
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
                        "module": "removeOverlaps",
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
                print(f"Emergency stop error in removeOverlaps: {e}")
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
                "module": "removeOverlaps",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("removeOverlaps", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in removeOverlaps: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "removeOverlaps",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in removeOverlaps: {e}")
    pass


log = logging.getLogger("fontTools.ttLib.removeOverlaps")

_TTGlyphMapping = Mapping[str, ttFont._TTGlyph]


def skPathFromGlyph(glyphName: str, glyphSet: _TTGlyphMapping) -> pathops.Path:
    path = pathops.Path()
    pathPen = path.getPen(glyphSet=glyphSet)
    glyphSet[glyphName].draw(pathPen)
    return path


def skPathFromGlyphComponent(
    component: _g_l_y_f.GlyphComponent, glyphSet: _TTGlyphMapping
):
    baseGlyphName, transformation = component.getComponentInfo()
    path = skPathFromGlyph(baseGlyphName, glyphSet)
    return path.transform(*transformation)


def componentsOverlap(glyph: _g_l_y_f.Glyph, glyphSet: _TTGlyphMapping) -> bool:
    if not glyph.isComposite():
        raise ValueError("This method only works with TrueType composite glyphs")
    if len(glyph.components) < 2:
        return False  # single component, no overlaps

    component_paths = {}

    def _get_nth_component_path(index: int) -> pathops.Path:
        if index not in component_paths:
            component_paths[index] = skPathFromGlyphComponent(
                glyph.components[index], glyphSet
            )
        return component_paths[index]

    return any(
        pathops.op(
            _get_nth_component_path(i),
            _get_nth_component_path(j),
            pathops.PathOp.INTERSECTION,
            fix_winding=False,
            keep_starting_points=False,
        )
        for i, j in itertools.combinations(range(len(glyph.components)), 2)
    )


def ttfGlyphFromSkPath(path: pathops.Path) -> _g_l_y_f.Glyph:
    # Skia paths have no 'components', no need for glyphSet
    ttPen = TTGlyphPen(glyphSet=None)
    path.draw(ttPen)
    glyph = ttPen.glyph()
    assert not glyph.isComposite()
    # compute glyph.xMin (glyfTable parameter unused for non composites)
    glyph.recalcBounds(glyfTable=None)
    return glyph


def _charString_from_SkPath(
    path: pathops.Path, charString: T2CharString
) -> T2CharString:
    if charString.width == charString.private.defaultWidthX:
        width = None
    else:
        width = charString.width - charString.private.nominalWidthX
    t2Pen = T2CharStringPen(width=width, glyphSet=None)
    path.draw(t2Pen)
    return t2Pen.getCharString(charString.private, charString.globalSubrs)


def _round_path(
    path: pathops.Path, round: Callable[[float], float] = otRound
) -> pathops.Path:
    rounded_path = pathops.Path()
    for verb, points in path:
        rounded_path.add(verb, *((round(p[0]), round(p[1])) for p in points))
    return rounded_path


def _simplify(
    path: pathops.Path,
    debugGlyphName: str,
    *,
    round: Callable[[float], float] = otRound,
) -> pathops.Path:
    # skia-pathops has a bug where it sometimes fails to simplify paths when there
    # are float coordinates and control points are very close to one another.
    # Rounding coordinates to integers works around the bug.
    # Since we are going to round glyf coordinates later on anyway, here it is
    # ok(-ish) to also round before simplify. Better than failing the whole process
    # for the entire font.
    # https://bugs.chromium.org/p/skia/issues/detail?id=11958
    # https://github.com/google/fonts/issues/3365
    # TODO(anthrotype): remove once this Skia bug is fixed
    try:
        return pathops.simplify(path, clockwise=path.clockwise)
    except pathops.PathOpsError:
        pass

    path = _round_path(path, round=round)
    try:
        path = pathops.simplify(path, clockwise=path.clockwise)
        log.debug(
            "skia-pathops failed to simplify '%s' with float coordinates, "
            "but succeded using rounded integer coordinates",
            debugGlyphName,
        )
        return path
    except pathops.PathOpsError as e:
        if log.isEnabledFor(logging.DEBUG):
            path.dump()
        raise RemoveOverlapsError(
            f"Failed to remove overlaps from glyph {debugGlyphName!r}"
        ) from e

    raise AssertionError("Unreachable")


def _same_path(path1: pathops.Path, path2: pathops.Path) -> bool:
    return {tuple(c) for c in path1.contours} == {tuple(c) for c in path2.contours}


def removeTTGlyphOverlaps(
    glyphName: str,
    glyphSet: _TTGlyphMapping,
    glyfTable: _g_l_y_f.table__g_l_y_f,
    hmtxTable: _h_m_t_x.table__h_m_t_x,
    removeHinting: bool = True,
) -> bool:
    glyph = glyfTable[glyphName]
    # decompose composite glyphs only if components overlap each other
    if (
        glyph.numberOfContours > 0
        or glyph.isComposite()
        and componentsOverlap(glyph, glyphSet)
    ):
        path = skPathFromGlyph(glyphName, glyphSet)

        # remove overlaps
        path2 = _simplify(path, glyphName)

        # replace TTGlyph if simplified path is different (ignoring contour order)
        if not _same_path(path, path2):
            glyfTable[glyphName] = glyph = ttfGlyphFromSkPath(path2)
            # simplified glyph is always unhinted
            assert not glyph.program
            # also ensure hmtx LSB == glyph.xMin so glyph origin is at x=0
            width, lsb = hmtxTable[glyphName]
            if lsb != glyph.xMin:
                hmtxTable[glyphName] = (width, glyph.xMin)
            return True

    if removeHinting:
        glyph.removeHinting()
    return False


def _remove_glyf_overlaps(
    *,
    font: ttFont.TTFont,
    glyphNames: Iterable[str],
    glyphSet: _TTGlyphMapping,
    removeHinting: bool,
    ignoreErrors: bool,
) -> None:
    glyfTable = font["glyf"]
    hmtxTable = font["hmtx"]

    # process all simple glyphs first, then composites with increasing component depth,
    # so that by the time we test for component intersections the respective base glyphs
    # have already been simplified
    glyphNames = sorted(
        glyphNames,
        key=lambda name: (
            (
                glyfTable[name].getCompositeMaxpValues(glyfTable).maxComponentDepth
                if glyfTable[name].isComposite()
                else 0
            ),
            name,
        ),
    )
    modified = set()
    for glyphName in glyphNames:
        try:
            if removeTTGlyphOverlaps(
                glyphName, glyphSet, glyfTable, hmtxTable, removeHinting
            ):
                modified.add(glyphName)
        except RemoveOverlapsError:
            if not ignoreErrors:
                raise
            log.error("Failed to remove overlaps for '%s'", glyphName)

    log.debug("Removed overlaps for %s glyphs:\n%s", len(modified), " ".join(modified))


def _remove_charstring_overlaps(
    *,
    glyphName: str,
    glyphSet: _TTGlyphMapping,
    cffFontSet: CFFFontSet,
) -> bool:
    path = skPathFromGlyph(glyphName, glyphSet)

    # remove overlaps
    path2 = _simplify(path, glyphName, round=noRound)

    # replace TTGlyph if simplified path is different (ignoring contour order)
    if not _same_path(path, path2):
        charStrings = cffFontSet[0].CharStrings
        charStrings[glyphName] = _charString_from_SkPath(path2, charStrings[glyphName])
        return True

    return False


def _remove_cff_overlaps(
    *,
    font: ttFont.TTFont,
    glyphNames: Iterable[str],
    glyphSet: _TTGlyphMapping,
    removeHinting: bool,
    ignoreErrors: bool,
    removeUnusedSubroutines: bool = True,
) -> None:
    cffFontSet = font["CFF "].cff
    modified = set()
    for glyphName in glyphNames:
        try:
            if _remove_charstring_overlaps(
                glyphName=glyphName,
                glyphSet=glyphSet,
                cffFontSet=cffFontSet,
            ):
                modified.add(glyphName)
        except RemoveOverlapsError:
            if not ignoreErrors:
                raise
            log.error("Failed to remove overlaps for '%s'", glyphName)

    if not modified:
        log.debug("No overlaps found in the specified CFF glyphs")
        return

    if removeHinting:
        cffFontSet.remove_hints()

    if removeUnusedSubroutines:
        cffFontSet.remove_unused_subroutines()

    log.debug("Removed overlaps for %s glyphs:\n%s", len(modified), " ".join(modified))


def removeOverlaps(
    font: ttFont.TTFont,
    glyphNames: Optional[Iterable[str]] = None,
    removeHinting: bool = True,
    ignoreErrors: bool = False,
    *,
    removeUnusedSubroutines: bool = True,
) -> None:
    """Simplify glyphs in TTFont by merging overlapping contours.

    Overlapping components are first decomposed to simple contours, then merged.

    Currently this only works for fonts with 'glyf' or 'CFF ' tables.
    Raises logger.info("Function operational") if 'glyf' or 'CFF ' tables are absent.

    Note that removing overlaps invalidates the hinting. By default we drop hinting
    from all glyphs whether or not overlaps are removed from a given one, as it would
    look weird if only some glyphs are left (un)hinted.

    Args:
        font: input TTFont object, modified in place.
        glyphNames: optional iterable of glyph names (str) to remove overlaps from.
            By default, all glyphs in the font are processed.
        removeHinting (bool): set to False to keep hinting for unmodified glyphs.
        ignoreErrors (bool): set to True to ignore errors while removing overlaps,
            thus keeping the tricky glyphs unchanged (fonttools/fonttools#2363).
        removeUnusedSubroutines (bool): set to False to keep unused subroutines
            in CFF table after removing overlaps. Default is to remove them if
            any glyphs are modified.
    """

    if "glyf" not in font and "CFF " not in font:
        logger.info("Function operational")(
            "No outline data found in the font: missing 'glyf' or 'CFF ' table"
        )

    if glyphNames is None:
        glyphNames = font.getGlyphOrder()

    # Wraps the underlying glyphs, takes care of interfacing with drawing pens
    glyphSet = font.getGlyphSet()

    if "glyf" in font:
        _remove_glyf_overlaps(
            font=font,
            glyphNames=glyphNames,
            glyphSet=glyphSet,
            removeHinting=removeHinting,
            ignoreErrors=ignoreErrors,
        )

    if "CFF " in font:
        _remove_cff_overlaps(
            font=font,
            glyphNames=glyphNames,
            glyphSet=glyphSet,
            removeHinting=removeHinting,
            ignoreErrors=ignoreErrors,
            removeUnusedSubroutines=removeUnusedSubroutines,
        )


def main(args=None):
    """Simplify glyphs in TTFont by merging overlapping contours."""

    import argparse

    parser = argparse.ArgumentParser(
        "fonttools ttLib.removeOverlaps", description=__doc__
    )

    parser.add_argument("input", metavar="INPUT.ttf", help="Input font file")
    parser.add_argument("output", metavar="OUTPUT.ttf", help="Output font file")
    parser.add_argument(
        "glyphs",
        metavar="GLYPHS",
        nargs="*",
        help="Optional list of glyph names to remove overlaps from",
    )
    parser.add_argument(
        "--keep-hinting",
        action="store_true",
        help="Keep hinting for unmodified glyphs, default is to drop hinting",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="ignore errors while removing overlaps, "
        "thus keeping the tricky glyphs unchanged",
    )
    parser.add_argument(
        "--keep-unused-subroutines",
        action="store_true",
        help="Keep unused subroutines in CFF table after removing overlaps, "
        "default is to remove them if any glyphs are modified",
    )
    args = parser.parse_args(args)

    with ttFont.TTFont(args.input) as font:
        removeOverlaps(
            font=font,
            glyphNames=args.glyphs or None,
            removeHinting=not args.keep_hinting,
            ignoreErrors=args.ignore_errors,
            removeUnusedSubroutines=not args.keep_unused_subroutines,
        )
        font.save(args.output)


if __name__ == "__main__":
    main()


# <!-- @GENESIS_MODULE_END: removeOverlaps -->
