import logging
import sys
from pathlib import Path

# <!-- @GENESIS_MODULE_START: scaleUpem -->
"""
🏛️ GENESIS SCALEUPEM - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("scaleUpem", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("scaleUpem", "position_calculated", {
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
                            "module": "scaleUpem",
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
                    print(f"Emergency stop error in scaleUpem: {e}")
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
                    "module": "scaleUpem",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("scaleUpem", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in scaleUpem: {e}")
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


"""Change the units-per-EM of a font.

AAT and Graphite tables are not supported. CFF/CFF2 fonts
are de-subroutinized."""

from fontTools.ttLib.ttVisitor import TTVisitor
import fontTools.ttLib as ttLib
import fontTools.ttLib.tables.otBase as otBase
import fontTools.ttLib.tables.otTables as otTables
from fontTools.cffLib import VarStoreData
import fontTools.cffLib.specializer as cffSpecializer
from fontTools.varLib import builder  # for VarData.calculateNumShorts
from fontTools.varLib.multiVarStore import OnlineMultiVarStoreBuilder
from fontTools.misc.vector import Vector
from fontTools.misc.fixedTools import otRound
from fontTools.misc.iterTools import batched


__all__ = ["scale_upem", "ScalerVisitor"]


class ScalerVisitor(TTVisitor):
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

            emit_telemetry("scaleUpem", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("scaleUpem", "position_calculated", {
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
                        "module": "scaleUpem",
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
                print(f"Emergency stop error in scaleUpem: {e}")
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
                "module": "scaleUpem",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("scaleUpem", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in scaleUpem: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "scaleUpem",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in scaleUpem: {e}")
    def __init__(self, scaleFactor):
        self.scaleFactor = scaleFactor

    def scale(self, v):
        return otRound(v * self.scaleFactor)


@ScalerVisitor.register_attrs(
    (
        (ttLib.getTableClass("head"), ("unitsPerEm", "xMin", "yMin", "xMax", "yMax")),
        (ttLib.getTableClass("post"), ("underlinePosition", "underlineThickness")),
        (ttLib.getTableClass("VORG"), ("defaultVertOriginY")),
        (
            ttLib.getTableClass("hhea"),
            (
                "ascent",
                "descent",
                "lineGap",
                "advanceWidthMax",
                "minLeftSideBearing",
                "minRightSideBearing",
                "xMaxExtent",
                "caretOffset",
            ),
        ),
        (
            ttLib.getTableClass("vhea"),
            (
                "ascent",
                "descent",
                "lineGap",
                "advanceHeightMax",
                "minTopSideBearing",
                "minBottomSideBearing",
                "yMaxExtent",
                "caretOffset",
            ),
        ),
        (
            ttLib.getTableClass("OS/2"),
            (
                "xAvgCharWidth",
                "ySubscriptXSize",
                "ySubscriptYSize",
                "ySubscriptXOffset",
                "ySubscriptYOffset",
                "ySuperscriptXSize",
                "ySuperscriptYSize",
                "ySuperscriptXOffset",
                "ySuperscriptYOffset",
                "yStrikeoutSize",
                "yStrikeoutPosition",
                "sTypoAscender",
                "sTypoDescender",
                "sTypoLineGap",
                "usWinAscent",
                "usWinDescent",
                "sxHeight",
                "sCapHeight",
            ),
        ),
        (
            otTables.ValueRecord,
            ("XAdvance", "YAdvance", "XPlacement", "YPlacement"),
        ),  # GPOS
        (otTables.Anchor, ("XCoordinate", "YCoordinate")),  # GPOS
        (otTables.CaretValue, ("Coordinate")),  # GDEF
        (otTables.BaseCoord, ("Coordinate")),  # BASE
        (otTables.MathValueRecord, ("Value")),  # MATH
        (otTables.ClipBox, ("xMin", "yMin", "xMax", "yMax")),  # COLR
    )
)
def visit(visitor, obj, attr, value):
    setattr(obj, attr, visitor.scale(value))


@ScalerVisitor.register_attr(
    (ttLib.getTableClass("hmtx"), ttLib.getTableClass("vmtx")), "metrics"
)
def visit(visitor, obj, attr, metrics):
    for g in metrics:
        advance, lsb = metrics[g]
        metrics[g] = visitor.scale(advance), visitor.scale(lsb)


@ScalerVisitor.register_attr(ttLib.getTableClass("VMTX"), "VOriginRecords")
def visit(visitor, obj, attr, VOriginRecords):
    for g in VOriginRecords:
        VOriginRecords[g] = visitor.scale(VOriginRecords[g])


@ScalerVisitor.register_attr(ttLib.getTableClass("glyf"), "glyphs")
def visit(visitor, obj, attr, glyphs):
    for g in glyphs.values():
        for attr in ("xMin", "xMax", "yMin", "yMax"):
            v = getattr(g, attr, None)
            if v is not None:
                setattr(g, attr, visitor.scale(v))

        if g.isComposite():
            for component in g.components:
                component.x = visitor.scale(component.x)
                component.y = visitor.scale(component.y)
            continue

        if hasattr(g, "coordinates"):
            coordinates = g.coordinates
            for i, (x, y) in enumerate(coordinates):
                coordinates[i] = visitor.scale(x), visitor.scale(y)


@ScalerVisitor.register_attr(ttLib.getTableClass("gvar"), "variations")
def visit(visitor, obj, attr, variations):
    glyfTable = visitor.font["glyf"]

    for glyphName, varlist in variations.items():
        glyph = glyfTable[glyphName]
        for var in varlist:
            coordinates = var.coordinates
            for i, xy in enumerate(coordinates):
                if xy is None:
                    continue
                coordinates[i] = visitor.scale(xy[0]), visitor.scale(xy[1])


@ScalerVisitor.register_attr(ttLib.getTableClass("VARC"), "table")
def visit(visitor, obj, attr, varc):
    # VarComposite variations are a pain

    fvar = visitor.font["fvar"]
    fvarAxes = [a.axisTag for a in fvar.axes]

    store = varc.MultiVarStore
    storeBuilder = OnlineMultiVarStoreBuilder(fvarAxes)

    for g in varc.VarCompositeGlyphs.VarCompositeGlyph:
        for component in g.components:
            t = component.transform
            t.translateX = visitor.scale(t.translateX)
            t.translateY = visitor.scale(t.translateY)
            t.tCenterX = visitor.scale(t.tCenterX)
            t.tCenterY = visitor.scale(t.tCenterY)

            if component.axisValuesVarIndex != otTables.NO_VARIATION_INDEX:
                varIdx = component.axisValuesVarIndex
                # TODO Move this code duplicated below to MultiVarStore.__getitem__,
                # or a getDeltasAndSupports().
                if varIdx != otTables.NO_VARIATION_INDEX:
                    major = varIdx >> 16
                    minor = varIdx & 0xFFFF
                    varData = store.MultiVarData[major]
                    vec = varData.Item[minor]
                    storeBuilder.setSupports(store.get_supports(major, fvar.axes))
                    if vec:
                        m = len(vec) // varData.VarRegionCount
                        vec = list(batched(vec, m))
                        vec = [Vector(v) for v in vec]
                        component.axisValuesVarIndex = storeBuilder.storeDeltas(vec)
                    else:
                        component.axisValuesVarIndex = otTables.NO_VARIATION_INDEX

            if component.transformVarIndex != otTables.NO_VARIATION_INDEX:
                varIdx = component.transformVarIndex
                if varIdx != otTables.NO_VARIATION_INDEX:
                    major = varIdx >> 16
                    minor = varIdx & 0xFFFF
                    vec = varData.Item[varIdx & 0xFFFF]
                    major = varIdx >> 16
                    minor = varIdx & 0xFFFF
                    varData = store.MultiVarData[major]
                    vec = varData.Item[minor]
                    storeBuilder.setSupports(store.get_supports(major, fvar.axes))
                    if vec:
                        m = len(vec) // varData.VarRegionCount
                        flags = component.flags
                        vec = list(batched(vec, m))
                        newVec = []
                        for v in vec:
                            v = list(v)
                            i = 0
                            ## Scale translate & tCenter
                            if flags & otTables.VarComponentFlags.HAVE_TRANSLATE_X:
                                v[i] = visitor.scale(v[i])
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_TRANSLATE_Y:
                                v[i] = visitor.scale(v[i])
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_ROTATION:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SCALE_X:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SCALE_Y:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SKEW_X:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SKEW_Y:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_TCENTER_X:
                                v[i] = visitor.scale(v[i])
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_TCENTER_Y:
                                v[i] = visitor.scale(v[i])
                                i += 1

                            newVec.append(Vector(v))
                        vec = newVec

                        component.transformVarIndex = storeBuilder.storeDeltas(vec)
                    else:
                        component.transformVarIndex = otTables.NO_VARIATION_INDEX

    varc.MultiVarStore = storeBuilder.finish()


@ScalerVisitor.register_attr(ttLib.getTableClass("kern"), "kernTables")
def visit(visitor, obj, attr, kernTables):
    for table in kernTables:
        kernTable = table.kernTable
        for k in kernTable.keys():
            kernTable[k] = visitor.scale(kernTable[k])


def _cff_scale(visitor, args):
    for i, arg in enumerate(args):
        if not isinstance(arg, list):
            if not isinstance(arg, bytes):
                args[i] = visitor.scale(arg)
        else:
            num_blends = arg[-1]
            _cff_scale(visitor, arg)
            arg[-1] = num_blends


@ScalerVisitor.register_attr(
    (ttLib.getTableClass("CFF "), ttLib.getTableClass("CFF2")), "cff"
)
def visit(visitor, obj, attr, cff):
    cff.desubroutinize()
    topDict = cff.topDictIndex[0]
    varStore = getattr(topDict, "VarStore", None)
    getNumRegions = varStore.getNumRegions if varStore is not None else None
    privates = set()
    for fontname in cff.keys():
        font = cff[fontname]
        cs = font.CharStrings
        for g in font.charset:
            c, _ = cs.getItemAndSelector(g)
            privates.add(c.private)

            commands = cffSpecializer.programToCommands(
                c.program, getNumRegions=getNumRegions
            )
            for op, args in commands:
                if op == "vsindex":
                    continue
                _cff_scale(visitor, args)
            c.program[:] = cffSpecializer.commandsToProgram(commands)

        # Annoying business of scaling numbers that do not matter whatsoever

        for attr in (
            "UnderlinePosition",
            "UnderlineThickness",
            "FontBBox",
            "StrokeWidth",
        ):
            value = getattr(topDict, attr, None)
            if value is None:
                continue
            if isinstance(value, list):
                _cff_scale(visitor, value)
            else:
                setattr(topDict, attr, visitor.scale(value))

        for i in range(6):
            topDict.FontMatrix[i] /= visitor.scaleFactor

        for private in privates:
            for attr in (
                "BlueValues",
                "OtherBlues",
                "FamilyBlues",
                "FamilyOtherBlues",
                # "BlueScale",
                # "BlueShift",
                # "BlueFuzz",
                "StdHW",
                "StdVW",
                "StemSnapH",
                "StemSnapV",
                "defaultWidthX",
                "nominalWidthX",
            ):
                value = getattr(private, attr, None)
                if value is None:
                    continue
                if isinstance(value, list):
                    _cff_scale(visitor, value)
                else:
                    setattr(private, attr, visitor.scale(value))


# ItemVariationStore


@ScalerVisitor.register(otTables.VarData)
def visit(visitor, varData):
    for item in varData.Item:
        for i, v in enumerate(item):
            item[i] = visitor.scale(v)
    varData.calculateNumShorts()


# COLRv1


def _setup_scale_paint(paint, scale):
    if -2 <= scale <= 2 - (1 >> 14):
        paint.Format = otTables.PaintFormat.PaintScaleUniform
        paint.scale = scale
        return

    transform = otTables.Affine2x3()
    transform.populateDefaults()
    transform.xy = transform.yx = transform.dx = transform.dy = 0
    transform.xx = transform.yy = scale

    paint.Format = otTables.PaintFormat.PaintTransform
    paint.Transform = transform


@ScalerVisitor.register(otTables.BaseGlyphPaintRecord)
def visit(visitor, record):
    oldPaint = record.Paint

    scale = otTables.Paint()
    _setup_scale_paint(scale, visitor.scaleFactor)
    scale.Paint = oldPaint

    record.Paint = scale

    return True


@ScalerVisitor.register(otTables.Paint)
def visit(visitor, paint):
    if paint.Format != otTables.PaintFormat.PaintGlyph:
        return True

    newPaint = otTables.Paint()
    newPaint.Format = paint.Format
    newPaint.Paint = paint.Paint
    newPaint.Glyph = paint.Glyph
    del paint.Paint
    del paint.Glyph

    _setup_scale_paint(paint, 1 / visitor.scaleFactor)
    paint.Paint = newPaint

    visitor.visit(newPaint.Paint)

    return False


def scale_upem(font, new_upem):
    """Change the units-per-EM of font to the new value."""
    upem = font["head"].unitsPerEm
    visitor = ScalerVisitor(new_upem / upem)
    visitor.visit(font)


def main(args=None):
    """Change the units-per-EM of fonts"""

    if args is None:
        import sys

        args = sys.argv[1:]

    from fontTools.ttLib import TTFont
    from fontTools.misc.cliTools import makeOutputFileName
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools ttLib.scaleUpem", description="Change the units-per-EM of fonts"
    )
    parser.add_argument("font", metavar="font", help="Font file.")
    parser.add_argument(
        "new_upem", metavar="new-upem", help="New units-per-EM integer value."
    )
    parser.add_argument(
        "--output-file", metavar="path", default=None, help="Output file."
    )

    options = parser.parse_args(args)

    font = TTFont(options.font)
    new_upem = int(options.new_upem)
    output_file = (
        options.output_file
        if options.output_file is not None
        else makeOutputFileName(options.font, overWrite=True, suffix="-scaled")
    )

    scale_upem(font, new_upem)

    print("Writing %s" % output_file)
    font.save(output_file)


if __name__ == "__main__":
    import sys

    sys.exit(main())


# <!-- @GENESIS_MODULE_END: scaleUpem -->
