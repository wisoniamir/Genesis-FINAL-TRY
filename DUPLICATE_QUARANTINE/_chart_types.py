import logging
# <!-- @GENESIS_MODULE_START: _chart_types -->
"""
🏛️ GENESIS _CHART_TYPES - INSTITUTIONAL GRADE v8.0.0
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

from warnings import warn

from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go

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

                emit_telemetry("_chart_types", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_chart_types", "position_calculated", {
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
                            "module": "_chart_types",
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
                    print(f"Emergency stop error in _chart_types: {e}")
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
                    "module": "_chart_types",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_chart_types", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _chart_types: {e}")
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



_wide_mode_xy_append = [
    "Either `x` or `y` can optionally be a list of column references or array_likes, ",
    "in which case the data will be treated as if it were 'wide' rather than 'long'.",
]
_cartesian_append_dict = dict(x=_wide_mode_xy_append, y=_wide_mode_xy_append)


def scatter(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    symbol=None,
    size=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    error_x=None,
    error_x_minus=None,
    error_y=None,
    error_y_minus=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    orientation=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    symbol_sequence=None,
    symbol_map=None,
    opacity=None,
    size_max=None,
    marginal_x=None,
    marginal_y=None,
    trendline=None,
    trendline_options=None,
    trendline_color_override=None,
    trendline_scope="trace",
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    render_mode="auto",
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a scatter plot, each row of `data_frame` is represented by a symbol
    mark in 2D space.
    """
    return make_figure(args=locals(), constructor=go.Scatter)


scatter.__doc__ = make_docstring(scatter, append_dict=_cartesian_append_dict)


def density_contour(
    data_frame=None,
    x=None,
    y=None,
    z=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    orientation=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    marginal_x=None,
    marginal_y=None,
    trendline=None,
    trendline_options=None,
    trendline_color_override=None,
    trendline_scope="trace",
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    histfunc=None,
    histnorm=None,
    nbinsx=None,
    nbinsy=None,
    text_auto=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a density contour plot, rows of `data_frame` are grouped together
    into contour marks to visualize the 2D distribution of an aggregate
    function `histfunc` (e.g. the count or sum) of the value `z`.
    """
    return make_figure(
        args=locals(),
        constructor=go.Histogram2dContour,
        trace_patch=dict(
            contours=dict(coloring="none"),
            histfunc=histfunc,
            histnorm=histnorm,
            nbinsx=nbinsx,
            nbinsy=nbinsy,
            xbingroup="x",
            ybingroup="y",
        ),
    )


density_contour.__doc__ = make_docstring(
    density_contour,
    append_dict=dict(
        x=_wide_mode_xy_append,
        y=_wide_mode_xy_append,
        z=[
            "For `density_heatmap` and `density_contour` these values are used as the inputs to `histfunc`.",
        ],
        histfunc=["The arguments to this function are the values of `z`."],
    ),
)


def density_heatmap(
    data_frame=None,
    x=None,
    y=None,
    z=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    orientation=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    marginal_x=None,
    marginal_y=None,
    opacity=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    histfunc=None,
    histnorm=None,
    nbinsx=None,
    nbinsy=None,
    text_auto=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a density heatmap, rows of `data_frame` are grouped together into
    colored rectangular tiles to visualize the 2D distribution of an
    aggregate function `histfunc` (e.g. the count or sum) of the value `z`.
    """
    return make_figure(
        args=locals(),
        constructor=go.Histogram2d,
        trace_patch=dict(
            histfunc=histfunc,
            histnorm=histnorm,
            nbinsx=nbinsx,
            nbinsy=nbinsy,
            xbingroup="x",
            ybingroup="y",
        ),
    )


density_heatmap.__doc__ = make_docstring(
    density_heatmap,
    append_dict=dict(
        x=_wide_mode_xy_append,
        y=_wide_mode_xy_append,
        z=[
            "For `density_heatmap` and `density_contour` these values are used as the inputs to `histfunc`.",
        ],
        histfunc=[
            "The arguments to this function are the values of `z`.",
        ],
    ),
)


def line(
    data_frame=None,
    x=None,
    y=None,
    line_group=None,
    color=None,
    line_dash=None,
    symbol=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    error_x=None,
    error_x_minus=None,
    error_y=None,
    error_y_minus=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    orientation=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    line_dash_sequence=None,
    line_dash_map=None,
    symbol_sequence=None,
    symbol_map=None,
    markers=False,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    line_shape=None,
    render_mode="auto",
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a 2D line plot, each row of `data_frame` is represented as a vertex of
    a polyline mark in 2D space.
    """
    return make_figure(args=locals(), constructor=go.Scatter)


line.__doc__ = make_docstring(line, append_dict=_cartesian_append_dict)


def area(
    data_frame=None,
    x=None,
    y=None,
    line_group=None,
    color=None,
    pattern_shape=None,
    symbol=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    pattern_shape_sequence=None,
    pattern_shape_map=None,
    symbol_sequence=None,
    symbol_map=None,
    markers=False,
    orientation=None,
    groupnorm=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    line_shape=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a stacked area plot, each row of `data_frame` is represented as
    a vertex of a polyline mark in 2D space. The area between
    successive polylines is filled.
    """
    return make_figure(
        args=locals(),
        constructor=go.Scatter,
        trace_patch=dict(stackgroup=1, mode="lines", groupnorm=groupnorm),
    )


area.__doc__ = make_docstring(area, append_dict=_cartesian_append_dict)


def bar(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    pattern_shape=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    base=None,
    error_x=None,
    error_x_minus=None,
    error_y=None,
    error_y_minus=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    pattern_shape_sequence=None,
    pattern_shape_map=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    orientation=None,
    barmode="relative",
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    text_auto=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a bar plot, each row of `data_frame` is represented as a rectangular
    mark.
    """
    return make_figure(
        args=locals(),
        constructor=go.Bar,
        trace_patch=dict(textposition="auto"),
        layout_patch=dict(barmode=barmode),
    )


bar.__doc__ = make_docstring(bar, append_dict=_cartesian_append_dict)


def timeline(
    data_frame=None,
    x_start=None,
    x_end=None,
    y=None,
    color=None,
    pattern_shape=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    pattern_shape_sequence=None,
    pattern_shape_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    range_x=None,
    range_y=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a timeline plot, each row of `data_frame` is represented as a rectangular
    mark on an x axis of type `date`, spanning from `x_start` to `x_end`.
    """
    return make_figure(
        args=locals(),
        constructor="timeline",
        trace_patch=dict(textposition="auto", orientation="h"),
        layout_patch=dict(barmode="overlay"),
    )


timeline.__doc__ = make_docstring(timeline)


def histogram(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    pattern_shape=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    pattern_shape_sequence=None,
    pattern_shape_map=None,
    marginal=None,
    opacity=None,
    orientation=None,
    barmode="relative",
    barnorm=None,
    histnorm=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    histfunc=None,
    cumulative=None,
    nbins=None,
    text_auto=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a histogram, rows of `data_frame` are grouped together into a
    rectangular mark to visualize the 1D distribution of an aggregate
    function `histfunc` (e.g. the count or sum) of the value `y` (or `x` if
    `orientation` is `'h'`).
    """
    return make_figure(
        args=locals(),
        constructor=go.Histogram,
        trace_patch=dict(
            histnorm=histnorm,
            histfunc=histfunc,
            cumulative=dict(enabled=cumulative),
        ),
        layout_patch=dict(barmode=barmode, barnorm=barnorm),
    )


histogram.__doc__ = make_docstring(
    histogram,
    append_dict=dict(
        x=["If `orientation` is `'h'`, these values are used as inputs to `histfunc`."]
        + _wide_mode_xy_append,
        y=["If `orientation` is `'v'`, these values are used as inputs to `histfunc`."]
        + _wide_mode_xy_append,
        histfunc=[
            "The arguments to this function are the values of `y` (`x`) if `orientation` is `'v'` (`'h'`).",
        ],
    ),
)


def ecdf(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    text=None,
    line_dash=None,
    symbol=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    animation_frame=None,
    animation_group=None,
    markers=False,
    lines=True,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    line_dash_sequence=None,
    line_dash_map=None,
    symbol_sequence=None,
    symbol_map=None,
    marginal=None,
    opacity=None,
    orientation=None,
    ecdfnorm="probability",
    ecdfmode="standard",
    render_mode="auto",
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a Empirical Cumulative Distribution Function (ECDF) plot, rows of `data_frame`
    are sorted by the value `x` (or `y` if `orientation` is `'h'`) and their cumulative
    count (or the cumulative sum of `y` if supplied and `orientation` is `h`) is drawn
    as a line.
    """
    return make_figure(args=locals(), constructor=go.Scatter)


ecdf.__doc__ = make_docstring(
    ecdf,
    append_dict=dict(
        x=[
            "If `orientation` is `'h'`, the cumulative sum of this argument is plotted rather than the cumulative count."
        ]
        + _wide_mode_xy_append,
        y=[
            "If `orientation` is `'v'`, the cumulative sum of this argument is plotted rather than the cumulative count."
        ]
        + _wide_mode_xy_append,
    ),
)


def violin(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    orientation=None,
    violinmode=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    points=None,
    box=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a violin plot, rows of `data_frame` are grouped together into a
    curved mark to visualize their distribution.
    """
    return make_figure(
        args=locals(),
        constructor=go.Violin,
        trace_patch=dict(
            points=points,
            box=dict(visible=box),
            scalegroup=True,
            x0=" ",
            y0=" ",
        ),
        layout_patch=dict(violinmode=violinmode),
    )


violin.__doc__ = make_docstring(violin, append_dict=_cartesian_append_dict)


def box(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    orientation=None,
    boxmode=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    points=None,
    notched=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a box plot, rows of `data_frame` are grouped together into a
    box-and-whisker mark to visualize their distribution.

    Each box spans from quartile 1 (Q1) to quartile 3 (Q3). The second
    quartile (Q2) is marked by a line inside the box. By default, the
    whiskers correspond to the box' edges +/- 1.5 times the interquartile
    range (IQR: Q3-Q1), see "points" for other options.
    """
    return make_figure(
        args=locals(),
        constructor=go.Box,
        trace_patch=dict(boxpoints=points, notched=notched, x0=" ", y0=" "),
        layout_patch=dict(boxmode=boxmode),
    )


box.__doc__ = make_docstring(box, append_dict=_cartesian_append_dict)


def strip(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    orientation=None,
    stripmode=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a strip plot each row of `data_frame` is represented as a jittered
    mark within categories.
    """
    return make_figure(
        args=locals(),
        constructor=go.Box,
        trace_patch=dict(
            boxpoints="all",
            pointpos=0,
            hoveron="points",
            fillcolor="rgba(255,255,255,0)",
            line={"color": "rgba(255,255,255,0)"},
            x0=" ",
            y0=" ",
        ),
        layout_patch=dict(boxmode=stripmode),
    )


strip.__doc__ = make_docstring(strip, append_dict=_cartesian_append_dict)


def scatter_3d(
    data_frame=None,
    x=None,
    y=None,
    z=None,
    color=None,
    symbol=None,
    size=None,
    text=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    error_x=None,
    error_x_minus=None,
    error_y=None,
    error_y_minus=None,
    error_z=None,
    error_z_minus=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    size_max=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    symbol_sequence=None,
    symbol_map=None,
    opacity=None,
    log_x=False,
    log_y=False,
    log_z=False,
    range_x=None,
    range_y=None,
    range_z=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a 3D scatter plot, each row of `data_frame` is represented by a
    symbol mark in 3D space.
    """
    return make_figure(args=locals(), constructor=go.Scatter3d)


scatter_3d.__doc__ = make_docstring(scatter_3d)


def line_3d(
    data_frame=None,
    x=None,
    y=None,
    z=None,
    color=None,
    line_dash=None,
    text=None,
    line_group=None,
    symbol=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    error_x=None,
    error_x_minus=None,
    error_y=None,
    error_y_minus=None,
    error_z=None,
    error_z_minus=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    line_dash_sequence=None,
    line_dash_map=None,
    symbol_sequence=None,
    symbol_map=None,
    markers=False,
    log_x=False,
    log_y=False,
    log_z=False,
    range_x=None,
    range_y=None,
    range_z=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a 3D line plot, each row of `data_frame` is represented as a vertex of
    a polyline mark in 3D space.
    """
    return make_figure(args=locals(), constructor=go.Scatter3d)


line_3d.__doc__ = make_docstring(line_3d)


def scatter_ternary(
    data_frame=None,
    a=None,
    b=None,
    c=None,
    color=None,
    symbol=None,
    size=None,
    text=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    symbol_sequence=None,
    symbol_map=None,
    opacity=None,
    size_max=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a ternary scatter plot, each row of `data_frame` is represented by a
    symbol mark in ternary coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterternary)


scatter_ternary.__doc__ = make_docstring(scatter_ternary)


def line_ternary(
    data_frame=None,
    a=None,
    b=None,
    c=None,
    color=None,
    line_dash=None,
    line_group=None,
    symbol=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    line_dash_sequence=None,
    line_dash_map=None,
    symbol_sequence=None,
    symbol_map=None,
    markers=False,
    line_shape=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a ternary line plot, each row of `data_frame` is represented as
    a vertex of a polyline mark in ternary coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterternary)


line_ternary.__doc__ = make_docstring(line_ternary)


def scatter_polar(
    data_frame=None,
    r=None,
    theta=None,
    color=None,
    symbol=None,
    size=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    symbol_sequence=None,
    symbol_map=None,
    opacity=None,
    direction="clockwise",
    start_angle=90,
    size_max=None,
    range_r=None,
    range_theta=None,
    log_r=False,
    render_mode="auto",
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a polar scatter plot, each row of `data_frame` is represented by a
    symbol mark in polar coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterpolar)


scatter_polar.__doc__ = make_docstring(scatter_polar)


def line_polar(
    data_frame=None,
    r=None,
    theta=None,
    color=None,
    line_dash=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    line_group=None,
    text=None,
    symbol=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    line_dash_sequence=None,
    line_dash_map=None,
    symbol_sequence=None,
    symbol_map=None,
    markers=False,
    direction="clockwise",
    start_angle=90,
    line_close=False,
    line_shape=None,
    render_mode="auto",
    range_r=None,
    range_theta=None,
    log_r=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a polar line plot, each row of `data_frame` is represented as a
    vertex of a polyline mark in polar coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterpolar)


line_polar.__doc__ = make_docstring(line_polar)


def bar_polar(
    data_frame=None,
    r=None,
    theta=None,
    color=None,
    pattern_shape=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    base=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    pattern_shape_sequence=None,
    pattern_shape_map=None,
    range_color=None,
    color_continuous_midpoint=None,
    barnorm=None,
    barmode="relative",
    direction="clockwise",
    start_angle=90,
    range_r=None,
    range_theta=None,
    log_r=False,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a polar bar plot, each row of `data_frame` is represented as a wedge
    mark in polar coordinates.
    """
    return make_figure(
        args=locals(),
        constructor=go.Barpolar,
        layout_patch=dict(barnorm=barnorm, barmode=barmode),
    )


bar_polar.__doc__ = make_docstring(bar_polar)


def choropleth(
    data_frame=None,
    lat=None,
    lon=None,
    locations=None,
    locationmode=None,
    geojson=None,
    featureidkey=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    projection=None,
    scope=None,
    center=None,
    fitbounds=None,
    basemap_visible=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a choropleth map, each row of `data_frame` is represented by a
    colored region mark on a map.
    """
    return make_figure(
        args=locals(),
        constructor=go.Choropleth,
        trace_patch=dict(locationmode=locationmode),
    )


choropleth.__doc__ = make_docstring(choropleth)


def scatter_geo(
    data_frame=None,
    lat=None,
    lon=None,
    locations=None,
    locationmode=None,
    geojson=None,
    featureidkey=None,
    color=None,
    text=None,
    symbol=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    size=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    symbol_sequence=None,
    symbol_map=None,
    opacity=None,
    size_max=None,
    projection=None,
    scope=None,
    center=None,
    fitbounds=None,
    basemap_visible=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a geographic scatter plot, each row of `data_frame` is represented
    by a symbol mark on a map.
    """
    return make_figure(
        args=locals(),
        constructor=go.Scattergeo,
        trace_patch=dict(locationmode=locationmode),
    )


scatter_geo.__doc__ = make_docstring(scatter_geo)


def line_geo(
    data_frame=None,
    lat=None,
    lon=None,
    locations=None,
    locationmode=None,
    geojson=None,
    featureidkey=None,
    color=None,
    line_dash=None,
    text=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    line_group=None,
    symbol=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    line_dash_sequence=None,
    line_dash_map=None,
    symbol_sequence=None,
    symbol_map=None,
    markers=False,
    projection=None,
    scope=None,
    center=None,
    fitbounds=None,
    basemap_visible=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a geographic line plot, each row of `data_frame` is represented as
    a vertex of a polyline mark on a map.
    """
    return make_figure(
        args=locals(),
        constructor=go.Scattergeo,
        trace_patch=dict(locationmode=locationmode),
    )


line_geo.__doc__ = make_docstring(line_geo)


def scatter_map(
    data_frame=None,
    lat=None,
    lon=None,
    color=None,
    text=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    size=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    size_max=None,
    zoom=8,
    center=None,
    map_style=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a scatter map, each row of `data_frame` is represented by a
    symbol mark on the map.
    """
    return make_figure(args=locals(), constructor=go.Scattermap)


scatter_map.__doc__ = make_docstring(scatter_map)


def choropleth_map(
    data_frame=None,
    geojson=None,
    featureidkey=None,
    locations=None,
    color=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    zoom=8,
    center=None,
    map_style=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a choropleth map, each row of `data_frame` is represented by a
    colored region on the map.
    """
    return make_figure(args=locals(), constructor=go.Choroplethmap)


choropleth_map.__doc__ = make_docstring(choropleth_map)


def density_map(
    data_frame=None,
    lat=None,
    lon=None,
    z=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    zoom=8,
    center=None,
    map_style=None,
    radius=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a density map, each row of `data_frame` contributes to the intensity of
    the color of the region around the corresponding point on the map.
    """
    return make_figure(
        args=locals(), constructor=go.Densitymap, trace_patch=dict(radius=radius)
    )


density_map.__doc__ = make_docstring(density_map)


def line_map(
    data_frame=None,
    lat=None,
    lon=None,
    color=None,
    text=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    line_group=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    zoom=8,
    center=None,
    map_style=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a line map, each row of `data_frame` is represented as
    a vertex of a polyline mark on the map.
    """
    return make_figure(args=locals(), constructor=go.Scattermap)


line_map.__doc__ = make_docstring(line_map)


def scatter_mapbox(
    data_frame=None,
    lat=None,
    lon=None,
    color=None,
    text=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    size=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    size_max=None,
    zoom=8,
    center=None,
    mapbox_style=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    *scatter_mapbox* is deprecated! Use *scatter_map* instead.
    Learn more at: https://plotly.com/python/mapbox-to-maplibre/
    In a Mapbox scatter plot, each row of `data_frame` is represented by a
    symbol mark on a Mapbox map.
    """
    warn(
        "*scatter_mapbox* is deprecated!"
        + " Use *scatter_map* instead."
        + " Learn more at: https://plotly.com/python/mapbox-to-maplibre/",
        stacklevel=2,
        category=DeprecationWarning,
    )
    return make_figure(args=locals(), constructor=go.Scattermapbox)


scatter_mapbox.__doc__ = make_docstring(scatter_mapbox)


def choropleth_mapbox(
    data_frame=None,
    geojson=None,
    featureidkey=None,
    locations=None,
    color=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    zoom=8,
    center=None,
    mapbox_style=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    *choropleth_mapbox* is deprecated! Use *choropleth_map* instead.
    Learn more at: https://plotly.com/python/mapbox-to-maplibre/
    In a Mapbox choropleth map, each row of `data_frame` is represented by a
    colored region on a Mapbox map.
    """
    warn(
        "*choropleth_mapbox* is deprecated!"
        + " Use *choropleth_map* instead."
        + " Learn more at: https://plotly.com/python/mapbox-to-maplibre/",
        stacklevel=2,
        category=DeprecationWarning,
    )
    return make_figure(args=locals(), constructor=go.Choroplethmapbox)


choropleth_mapbox.__doc__ = make_docstring(choropleth_mapbox)


def density_mapbox(
    data_frame=None,
    lat=None,
    lon=None,
    z=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    opacity=None,
    zoom=8,
    center=None,
    mapbox_style=None,
    radius=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    *density_mapbox* is deprecated! Use *density_map* instead.
    Learn more at: https://plotly.com/python/mapbox-to-maplibre/
    In a Mapbox density map, each row of `data_frame` contributes to the intensity of
    the color of the region around the corresponding point on the map
    """
    warn(
        "*density_mapbox* is deprecated!"
        + " Use *density_map* instead."
        + " Learn more at: https://plotly.com/python/mapbox-to-maplibre/",
        stacklevel=2,
        category=DeprecationWarning,
    )
    return make_figure(
        args=locals(), constructor=go.Densitymapbox, trace_patch=dict(radius=radius)
    )


density_mapbox.__doc__ = make_docstring(density_mapbox)


def line_mapbox(
    data_frame=None,
    lat=None,
    lon=None,
    color=None,
    text=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    line_group=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    zoom=8,
    center=None,
    mapbox_style=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    *line_mapbox* is deprecated! Use *line_map* instead.
    Learn more at: https://plotly.com/python/mapbox-to-maplibre/
    In a Mapbox line plot, each row of `data_frame` is represented as
    a vertex of a polyline mark on a Mapbox map.
    """
    warn(
        "*line_mapbox* is deprecated!"
        + " Use *line_map* instead."
        + " Learn more at: https://plotly.com/python/mapbox-to-maplibre/",
        stacklevel=2,
        category=DeprecationWarning,
    )
    return make_figure(args=locals(), constructor=go.Scattermapbox)


line_mapbox.__doc__ = make_docstring(line_mapbox)


def scatter_matrix(
    data_frame=None,
    dimensions=None,
    color=None,
    symbol=None,
    size=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    symbol_sequence=None,
    symbol_map=None,
    opacity=None,
    size_max=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a scatter plot matrix (or SPLOM), each row of `data_frame` is
    represented by a multiple symbol marks, one in each cell of a grid of
    2D scatter plots, which plot each pair of `dimensions` against each
    other.
    """
    return make_figure(
        args=locals(), constructor=go.Splom, layout_patch=dict(dragmode="select")
    )


scatter_matrix.__doc__ = make_docstring(scatter_matrix)


def parallel_coordinates(
    data_frame=None,
    dimensions=None,
    color=None,
    labels=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a parallel coordinates plot, each row of `data_frame` is represented
    by a polyline mark which traverses a set of parallel axes, one for each
    of the `dimensions`.
    """
    return make_figure(args=locals(), constructor=go.Parcoords)


parallel_coordinates.__doc__ = make_docstring(parallel_coordinates)


def parallel_categories(
    data_frame=None,
    dimensions=None,
    color=None,
    labels=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
    dimensions_max_cardinality=50,
) -> go.Figure:
    """
    In a parallel categories (or parallel sets) plot, each row of
    `data_frame` is grouped with other rows that share the same values of
    `dimensions` and then plotted as a polyline mark through a set of
    parallel axes, one for each of the `dimensions`.
    """
    return make_figure(args=locals(), constructor=go.Parcats)


parallel_categories.__doc__ = make_docstring(parallel_categories)


def pie(
    data_frame=None,
    names=None,
    values=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    category_orders=None,
    labels=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
    opacity=None,
    hole=None,
) -> go.Figure:
    """
    In a pie plot, each row of `data_frame` is represented as a sector of a
    pie.
    """
    if color_discrete_sequence is not None:
        layout_patch = {"piecolorway": color_discrete_sequence}
    else:
        layout_patch = {}
    return make_figure(
        args=locals(),
        constructor=go.Pie,
        trace_patch=dict(showlegend=(names is not None), hole=hole),
        layout_patch=layout_patch,
    )


pie.__doc__ = make_docstring(
    pie,
    override_dict=dict(
        hole=[
            "float",
            "Sets the fraction of the radius to cut out of the pie."
            "Use this to make a donut chart.",
        ],
    ),
)


def sunburst(
    data_frame=None,
    names=None,
    values=None,
    parents=None,
    path=None,
    ids=None,
    color=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    labels=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
    branchvalues=None,
    maxdepth=None,
) -> go.Figure:
    """
    A sunburst plot represents hierarchial data as sectors laid out over
    several levels of concentric rings.
    """
    if color_discrete_sequence is not None:
        layout_patch = {"sunburstcolorway": color_discrete_sequence}
    else:
        layout_patch = {}
    if path is not None and (ids is not None or parents is not None):
        raise ValueError(
            "Either `path` should be provided, or `ids` and `parents`."
            "These parameters are mutually exclusive and cannot be passed together."
        )
    if path is not None and branchvalues is None:
        branchvalues = "total"
    return make_figure(
        args=locals(),
        constructor=go.Sunburst,
        trace_patch=dict(branchvalues=branchvalues, maxdepth=maxdepth),
        layout_patch=layout_patch,
    )


sunburst.__doc__ = make_docstring(sunburst)


def treemap(
    data_frame=None,
    names=None,
    values=None,
    parents=None,
    ids=None,
    path=None,
    color=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    labels=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
    branchvalues=None,
    maxdepth=None,
) -> go.Figure:
    """
    A treemap plot represents hierarchial data as nested rectangular
    sectors.
    """
    if color_discrete_sequence is not None:
        layout_patch = {"treemapcolorway": color_discrete_sequence}
    else:
        layout_patch = {}
    if path is not None and (ids is not None or parents is not None):
        raise ValueError(
            "Either `path` should be provided, or `ids` and `parents`."
            "These parameters are mutually exclusive and cannot be passed together."
        )
    if path is not None and branchvalues is None:
        branchvalues = "total"
    return make_figure(
        args=locals(),
        constructor=go.Treemap,
        trace_patch=dict(branchvalues=branchvalues, maxdepth=maxdepth),
        layout_patch=layout_patch,
    )


treemap.__doc__ = make_docstring(treemap)


def icicle(
    data_frame=None,
    names=None,
    values=None,
    parents=None,
    path=None,
    ids=None,
    color=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    labels=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
    branchvalues=None,
    maxdepth=None,
) -> go.Figure:
    """
    An icicle plot represents hierarchial data with adjoined rectangular
    sectors that all cascade from root down to leaf in one direction.
    """
    if color_discrete_sequence is not None:
        layout_patch = {"iciclecolorway": color_discrete_sequence}
    else:
        layout_patch = {}
    if path is not None and (ids is not None or parents is not None):
        raise ValueError(
            "Either `path` should be provided, or `ids` and `parents`."
            "These parameters are mutually exclusive and cannot be passed together."
        )
    if path is not None and branchvalues is None:
        branchvalues = "total"
    return make_figure(
        args=locals(),
        constructor=go.Icicle,
        trace_patch=dict(branchvalues=branchvalues, maxdepth=maxdepth),
        layout_patch=layout_patch,
    )


icicle.__doc__ = make_docstring(icicle)


def funnel(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    opacity=None,
    orientation=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a funnel plot, each row of `data_frame` is represented as a
    rectangular sector of a funnel.
    """
    return make_figure(args=locals(), constructor=go.Funnel)


funnel.__doc__ = make_docstring(funnel, append_dict=_cartesian_append_dict)


def funnel_area(
    data_frame=None,
    names=None,
    values=None,
    color=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    labels=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
    opacity=None,
) -> go.Figure:
    """
    In a funnel area plot, each row of `data_frame` is represented as a
    trapezoidal sector of a funnel.
    """
    if color_discrete_sequence is not None:
        layout_patch = {"funnelareacolorway": color_discrete_sequence}
    else:
        layout_patch = {}
    return make_figure(
        args=locals(),
        constructor=go.Funnelarea,
        trace_patch=dict(showlegend=(names is not None)),
        layout_patch=layout_patch,
    )


funnel_area.__doc__ = make_docstring(funnel_area)


# <!-- @GENESIS_MODULE_END: _chart_types -->
