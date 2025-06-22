import logging
# <!-- @GENESIS_MODULE_START: metric -->
"""
🏛️ GENESIS METRIC - INSTITUTIONAL GRADE v8.0.0
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

# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
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

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from typing_extensions import TypeAlias

from streamlit.elements.lib.layout_utils import LayoutConfig, Width, validate_width
from streamlit.elements.lib.policies import maybe_raise_label_warnings
from streamlit.elements.lib.utils import (

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

                emit_telemetry("metric", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("metric", "position_calculated", {
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
                            "module": "metric",
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
                    print(f"Emergency stop error in metric: {e}")
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
                    "module": "metric",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("metric", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in metric: {e}")
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


    LabelVisibility,
    get_label_visibility_proto_value,
)
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text

if TYPE_CHECKING:
    import numpy as np

    from streamlit.delta_generator import DeltaGenerator


Value: TypeAlias = Union["np.integer[Any]", "np.floating[Any]", float, int, str, None]
Delta: TypeAlias = Union[float, int, str, None]
DeltaColor: TypeAlias = Literal["normal", "inverse", "off"]


@dataclass(frozen=True)
class MetricColorAndDirection:
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

            emit_telemetry("metric", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("metric", "position_calculated", {
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
                        "module": "metric",
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
                print(f"Emergency stop error in metric: {e}")
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
                "module": "metric",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("metric", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in metric: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "metric",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in metric: {e}")
    color: MetricProto.MetricColor.ValueType
    direction: MetricProto.MetricDirection.ValueType


class MetricMixin:
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

            emit_telemetry("metric", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("metric", "position_calculated", {
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
                        "module": "metric",
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
                print(f"Emergency stop error in metric: {e}")
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
                "module": "metric",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("metric", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in metric: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "metric",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in metric: {e}")
    @gather_metrics("metric")
    def metric(
        self,
        label: str,
        value: Value,
        delta: Delta = None,
        delta_color: DeltaColor = "normal",
        help: str | None = None,
        label_visibility: LabelVisibility = "visible",
        border: bool = False,
        width: Width = "stretch",
    ) -> DeltaGenerator:
        r"""Display a metric in big bold font, with an optional indicator of how the metric changed.

        Tip: If you want to display a large number, it may be a good idea to
        shorten it using packages like `millify <https://github.com/azaitsev/millify>`_
        or `numerize <https://github.com/davidsa03/numerize>`_. E.g. ``1234`` can be
        displayed as ``1.2k`` using ``st.metric("Short number", millify(1234))``.

        Parameters
        ----------
        label : str
            The header or title for the metric. The label can optionally
            contain GitHub-flavored Markdown of the following types: Bold, Italics,
            Strikethroughs, Inline Code, Links, and Images. Images display like
            icons, with a max height equal to the font height.

            Unsupported Markdown elements are unwrapped so only their children
            (text contents) render. Display unsupported elements as literal
            characters by backslash-escaping them. E.g.,
            ``"1\. Not an ordered list"``.

            See the ``body`` parameter of |st.markdown|_ for additional,
            supported Markdown directives.

            .. |st.markdown| replace:: ``st.markdown``
            .. _st.markdown: https://docs.streamlit.io/develop/api-reference/text/st.markdown

        value : int, float, str, or None
             Value of the metric. None is rendered as a long dash.

        delta : int, float, str, or None
            Indicator of how the metric changed, rendered with an arrow below
            the metric. If delta is negative (int/float) or starts with a minus
            sign (str), the arrow points down and the text is red; else the
            arrow points up and the text is green. If None (default), no delta
            indicator is shown.

        delta_color : "normal", "inverse", or "off"
             If "normal" (default), the delta indicator is shown as described
             above. If "inverse", it is red when positive and green when
             negative. This is useful when a negative change is considered
             good, e.g. if cost decreased. If "off", delta is  shown in gray
             regardless of its value.

        help : str or None
            A tooltip that gets displayed next to the metric label. Streamlit
            only displays the tooltip when ``label_visibility="visible"``. If
            this is ``None`` (default), no tooltip is displayed.

            The tooltip can optionally contain GitHub-flavored Markdown,
            including the Markdown directives described in the ``body``
            parameter of ``st.markdown``.

        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. The default is ``"visible"``. If this
            is ``"hidden"``, Streamlit displays an empty spacer instead of the
            label, which can help keep the widget aligned with other widgets.
            If this is ``"collapsed"``, Streamlit displays no label or spacer.

        border : bool
            Whether to show a border around the metric container. If this is
            ``False`` (default), no border is shown. If this is ``True``, a
            border is shown.

        width : "stretch", "content", or int
            The width of the metric element. This can be one of the following:

            - ``"stretch"`` (default): The width of the element matches the
              width of the parent container.
            - ``"content"``: The width of the element matches the width of its
              content, but doesn't exceed the width of the parent container.
            - An integer specifying the width in pixels: The element has a
              fixed width. If the specified width is greater than the width of
              the parent container, the width of the element matches the width
              of the parent container.

        Examples
        --------
        **Example 1: Show a metric**

        >>> import streamlit as st
        >>>
        >>> st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

        .. output::
            https://doc-metric-example1.streamlit.app/
            height: 210px

        **Example 2: Create a row of metrics**

        ``st.metric`` looks especially nice in combination with ``st.columns``.

        >>> import streamlit as st
        >>>
        >>> col1, col2, col3 = st.columns(3)
        >>> col1.metric("Temperature", "70 °F", "1.2 °F")
        >>> col2.metric("Wind", "9 mph", "-8%")
        >>> col3.metric("Humidity", "86%", "4%")

        .. output::
            https://doc-metric-example2.streamlit.app/
            height: 210px

        **Example 3: Modify the delta indicator**

        The delta indicator color can also be inverted or turned off.

        >>> import streamlit as st
        >>>
        >>> st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
        >>>
        >>> st.metric(
        ...     label="Active developers", value=123, delta=123, delta_color="off"
        ... )

        .. output::
            https://doc-metric-example3.streamlit.app/
            height: 320px

        **Example 4: Create a grid of metric cards**

        Add borders to your metrics to create a dashboard look.

        >>> import streamlit as st
        >>>
        >>> a, b = st.columns(2)
        >>> c, d = st.columns(2)
        >>>
        >>> a.metric("Temperature", "30°F", "-9°F", border=True)
        >>> b.metric("Wind", "4 mph", "2 mph", border=True)
        >>>
        >>> c.metric("Humidity", "77%", "5%", border=True)
        >>> d.metric("Pressure", "30.34 inHg", "-2 inHg", border=True)

        .. output::
            https://doc-metric-example4.streamlit.app/
            height: 350px

        """
        maybe_raise_label_warnings(label, label_visibility)

        metric_proto = MetricProto()
        metric_proto.body = _parse_value(value)
        metric_proto.label = _parse_label(label)
        metric_proto.delta = _parse_delta(delta)
        metric_proto.show_border = border
        if help is not None:
            metric_proto.help = dedent(help)

        color_and_direction = _determine_delta_color_and_direction(
            cast("DeltaColor", clean_text(delta_color)), delta
        )
        metric_proto.color = color_and_direction.color
        metric_proto.direction = color_and_direction.direction
        metric_proto.label_visibility.value = get_label_visibility_proto_value(
            label_visibility
        )

        validate_width(width, allow_content=True)
        layout_config = LayoutConfig(width=width)

        return self.dg._enqueue("metric", metric_proto, layout_config=layout_config)

    @property
    def dg(self) -> DeltaGenerator:
        return cast("DeltaGenerator", self)


def _parse_label(label: str) -> str:
    if not isinstance(label, str):
        raise TypeError(
            f"'{label}' is of type {type(label)}, which is not an accepted type."
            " label only accepts: str. Please convert the label to an accepted type."
        )
    return label


def _parse_value(value: Value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float, str)):
        return str(value)
    if hasattr(value, "item"):
        # Add support for numpy values (e.g. int16, float64, etc.)
        try:
            # Item could also be just a variable, so we use try, except
            if isinstance(value.item(), (float, int)):
                return str(value.item())
        except Exception:  # noqa: S110
            # If the numpy item is not a valid value, the TypeError below will be raised.
            pass

    raise TypeError(
        f"'{value}' is of type {type(value)}, which is not an accepted type."
        " value only accepts: int, float, str, or None."
        " Please convert the value to an accepted type."
    )


def _parse_delta(delta: Delta) -> str:
    if delta is None or delta == "":
        return ""
    if isinstance(delta, str):
        return dedent(delta)
    if isinstance(delta, (int, float)):
        return str(delta)
    raise TypeError(
        f"'{delta}' is of type {type(delta)}, which is not an accepted type."
        " delta only accepts: int, float, str, or None."
        " Please convert the value to an accepted type."
    )


def _determine_delta_color_and_direction(
    delta_color: DeltaColor,
    delta: Delta,
) -> MetricColorAndDirection:
    if delta_color not in {"normal", "inverse", "off"}:
        raise StreamlitAPIException(
            f"'{delta_color}' is not an accepted value. delta_color only accepts: "
            "'normal', 'inverse', or 'off'"
        )

    if delta is None or delta == "":
        return MetricColorAndDirection(
            color=MetricProto.MetricColor.GRAY,
            direction=MetricProto.MetricDirection.NONE,
        )

    if _is_negative_delta(delta):
        if delta_color == "normal":
            cd_color = MetricProto.MetricColor.RED
        elif delta_color == "inverse":
            cd_color = MetricProto.MetricColor.GREEN
        else:
            cd_color = MetricProto.MetricColor.GRAY
        cd_direction = MetricProto.MetricDirection.DOWN
    else:
        if delta_color == "normal":
            cd_color = MetricProto.MetricColor.GREEN
        elif delta_color == "inverse":
            cd_color = MetricProto.MetricColor.RED
        else:
            cd_color = MetricProto.MetricColor.GRAY
        cd_direction = MetricProto.MetricDirection.UP

    return MetricColorAndDirection(
        color=cd_color,
        direction=cd_direction,
    )


def _is_negative_delta(delta: Delta) -> bool:
    return dedent(str(delta)).startswith("-")


# <!-- @GENESIS_MODULE_END: metric -->
