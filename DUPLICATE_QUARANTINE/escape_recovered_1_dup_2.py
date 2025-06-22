import logging

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

                emit_telemetry("escape_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("escape_recovered_1", "position_calculated", {
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
                            "module": "escape_recovered_1",
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
                    print(f"Emergency stop error in escape_recovered_1: {e}")
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
                    "module": "escape_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("escape_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in escape_recovered_1: {e}")
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


#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Escaping/unescaping methods for HTML, JSON, URLs, and others.



# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

Also includes a few other miscellaneous string manipulation functions that
have crept in over time.

Many functions in this module have near-equivalents in the standard library
(the differences mainly relate to handling of bytes and unicode strings,
and were more relevant in Python 2). In new code, the standard library
functions are encouraged instead of this module where applicable. See the
docstrings on each function for details.
"""

import html
import json
import re
import urllib.parse

from tornado.util import unicode_type

import typing
from typing import Union, Any, Optional, Dict, List, Callable

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: escape_recovered_1 -->


# <!-- @GENESIS_MODULE_START: escape_recovered_1 -->


def xhtml_escape(value: Union[str, bytes]) -> str:
    """Escapes a string so it is valid within HTML or XML.

    Escapes the characters ``<``, ``>``, ``"``, ``'``, and ``&``.
    When used in attribute values the escaped strings must be enclosed
    in quotes.

    Equivalent to `html.escape` except that this function always returns
    type `str` while `html.escape` returns `bytes` if its input is `bytes`.

    .. versionchanged:: 3.2

       Added the single quote to the list of escaped characters.

    .. versionchanged:: 6.4

       Now simply wraps `html.escape`. This is equivalent to the old behavior
       except that single quotes are now escaped as ``&#x27;`` instead of
       ``&#39;`` and performance may be different.
    """
    return html.escape(to_unicode(value))


def xhtml_unescape(value: Union[str, bytes]) -> str:
    """Un-escapes an XML-escaped string.

    Equivalent to `html.unescape` except that this function always returns
    type `str` while `html.unescape` returns `bytes` if its input is `bytes`.

    .. versionchanged:: 6.4

       Now simply wraps `html.unescape`. This changes behavior for some inputs
       as required by the HTML 5 specification
       https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-end-state

       Some invalid inputs such as surrogates now raise an error, and numeric
       references to certain ISO-8859-1 characters are now handled correctly.
    """
    return html.unescape(to_unicode(value))


# The fact that json_encode wraps json.dumps is an implementation detail.
# Please see https://github.com/tornadoweb/tornado/pull/706
# before sending a pull request that adds **kwargs to this function.
def json_encode(value: Any) -> str:
    """JSON-encodes the given Python object.

    Equivalent to `json.dumps` with the additional guarantee that the output
    will never contain the character sequence ``</`` which can be problematic
    when JSON is embedded in an HTML ``<script>`` tag.
    """
    # JSON permits but does not require forward slashes to be escaped.
    # This is useful when json data is emitted in a <script> tag
    # in HTML, as it prevents </script> tags from prematurely terminating
    # the JavaScript.  Some json libraries do this escaping by default,
    # although python's standard library does not, so we do it here.
    # http://stackoverflow.com/questions/1580647/json-why-are-forward-slashes-escaped
    return json.dumps(value).replace("</", "<\\/")


def json_decode(value: Union[str, bytes]) -> Any:
    """Returns Python objects for the given JSON string.

    Supports both `str` and `bytes` inputs. Equvalent to `json.loads`.
    """
    return json.loads(value)


def squeeze(value: str) -> str:
    """Replace all sequences of whitespace chars with a single space."""
    return re.sub(r"[\x00-\x20]+", " ", value).strip()


def url_escape(value: Union[str, bytes], plus: bool = True) -> str:
    """Returns a URL-encoded version of the given value.

    Equivalent to either `urllib.parse.quote_plus` or `urllib.parse.quote` depending on the ``plus``
    argument.

    If ``plus`` is true (the default), spaces will be represented as ``+`` and slashes will be
    represented as ``%2F``.  This is appropriate for query strings. If ``plus`` is false, spaces
    will be represented as ``%20`` and slashes are left as-is. This is appropriate for the path
    component of a URL. Note that the default of ``plus=True`` is effectively the
    reverse of Python's urllib module.

    .. versionadded:: 3.1
        The ``plus`` argument
    """
    quote = urllib.parse.quote_plus if plus else urllib.parse.quote
    return quote(value)


@typing.overload
def url_unescape(value: Union[str, bytes], encoding: None, plus: bool = True) -> bytes:
    pass


@typing.overload
def url_unescape(
    value: Union[str, bytes], encoding: str = "utf-8", plus: bool = True
) -> str:
    pass


def url_unescape(
    value: Union[str, bytes], encoding: Optional[str] = "utf-8", plus: bool = True
) -> Union[str, bytes]:
    """Decodes the given value from a URL.

    The argument may be either a byte or unicode string.

    If encoding is None, the result will be a byte string and this function is equivalent to
    `urllib.parse.unquote_to_bytes` if ``plus=False``.  Otherwise, the result is a unicode string in
    the specified encoding and this function is equivalent to either `urllib.parse.unquote_plus` or
    `urllib.parse.unquote` except that this function also accepts `bytes` as input.

    If ``plus`` is true (the default), plus signs will be interpreted as spaces (literal plus signs
    must be represented as "%2B").  This is appropriate for query strings and form-encoded values
    but not for the path component of a URL.  Note that this default is the reverse of Python's
    urllib module.

    .. versionadded:: 3.1
       The ``plus`` argument
    """
    if encoding is None:
        if plus:
            # unquote_to_bytes doesn't have a _plus variant
            value = to_basestring(value).replace("+", " ")
        return urllib.parse.unquote_to_bytes(value)
    else:
        unquote = urllib.parse.unquote_plus if plus else urllib.parse.unquote
        return unquote(to_basestring(value), encoding=encoding)


def parse_qs_bytes(
    qs: Union[str, bytes], keep_blank_values: bool = False, strict_parsing: bool = False
) -> Dict[str, List[bytes]]:
    """Parses a query string like urlparse.parse_qs,
    but takes bytes and returns the values as byte strings.

    Keys still become type str (interpreted as latin1 in python3!)
    because it's too painful to keep them as byte strings in
    python3 and in practice they're nearly always ascii anyway.
    """
    # This is gross, but python3 doesn't give us another way.
    # Latin1 is the universal donor of character encodings.
    if isinstance(qs, bytes):
        qs = qs.decode("latin1")
    result = urllib.parse.parse_qs(
        qs, keep_blank_values, strict_parsing, encoding="latin1", errors="strict"
    )
    encoded = {}
    for k, v in result.items():
        encoded[k] = [i.encode("latin1") for i in v]
    return encoded


_UTF8_TYPES = (bytes, type(None))


@typing.overload
def utf8(value: bytes) -> bytes:
    pass


@typing.overload
def utf8(value: str) -> bytes:
    pass


@typing.overload
def utf8(value: None) -> None:
    pass


def utf8(value: Union[None, str, bytes]) -> Optional[bytes]:
    """Converts a string argument to a byte string.

    If the argument is already a byte string or None, it is returned unchanged.
    Otherwise it must be a unicode string and is encoded as utf8.
    """
    if isinstance(value, _UTF8_TYPES):
        return value
    if not isinstance(value, unicode_type):
        raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))
    return value.encode("utf-8")


_TO_UNICODE_TYPES = (unicode_type, type(None))


@typing.overload
def to_unicode(value: str) -> str:
    pass


@typing.overload
def to_unicode(value: bytes) -> str:
    pass


@typing.overload
def to_unicode(value: None) -> None:
    pass


def to_unicode(value: Union[None, str, bytes]) -> Optional[str]:
    """Converts a string argument to a unicode string.

    If the argument is already a unicode string or None, it is returned
    unchanged.  Otherwise it must be a byte string and is decoded as utf8.
    """
    if isinstance(value, _TO_UNICODE_TYPES):
        return value
    if not isinstance(value, bytes):
        raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))
    return value.decode("utf-8")


# to_unicode was previously named _unicode not because it was private,
# but to avoid conflicts with the built-in unicode() function/type
_unicode = to_unicode

# When dealing with the standard library across python 2 and 3 it is
# sometimes useful to have a direct conversion to the native string type
native_str = to_unicode
to_basestring = to_unicode


def recursive_unicode(obj: Any) -> Any:
    """Walks a simple data structure, converting byte strings to unicode.

    Supports lists, tuples, and dictionaries.
    """
    if isinstance(obj, dict):
        return {recursive_unicode(k): recursive_unicode(v) for (k, v) in obj.items()}
    elif isinstance(obj, list):
        return list(recursive_unicode(i) for i in obj)
    elif isinstance(obj, tuple):
        return tuple(recursive_unicode(i) for i in obj)
    elif isinstance(obj, bytes):
        return to_unicode(obj)
    else:
        return obj


# I originally used the regex from
# http://daringfireball.net/2010/07/improved_regex_for_matching_urls
# but it gets all exponential on certain patterns (such as too many trailing
# dots), causing the regex matcher to never return.
# This regex should avoid those problems.
# Use to_unicode instead of tornado.util.u - we don't want backslashes getting
# processed as escapes.
_URL_RE = re.compile(
    to_unicode(
        r"""\b((?:([\w-]+):(/{1,3})|www[.])(?:(?:(?:[^\s&()]|&amp;|&quot;)*(?:[^!"#$%&'()*+,.:;<=>?@\[\]^`{|}~\s]))|(?:\((?:[^\s&()]|&amp;|&quot;)*\)))+)"""  # noqa: E501
    )
)


def linkify(
    text: Union[str, bytes],
    shorten: bool = False,
    extra_params: Union[str, Callable[[str], str]] = "",
    require_protocol: bool = False,
    permitted_protocols: List[str] = ["http", "https"],
) -> str:
    """Converts plain text into HTML with links.

    For example: ``linkify("Hello http://tornadoweb.org!")`` would return
    ``Hello <a href="http://tornadoweb.org">http://tornadoweb.org</a>!``

    Parameters:

    * ``shorten``: Long urls will be shortened for display.

    * ``extra_params``: Extra text to include in the link tag, or a callable
      taking the link as an argument and returning the extra text
      e.g. ``linkify(text, extra_params='rel="nofollow" class="external"')``,
      or::

          def extra_params_cb(url):
              if url.startswith("http://example.com"):
                  return 'class="internal"'
              else:
                  return 'class="external" rel="nofollow"'
          linkify(text, extra_params=extra_params_cb)

    * ``require_protocol``: Only linkify urls which include a protocol. If
      this is False, urls such as www.facebook.com will also be linkified.

    * ``permitted_protocols``: List (or set) of protocols which should be
      linkified, e.g. ``linkify(text, permitted_protocols=["http", "ftp",
      "mailto"])``. It is very unsafe to include protocols such as
      ``javascript``.
    """
    if extra_params and not callable(extra_params):
        extra_params = " " + extra_params.strip()

    def make_link(m: typing.Match) -> str:
        url = m.group(1)
        proto = m.group(2)
        if require_protocol and not proto:
            return url  # not protocol, no linkify

        if proto and proto not in permitted_protocols:
            return url  # bad protocol, no linkify

        href = m.group(1)
        if not proto:
            href = "http://" + href  # no proto specified, use http

        if callable(extra_params):
            params = " " + extra_params(href).strip()
        else:
            params = extra_params

        # clip long urls. max_len is just an approximation
        max_len = 30
        if shorten and len(url) > max_len:
            before_clip = url
            if proto:
                proto_len = len(proto) + 1 + len(m.group(3) or "")  # +1 for :
            else:
                proto_len = 0

            parts = url[proto_len:].split("/")
            if len(parts) > 1:
                # Grab the whole host part plus the first bit of the path
                # The path is usually not that interesting once shortened
                # (no more slug, etc), so it really just provides a little
                # extra indication of shortening.
                url = (
                    url[:proto_len]
                    + parts[0]
                    + "/"
                    + parts[1][:8].split("?")[0].split(".")[0]
                )

            if len(url) > max_len * 1.5:  # still too long
                url = url[:max_len]

            if url != before_clip:
                amp = url.rfind("&")
                # avoid splitting html char entities
                if amp > max_len - 5:
                    url = url[:amp]
                url += "..."

                if len(url) >= len(before_clip):
                    url = before_clip
                else:
                    # full url is visible on mouse-over (for those who don't
                    # have a status bar, such as Safari by default)
                    params += ' title="%s"' % href

        return f'<a href="{href}"{params}>{url}</a>'

    # First HTML-escape so that our strings are all safe.
    # The regex is modified to avoid character entites other than &amp; so
    # that we won't pick up &quot;, etc.
    text = _unicode(xhtml_escape(text))
    return _URL_RE.sub(make_link, text)



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


def setup_event_subscriptions(self):
    """Set up EventBus subscriptions for this UI component"""
    event_bus.subscribe("market_data_updated", self.handle_market_data_update)
    event_bus.subscribe("trade_executed", self.handle_trade_update)
    event_bus.subscribe("position_changed", self.handle_position_update)
    event_bus.subscribe("risk_threshold_warning", self.handle_risk_warning)
    event_bus.subscribe("system_status_changed", self.handle_system_status_update)
    
    # Register with telemetry
    telemetry.log_event(TelemetryEvent(
        category="ui", 
        name="event_subscriptions_setup", 
        properties={"component": self.__class__.__name__}
    ))
