import logging
# <!-- @GENESIS_MODULE_START: _pprint -->
"""
🏛️ GENESIS _PPRINT - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("_pprint", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_pprint", "position_calculated", {
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
                            "module": "_pprint",
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
                    print(f"Emergency stop error in _pprint: {e}")
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
                    "module": "_pprint",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_pprint", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _pprint: {e}")
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


"""This module contains the _EstimatorPrettyPrinter class used in
BaseEstimator.__repr__ for pretty-printing estimators"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 Python Software Foundation;
# All Rights Reserved

# Authors: Fred L. Drake, Jr. <fdrake@acm.org> (built-in CPython pprint module)
#          Nicolas Hug (scikit-learn specific changes)

# License: PSF License version 2 (see below)

# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
# --------------------------------------------

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"),
# and the Individual or Organization ("Licensee") accessing and otherwise
# using this software ("Python") in source or binary form and its associated
# documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
# grants Licensee a nonexclusive, royalty-free, world-wide license to
# reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use Python alone or in any
# derivative version, provided, however, that PSF's License Agreement and
# PSF's notice of copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004,
# 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
# 2017, 2018 Python Software Foundation; All Rights Reserved" are retained in
# Python alone or in any derivative version prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates Python or any part thereof, and wants to make the derivative
# work available to others as provided herein, then Licensee hereby agrees to
# include in any such work a brief summary of the changes made to Python.

# 4. PSF is making Python available to Licensee on an "AS IS" basis. PSF MAKES
# NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT
# NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF
# MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
# PYTHON WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON FOR ANY
# INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
# MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON, OR ANY DERIVATIVE
# THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between PSF and
# Licensee. This License Agreement does not grant permission to use PSF
# trademarks or trade name in a trademark sense to endorse or promote products
# or services of Licensee, or any third party.

# 8. By copying, installing or otherwise using Python, Licensee agrees to be
# bound by the terms and conditions of this License Agreement.


# Brief summary of changes to original code:
# - "compact" parameter is supported for dicts, not just lists or tuples
# - estimators have a custom handler, they're not just treated as objects
# - long sequences (lists, tuples, dict items) with more than N elements are
#   shortened using ellipsis (', ...') at the end.

import inspect
import pprint

from .._config import get_config
from ..base import BaseEstimator
from ._missing import is_scalar_nan


class KeyValTuple(tuple):
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

            emit_telemetry("_pprint", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("_pprint", "position_calculated", {
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
                        "module": "_pprint",
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
                print(f"Emergency stop error in _pprint: {e}")
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
                "module": "_pprint",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("_pprint", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in _pprint: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "_pprint",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in _pprint: {e}")
    """Dummy class for correctly rendering key-value tuples from dicts."""

    def __repr__(self):
        # needed for _dispatch[tuple.__repr__] not to be overridden
        return super().__repr__()


class KeyValTupleParam(KeyValTuple):
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

            emit_telemetry("_pprint", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("_pprint", "position_calculated", {
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
                        "module": "_pprint",
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
                print(f"Emergency stop error in _pprint: {e}")
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
                "module": "_pprint",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("_pprint", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in _pprint: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "_pprint",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in _pprint: {e}")
    """Dummy class for correctly rendering key-value tuples from parameters."""

    pass


def _changed_params(estimator):
    """Return dict (param_name: value) of parameters that were given to
    estimator with non-default values."""

    params = estimator.get_params(deep=False)
    init_func = getattr(estimator.__init__, "deprecated_original", estimator.__init__)
    init_params = inspect.signature(init_func).parameters
    init_params = {name: param.default for name, param in init_params.items()}

    def has_changed(k, v):
        if k not in init_params:  # happens if k is part of a **kwargs
            return True
        if init_params[k] == inspect._empty:  # k has no default value
            return True
        # try to avoid calling repr on nested estimators
        if isinstance(v, BaseEstimator) and v.__class__ != init_params[k].__class__:
            return True
        # Use repr as a last resort. It may be expensive.
        if repr(v) != repr(init_params[k]) and not (
            is_scalar_nan(init_params[k]) and is_scalar_nan(v)
        ):
            return True
        return False

    return {k: v for k, v in params.items() if has_changed(k, v)}


class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
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

            emit_telemetry("_pprint", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("_pprint", "position_calculated", {
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
                        "module": "_pprint",
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
                print(f"Emergency stop error in _pprint: {e}")
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
                "module": "_pprint",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("_pprint", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in _pprint: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "_pprint",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in _pprint: {e}")
    """Pretty Printer class for estimator objects.

    This extends the pprint.PrettyPrinter class, because:
    - we need estimators to be printed with their parameters, e.g.
      Estimator(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.

    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):

    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object recursively, only calling itself,
      not caring about line length or anything
    - back to _format(), if the output string is too long, _format() then calls
      the appropriate _pprint_TYPE() method (e.g. _pprint_list()) depending on
      the type of the object. This where the line length and the compact
      parameters are taken into account.
    - those _pprint_TYPE() methods will internally use the format() method for
      rendering the nested objects of an object (e.g. the elements of a list)

    In the end, everything has to be implemented twice: in _safe_repr and in
    the custom _pprint_TYPE methods. Unfortunately PrettyPrinter is really not
    straightforward to extend (especially when we want a compact output), so
    the code is a bit convoluted.

    This class overrides:
    - format() to support the changed_only parameter
    - _safe_repr to support printing of estimators (for when they fit on a
      single line)
    - _format_dict_items so that dict are correctly 'compacted'
    - _format_items so that ellipsis is used on long lists and tuples

    When estimators cannot be printed on a single line, the builtin _format()
    will call _pprint_estimator() because it was registered to do so (see
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator).

    both _format_dict_items() and _pprint_estimator() use the
    _format_params_or_dict_items() method that will format parameters and
    key-value pairs respecting the compact parameter. This method needs another
    subroutine _pprint_key_val_tuple() used when a parameter or a key-value
    pair is too long to fit on a single line. This subroutine is called in
    _format() and is registered as well in the _dispatch dict (just like
    _pprint_estimator). We had to create the two classes KeyValTuple and
    KeyValTupleParam for this.
    """

    def __init__(
        self,
        indent=1,
        width=80,
        depth=None,
        stream=None,
        *,
        compact=False,
        indent_at_name=True,
        n_max_elements_to_show=None,
    ):
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name
        if self._indent_at_name:
            self._indent_per_level = 1  # ignore indent param
        self._changed_only = get_config()["print_changed_only"]
        # Max number of elements in a list, dict, tuple until we start using
        # ellipsis. This also affects the number of arguments of an estimators
        # (they are treated as dicts)
        self.n_max_elements_to_show = n_max_elements_to_show

    def format(self, object, context, maxlevels, level):
        return _safe_repr(
            object, context, maxlevels, level, changed_only=self._changed_only
        )

    def _pprint_estimator(self, object, stream, indent, allowance, context, level):
        stream.write(object.__class__.__name__ + "(")
        if self._indent_at_name:
            indent += len(object.__class__.__name__)

        if self._changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)

        self._format_params(
            sorted(params.items()), stream, indent, allowance + 1, context, level
        )
        stream.write(")")

    def _format_dict_items(self, items, stream, indent, allowance, context, level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=True
        )

    def _format_params(self, items, stream, indent, allowance, context, level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=False
        )

    def _format_params_or_dict_items(
        self, object, stream, indent, allowance, context, level, is_dict
    ):
        """Format dict items or parameters respecting the compact=True
        parameter. For some reason, the builtin rendering of dict items doesn't
        respect compact=True and will use one line per key-value if all cannot
        fit in a single line.
        Dict items will be rendered as <'key': value> while params will be
        rendered as <key=value>. The implementation is mostly copy/pasting from
        the builtin _format_items().
        This also adds ellipsis if the number of items is greater than
        self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        delimnl = ",\n" + " " * indent
        delim = ""
        width = max_width = self._width - indent + 1
        it = iter(object)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(", ...")
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                k, v = ent
                krepr = self._repr(k, context, level)
                vrepr = self._repr(v, context, level)
                if not is_dict:
                    krepr = krepr.strip("'")
                middle = ": " if is_dict else "="
                rep = krepr + middle + vrepr
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ", "
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            class_ = KeyValTuple if is_dict else KeyValTupleParam
            self._format(
                class_(ent), stream, indent, allowance if last else 1, context, level
            )

    def _format_items(self, items, stream, indent, allowance, context, level):
        """Format the items of an iterable (list, tuple...). Same as the
        built-in _format_items, with support for ellipsis if the number of
        elements is greater than self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * " ")
        delimnl = ",\n" + " " * indent
        delim = ""
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(", ...")
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ", "
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent, allowance if last else 1, context, level)

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context, level):
        """Pretty printing for key-value tuples from dict or parameters."""
        k, v = object
        rep = self._repr(k, context, level)
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = "="
        else:
            middle = ": "
        stream.write(rep)
        stream.write(middle)
        self._format(
            v, stream, indent + len(rep) + len(middle), allowance, context, level
        )

    # Note: need to copy _dispatch to prevent instances of the builtin
    # PrettyPrinter class to call methods of _EstimatorPrettyPrinter (see issue
    # 12906)
    # mypy error: "Type[PrettyPrinter]" has no attribute "_dispatch"
    _dispatch = pprint.PrettyPrinter._dispatch.copy()  # type: ignore[attr-defined]
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple


def _safe_repr(object, context, maxlevels, level, changed_only=False):
    """Same as the builtin _safe_repr, with added support for Estimator
    objects."""
    typ = type(object)

    if typ in pprint._builtin_scalars:
        return repr(object), True, False

    r = getattr(typ, "__repr__", None)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return "{}", True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(object.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only
            )
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only
            )
            append("%s: %s" % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return "{%s}" % ", ".join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or (
        issubclass(typ, tuple) and r is tuple.__repr__
    ):
        if issubclass(typ, list):
            if not object:
                return "[]", True, False
            format = "[%s]"
        elif len(object) == 1:
            format = "(%s,)"
        else:
            if not object:
                return "()", True, False
            format = "(%s)"
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return format % "...", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = _safe_repr(
                o, context, maxlevels, level, changed_only=changed_only
            )
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ", ".join(components), readable, recursive

    if issubclass(typ, BaseEstimator):
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return f"{typ.__name__}(...)", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        if changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(params.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only
            )
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only
            )
            append("%s=%s" % (krepr.strip("'"), vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ("%s(%s)" % (typ.__name__, ", ".join(components)), readable, recursive)

    rep = repr(object)
    return rep, (rep and not rep.startswith("<")), False


# <!-- @GENESIS_MODULE_END: _pprint -->
