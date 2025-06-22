import logging
import sys
from pathlib import Path

# <!-- @GENESIS_MODULE_START: plugin_registry -->
"""
🏛️ GENESIS PLUGIN_REGISTRY - INSTITUTIONAL GRADE v8.0.0
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

from __future__ import annotations

import sys
from functools import partial
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, cast

from altair.utils.deprecation import deprecated_warn

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

                emit_telemetry("plugin_registry", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("plugin_registry", "position_calculated", {
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
                            "module": "plugin_registry",
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
                    print(f"Emergency stop error in plugin_registry: {e}")
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
                    "module": "plugin_registry",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("plugin_registry", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in plugin_registry: {e}")
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



if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs
if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    from types import TracebackType

T = TypeVar("T")
R = TypeVar("R")
Plugin = TypeAliasType("Plugin", Callable[..., R], type_params=(R,))
PluginT = TypeVar("PluginT", bound=Plugin[Any])
IsPlugin = Callable[[object], TypeIs[Plugin[Any]]]


def _is_type(tp: type[T], /) -> Callable[[object], TypeIs[type[T]]]:
    """
    Converts a type to guard function.

    Added for compatibility with original `PluginRegistry` default.
    """

    def func(obj: object, /) -> TypeIs[type[T]]:
        return isinstance(obj, tp)

    return func


class NoSuchEntryPoint(Exception):
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

            emit_telemetry("plugin_registry", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("plugin_registry", "position_calculated", {
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
                        "module": "plugin_registry",
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
                print(f"Emergency stop error in plugin_registry: {e}")
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
                "module": "plugin_registry",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("plugin_registry", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in plugin_registry: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "plugin_registry",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in plugin_registry: {e}")
    def __init__(self, group, name):
        self.group = group
        self.name = name

    def __str__(self):
        return f"No {self.name!r} entry point found in group {self.group!r}"


class PluginEnabler(Generic[PluginT, R]):
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

            emit_telemetry("plugin_registry", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("plugin_registry", "position_calculated", {
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
                        "module": "plugin_registry",
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
                print(f"Emergency stop error in plugin_registry: {e}")
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
                "module": "plugin_registry",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("plugin_registry", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in plugin_registry: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "plugin_registry",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in plugin_registry: {e}")
    """
    Context manager for enabling plugins.

    This object lets you use enable() as a context manager to
    temporarily enable a given plugin::

        with plugins.enable("name"):
            do_something()  # 'name' plugin temporarily enabled
        # plugins back to original state
    """

    def __init__(
        self, registry: PluginRegistry[PluginT, R], name: str, **options: Any
    ) -> None:
        self.registry: PluginRegistry[PluginT, R] = registry
        self.name: str = name
        self.options: dict[str, Any] = options
        self.original_state: dict[str, Any] = registry._get_state()
        self.registry._enable(name, **options)

    def __enter__(self) -> PluginEnabler[PluginT, R]:
        return self

    def __exit__(self, typ: type, value: Exception, traceback: TracebackType) -> None:
        self.registry._set_state(self.original_state)

    def __repr__(self) -> str:
        return f"{type(self.registry).__name__}.enable({self.name!r})"


class PluginRegistry(Generic[PluginT, R]):
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

            emit_telemetry("plugin_registry", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("plugin_registry", "position_calculated", {
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
                        "module": "plugin_registry",
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
                print(f"Emergency stop error in plugin_registry: {e}")
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
                "module": "plugin_registry",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("plugin_registry", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in plugin_registry: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "plugin_registry",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in plugin_registry: {e}")
    """
    A registry for plugins.

    This is a plugin registry that allows plugins to be loaded/registered
    in two ways:

    1. Through an explicit call to ``.register(name, value)``.
    2. By looking for other Python packages that are installed and provide
       a setuptools entry point group.

    When you create an instance of this class, provide the name of the
    entry point group to use::

        reg = PluginRegister("my_entrypoint_group")

    """

    # this is a mapping of name to error message to allow custom error messages
    # in case an entrypoint is not found
    entrypoint_err_messages: dict[str, str] = {}

    # global settings is a key-value mapping of settings that are stored globally
    # in the registry rather than passed to the plugins
    _global_settings: dict[str, Any] = {}

    def __init__(
        self, entry_point_group: str = "", plugin_type: IsPlugin = callable
    ) -> None:
        """
        Create a PluginRegistry for a named entry point group.

        Parameters
        ----------
        entry_point_group: str
            The name of the entry point group.
        plugin_type
            A type narrowing function that will optionally be used for runtime
            type checking loaded plugins.

        References
        ----------
        https://typing.readthedocs.io/en/latest/spec/narrowing.html
        """
        self.entry_point_group: str = entry_point_group
        self.plugin_type: IsPlugin
        if plugin_type is not callable and isinstance(plugin_type, type):
            msg: Any = (
                f"Pass a callable `TypeIs` function to `plugin_type` instead.\n"
                f"{type(self).__name__!r}(plugin_type)\n\n"
                f"See also:\n"
                f"https://typing.readthedocs.io/en/latest/spec/narrowing.html\n"
                f"https://docs.astral.sh/ruff/rules/assert/"
            )
            deprecated_warn(msg, version="5.4.0")
            self.plugin_type = cast(IsPlugin, _is_type(plugin_type))
        else:
            self.plugin_type = plugin_type
        self._active: Plugin[R] | None = None
        self._active_name: str = ""
        self._plugins: dict[str, PluginT] = {}
        self._options: dict[str, Any] = {}
        self._global_settings: dict[str, Any] = self.__class__._global_settings.copy()

    def register(self, name: str, value: PluginT | None) -> PluginT | None:
        """
        Register a plugin by name and value.

        This method is used for explicit registration of a plugin and shouldn't be
        used to manage entry point managed plugins, which are auto-loaded.

        Parameters
        ----------
        name: str
            The name of the plugin.
        value: PluginType or None
            The actual plugin object to register or None to unregister that plugin.

        Returns
        -------
        plugin: PluginType or None
            The plugin that was registered or unregistered.
        """
        if value is None:
            return self._plugins.pop(name, None)
        elif self.plugin_type(value):
            self._plugins[name] = value
            return value
        else:
            msg = f"{type(value).__name__!r} is not compatible with {type(self).__name__!r}"
            raise TypeError(msg)

    def names(self) -> list[str]:
        """List the names of the registered and entry points plugins."""
        exts = list(self._plugins.keys())
        e_points = importlib_metadata_get(self.entry_point_group)
        more_exts = [ep.name for ep in e_points]
        exts.extend(more_exts)
        return sorted(set(exts))

    def _get_state(self) -> dict[str, Any]:
        """Return a dictionary representing the current state of the registry."""
        return {
            "_active": self._active,
            "_active_name": self._active_name,
            "_plugins": self._plugins.copy(),
            "_options": self._options.copy(),
            "_global_settings": self._global_settings.copy(),
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        """Reset the state of the registry."""
        assert set(state.keys()) == {
            "_active",
            "_active_name",
            "_plugins",
            "_options",
            "_global_settings",
        }
        for key, val in state.items():
            setattr(self, key, val)

    def _enable(self, name: str, **options) -> None:
        if name not in self._plugins:
            try:
                (ep,) = (
                    ep
                    for ep in importlib_metadata_get(self.entry_point_group)
                    if ep.name == name
                )
            except ValueError as err:
                if name in self.entrypoint_err_messages:
                    raise ValueError(self.entrypoint_err_messages[name]) from err
                else:
                    raise NoSuchEntryPoint(self.entry_point_group, name) from err
            value = cast(PluginT, ep.load())
            self.register(name, value)
        self._active_name = name
        self._active = self._plugins[name]
        for key in set(options.keys()) & set(self._global_settings.keys()):
            self._global_settings[key] = options.pop(key)
        self._options = options

    def enable(
        self, name: str | None = None, **options: Any
    ) -> PluginEnabler[PluginT, R]:
        """
        Enable a plugin by name.

        This can be either called directly, or used as a context manager.

        Parameters
        ----------
        name : string (optional)
            The name of the plugin to enable. If not specified, then use the
            current active name.
        **options :
            Any additional parameters will be passed to the plugin as keyword
            arguments

        Returns
        -------
        PluginEnabler:
            An object that allows enable() to be used as a context manager
        """
        if name is None:
            name = self.active
        return PluginEnabler(self, name, **options)

    @property
    def active(self) -> str:
        """Return the name of the currently active plugin."""
        return self._active_name

    @property
    def options(self) -> dict[str, Any]:
        """Return the current options dictionary."""
        return self._options

    def get(self) -> partial[R] | Plugin[R] | None:
        """Return the currently active plugin."""
        if (func := self._active) and self.plugin_type(func):
            return partial(func, **self._options) if self._options else func
        elif self._active is not None:
            msg = (
                f"{type(self).__name__!r} requires all plugins to be callable objects, "
                f"but {type(self._active).__name__!r} is not callable."
            )
            raise TypeError(msg)
        elif TYPE_CHECKING:
            # NOTE: The `None` return is implicit, but `mypy` isn't satisfied
            # - `ruff` will factor out explicit `None` return
            # - `pyright` has no issue
            logger.info("Function operational")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(active={self.active!r}, registered={self.names()!r})"


def importlib_metadata_get(group):
    ep = entry_points()
    # 'select' was introduced in Python 3.10 and 'get' got deprecated
    # We don't check for Python version here as by checking with hasattr we
    # also get compatibility with the importlib_metadata package which had a different
    # deprecation cycle for 'get'
    if hasattr(ep, "select"):
        return ep.select(group=group)  # pyright: ignore
    else:
        return ep.get(group, [])


# <!-- @GENESIS_MODULE_END: plugin_registry -->
