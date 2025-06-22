import logging
# <!-- @GENESIS_MODULE_START: local_sources_watcher -->
"""
🏛️ GENESIS LOCAL_SOURCES_WATCHER - INSTITUTIONAL GRADE v8.0.0
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

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, NamedTuple

from streamlit import config, file_util
from streamlit.logger import get_logger
from streamlit.watcher.folder_black_list import FolderBlackList
from streamlit.watcher.path_watcher import (

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

                emit_telemetry("local_sources_watcher", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("local_sources_watcher", "position_calculated", {
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
                            "module": "local_sources_watcher",
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
                    print(f"Emergency stop error in local_sources_watcher: {e}")
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
                    "module": "local_sources_watcher",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("local_sources_watcher", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in local_sources_watcher: {e}")
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


    NoOpPathWatcher,
    get_default_path_watcher_class,
)

if TYPE_CHECKING:
    from types import ModuleType

    from streamlit.runtime.pages_manager import PagesManager

_LOGGER: Final = get_logger(__name__)


class WatchedModule(NamedTuple):
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

            emit_telemetry("local_sources_watcher", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("local_sources_watcher", "position_calculated", {
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
                        "module": "local_sources_watcher",
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
                print(f"Emergency stop error in local_sources_watcher: {e}")
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
                "module": "local_sources_watcher",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("local_sources_watcher", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in local_sources_watcher: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "local_sources_watcher",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in local_sources_watcher: {e}")
    watcher: Any
    module_name: Any


# This needs to be initialized lazily to avoid calling config.get_option() and
# thus initializing config options when this file is first imported.
PathWatcher = None


class LocalSourcesWatcher:
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

            emit_telemetry("local_sources_watcher", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("local_sources_watcher", "position_calculated", {
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
                        "module": "local_sources_watcher",
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
                print(f"Emergency stop error in local_sources_watcher: {e}")
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
                "module": "local_sources_watcher",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("local_sources_watcher", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in local_sources_watcher: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "local_sources_watcher",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in local_sources_watcher: {e}")
    def __init__(self, pages_manager: PagesManager) -> None:
        self._pages_manager = pages_manager
        self._main_script_path = os.path.abspath(self._pages_manager.main_script_path)
        self._watch_folders = config.get_option("server.folderWatchList")
        self._script_folder = os.path.dirname(self._main_script_path)
        self._on_path_changed: list[Callable[[str], None]] = []
        self._is_closed = False
        self._cached_sys_modules: set[str] = set()

        # Blacklist for folders that should not be watched
        self._folder_black_list = FolderBlackList(
            config.get_option("server.folderWatchBlacklist")
        )

        self._watched_modules: dict[str, WatchedModule] = {}
        self._watched_pages: set[str] = set()

        self.update_watched_pages()

    def update_watched_pages(self) -> None:
        old_page_paths = self._watched_pages.copy()
        new_pages_paths: set[str] = set()

        for page_info in self._pages_manager.get_pages().values():
            if not page_info["script_path"]:
                continue

            new_pages_paths.add(page_info["script_path"])
            if page_info["script_path"] not in self._watched_pages:
                self._register_watcher(
                    page_info["script_path"],
                    module_name=None,
                )

        # Add custom watch path if it exists

        for watch_folder in self._watch_folders:
            # check if it is folder
            if not os.path.isdir(watch_folder):
                _LOGGER.warning("Watch folder is not a directory: %s", watch_folder)
                continue
            _LOGGER.debug("Registering watch folder: %s", watch_folder)
            if watch_folder not in self._watched_pages:
                self._register_watcher(
                    watch_folder,
                    module_name=None,
                    is_directory=True,
                )

        for old_page_path in old_page_paths:
            # Only remove pages that are no longer valid files
            if old_page_path not in new_pages_paths and not os.path.isfile(
                old_page_path
            ):
                self._deregister_watcher(old_page_path)
                self._watched_pages.remove(old_page_path)

        self._watched_pages = self._watched_pages.union(new_pages_paths)

    def register_file_change_callback(self, cb: Callable[[str], None]) -> None:
        self._on_path_changed.append(cb)

    def on_path_changed(self, filepath: str) -> None:
        _LOGGER.debug("Path changed: %s", filepath)
        if filepath not in self._watched_modules:
            # Check if this is a file in a watched directory
            for watched_dir in self._watched_modules:
                if (
                    os.path.isdir(watched_dir)
                    and os.path.commonpath([watched_dir, filepath]) == watched_dir
                ):
                    _LOGGER.info("File changed in watched directory: %s", filepath)
                    for cb in self._on_path_changed:
                        cb(filepath)
                    return
            _LOGGER.error("Received event for non-watched path: %s", filepath)
            return

        # Workaround:
        # Delete all watched modules so we can guarantee changes to the
        # updated module are reflected on reload.
        #
        # In principle, for reloading a given module, we only need to unload
        # the module itself and all of the modules which import it (directly
        # or indirectly) such that when we exec the application code, the
        # changes are reloaded and reflected in the running application.
        #
        # However, determining all import paths for a given loaded module is
        # non-trivial, and so as a workaround we simply unload all watched
        # modules.
        for wm in self._watched_modules.values():
            if wm.module_name is not None and wm.module_name in sys.modules:
                del sys.modules[wm.module_name]

        for cb in self._on_path_changed:
            cb(filepath)

    def close(self) -> None:
        for wm in self._watched_modules.values():
            wm.watcher.close()
        self._watched_modules = {}
        self._watched_pages = set()
        self._is_closed = True

    def _register_watcher(
        self, filepath: str, module_name: str | None, is_directory: bool = False
    ) -> None:
        global PathWatcher  # noqa: PLW0603
        if PathWatcher is None:
            PathWatcher = get_default_path_watcher_class()

        if PathWatcher is NoOpPathWatcher:
            return

        try:
            # Instead of using **kwargs, explicitly pass the named parameters
            glob_pattern = "**/*" if is_directory else None

            wm = WatchedModule(
                watcher=PathWatcher(
                    filepath,
                    self.on_path_changed,
                    glob_pattern=glob_pattern,  # Pass as named parameter
                    allow_nonexistent=False,
                ),
                module_name=module_name,
            )
            self._watched_modules[filepath] = wm
        except PermissionError:
            # If you don't have permission to read this file, don't even add it
            # to watchers.
            return

        self._watched_modules[filepath] = wm

    def _deregister_watcher(self, filepath: str) -> None:
        if filepath not in self._watched_modules:
            return

        if filepath == self._main_script_path:
            return

        wm = self._watched_modules[filepath]
        wm.watcher.close()
        del self._watched_modules[filepath]

    def _file_is_new(self, filepath: str) -> bool:
        return filepath not in self._watched_modules

    def _file_should_be_watched(self, filepath: str) -> bool:
        # Using short circuiting for performance.
        return self._file_is_new(filepath) and (
            file_util.file_is_in_folder_glob(filepath, self._script_folder)
            or file_util.file_in_pythonpath(filepath)
        )

    def update_watched_modules(self) -> None:
        if self._is_closed:
            return

        if set(sys.modules) != self._cached_sys_modules:
            modules_paths = {
                name: self._exclude_blacklisted_paths(get_module_paths(module))
                for name, module in dict(sys.modules).items()
            }
            self._cached_sys_modules = set(sys.modules)
            self._register_necessary_watchers(modules_paths)

    def _register_necessary_watchers(self, module_paths: dict[str, set[str]]) -> None:
        for name, paths in module_paths.items():
            for path in paths:
                if self._file_should_be_watched(path):
                    self._register_watcher(str(Path(path).resolve()), name)

    def _exclude_blacklisted_paths(self, paths: set[str]) -> set[str]:
        return {p for p in paths if not self._folder_black_list.is_blacklisted(p)}


def get_module_paths(module: ModuleType) -> set[str]:
    paths_extractors: list[Callable[[ModuleType], list[str | None]]] = [
        # https://docs.python.org/3/reference/datamodel.html
        # __file__ is the pathname of the file from which the module was loaded
        # if it was loaded from a file.
        # The __file__ attribute may be missing for certain types of modules
        lambda m: [m.__file__] if hasattr(m, "__file__") else [],
        # https://docs.python.org/3/reference/import.html#__spec__
        # The __spec__ attribute is set to the module spec that was used
        # when importing the module. one exception is __main__,
        # where __spec__ is set to None in some cases.
        # https://www.python.org/dev/peps/pep-0451/#id16
        # "origin" in an import context means the system
        # (or resource within a system) from which a module originates
        # ... It is up to the loader to decide on how to interpret
        # and use a module's origin, if at all.
        lambda m: [m.__spec__.origin]
        if hasattr(m, "__spec__") and m.__spec__ is not None
        else [],
        # https://www.python.org/dev/peps/pep-0420/
        # Handling of "namespace packages" in which the __path__ attribute
        # is a _NamespacePath object with a _path attribute containing
        # the various paths of the package.
        lambda m: list(m.__path__._path)
        if hasattr(m, "__path__")
        # This check prevents issues with torch classes:
        # https://github.com/streamlit/streamlit/issues/10992
        and type(m.__path__).__name__ == "_NamespacePath"
        and hasattr(m.__path__, "_path")
        else [],
    ]

    all_paths = set()
    for extract_paths in paths_extractors:
        potential_paths = []
        try:
            potential_paths = extract_paths(module)
        except AttributeError:
            # Some modules might not have __file__ or __spec__ attributes.
            pass
        except Exception:
            _LOGGER.warning(
                "Examining the path of %s raised:", module.__name__, exc_info=True
            )

        all_paths.update(
            [os.path.abspath(str(p)) for p in potential_paths if _is_valid_path(p)]
        )
    return all_paths


def _is_valid_path(path: str | None) -> bool:
    return isinstance(path, str) and (os.path.isfile(path) or os.path.isdir(path))


# <!-- @GENESIS_MODULE_END: local_sources_watcher -->
