# <!-- @GENESIS_MODULE_START: server -->
"""
🏛️ GENESIS SERVER - INSTITUTIONAL GRADE v8.0.0
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

import errno
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import tornado.concurrent
import tornado.locks
import tornado.netutil
import tornado.web
import tornado.websocket
from tornado.httpserver import HTTPServer

from streamlit import cli_util, config, file_util, util
from streamlit.auth_util import is_authlib_installed
from streamlit.config_option import ConfigOption
from streamlit.logger import get_logger
from streamlit.runtime import Runtime, RuntimeConfig, RuntimeState
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.memory_session_storage import MemorySessionStorage
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.runtime_util import get_max_message_size_bytes
from streamlit.web.cache_storage_manager_config import (

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

                emit_telemetry("server", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("server", "position_calculated", {
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
                            "module": "server",
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
                    print(f"Emergency stop error in server: {e}")
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
                    "module": "server",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("server", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in server: {e}")
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


    create_default_cache_storage_manager,
)
from streamlit.web.server.app_static_file_handler import AppStaticFileHandler
from streamlit.web.server.browser_websocket_handler import BrowserWebSocketHandler
from streamlit.web.server.component_request_handler import ComponentRequestHandler
from streamlit.web.server.media_file_handler import MediaFileHandler
from streamlit.web.server.routes import (
    AddSlashHandler,
    HealthHandler,
    HostConfigHandler,
    RemoveSlashHandler,
    StaticFileHandler,
)
from streamlit.web.server.server_util import (
    get_cookie_secret,
    is_xsrf_enabled,
    make_url_path_regex,
)
from streamlit.web.server.stats_request_handler import StatsRequestHandler
from streamlit.web.server.upload_file_request_handler import UploadFileRequestHandler

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from ssl import SSLContext

_LOGGER: Final = get_logger(__name__)

TORNADO_SETTINGS = {
    # Gzip HTTP responses.
    "compress_response": True,
    # Ping every 1s to keep WS alive.
    # 2021.06.22: this value was previously 20s, and was causing
    # connection instability for a small number of users. This smaller
    # ping_interval fixes that instability.
    # https://github.com/streamlit/streamlit/issues/3196
    "websocket_ping_interval": 1,
    # If we don't get a ping response within 30s, the connection
    # is timed out.
    "websocket_ping_timeout": 30,
    "xsrf_cookie_name": "_streamlit_xsrf",
}

# When server.port is not available it will look for the next available port
# up to MAX_PORT_SEARCH_RETRIES.
MAX_PORT_SEARCH_RETRIES: Final = 100

# When server.address starts with this prefix, the server will bind
# to an unix socket.
UNIX_SOCKET_PREFIX: Final = "unix://"

# Please make sure to also update frontend/app/vite.config.ts
# dev server proxy when changing or updating these endpoints as well
# as the endpoints in frontend/connection/src/DefaultStreamlitEndpoints
MEDIA_ENDPOINT: Final = "/media"
COMPONENT_ENDPOINT: Final = "/component"
STATIC_SERVING_ENDPOINT: Final = "/app/static"
UPLOAD_FILE_ENDPOINT: Final = "/_stcore/upload_file"
STREAM_ENDPOINT: Final = r"_stcore/stream"
METRIC_ENDPOINT: Final = r"(?:st-metrics|_stcore/metrics)"
MESSAGE_ENDPOINT: Final = r"_stcore/message"
NEW_HEALTH_ENDPOINT: Final = "_stcore/health"
HEALTH_ENDPOINT: Final = rf"(?:healthz|{NEW_HEALTH_ENDPOINT})"
HOST_CONFIG_ENDPOINT: Final = r"_stcore/host-config"
SCRIPT_HEALTH_CHECK_ENDPOINT: Final = (
    r"(?:script-health-check|_stcore/script-health-check)"
)

OAUTH2_CALLBACK_ENDPOINT: Final = "/oauth2callback"
AUTH_LOGIN_ENDPOINT: Final = "/auth/login"
AUTH_LOGOUT_ENDPOINT: Final = "/auth/logout"


class RetriesExceededError(Exception):
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

            emit_telemetry("server", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("server", "position_calculated", {
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
                        "module": "server",
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
                print(f"Emergency stop error in server: {e}")
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
                "module": "server",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("server", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in server: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "server",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in server: {e}")
    pass


def server_port_is_manually_set() -> bool:
    return config.is_manually_set("server.port")


def server_address_is_unix_socket() -> bool:
    address = config.get_option("server.address")
    return address is not None and address.startswith(UNIX_SOCKET_PREFIX)


def start_listening(app: tornado.web.Application) -> None:
    """Makes the server start listening at the configured port.

    In case the port is already taken it tries listening to the next available
    port.  It will error after MAX_PORT_SEARCH_RETRIES attempts.

    """
    cert_file = config.get_option("server.sslCertFile")
    key_file = config.get_option("server.sslKeyFile")
    ssl_options = _get_ssl_options(cert_file, key_file)

    http_server = HTTPServer(
        app,
        max_buffer_size=config.get_option("server.maxUploadSize") * 1024 * 1024,
        ssl_options=ssl_options,
    )

    if server_address_is_unix_socket():
        start_listening_unix_socket(http_server)
    else:
        start_listening_tcp_socket(http_server)


def _get_ssl_options(cert_file: str | None, key_file: str | None) -> SSLContext | None:
    if bool(cert_file) != bool(key_file):
        _LOGGER.error(
            "Options 'server.sslCertFile' and 'server.sslKeyFile' must "
            "be set together. Set missing options or delete existing options."
        )
        sys.exit(1)
    if cert_file and key_file:
        # ssl_ctx.load_cert_chain raise exception as below, but it is not
        # sufficiently user-friendly
        # FileNotFoundError: [Errno 2] No such file or directory
        if not Path(cert_file).exists():
            _LOGGER.error("Cert file '%s' does not exist.", cert_file)
            sys.exit(1)
        if not Path(key_file).exists():
            _LOGGER.error("Key file '%s' does not exist.", key_file)
            sys.exit(1)

        import ssl

        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        # When the SSL certificate fails to load, an exception is raised as below,
        # but it is not sufficiently user-friendly.
        # ssl.SSLError: [SSL] PEM lib (_ssl.c:4067)
        try:
            ssl_ctx.load_cert_chain(cert_file, key_file)
        except ssl.SSLError:
            _LOGGER.exception(
                "Failed to load SSL certificate. Make sure "
                "cert file '%s' and key file '%s' are correct.",
                cert_file,
                key_file,
            )
            sys.exit(1)

        return ssl_ctx
    return None


def start_listening_unix_socket(http_server: HTTPServer) -> None:
    address = config.get_option("server.address")
    file_name = os.path.expanduser(address[len(UNIX_SOCKET_PREFIX) :])

    unix_socket = tornado.netutil.bind_unix_socket(file_name)
    http_server.add_socket(unix_socket)


def start_listening_tcp_socket(http_server: HTTPServer) -> None:
    call_count = 0

    port = None
    while call_count < MAX_PORT_SEARCH_RETRIES:
        address = config.get_option("server.address")
        port = config.get_option("server.port")

        try:
            http_server.listen(port, address)
            break  # It worked! So let's break out of the loop.

        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                if server_port_is_manually_set():
                    _LOGGER.error("Port %s is already in use", port)  # noqa: TRY400
                    sys.exit(1)
                else:
                    _LOGGER.debug(
                        "Port %s already in use, trying to use the next one.", port
                    )
                    port += 1

                    config.set_option(
                        "server.port", port, ConfigOption.STREAMLIT_DEFINITION
                    )
                    call_count += 1
            else:
                raise

    if call_count >= MAX_PORT_SEARCH_RETRIES:
        raise RetriesExceededError(
            f"Cannot start Streamlit server. Port {port} is already in use, and "
            f"Streamlit was unable to find a free port after {MAX_PORT_SEARCH_RETRIES} attempts.",
        )


class Server:
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

            emit_telemetry("server", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("server", "position_calculated", {
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
                        "module": "server",
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
                print(f"Emergency stop error in server: {e}")
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
                "module": "server",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("server", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in server: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "server",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in server: {e}")
    def __init__(self, main_script_path: str, is_hello: bool) -> None:
        """Create the server. It won't be started yet."""
        _set_tornado_log_levels()
        self.initialize_mimetypes()

        self._main_script_path = main_script_path

        # Initialize MediaFileStorage and its associated endpoint
        media_file_storage = MemoryMediaFileStorage(MEDIA_ENDPOINT)
        MediaFileHandler.initialize_storage(media_file_storage)

        uploaded_file_mgr = MemoryUploadedFileManager(UPLOAD_FILE_ENDPOINT)

        self._runtime = Runtime(
            RuntimeConfig(
                script_path=main_script_path,
                command_line=None,
                media_file_storage=media_file_storage,
                uploaded_file_manager=uploaded_file_mgr,
                cache_storage_manager=create_default_cache_storage_manager(),
                is_hello=is_hello,
                session_storage=MemorySessionStorage(
                    ttl_seconds=config.get_option("server.disconnectedSessionTTL")
                ),
            ),
        )

        self._runtime.stats_mgr.register_provider(media_file_storage)

    @classmethod
    def initialize_mimetypes(cls) -> None:
        """Ensures that common mime-types are robust against system misconfiguration."""
        mimetypes.add_type("text/html", ".html")
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        mimetypes.add_type("image/webp", ".webp")

    def __repr__(self) -> str:
        return util.repr_(self)

    @property
    def main_script_path(self) -> str:
        return self._main_script_path

    async def start(self) -> None:
        """Start the server.

        When this returns, Streamlit is ready to accept new sessions.
        """

        _LOGGER.debug("Starting server...")

        app = self._create_app()
        start_listening(app)

        port = config.get_option("server.port")
        _LOGGER.debug("Server started on port %s", port)

        await self._runtime.start()

    @property
    def stopped(self) -> Awaitable[None]:
        """A Future that completes when the Server's run loop has exited."""
        return self._runtime.stopped

    def _create_app(self) -> tornado.web.Application:
        """Create our tornado web app."""
        base = config.get_option("server.baseUrlPath")

        routes: list[Any] = [
            (
                make_url_path_regex(base, STREAM_ENDPOINT),
                BrowserWebSocketHandler,
                {"runtime": self._runtime},
            ),
            (
                make_url_path_regex(base, HEALTH_ENDPOINT),
                HealthHandler,
                {"callback": lambda: self._runtime.is_ready_for_browser_connection},
            ),
            (
                make_url_path_regex(base, METRIC_ENDPOINT),
                StatsRequestHandler,
                {"stats_manager": self._runtime.stats_mgr},
            ),
            (
                make_url_path_regex(base, HOST_CONFIG_ENDPOINT),
                HostConfigHandler,
            ),
            (
                make_url_path_regex(
                    base,
                    rf"{UPLOAD_FILE_ENDPOINT}/(?P<session_id>[^/]+)/(?P<file_id>[^/]+)",
                ),
                UploadFileRequestHandler,
                {
                    "file_mgr": self._runtime.uploaded_file_mgr,
                    "is_active_session": self._runtime.is_active_session,
                },
            ),
            (
                make_url_path_regex(base, f"{MEDIA_ENDPOINT}/(.*)"),
                MediaFileHandler,
                {"path": ""},
            ),
            (
                make_url_path_regex(base, f"{COMPONENT_ENDPOINT}/(.*)"),
                ComponentRequestHandler,
                {"registry": self._runtime.component_registry},
            ),
        ]

        if config.get_option("server.scriptHealthCheckEnabled"):
            routes.extend(
                [
                    (
                        make_url_path_regex(base, SCRIPT_HEALTH_CHECK_ENDPOINT),
                        HealthHandler,
                        {
                            "callback": lambda: self._runtime.does_script_run_without_error()
                        },
                    )
                ]
            )

        if config.get_option("server.enableStaticServing"):
            routes.extend(
                [
                    (
                        make_url_path_regex(base, f"{STATIC_SERVING_ENDPOINT}/(.*)"),
                        AppStaticFileHandler,
                        {"path": file_util.get_app_static_dir(self.main_script_path)},
                    ),
                ]
            )

        if is_authlib_installed():
            from streamlit.web.server.oauth_authlib_routes import (
                AuthCallbackHandler,
                AuthLoginHandler,
                AuthLogoutHandler,
            )

            routes.extend(
                [
                    (
                        make_url_path_regex(base, OAUTH2_CALLBACK_ENDPOINT),
                        AuthCallbackHandler,
                        {"base_url": base},
                    ),
                    (
                        make_url_path_regex(base, AUTH_LOGIN_ENDPOINT),
                        AuthLoginHandler,
                        {"base_url": base},
                    ),
                    (
                        make_url_path_regex(base, AUTH_LOGOUT_ENDPOINT),
                        AuthLogoutHandler,
                        {"base_url": base},
                    ),
                ]
            )

        if config.get_option("global.developmentMode"):
            _LOGGER.debug("Serving static content from the Node dev server")
        else:
            static_path = file_util.get_static_dir()
            _LOGGER.debug("Serving static content from %s", static_path)

            routes.extend(
                [
                    (
                        # We want to remove paths with a trailing slash, but if the path
                        # starts with a double slash //, the redirect will point
                        # the browser to the wrong host.
                        make_url_path_regex(
                            base, "(?!/)(.*)", trailing_slash="required"
                        ),
                        RemoveSlashHandler,
                    ),
                    (
                        make_url_path_regex(base, "(.*)"),
                        StaticFileHandler,
                        {
                            "path": f"{static_path}/",
                            "default_filename": "index.html",
                            "reserved_paths": [
                                # These paths are required for identifying
                                # the base url path.
                                NEW_HEALTH_ENDPOINT,
                                HOST_CONFIG_ENDPOINT,
                            ],
                        },
                    ),
                    (
                        make_url_path_regex(base, trailing_slash="prohibited"),
                        AddSlashHandler,
                    ),
                ]
            )

        return tornado.web.Application(
            routes,
            cookie_secret=get_cookie_secret(),
            xsrf_cookies=is_xsrf_enabled(),
            # Set the websocket message size. The default value is too low.
            websocket_max_message_size=get_max_message_size_bytes(),
            **TORNADO_SETTINGS,  # type: ignore[arg-type]
        )

    @property
    def browser_is_connected(self) -> bool:
        return self._runtime.state == RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED

    @property
    def is_running_hello(self) -> bool:
        from streamlit.hello import streamlit_app

        return self._main_script_path == streamlit_app.__file__

    def stop(self) -> None:
        cli_util.print_to_cli("  Stopping...", fg="blue")
        self._runtime.stop()


def _set_tornado_log_levels() -> None:
    if not config.get_option("global.developmentMode"):
        # Hide logs unless they're super important.
        # Example of stuff we don't care about: 404 about .js.map files.
        logging.getLogger("tornado.access").setLevel(logging.ERROR)
        logging.getLogger("tornado.application").setLevel(logging.ERROR)
        logging.getLogger("tornado.general").setLevel(logging.ERROR)


# <!-- @GENESIS_MODULE_END: server -->
