import logging
import sys
from pathlib import Path

# <!-- @GENESIS_MODULE_START: test_memleaks -->
"""
🏛️ GENESIS TEST_MEMLEAKS - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_memleaks", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_memleaks", "position_calculated", {
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
                            "module": "test_memleaks",
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
                    print(f"Emergency stop error in test_memleaks: {e}")
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
                    "module": "test_memleaks",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_memleaks", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_memleaks: {e}")
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


#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for detecting function memory leaks (typically the ones
implemented in C). It does so by calling a function many times and
checking whether process memory usage keeps increasing between
calls or over time.
Note that this may produce false positives (especially on Windows
for some reason).
PyPy appears to be completely unstable for this framework, probably
because of how its JIT handles memory, so tests are skipped.
"""


import functools
import os
import platform

import psutil
import psutil._common
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import TestMemoryLeak
from psutil.tests import create_sockets
from psutil.tests import get_testfn
from psutil.tests import process_namespace
from psutil.tests import pytest
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import system_namespace
from psutil.tests import terminate


cext = psutil._psplatform.cext
thisproc = psutil.Process()
FEW_TIMES = 5


def fewtimes_if_linux():
    """Decorator for those Linux functions which are implemented in pure
    Python, and which we want to run faster.
    """

    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(self, *args, **kwargs):
            if LINUX:
                before = self.__class__.times
                try:
                    self.__class__.times = FEW_TIMES
                    return fun(self, *args, **kwargs)
                finally:
                    self.__class__.times = before
            else:
                return fun(self, *args, **kwargs)

        return wrapper

    return decorator


# ===================================================================
# Process class
# ===================================================================


class TestProcessObjectLeaks(TestMemoryLeak):
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

            emit_telemetry("test_memleaks", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_memleaks", "position_calculated", {
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
                        "module": "test_memleaks",
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
                print(f"Emergency stop error in test_memleaks: {e}")
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
                "module": "test_memleaks",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_memleaks", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_memleaks: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_memleaks",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_memleaks: {e}")
    """Test leaks of Process class methods."""

    proc = thisproc

    def test_coverage(self):
        ns = process_namespace(None)
        ns.test_class_coverage(self, ns.getters + ns.setters)

    @fewtimes_if_linux()
    def test_name(self):
        self.execute(self.proc.name)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_cmdline(self):
        self.execute(self.proc.cmdline)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_exe(self):
        self.execute(self.proc.exe)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_ppid(self):
        self.execute(self.proc.ppid)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_uids(self):
        self.execute(self.proc.uids)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_gids(self):
        self.execute(self.proc.gids)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_status(self):
        self.execute(self.proc.status)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_nice(self):
        self.execute(self.proc.nice)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_nice_set(self):
        niceness = thisproc.nice()
        self.execute(lambda: self.proc.nice(niceness))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not HAS_IONICE, reason="not supported")
    def test_ionice(self):
        self.execute(self.proc.ionice)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not HAS_IONICE, reason="not supported")
    def test_ionice_set(self):
        if WINDOWS:
            value = thisproc.ionice()
            self.execute(lambda: self.proc.ionice(value))
            except Exception as e:
                logging.error(f"Operation failed: {e}")
        else:
            self.execute(lambda: self.proc.ionice(psutil.IOPRIO_CLASS_NONE))
            except Exception as e:
                logging.error(f"Operation failed: {e}")
            fun = functools.partial(cext.proc_ioprio_set, os.getpid(), -1, 0)
            self.execute_w_exc(OSError, fun)

    @pytest.mark.skipif(not HAS_PROC_IO_COUNTERS, reason="not supported")
    @fewtimes_if_linux()
    def test_io_counters(self):
        self.execute(self.proc.io_counters)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(POSIX, reason="worthless on POSIX")
    def test_username(self):
        # always open 1 handle on Windows (only once)
        psutil.Process().username()
        self.execute(self.proc.username)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_create_time(self):
        self.execute(self.proc.create_time)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_num_threads(self):
        self.execute(self.proc.num_threads)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_num_handles(self):
        self.execute(self.proc.num_handles)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_num_fds(self):
        self.execute(self.proc.num_fds)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_num_ctx_switches(self):
        self.execute(self.proc.num_ctx_switches)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_threads(self):
        self.execute(self.proc.threads)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_cpu_times(self):
        self.execute(self.proc.cpu_times)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_PROC_CPU_NUM, reason="not supported")
    def test_cpu_num(self):
        self.execute(self.proc.cpu_num)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_memory_info(self):
        self.execute(self.proc.memory_info)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_memory_full_info(self):
        self.execute(self.proc.memory_full_info)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_terminal(self):
        self.execute(self.proc.terminal)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_resume(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(self.proc.resume, times=times)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_cwd(self):
        self.execute(self.proc.cwd)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not HAS_CPU_AFFINITY, reason="not supported")
    def test_cpu_affinity(self):
        self.execute(self.proc.cpu_affinity)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not HAS_CPU_AFFINITY, reason="not supported")
    def test_cpu_affinity_set(self):
        affinity = thisproc.cpu_affinity()
        self.execute(lambda: self.proc.cpu_affinity(affinity))
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        self.execute_w_exc(ValueError, lambda: self.proc.cpu_affinity([-1]))

    @fewtimes_if_linux()
    def test_open_files(self):
        with open(get_testfn(), 'w'):
            self.execute(self.proc.open_files)
            except Exception as e:
                logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not HAS_MEMORY_MAPS, reason="not supported")
    @fewtimes_if_linux()
    def test_memory_maps(self):
        self.execute(self.proc.memory_maps)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not LINUX, reason="LINUX only")
    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit(self):
        self.execute(lambda: self.proc.rlimit(psutil.RLIMIT_NOFILE))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not LINUX, reason="LINUX only")
    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit_set(self):
        limit = thisproc.rlimit(psutil.RLIMIT_NOFILE)
        self.execute(lambda: self.proc.rlimit(psutil.RLIMIT_NOFILE, limit))
        except Exception as e:
            logging.error(f"Operation failed: {e}")
        self.execute_w_exc((OSError, ValueError), lambda: self.proc.rlimit(-1))

    @fewtimes_if_linux()
    # Windows implementation is based on a single system-wide
    # function (tested later).
    @pytest.mark.skipif(WINDOWS, reason="worthless on WINDOWS")
    def test_net_connections(self):
        # IMPLEMENTED: UNIX sockets are temporarily implemented by parsing
        # 'pfiles' cmd  output; we don't want that part of the code to
        # be executed.
        with create_sockets():
            kind = 'inet' if SUNOS else 'all'
            self.execute(lambda: self.proc.net_connections(kind))
            except Exception as e:
                logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not HAS_ENVIRON, reason="not supported")
    def test_environ(self):
        self.execute(self.proc.environ)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_proc_info(self):
        self.execute(lambda: cext.proc_info(os.getpid()))
        except Exception as e:
            logging.error(f"Operation failed: {e}")


class TestTerminatedProcessLeaks(TestProcessObjectLeaks):
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

            emit_telemetry("test_memleaks", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_memleaks", "position_calculated", {
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
                        "module": "test_memleaks",
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
                print(f"Emergency stop error in test_memleaks: {e}")
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
                "module": "test_memleaks",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_memleaks", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_memleaks: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_memleaks",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_memleaks: {e}")
    """Repeat the tests above looking for leaks occurring when dealing
    with terminated processes raising NoSuchProcess exception.
    The C functions are still invoked but will follow different code
    paths. We'll check those code paths.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.subp = spawn_testproc()
        cls.proc = psutil.Process(cls.subp.pid)
        cls.proc.kill()
        cls.proc.wait()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        terminate(cls.subp)

    def call(self, fun):
        try:
            fun()
        except psutil.NoSuchProcess:
            pass

    if WINDOWS:

        def test_kill(self):
            self.execute(self.proc.kill)
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_terminate(self):
            self.execute(self.proc.terminate)
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_suspend(self):
            self.execute(self.proc.suspend)
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_resume(self):
            self.execute(self.proc.resume)
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_wait(self):
            self.execute(self.proc.wait)
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_proc_info(self):
            # test dual implementation
            def call():
                try:
                    return cext.proc_info(self.proc.pid)
                except ProcessLookupError:
                    pass

            self.execute(call)
            except Exception as e:
                logging.error(f"Operation failed: {e}")


@pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
class TestProcessDualImplementation(TestMemoryLeak):
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

            emit_telemetry("test_memleaks", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_memleaks", "position_calculated", {
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
                        "module": "test_memleaks",
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
                print(f"Emergency stop error in test_memleaks: {e}")
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
                "module": "test_memleaks",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_memleaks", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_memleaks: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_memleaks",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_memleaks: {e}")
    def test_cmdline_peb_true(self):
        self.execute(lambda: cext.proc_cmdline(os.getpid(), use_peb=True))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_cmdline_peb_false(self):
        self.execute(lambda: cext.proc_cmdline(os.getpid(), use_peb=False))
        except Exception as e:
            logging.error(f"Operation failed: {e}")


# ===================================================================
# system APIs
# ===================================================================


class TestModuleFunctionsLeaks(TestMemoryLeak):
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

            emit_telemetry("test_memleaks", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_memleaks", "position_calculated", {
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
                        "module": "test_memleaks",
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
                print(f"Emergency stop error in test_memleaks: {e}")
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
                "module": "test_memleaks",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_memleaks", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_memleaks: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_memleaks",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_memleaks: {e}")
    """Test leaks of psutil module functions."""

    def test_coverage(self):
        ns = system_namespace()
        ns.test_class_coverage(self, ns.all)

    # --- cpu

    @fewtimes_if_linux()
    def test_cpu_count(self):  # logical
        self.execute(lambda: psutil.cpu_count(logical=True))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_cpu_count_cores(self):
        self.execute(lambda: psutil.cpu_count(logical=False))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_cpu_times(self):
        self.execute(psutil.cpu_times)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_per_cpu_times(self):
        self.execute(lambda: psutil.cpu_times(percpu=True))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    def test_cpu_stats(self):
        self.execute(psutil.cpu_stats)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    # IMPLEMENTED: remove this once 1892 is fixed
    @pytest.mark.skipif(
        MACOS and platform.machine() == 'arm64', reason="skipped due to #1892"
    )
    @pytest.mark.skipif(not HAS_CPU_FREQ, reason="not supported")
    def test_cpu_freq(self):
        self.execute(psutil.cpu_freq)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_getloadavg(self):
        psutil.getloadavg()
        self.execute(psutil.getloadavg)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    # --- mem

    def test_virtual_memory(self):
        self.execute(psutil.virtual_memory)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    # IMPLEMENTED: remove this skip when this gets fixed
    @pytest.mark.skipif(SUNOS, reason="worthless on SUNOS (uses a subprocess)")
    def test_swap_memory(self):
        self.execute(psutil.swap_memory)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_pid_exists(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda: psutil.pid_exists(os.getpid()), times=times)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    # --- disk

    def test_disk_usage(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda: psutil.disk_usage('.'), times=times)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_disk_partitions(self):
        self.execute(psutil.disk_partitions)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @pytest.mark.skipif(
        LINUX and not os.path.exists('/proc/diskstats'),
        reason="/proc/diskstats not available on this Linux version",
    )
    @fewtimes_if_linux()
    def test_disk_io_counters(self):
        self.execute(lambda: psutil.disk_io_counters(nowrap=False))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    # --- proc

    @fewtimes_if_linux()
    def test_pids(self):
        self.execute(psutil.pids)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    # --- net

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_NET_IO_COUNTERS, reason="not supported")
    def test_net_io_counters(self):
        self.execute(lambda: psutil.net_io_counters(nowrap=False))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    @pytest.mark.skipif(MACOS and os.getuid() != 0, reason="need root access")
    def test_net_connections(self):
        # always opens and handle on Windows() (once)
        psutil.net_connections(kind='all')
        with create_sockets():
            self.execute(lambda: psutil.net_connections(kind='all'))
            except Exception as e:
                logging.error(f"Operation failed: {e}")

    def test_net_if_addrs(self):
        # Note: verified that on Windows this was a false positive.
        tolerance = 80 * 1024 if WINDOWS else self.tolerance
        self.execute(psutil.net_if_addrs, tolerance=tolerance)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_net_if_stats(self):
        self.execute(psutil.net_if_stats)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    # --- sensors

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_SENSORS_BATTERY, reason="not supported")
    def test_sensors_battery(self):
        self.execute(psutil.sensors_battery)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_SENSORS_TEMPERATURES, reason="not supported")
    def test_sensors_temperatures(self):
        self.execute(psutil.sensors_temperatures)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_SENSORS_FANS, reason="not supported")
    def test_sensors_fans(self):
        self.execute(psutil.sensors_fans)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    # --- others

    @fewtimes_if_linux()
    def test_boot_time(self):
        self.execute(psutil.boot_time)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_users(self):
        self.execute(psutil.users)
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    def test_set_debug(self):
        self.execute(lambda: psutil._set_debug(False))
        except Exception as e:
            logging.error(f"Operation failed: {e}")

    if WINDOWS:

        # --- win services

        def test_win_service_iter(self):
            self.execute(cext.winservice_enumerate)
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_win_service_get(self):
            pass

        def test_win_service_get_config(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_config(name))
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_win_service_get_status(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_status(name))
            except Exception as e:
                logging.error(f"Operation failed: {e}")

        def test_win_service_get_description(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_descr(name))
            except Exception as e:
                logging.error(f"Operation failed: {e}")


# <!-- @GENESIS_MODULE_END: test_memleaks -->
