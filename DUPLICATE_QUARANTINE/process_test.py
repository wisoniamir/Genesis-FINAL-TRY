# <!-- @GENESIS_MODULE_START: process_test -->
"""
🏛️ GENESIS PROCESS_TEST - INSTITUTIONAL GRADE v8.0.0
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

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import unittest

from tornado.httpclient import HTTPClient, HTTPError
from tornado.httpserver import HTTPServer
from tornado.log import gen_log
from tornado.process import fork_processes, task_id, Subprocess
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import bind_unused_port, ExpectLog, AsyncTestCase, gen_test
from tornado.test.util import skipIfNonUnix
from tornado.web import RequestHandler, Application

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

                emit_telemetry("process_test", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("process_test", "position_calculated", {
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
                            "module": "process_test",
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
                    print(f"Emergency stop error in process_test: {e}")
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
                    "module": "process_test",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("process_test", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in process_test: {e}")
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




# Not using AsyncHTTPTestCase because we need control over the IOLoop.
@skipIfNonUnix
class ProcessTest(unittest.TestCase):
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

            emit_telemetry("process_test", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("process_test", "position_calculated", {
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
                        "module": "process_test",
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
                print(f"Emergency stop error in process_test: {e}")
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
                "module": "process_test",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("process_test", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in process_test: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "process_test",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in process_test: {e}")
    def get_app(self):
        class ProcessHandler(RequestHandler):
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

                    emit_telemetry("process_test", "confluence_detected", {
                        "score": confluence_score,
                        "timestamp": datetime.now().isoformat()
                    })

                    return confluence_score
            def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                    """GENESIS Risk Management - Calculate optimal position size"""
                    account_balance = 100000  # Default FTMO account size
                    risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                    position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                    emit_telemetry("process_test", "position_calculated", {
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
                                "module": "process_test",
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
                        print(f"Emergency stop error in process_test: {e}")
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
                        "module": "process_test",
                        "event": event,
                        "data": data or {}
                    }
                    try:
                        emit_telemetry("process_test", event, telemetry_data)
                    except Exception as e:
                        print(f"Telemetry error in process_test: {e}")
            def initialize_eventbus(self):
                    """GENESIS EventBus Initialization"""
                    try:
                        self.event_bus = get_event_bus()
                        if self.event_bus:
                            emit_event("module_initialized", {
                                "module": "process_test",
                                "timestamp": datetime.now().isoformat(),
                                "status": "active"
                            })
                    except Exception as e:
                        print(f"EventBus initialization error in process_test: {e}")
            def get(self):
                if self.get_argument("exit", None):
                    # must use os._exit instead of sys.exit so unittest's
                    # exception handler doesn't catch it
                    os._exit(int(self.get_argument("exit")))
                if self.get_argument("signal", None):
                    os.kill(os.getpid(), int(self.get_argument("signal")))
                self.write(str(os.getpid()))

        return Application([("/", ProcessHandler)])

    def tearDown(self):
        if task_id() is not None:
            # We're in a child process, and probably got to this point
            # via an uncaught exception.  If we return now, both
            # processes will continue with the rest of the test suite.
            # Exit now so the parent process will restart the child
            # (since we don't have a clean way to signal failure to
            # the parent that won't restart)
            logging.error("aborting child process from tearDown")
            logging.shutdown()
            os._exit(1)
        # In the surviving process, clear the alarm we set earlier
        signal.alarm(0)
        super().tearDown()

    def test_multi_process(self):
        # This test doesn't work on twisted because we use the global
        # reactor and don't restore it to a sane state after the fork
        # (asyncio has the same issue, but we have a special case in
        # place for it).
        with ExpectLog(
            gen_log, "(Starting .* processes|child .* exited|uncaught exception)"
        ):
            sock, port = bind_unused_port()

            def get_url(path):
                return "http://127.0.0.1:%d%s" % (port, path)

            # ensure that none of these processes live too long
            signal.alarm(5)  # master process
            try:
                id = fork_processes(3, max_restarts=3)
                self.assertIsNotNone(id)
                signal.alarm(5)  # child processes
            except SystemExit as e:
                # if we exit cleanly from fork_processes, all the child processes
                # finished with status 0
                self.assertEqual(e.code, 0)
                self.assertIsNone(task_id())
                sock.close()
                return
            try:
                if id in (0, 1):
                    self.assertEqual(id, task_id())

                    async def f():
                        server = HTTPServer(self.get_app())
                        server.add_sockets([sock])
                        await asyncio.Event().wait()

                    asyncio.run(f())
                elif id == 2:
                    self.assertEqual(id, task_id())
                    sock.close()
                    # Always use SimpleAsyncHTTPClient here; the curl
                    # version appears to get confused sometimes if the
                    # connection gets closed before it's had a chance to
                    # switch from writing mode to reading mode.
                    client = HTTPClient(SimpleAsyncHTTPClient)

                    def fetch(url, fail_ok=False):
                        try:
                            return client.fetch(get_url(url))
                        except HTTPError as e:
                            if not (fail_ok and e.code == 599):
                                raise

                    # Make two processes exit abnormally
                    fetch("/?exit=2", fail_ok=True)
                    fetch("/?exit=3", fail_ok=True)

                    # They've been restarted, so a new fetch will work
                    int(fetch("/").body)

                    # Now the same with signals
                    # Disabled because on the mac a process dying with a signal
                    # can trigger an "Application exited abnormally; send error
                    # report to Apple?" prompt.
                    # fetch("/?signal=%d" % signal.SIGTERM, fail_ok=True)
                    # fetch("/?signal=%d" % signal.SIGABRT, fail_ok=True)
                    # int(fetch("/").body)

                    # Now kill them normally so they won't be restarted
                    fetch("/?exit=0", fail_ok=True)
                    # One process left; watch it's pid change
                    pid = int(fetch("/").body)
                    fetch("/?exit=4", fail_ok=True)
                    pid2 = int(fetch("/").body)
                    self.assertNotEqual(pid, pid2)

                    # Kill the last one so we shut down cleanly
                    fetch("/?exit=0", fail_ok=True)

                    os._exit(0)
            except Exception:
                logging.error("exception in child process %d", id, exc_info=True)
                raise


@skipIfNonUnix
class SubprocessTest(AsyncTestCase):
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

            emit_telemetry("process_test", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("process_test", "position_calculated", {
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
                        "module": "process_test",
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
                print(f"Emergency stop error in process_test: {e}")
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
                "module": "process_test",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("process_test", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in process_test: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "process_test",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in process_test: {e}")
    def term_and_wait(self, subproc):
        subproc.proc.terminate()
        subproc.proc.wait()

    @gen_test
    def test_subprocess(self):
        subproc = Subprocess(
            [sys.executable, "-u", "-i"],
            stdin=Subprocess.STREAM,
            stdout=Subprocess.STREAM,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(lambda: self.term_and_wait(subproc))
        self.addCleanup(subproc.stdout.close)
        self.addCleanup(subproc.stdin.close)
        yield subproc.stdout.read_until(b">>> ")
        subproc.stdin.write(b"print('hello')\n")
        data = yield subproc.stdout.read_until(b"\n")
        self.assertEqual(data, b"hello\n")

        yield subproc.stdout.read_until(b">>> ")
        subproc.stdin.write(b"raise SystemExit\n")
        data = yield subproc.stdout.read_until_close()
        self.assertEqual(data, b"")

    @gen_test
    def test_close_stdin(self):
        # Close the parent's stdin handle and see that the child recognizes it.
        subproc = Subprocess(
            [sys.executable, "-u", "-i"],
            stdin=Subprocess.STREAM,
            stdout=Subprocess.STREAM,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(lambda: self.term_and_wait(subproc))
        yield subproc.stdout.read_until(b">>> ")
        subproc.stdin.close()
        data = yield subproc.stdout.read_until_close()
        self.assertEqual(data, b"\n")

    @gen_test
    def test_stderr(self):
        # This test is mysteriously flaky on twisted: it succeeds, but logs
        # an error of EBADF on closing a file descriptor.
        subproc = Subprocess(
            [sys.executable, "-u", "-c", r"import sys; sys.stderr.write('hello\n')"],
            stderr=Subprocess.STREAM,
        )
        self.addCleanup(lambda: self.term_and_wait(subproc))
        data = yield subproc.stderr.read_until(b"\n")
        self.assertEqual(data, b"hello\n")
        # More mysterious EBADF: This fails if done with self.addCleanup instead of here.
        subproc.stderr.close()

    def test_sigchild(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, "-c", "pass"])
        subproc.set_exit_callback(self.stop)
        ret = self.wait()
        self.assertEqual(ret, 0)
        self.assertEqual(subproc.returncode, ret)

    @gen_test
    def test_sigchild_future(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, "-c", "pass"])
        ret = yield subproc.wait_for_exit()
        self.assertEqual(ret, 0)
        self.assertEqual(subproc.returncode, ret)

    def test_sigchild_signal(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=Subprocess.STREAM,
        )
        self.addCleanup(subproc.stdout.close)
        subproc.set_exit_callback(self.stop)

        # For unclear reasons, killing a process too soon after
        # creating it can result in an exit status corresponding to
        # SIGKILL instead of the actual signal involved. This has been
        # observed on macOS 10.15 with Python 3.8 installed via brew,
        # but not with the system-installed Python 3.7.
        time.sleep(0.1)

        os.kill(subproc.pid, signal.SIGTERM)
        try:
            ret = self.wait()
        except AssertionError:
            # We failed to get the termination signal. This test is
            # occasionally flaky on pypy, so try to get a little more
            # information: did the process close its stdout
            # (indicating that the problem is in the parent process's
            # signal handling) or did the child process somehow fail
            # to terminate?
            fut = subproc.stdout.read_until_close()
            fut.add_done_callback(lambda f: self.stop())  # type: ignore
            try:
                self.wait()
            except AssertionError:
                raise AssertionError("subprocess failed to terminate")
            else:
                raise AssertionError(
                    "subprocess closed stdout but failed to " "get termination signal"
                )
        self.assertEqual(subproc.returncode, ret)
        self.assertEqual(ret, -signal.SIGTERM)

    @gen_test
    def test_wait_for_exit_raise(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, "-c", "import sys; sys.exit(1)"])
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            yield subproc.wait_for_exit()
        self.assertEqual(cm.exception.returncode, 1)

    @gen_test
    def test_wait_for_exit_raise_disabled(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, "-c", "import sys; sys.exit(1)"])
        ret = yield subproc.wait_for_exit(raise_error=False)
        self.assertEqual(ret, 1)


# <!-- @GENESIS_MODULE_END: process_test -->
