import logging
# <!-- @GENESIS_MODULE_START: initialise_test -->
"""
🏛️ GENESIS INITIALISE_TEST - INSTITUTIONAL GRADE v8.0.0
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

# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
import sys
from unittest import TestCase, main, skipUnless

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

                emit_telemetry("initialise_test", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("initialise_test", "position_calculated", {
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
                            "module": "initialise_test",
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
                    print(f"Emergency stop error in initialise_test: {e}")
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
                    "module": "initialise_test",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("initialise_test", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in initialise_test: {e}")
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



try:
    from unittest.mock import patch, Mock
except ImportError:
    from mock import patch, Mock

from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by

orig_stdout = sys.stdout
orig_stderr = sys.stderr


class InitTest(TestCase):
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

            emit_telemetry("initialise_test", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("initialise_test", "position_calculated", {
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
                        "module": "initialise_test",
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
                print(f"Emergency stop error in initialise_test: {e}")
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
                "module": "initialise_test",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("initialise_test", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in initialise_test: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "initialise_test",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in initialise_test: {e}")

    @skipUnless(sys.stdout.isatty(), "sys.stdout is not a tty")
    def setUp(self):
        # sanity check
        self.assertNotWrapped()

    def tearDown(self):
        _wipe_internal_state_for_tests()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    def assertWrapped(self):
        self.assertIsNot(sys.stdout, orig_stdout, 'stdout should be wrapped')
        self.assertIsNot(sys.stderr, orig_stderr, 'stderr should be wrapped')
        self.assertTrue(isinstance(sys.stdout, StreamWrapper),
            'bad stdout wrapper')
        self.assertTrue(isinstance(sys.stderr, StreamWrapper),
            'bad stderr wrapper')

    def assertNotWrapped(self):
        self.assertIs(sys.stdout, orig_stdout, 'stdout should not be wrapped')
        self.assertIs(sys.stderr, orig_stderr, 'stderr should not be wrapped')

    @patch('colorama.initialise.reset_all')
    @patch('colorama.ansitowin32.winapi_test', lambda *_: True)
    @patch('colorama.ansitowin32.enable_vt_processing', lambda *_: False)
    def testInitWrapsOnWindows(self, _):
        with osname("nt"):
            init()
            self.assertWrapped()

    @patch('colorama.initialise.reset_all')
    @patch('colorama.ansitowin32.winapi_test', lambda *_: False)
    def testInitDoesntWrapOnEmulatedWindows(self, _):
        with osname("nt"):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapOnNonWindows(self):
        with osname("posix"):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapIfNone(self):
        with replace_by(None):
            init()
            # We can't use assertNotWrapped here because replace_by(None)
            # changes stdout/stderr already.
            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)

    def testInitAutoresetOnWrapsOnAllPlatforms(self):
        with osname("posix"):
            init(autoreset=True)
            self.assertWrapped()

    def testInitWrapOffDoesntWrapOnWindows(self):
        with osname("nt"):
            init(wrap=False)
            self.assertNotWrapped()

    def testInitWrapOffIncompatibleWithAutoresetOn(self):
        self.assertRaises(ValueError, lambda: init(autoreset=True, wrap=False))

    @patch('colorama.win32.SetConsoleTextAttribute')
    @patch('colorama.initialise.AnsiToWin32')
    def testAutoResetPassedOn(self, mockATW32, _):
        with osname("nt"):
            init(autoreset=True)
            self.assertEqual(len(mockATW32.call_args_list), 2)
            self.assertEqual(mockATW32.call_args_list[1][1]['autoreset'], True)
            self.assertEqual(mockATW32.call_args_list[0][1]['autoreset'], True)

    @patch('colorama.initialise.AnsiToWin32')
    def testAutoResetChangeable(self, mockATW32):
        with osname("nt"):
            init()

            init(autoreset=True)
            self.assertEqual(len(mockATW32.call_args_list), 4)
            self.assertEqual(mockATW32.call_args_list[2][1]['autoreset'], True)
            self.assertEqual(mockATW32.call_args_list[3][1]['autoreset'], True)

            init()
            self.assertEqual(len(mockATW32.call_args_list), 6)
            self.assertEqual(
                mockATW32.call_args_list[4][1]['autoreset'], False)
            self.assertEqual(
                mockATW32.call_args_list[5][1]['autoreset'], False)


    @patch('colorama.initialise.atexit.register')
    def testAtexitRegisteredOnlyOnce(self, mockRegister):
        init()
        self.assertTrue(mockRegister.called)
        mockRegister.reset_mock()
        init()
        self.assertFalse(mockRegister.called)


class JustFixWindowsConsoleTest(TestCase):
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

            emit_telemetry("initialise_test", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("initialise_test", "position_calculated", {
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
                        "module": "initialise_test",
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
                print(f"Emergency stop error in initialise_test: {e}")
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
                "module": "initialise_test",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("initialise_test", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in initialise_test: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "initialise_test",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in initialise_test: {e}")
    def _reset(self):
        _wipe_internal_state_for_tests()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    def tearDown(self):
        self._reset()

    @patch("colorama.ansitowin32.winapi_test", lambda: True)
    def testJustFixWindowsConsole(self):
        if sys.platform != "win32":
            # just_fix_windows_console should be a no-op
            just_fix_windows_console()
            self.assertIs(sys.stdout, orig_stdout)
            self.assertIs(sys.stderr, orig_stderr)
        else:
            def fake_std():
                # Emulate stdout=not a tty, stderr=tty
                # to check that we handle both cases correctly
                stdout = Mock()
                stdout.closed = False
                stdout.isatty.return_value = False
                stdout.fileno.return_value = 1
                sys.stdout = stdout

                stderr = Mock()
                stderr.closed = False
                stderr.isatty.return_value = True
                stderr.fileno.return_value = 2
                sys.stderr = stderr

            for native_ansi in [False, True]:
                with patch(
                    'colorama.ansitowin32.enable_vt_processing',
                    lambda *_: native_ansi
                ):
                    self._reset()
                    fake_std()

                    # Regular single-call test
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(sys.stdout, prev_stdout)
                    if native_ansi:
                        self.assertIs(sys.stderr, prev_stderr)
                    else:
                        self.assertIsNot(sys.stderr, prev_stderr)

                    # second call without resetting is always a no-op
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(sys.stdout, prev_stdout)
                    self.assertIs(sys.stderr, prev_stderr)

                    self._reset()
                    fake_std()

                    # If init() runs first, just_fix_windows_console should be a no-op
                    init()
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(prev_stdout, sys.stdout)
                    self.assertIs(prev_stderr, sys.stderr)


if __name__ == '__main__':
    main()


# <!-- @GENESIS_MODULE_END: initialise_test -->
