import logging
# <!-- @GENESIS_MODULE_START: test_censored_data -->
"""
🏛️ GENESIS TEST_CENSORED_DATA - INSTITUTIONAL GRADE v8.0.0
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

# Tests for the CensoredData class.

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData

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

                emit_telemetry("test_censored_data", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_censored_data", "position_calculated", {
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
                            "module": "test_censored_data",
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
                    print(f"Emergency stop error in test_censored_data: {e}")
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
                    "module": "test_censored_data",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_censored_data", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_censored_data: {e}")
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




class TestCensoredData:
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

            emit_telemetry("test_censored_data", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_censored_data", "position_calculated", {
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
                        "module": "test_censored_data",
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
                print(f"Emergency stop error in test_censored_data: {e}")
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
                "module": "test_censored_data",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_censored_data", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_censored_data: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_censored_data",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_censored_data: {e}")

    def test_basic(self):
        uncensored = [1]
        left = [0]
        right = [2, 5]
        interval = [[2, 3]]
        data = CensoredData(uncensored, left=left, right=right,
                            interval=interval)
        assert_equal(data._uncensored, uncensored)
        assert_equal(data._left, left)
        assert_equal(data._right, right)
        assert_equal(data._interval, interval)

        udata = data._uncensor()
        assert_equal(udata, np.concatenate((uncensored, left, right,
                                            np.mean(interval, axis=1))))

    def test_right_censored(self):
        x = np.array([0, 3, 2.5])
        is_censored = np.array([0, 1, 0], dtype=bool)
        data = CensoredData.right_censored(x, is_censored)
        assert_equal(data._uncensored, x[~is_censored])
        assert_equal(data._right, x[is_censored])
        assert_equal(data._left, [])
        assert_equal(data._interval, np.empty((0, 2)))

    def test_left_censored(self):
        x = np.array([0, 3, 2.5])
        is_censored = np.array([0, 1, 0], dtype=bool)
        data = CensoredData.left_censored(x, is_censored)
        assert_equal(data._uncensored, x[~is_censored])
        assert_equal(data._left, x[is_censored])
        assert_equal(data._right, [])
        assert_equal(data._interval, np.empty((0, 2)))

    def test_interval_censored_basic(self):
        a = [0.5, 2.0, 3.0, 5.5]
        b = [1.0, 2.5, 3.5, 7.0]
        data = CensoredData.interval_censored(low=a, high=b)
        assert_array_equal(data._interval, np.array(list(zip(a, b))))
        assert data._uncensored.shape == (0,)
        assert data._left.shape == (0,)
        assert data._right.shape == (0,)

    def test_interval_censored_mixed(self):
        # This is actually a mix of uncensored, left-censored, right-censored
        # and interval-censored data.  Check that when the `interval_censored`
        # class method is used, the data is correctly separated into the
        # appropriate arrays.
        a = [0.5, -np.inf, -13.0, 2.0, 1.0, 10.0, -1.0]
        b = [0.5, 2500.0, np.inf, 3.0, 1.0, 11.0, np.inf]
        data = CensoredData.interval_censored(low=a, high=b)
        assert_array_equal(data._interval, [[2.0, 3.0], [10.0, 11.0]])
        assert_array_equal(data._uncensored, [0.5, 1.0])
        assert_array_equal(data._left, [2500.0])
        assert_array_equal(data._right, [-13.0, -1.0])

    def test_interval_to_other_types(self):
        # The interval parameter can represent uncensored and
        # left- or right-censored data.  Test the conversion of such
        # an example to the canonical form in which the different
        # types have been split into the separate arrays.
        interval = np.array([[0, 1],        # interval-censored
                             [2, 2],        # not censored
                             [3, 3],        # not censored
                             [9, np.inf],   # right-censored
                             [8, np.inf],   # right-censored
                             [-np.inf, 0],  # left-censored
                             [1, 2]])       # interval-censored
        data = CensoredData(interval=interval)
        assert_equal(data._uncensored, [2, 3])
        assert_equal(data._left, [0])
        assert_equal(data._right, [9, 8])
        assert_equal(data._interval, [[0, 1], [1, 2]])

    def test_empty_arrays(self):
        data = CensoredData(uncensored=[], left=[], right=[], interval=[])
        assert data._uncensored.shape == (0,)
        assert data._left.shape == (0,)
        assert data._right.shape == (0,)
        assert data._interval.shape == (0, 2)
        assert len(data) == 0

    def test_invalid_constructor_args(self):
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(uncensored=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(left=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(right=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a two-dimensional'):
            CensoredData(interval=[[1, 2, 3]])

        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(uncensored=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(left=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(right=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(interval=[[1, np.nan], [2, 3]])

        with pytest.raises(ValueError,
                           match='both values must not be infinite'):
            CensoredData(interval=[[1, 3], [2, 9], [np.inf, np.inf]])

        with pytest.raises(ValueError,
                           match='left value must not exceed the right'):
            CensoredData(interval=[[1, 0], [2, 2]])

    @pytest.mark.parametrize('func', [CensoredData.left_censored,
                                      CensoredData.right_censored])
    def test_invalid_left_right_censored_args(self, func):
        with pytest.raises(ValueError,
                           match='`x` must be one-dimensional'):
            func([[1, 2, 3]], [0, 1, 1])
        with pytest.raises(ValueError,
                           match='`censored` must be one-dimensional'):
            func([1, 2, 3], [[0, 1, 1]])
        with pytest.raises(ValueError, match='`x` must not contain'):
            func([1, 2, np.nan], [0, 1, 1])
        with pytest.raises(ValueError, match='must have the same length'):
            func([1, 2, 3], [0, 0, 1, 1])

    def test_invalid_censored_args(self):
        with pytest.raises(ValueError,
                           match='`low` must be a one-dimensional'):
            CensoredData.interval_censored(low=[[3]], high=[4, 5])
        with pytest.raises(ValueError,
                           match='`high` must be a one-dimensional'):
            CensoredData.interval_censored(low=[3], high=[[4, 5]])
        with pytest.raises(ValueError, match='`low` must not contain'):
            CensoredData.interval_censored([1, 2, np.nan], [0, 1, 1])
        with pytest.raises(ValueError, match='must have the same length'):
            CensoredData.interval_censored([1, 2, 3], [0, 0, 1, 1])

    def test_count_censored(self):
        x = [1, 2, 3]
        # data1 has no censored data.
        data1 = CensoredData(x)
        assert data1.num_censored() == 0
        data2 = CensoredData(uncensored=[2.5], left=[10], interval=[[0, 1]])
        assert data2.num_censored() == 2


# <!-- @GENESIS_MODULE_END: test_censored_data -->
