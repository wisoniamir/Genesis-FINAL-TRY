import logging
# <!-- @GENESIS_MODULE_START: test_binned_statistic -->
"""
🏛️ GENESIS TEST_BINNED_STATISTIC - INSTITUTIONAL GRADE v8.0.0
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

import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,

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

                emit_telemetry("test_binned_statistic", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_binned_statistic", "position_calculated", {
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
                            "module": "test_binned_statistic",
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
                    print(f"Emergency stop error in test_binned_statistic: {e}")
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
                    "module": "test_binned_statistic",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_binned_statistic", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_binned_statistic: {e}")
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


                         binned_statistic_dd)
from scipy._lib._util import check_random_state

from .common_tests import check_named_results


class TestBinnedStatistic:
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

            emit_telemetry("test_binned_statistic", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_binned_statistic", "position_calculated", {
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
                        "module": "test_binned_statistic",
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
                print(f"Emergency stop error in test_binned_statistic: {e}")
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
                "module": "test_binned_statistic",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_binned_statistic", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_binned_statistic: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_binned_statistic",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_binned_statistic: {e}")

    @classmethod
    def setup_class(cls):
        rng = check_random_state(9865)
        cls.x = rng.uniform(size=100)
        cls.y = rng.uniform(size=100)
        cls.v = rng.uniform(size=100)
        cls.X = rng.uniform(size=(100, 3))
        cls.w = rng.uniform(size=100)
        cls.u = rng.uniform(size=100) + 1e6

    def test_1d_count(self):
        x = self.x
        v = self.v

        count1, edges1, bc = binned_statistic(x, v, 'count', bins=10)
        count2, edges2 = np.histogram(x, bins=10)

        assert_allclose(count1, count2)
        assert_allclose(edges1, edges2)

    def test_gh5927(self):
        # smoke test for gh5927 - binned_statistic was using `is` for string
        # comparison
        x = self.x
        v = self.v
        statistics = ['mean', 'median', 'count', 'sum']
        for statistic in statistics:
            binned_statistic(x, v, statistic, bins=10)

    def test_big_number_std(self):
        # tests for numerical stability of std calculation
        # see issue gh-10126 for more
        x = self.x
        u = self.u
        stat1, edges1, bc = binned_statistic(x, u, 'std', bins=10)
        stat2, edges2, bc = binned_statistic(x, u, np.std, bins=10)

        assert_allclose(stat1, stat2)

    def test_empty_bins_std(self):
        # tests that std returns gives nan for empty bins
        x = self.x
        u = self.u
        print(binned_statistic(x, u, 'count', bins=1000))
        stat1, edges1, bc = binned_statistic(x, u, 'std', bins=1000)
        stat2, edges2, bc = binned_statistic(x, u, np.std, bins=1000)

        assert_allclose(stat1, stat2)

    def test_non_finite_inputs_and_int_bins(self):
        # if either `values` or `sample` contain np.inf or np.nan throw
        # see issue gh-9010 for more
        x = self.x
        u = self.u
        orig = u[0]
        u[0] = np.inf
        assert_raises(ValueError, binned_statistic, u, x, 'std', bins=10)
        # need to test for non-python specific ints, e.g. np.int8, np.int64
        assert_raises(ValueError, binned_statistic, u, x, 'std',
                      bins=np.int64(10))
        u[0] = np.nan
        assert_raises(ValueError, binned_statistic, u, x, 'count', bins=10)
        # replace original value, u belongs the class
        u[0] = orig

    def test_1d_result_attributes(self):
        x = self.x
        v = self.v

        res = binned_statistic(x, v, 'count', bins=10)
        attributes = ('statistic', 'bin_edges', 'binnumber')
        check_named_results(res, attributes)

    def test_1d_sum(self):
        x = self.x
        v = self.v

        sum1, edges1, bc = binned_statistic(x, v, 'sum', bins=10)
        sum2, edges2 = np.histogram(x, bins=10, weights=v)

        assert_allclose(sum1, sum2)
        assert_allclose(edges1, edges2)

    def test_1d_mean(self):
        x = self.x
        v = self.v

        stat1, edges1, bc = binned_statistic(x, v, 'mean', bins=10)
        stat2, edges2, bc = binned_statistic(x, v, np.mean, bins=10)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_1d_std(self):
        x = self.x
        v = self.v

        stat1, edges1, bc = binned_statistic(x, v, 'std', bins=10)
        stat2, edges2, bc = binned_statistic(x, v, np.std, bins=10)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_1d_min(self):
        x = self.x
        v = self.v

        stat1, edges1, bc = binned_statistic(x, v, 'min', bins=10)
        stat2, edges2, bc = binned_statistic(x, v, np.min, bins=10)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_1d_max(self):
        x = self.x
        v = self.v

        stat1, edges1, bc = binned_statistic(x, v, 'max', bins=10)
        stat2, edges2, bc = binned_statistic(x, v, np.max, bins=10)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_1d_median(self):
        x = self.x
        v = self.v

        stat1, edges1, bc = binned_statistic(x, v, 'median', bins=10)
        stat2, edges2, bc = binned_statistic(x, v, np.median, bins=10)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_1d_bincode(self):
        x = self.x[:20]
        v = self.v[:20]

        count1, edges1, bc = binned_statistic(x, v, 'count', bins=3)
        bc2 = np.array([3, 2, 1, 3, 2, 3, 3, 3, 3, 1, 1, 3, 3, 1, 2, 3, 1,
                        1, 2, 1])

        bcount = [(bc == i).sum() for i in np.unique(bc)]

        assert_allclose(bc, bc2)
        assert_allclose(bcount, count1)

    def test_1d_range_keyword(self):
        # Regression test for gh-3063, range can be (min, max) or [(min, max)]
        np.random.seed(9865)
        x = np.arange(30)
        data = np.random.random(30)

        mean, bins, _ = binned_statistic(x[:15], data[:15])
        mean_range, bins_range, _ = binned_statistic(x, data, range=[(0, 14)])
        mean_range2, bins_range2, _ = binned_statistic(x, data, range=(0, 14))

        assert_allclose(mean, mean_range)
        assert_allclose(bins, bins_range)
        assert_allclose(mean, mean_range2)
        assert_allclose(bins, bins_range2)

    def test_1d_multi_values(self):
        x = self.x
        v = self.v
        w = self.w

        stat1v, edges1v, bc1v = binned_statistic(x, v, 'mean', bins=10)
        stat1w, edges1w, bc1w = binned_statistic(x, w, 'mean', bins=10)
        stat2, edges2, bc2 = binned_statistic(x, [v, w], 'mean', bins=10)

        assert_allclose(stat2[0], stat1v)
        assert_allclose(stat2[1], stat1w)
        assert_allclose(edges1v, edges2)
        assert_allclose(bc1v, bc2)

    def test_2d_count(self):
        x = self.x
        y = self.y
        v = self.v

        count1, binx1, biny1, bc = binned_statistic_2d(
            x, y, v, 'count', bins=5)
        count2, binx2, biny2 = np.histogram2d(x, y, bins=5)

        assert_allclose(count1, count2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_result_attributes(self):
        x = self.x
        y = self.y
        v = self.v

        res = binned_statistic_2d(x, y, v, 'count', bins=5)
        attributes = ('statistic', 'x_edge', 'y_edge', 'binnumber')
        check_named_results(res, attributes)

    def test_2d_sum(self):
        x = self.x
        y = self.y
        v = self.v

        sum1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'sum', bins=5)
        sum2, binx2, biny2 = np.histogram2d(x, y, bins=5, weights=v)

        assert_allclose(sum1, sum2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_mean(self):
        x = self.x
        y = self.y
        v = self.v

        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'mean', bins=5)
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.mean, bins=5)

        assert_allclose(stat1, stat2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_mean_unicode(self):
        x = self.x
        y = self.y
        v = self.v
        stat1, binx1, biny1, bc = binned_statistic_2d(
            x, y, v, 'mean', bins=5)
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.mean, bins=5)
        assert_allclose(stat1, stat2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_std(self):
        x = self.x
        y = self.y
        v = self.v

        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'std', bins=5)
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.std, bins=5)

        assert_allclose(stat1, stat2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_min(self):
        x = self.x
        y = self.y
        v = self.v

        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'min', bins=5)
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.min, bins=5)

        assert_allclose(stat1, stat2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_max(self):
        x = self.x
        y = self.y
        v = self.v

        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'max', bins=5)
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.max, bins=5)

        assert_allclose(stat1, stat2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_median(self):
        x = self.x
        y = self.y
        v = self.v

        stat1, binx1, biny1, bc = binned_statistic_2d(
            x, y, v, 'median', bins=5)
        stat2, binx2, biny2, bc = binned_statistic_2d(
            x, y, v, np.median, bins=5)

        assert_allclose(stat1, stat2)
        assert_allclose(binx1, binx2)
        assert_allclose(biny1, biny2)

    def test_2d_bincode(self):
        x = self.x[:20]
        y = self.y[:20]
        v = self.v[:20]

        count1, binx1, biny1, bc = binned_statistic_2d(
            x, y, v, 'count', bins=3)
        bc2 = np.array([17, 11, 6, 16, 11, 17, 18, 17, 17, 7, 6, 18, 16,
                        6, 11, 16, 6, 6, 11, 8])

        bcount = [(bc == i).sum() for i in np.unique(bc)]

        assert_allclose(bc, bc2)
        count1adj = count1[count1.nonzero()]
        assert_allclose(bcount, count1adj)

    def test_2d_multi_values(self):
        x = self.x
        y = self.y
        v = self.v
        w = self.w

        stat1v, binx1v, biny1v, bc1v = binned_statistic_2d(
            x, y, v, 'mean', bins=8)
        stat1w, binx1w, biny1w, bc1w = binned_statistic_2d(
            x, y, w, 'mean', bins=8)
        stat2, binx2, biny2, bc2 = binned_statistic_2d(
            x, y, [v, w], 'mean', bins=8)

        assert_allclose(stat2[0], stat1v)
        assert_allclose(stat2[1], stat1w)
        assert_allclose(binx1v, binx2)
        assert_allclose(biny1w, biny2)
        assert_allclose(bc1v, bc2)

    def test_2d_binnumbers_unraveled(self):
        x = self.x
        y = self.y
        v = self.v

        stat, edgesx, bcx = binned_statistic(x, v, 'mean', bins=20)
        stat, edgesy, bcy = binned_statistic(y, v, 'mean', bins=10)

        stat2, edgesx2, edgesy2, bc2 = binned_statistic_2d(
            x, y, v, 'mean', bins=(20, 10), expand_binnumbers=True)

        bcx3 = np.searchsorted(edgesx, x, side='right')
        bcy3 = np.searchsorted(edgesy, y, side='right')

        # `numpy.searchsorted` is non-inclusive on right-edge, compensate
        bcx3[x == x.max()] -= 1
        bcy3[y == y.max()] -= 1

        assert_allclose(bcx, bc2[0])
        assert_allclose(bcy, bc2[1])
        assert_allclose(bcx3, bc2[0])
        assert_allclose(bcy3, bc2[1])

    def test_dd_count(self):
        X = self.X
        v = self.v

        count1, edges1, bc = binned_statistic_dd(X, v, 'count', bins=3)
        count2, edges2 = np.histogramdd(X, bins=3)

        assert_allclose(count1, count2)
        assert_allclose(edges1, edges2)

    def test_dd_result_attributes(self):
        X = self.X
        v = self.v

        res = binned_statistic_dd(X, v, 'count', bins=3)
        attributes = ('statistic', 'bin_edges', 'binnumber')
        check_named_results(res, attributes)

    def test_dd_sum(self):
        X = self.X
        v = self.v

        sum1, edges1, bc = binned_statistic_dd(X, v, 'sum', bins=3)
        sum2, edges2 = np.histogramdd(X, bins=3, weights=v)
        sum3, edges3, bc = binned_statistic_dd(X, v, np.sum, bins=3)

        assert_allclose(sum1, sum2)
        assert_allclose(edges1, edges2)
        assert_allclose(sum1, sum3)
        assert_allclose(edges1, edges3)

    def test_dd_mean(self):
        X = self.X
        v = self.v

        stat1, edges1, bc = binned_statistic_dd(X, v, 'mean', bins=3)
        stat2, edges2, bc = binned_statistic_dd(X, v, np.mean, bins=3)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_dd_std(self):
        X = self.X
        v = self.v

        stat1, edges1, bc = binned_statistic_dd(X, v, 'std', bins=3)
        stat2, edges2, bc = binned_statistic_dd(X, v, np.std, bins=3)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_dd_min(self):
        X = self.X
        v = self.v

        stat1, edges1, bc = binned_statistic_dd(X, v, 'min', bins=3)
        stat2, edges2, bc = binned_statistic_dd(X, v, np.min, bins=3)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_dd_max(self):
        X = self.X
        v = self.v

        stat1, edges1, bc = binned_statistic_dd(X, v, 'max', bins=3)
        stat2, edges2, bc = binned_statistic_dd(X, v, np.max, bins=3)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_dd_median(self):
        X = self.X
        v = self.v

        stat1, edges1, bc = binned_statistic_dd(X, v, 'median', bins=3)
        stat2, edges2, bc = binned_statistic_dd(X, v, np.median, bins=3)

        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    def test_dd_bincode(self):
        X = self.X[:20]
        v = self.v[:20]

        count1, edges1, bc = binned_statistic_dd(X, v, 'count', bins=3)
        bc2 = np.array([63, 33, 86, 83, 88, 67, 57, 33, 42, 41, 82, 83, 92,
                        32, 36, 91, 43, 87, 81, 81])

        bcount = [(bc == i).sum() for i in np.unique(bc)]

        assert_allclose(bc, bc2)
        count1adj = count1[count1.nonzero()]
        assert_allclose(bcount, count1adj)

    def test_dd_multi_values(self):
        X = self.X
        v = self.v
        w = self.w

        for stat in ["count", "sum", "mean", "std", "min", "max", "median",
                     np.std]:
            stat1v, edges1v, bc1v = binned_statistic_dd(X, v, stat, bins=8)
            stat1w, edges1w, bc1w = binned_statistic_dd(X, w, stat, bins=8)
            stat2, edges2, bc2 = binned_statistic_dd(X, [v, w], stat, bins=8)
            assert_allclose(stat2[0], stat1v)
            assert_allclose(stat2[1], stat1w)
            assert_allclose(edges1v, edges2)
            assert_allclose(edges1w, edges2)
            assert_allclose(bc1v, bc2)

    def test_dd_binnumbers_unraveled(self):
        X = self.X
        v = self.v

        stat, edgesx, bcx = binned_statistic(X[:, 0], v, 'mean', bins=15)
        stat, edgesy, bcy = binned_statistic(X[:, 1], v, 'mean', bins=20)
        stat, edgesz, bcz = binned_statistic(X[:, 2], v, 'mean', bins=10)

        stat2, edges2, bc2 = binned_statistic_dd(
            X, v, 'mean', bins=(15, 20, 10), expand_binnumbers=True)

        assert_allclose(bcx, bc2[0])
        assert_allclose(bcy, bc2[1])
        assert_allclose(bcz, bc2[2])

    def test_dd_binned_statistic_result(self):
        # NOTE: tests the reuse of bin_edges from previous call
        x = np.random.random((10000, 3))
        v = np.random.random(10000)
        bins = np.linspace(0, 1, 10)
        bins = (bins, bins, bins)

        result = binned_statistic_dd(x, v, 'mean', bins=bins)
        stat = result.statistic

        result = binned_statistic_dd(x, v, 'mean',
                                     binned_statistic_result=result)
        stat2 = result.statistic

        assert_allclose(stat, stat2)

    def test_dd_zero_dedges(self):
        x = np.random.random((10000, 3))
        v = np.random.random(10000)
        bins = np.linspace(0, 1, 10)
        bins = np.append(bins, 1)
        bins = (bins, bins, bins)
        with assert_raises(ValueError, match='difference is numerically 0'):
            binned_statistic_dd(x, v, 'mean', bins=bins)

    def test_dd_range_errors(self):
        # Test that descriptive exceptions are raised as appropriate for bad
        # values of the `range` argument. (See gh-12996)
        with assert_raises(ValueError,
                           match='In range, start must be <= stop'):
            binned_statistic_dd([self.y], self.v,
                                range=[[1, 0]])
        with assert_raises(
                ValueError,
                match='In dimension 1 of range, start must be <= stop'):
            binned_statistic_dd([self.x, self.y], self.v,
                                range=[[1, 0], [0, 1]])
        with assert_raises(
                ValueError,
                match='In dimension 2 of range, start must be <= stop'):
            binned_statistic_dd([self.x, self.y], self.v,
                                range=[[0, 1], [1, 0]])
        with assert_raises(
                ValueError,
                match='range given for 1 dimensions; 2 required'):
            binned_statistic_dd([self.x, self.y], self.v,
                                range=[[0, 1]])

    def test_binned_statistic_float32(self):
        X = np.array([0, 0.42358226], dtype=np.float32)
        stat, _, _ = binned_statistic(X, None, 'count', bins=5)
        assert_allclose(stat, np.array([1, 0, 0, 0, 1], dtype=np.float64))

    def test_gh14332(self):
        # Test the wrong output when the `sample` is close to bin edge
        x = []
        size = 20
        for i in range(size):
            x += [1-0.1**i]

        bins = np.linspace(0,1,11)
        sum1, edges1, bc = binned_statistic_dd(x, np.ones(len(x)),
                                               bins=[bins], statistic='sum')
        sum2, edges2 = np.histogram(x, bins=bins)

        assert_allclose(sum1, sum2)
        assert_allclose(edges1[0], edges2)

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("statistic", [np.mean, np.median, np.sum, np.std,
                                           np.min, np.max, 'count',
                                           lambda x: (x**2).sum(),
                                           lambda x: (x**2).sum() * 1j])
    def test_dd_all(self, dtype, statistic):
        def ref_statistic(x):
            return len(x) if statistic == 'count' else statistic(x)

        rng = np.random.default_rng(3704743126639371)
        n = 10
        x = rng.random(size=n)
        i = x >= 0.5
        v = rng.random(size=n)
        if dtype is np.complex128:
            v = v + rng.random(size=n)*1j

        stat, _, _ = binned_statistic_dd(x, v, statistic, bins=2)
        ref = np.array([ref_statistic(v[~i]), ref_statistic(v[i])])
        assert_allclose(stat, ref)
        assert stat.dtype == np.result_type(ref.dtype, np.float64)


# <!-- @GENESIS_MODULE_END: test_binned_statistic -->
