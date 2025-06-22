import logging
# <!-- @GENESIS_MODULE_START: test_rank -->
"""
🏛️ GENESIS TEST_RANK - INSTITUTIONAL GRADE v8.0.0
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
from numpy.testing import assert_equal, assert_array_equal
import pytest

from scipy.conftest import skip_xp_invalid_arg
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long

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

                emit_telemetry("test_rank", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_rank", "position_calculated", {
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
                            "module": "test_rank",
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
                    print(f"Emergency stop error in test_rank: {e}")
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
                    "module": "test_rank",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_rank", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_rank: {e}")
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




class TestTieCorrect:
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

            emit_telemetry("test_rank", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_rank", "position_calculated", {
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
                        "module": "test_rank",
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
                print(f"Emergency stop error in test_rank: {e}")
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
                "module": "test_rank",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_rank", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_rank: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_rank",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_rank: {e}")

    def test_empty(self):
        """An empty array requires no correction, should return 1.0."""
        ranks = np.array([], dtype=np.float64)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_one(self):
        """A single element requires no correction, should return 1.0."""
        ranks = np.array([1.0], dtype=np.float64)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_no_correction(self):
        """Arrays with no ties require no correction."""
        ranks = np.arange(2.0)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)
        ranks = np.arange(3.0)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_basic(self):
        """Check a few basic examples of the tie correction factor."""
        # One tie of two elements
        ranks = np.array([1.0, 2.5, 2.5])
        c = tiecorrect(ranks)
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)
        assert_equal(c, expected)

        # One tie of two elements (same as above, but tie is not at the end)
        ranks = np.array([1.5, 1.5, 3.0])
        c = tiecorrect(ranks)
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)
        assert_equal(c, expected)

        # One tie of three elements
        ranks = np.array([1.0, 3.0, 3.0, 3.0])
        c = tiecorrect(ranks)
        T = 3.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)
        assert_equal(c, expected)

        # Two ties, lengths 2 and 3.
        ranks = np.array([1.5, 1.5, 4.0, 4.0, 4.0])
        c = tiecorrect(ranks)
        T1 = 2.0
        T2 = 3.0
        N = ranks.size
        expected = 1.0 - ((T1**3 - T1) + (T2**3 - T2)) / (N**3 - N)
        assert_equal(c, expected)

    def test_overflow(self):
        ntie, k = 2000, 5
        a = np.repeat(np.arange(k), ntie)
        n = a.size  # ntie * k
        out = tiecorrect(rankdata(a))
        assert_equal(out, 1.0 - k * (ntie**3 - ntie) / float(n**3 - n))


class TestRankData:
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

            emit_telemetry("test_rank", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_rank", "position_calculated", {
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
                        "module": "test_rank",
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
                print(f"Emergency stop error in test_rank: {e}")
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
                "module": "test_rank",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_rank", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_rank: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_rank",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_rank: {e}")

    def test_empty(self):
        """stats.rankdata([]) should return an empty array."""
        a = np.array([], dtype=int)
        r = rankdata(a)
        assert_array_equal(r, np.array([], dtype=np.float64))
        r = rankdata([])
        assert_array_equal(r, np.array([], dtype=np.float64))

    @pytest.mark.parametrize("shape", [(0, 1, 2)])
    @pytest.mark.parametrize("axis", [None, *range(3)])
    def test_empty_multidim(self, shape, axis):
        a = np.empty(shape, dtype=int)
        r = rankdata(a, axis=axis)
        expected_shape = (0,) if axis is None else shape
        assert_equal(r.shape, expected_shape)
        assert_equal(r.dtype, np.float64)

    def test_one(self):
        """Check stats.rankdata with an array of length 1."""
        data = [100]
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, np.array([1.0], dtype=np.float64))
        r = rankdata(data)
        assert_array_equal(r, np.array([1.0], dtype=np.float64))

    def test_basic(self):
        """Basic tests of stats.rankdata."""
        data = [100, 10, 50]
        expected = np.array([3.0, 1.0, 2.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)

        data = [40, 10, 30, 10, 50]
        expected = np.array([4.0, 1.5, 3.0, 1.5, 5.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)

        data = [20, 20, 20, 10, 10, 10]
        expected = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)
        # The docstring states explicitly that the argument is flattened.
        a2d = a.reshape(2, 3)
        r = rankdata(a2d)
        assert_array_equal(r, expected)

    @skip_xp_invalid_arg
    def test_rankdata_object_string(self):

        def min_rank(a):
            return [1 + sum(i < j for i in a) for j in a]

        def max_rank(a):
            return [sum(i <= j for i in a) for j in a]

        def ordinal_rank(a):
            return min_rank([(x, i) for i, x in enumerate(a)])

        def average_rank(a):
            return [(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))]

        def dense_rank(a):
            b = np.unique(a)
            return [1 + sum(i < j for i in b) for j in a]

        rankf = dict(min=min_rank, max=max_rank, ordinal=ordinal_rank,
                     average=average_rank, dense=dense_rank)

        def check_ranks(a):
            for method in 'min', 'max', 'dense', 'ordinal', 'average':
                out = rankdata(a, method=method)
                assert_array_equal(out, rankf[method](a))

        val = ['foo', 'bar', 'qux', 'xyz', 'abc', 'efg', 'ace', 'qwe', 'qaz']
        check_ranks(np.random.choice(val, 200))
        check_ranks(np.random.choice(val, 200).astype('object'))

        val = np.array([0, 1, 2, 2.718, 3, 3.141], dtype='object')
        check_ranks(np.random.choice(val, 200).astype('object'))

    def test_large_int(self):
        data = np.array([2**60, 2**60+1], dtype=np.uint64)
        r = rankdata(data)
        assert_array_equal(r, [1.0, 2.0])

        data = np.array([2**60, 2**60+1], dtype=np.int64)
        r = rankdata(data)
        assert_array_equal(r, [1.0, 2.0])

        data = np.array([2**60, -2**60+1], dtype=np.int64)
        r = rankdata(data)
        assert_array_equal(r, [2.0, 1.0])

    def test_big_tie(self):
        for n in [10000, 100000, 1000000]:
            data = np.ones(n, dtype=int)
            r = rankdata(data)
            expected_rank = 0.5 * (n + 1)
            assert_array_equal(r, expected_rank * data,
                               "test failed with n=%d" % n)

    def test_axis(self):
        data = [[0, 2, 1],
                [4, 2, 2]]
        expected0 = [[1., 1.5, 1.],
                     [2., 1.5, 2.]]
        r0 = rankdata(data, axis=0)
        assert_array_equal(r0, expected0)
        expected1 = [[1., 3., 2.],
                     [3., 1.5, 1.5]]
        r1 = rankdata(data, axis=1)
        assert_array_equal(r1, expected1)

    methods = ["average", "min", "max", "dense", "ordinal"]
    dtypes = [np.float64] + [np_long]*4

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("method, dtype", zip(methods, dtypes))
    def test_size_0_axis(self, axis, method, dtype):
        shape = (3, 0)
        data = np.zeros(shape)
        r = rankdata(data, method=method, axis=axis)
        assert_equal(r.shape, shape)
        assert_equal(r.dtype, dtype)

    @pytest.mark.parametrize('axis', range(3))
    @pytest.mark.parametrize('method', methods)
    def test_nan_policy_omit_3d(self, axis, method):
        shape = (20, 21, 22)
        rng = np.random.RandomState(23983242)

        a = rng.random(size=shape)
        i = rng.random(size=shape) < 0.4
        j = rng.random(size=shape) < 0.1
        k = rng.random(size=shape) < 0.1
        a[i] = np.nan
        a[j] = -np.inf
        a[k] - np.inf

        def rank_1d_omit(a, method):
            out = np.zeros_like(a)
            i = np.isnan(a)
            a_compressed = a[~i]
            res = rankdata(a_compressed, method)
            out[~i] = res
            out[i] = np.nan
            return out

        def rank_omit(a, method, axis):
            return np.apply_along_axis(lambda a: rank_1d_omit(a, method),
                                       axis, a)

        res = rankdata(a, method, axis=axis, nan_policy='omit')
        res0 = rank_omit(a, method, axis=axis)

        assert_array_equal(res, res0)

    def test_nan_policy_2d_axis_none(self):
        # 2 2d-array test with axis=None
        data = [[0, np.nan, 3],
                [4, 2, np.nan],
                [1, 2, 2]]
        assert_array_equal(rankdata(data, axis=None, nan_policy='omit'),
                           [1., np.nan, 6., 7., 4., np.nan, 2., 4., 4.])
        assert_array_equal(rankdata(data, axis=None, nan_policy='propagate'),
                           [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan])

    def test_nan_policy_raise(self):
        # 1 1d-array test
        data = [0, 2, 3, -2, np.nan, np.nan]
        with pytest.raises(ValueError, match="The input contains nan"):
            rankdata(data, nan_policy='raise')

        # 2 2d-array test
        data = [[0, np.nan, 3],
                [4, 2, np.nan],
                [np.nan, 2, 2]]

        with pytest.raises(ValueError, match="The input contains nan"):
            rankdata(data, axis=0, nan_policy="raise")

        with pytest.raises(ValueError, match="The input contains nan"):
            rankdata(data, axis=1, nan_policy="raise")

    def test_nan_policy_propagate(self):
        # 1 1d-array test
        data = [0, 2, 3, -2, np.nan, np.nan]
        assert_array_equal(rankdata(data, nan_policy='propagate'),
                           [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        # 2 2d-array test
        data = [[0, np.nan, 3],
                [4, 2, np.nan],
                [1, 2, 2]]
        assert_array_equal(rankdata(data, axis=0, nan_policy='propagate'),
                           [[1, np.nan, np.nan],
                            [3, np.nan, np.nan],
                            [2, np.nan, np.nan]])
        assert_array_equal(rankdata(data, axis=1, nan_policy='propagate'),
                           [[np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan],
                            [1, 2.5, 2.5]])


_cases = (
    # values, method, expected
    ([], 'average', []),
    ([], 'min', []),
    ([], 'max', []),
    ([], 'dense', []),
    ([], 'ordinal', []),
    #
    ([100], 'average', [1.0]),
    ([100], 'min', [1.0]),
    ([100], 'max', [1.0]),
    ([100], 'dense', [1.0]),
    ([100], 'ordinal', [1.0]),
    #
    ([100, 100, 100], 'average', [2.0, 2.0, 2.0]),
    ([100, 100, 100], 'min', [1.0, 1.0, 1.0]),
    ([100, 100, 100], 'max', [3.0, 3.0, 3.0]),
    ([100, 100, 100], 'dense', [1.0, 1.0, 1.0]),
    ([100, 100, 100], 'ordinal', [1.0, 2.0, 3.0]),
    #
    ([100, 300, 200], 'average', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'min', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'max', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'dense', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'ordinal', [1.0, 3.0, 2.0]),
    #
    ([100, 200, 300, 200], 'average', [1.0, 2.5, 4.0, 2.5]),
    ([100, 200, 300, 200], 'min', [1.0, 2.0, 4.0, 2.0]),
    ([100, 200, 300, 200], 'max', [1.0, 3.0, 4.0, 3.0]),
    ([100, 200, 300, 200], 'dense', [1.0, 2.0, 3.0, 2.0]),
    ([100, 200, 300, 200], 'ordinal', [1.0, 2.0, 4.0, 3.0]),
    #
    ([100, 200, 300, 200, 100], 'average', [1.5, 3.5, 5.0, 3.5, 1.5]),
    ([100, 200, 300, 200, 100], 'min', [1.0, 3.0, 5.0, 3.0, 1.0]),
    ([100, 200, 300, 200, 100], 'max', [2.0, 4.0, 5.0, 4.0, 2.0]),
    ([100, 200, 300, 200, 100], 'dense', [1.0, 2.0, 3.0, 2.0, 1.0]),
    ([100, 200, 300, 200, 100], 'ordinal', [1.0, 3.0, 5.0, 4.0, 2.0]),
    #
    ([10] * 30, 'ordinal', np.arange(1.0, 31.0)),
)


def test_cases():
    for values, method, expected in _cases:
        r = rankdata(values, method=method)
        assert_array_equal(r, expected)


# <!-- @GENESIS_MODULE_END: test_rank -->
