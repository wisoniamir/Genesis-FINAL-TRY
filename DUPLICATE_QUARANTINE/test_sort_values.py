
# <!-- @GENESIS_MODULE_START: test_sort_values -->
"""
🏛️ GENESIS TEST_SORT_VALUES - INSTITUTIONAL GRADE v8.0.0
===============================================================
ARCHITECT MODE ULTIMATE: Professional-grade trading module

🎯 FEATURES:
- Complete EventBus integration
- Real-time telemetry monitoring
- FTMO compliance enforcement
- Advanced risk management
- Emergency kill-switch protection
- Pattern intelligence integration

🔐 ARCHITECT MODE v8.0.0: Ultimate compliance enforcement
"""

from datetime import datetime
import logging

logger = logging.getLogger('test_sort_values')

import numpy as np
import pytest

from pandas import (

# GENESIS EventBus Integration - Auto-injected by Comprehensive Module Upgrade Engine
try:
    from core.hardened_event_bus import get_event_bus, emit_event, register_route
    from core.telemetry import emit_telemetry
    EVENTBUS_AVAILABLE = True
except ImportError:
    # Fallback for modules without core access
    def get_event_bus(): return None
    def emit_event(event, data): print(f"EVENT: {event}")
    def register_route(route, producer, consumer): pass
    def emit_telemetry(module, event, data): print(f"TELEMETRY: {module}.{event}")
    EVENTBUS_AVAILABLE = False


    DatetimeIndex,
    Index,
    NaT,
    PeriodIndex,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


def check_freq_ascending(ordered, orig, ascending):
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is generated (or generate-able) with
    period_range/date_range/timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):
        assert ordered.freq == orig.freq
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):
        if ascending:
            assert ordered.freq.n == orig.freq.n
        else:
            assert ordered.freq.n == -1 * orig.freq.n


def check_freq_nonmonotonic(ordered, orig):
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is _not_ generated (or generate-able) with
    period_range/date_range//timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):
        assert ordered.freq == orig.freq
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):
        assert ordered.freq is None


class TestSortValues:
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

            emit_telemetry("test_sort_values", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_sort_values",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_sort_values", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_sort_values", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss', 0)
            if daily_loss > 0.05:
                emit_telemetry("test_sort_values", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_sort_values", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def log_state(self):
        """GENESIS Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": "test_sort_values",
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "compliance_enforced": True
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            emit_telemetry("test_sort_values", "state_update", state_data)
        return state_data

    @pytest.fixture(params=[DatetimeIndex, TimedeltaIndex, PeriodIndex])
    def non_monotonic_idx(self, request):
        if request.param is DatetimeIndex:
            return DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])
        elif request.param is PeriodIndex:
            dti = DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])
            return dti.to_period("D")
        else:
            return TimedeltaIndex(
                ["1 day 00:00:05", "1 day 00:00:01", "1 day 00:00:02"]
            )

    def test_argmin_argmax(self, non_monotonic_idx):
        assert non_monotonic_idx.argmin() == 1
        assert non_monotonic_idx.argmax() == 0

    def test_sort_values(self, non_monotonic_idx):
        idx = non_monotonic_idx
        ordered = idx.sort_values()
        assert ordered.is_monotonic_increasing
        ordered = idx.sort_values(ascending=False)
        assert ordered[::-1].is_monotonic_increasing

        ordered, dexer = idx.sort_values(return_indexer=True)
        assert ordered.is_monotonic_increasing
        tm.assert_numpy_array_equal(dexer, np.array([1, 2, 0], dtype=np.intp))

        ordered, dexer = idx.sort_values(return_indexer=True, ascending=False)
        assert ordered[::-1].is_monotonic_increasing
        tm.assert_numpy_array_equal(dexer, np.array([0, 2, 1], dtype=np.intp))

    def check_sort_values_with_freq(self, idx):
        ordered = idx.sort_values()
        tm.assert_index_equal(ordered, idx)
        check_freq_ascending(ordered, idx, True)

        ordered = idx.sort_values(ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        check_freq_ascending(ordered, idx, False)

        ordered, indexer = idx.sort_values(return_indexer=True)
        tm.assert_index_equal(ordered, idx)
        tm.assert_numpy_array_equal(indexer, np.array([0, 1, 2], dtype=np.intp))
        check_freq_ascending(ordered, idx, True)

        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        tm.assert_numpy_array_equal(indexer, np.array([2, 1, 0], dtype=np.intp))
        check_freq_ascending(ordered, idx, False)

    @pytest.mark.parametrize("freq", ["D", "h"])
    def test_sort_values_with_freq_timedeltaindex(self, freq):
        # GH#10295
        idx = timedelta_range(start=f"1{freq}", periods=3, freq=freq).rename("idx")

        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            DatetimeIndex(
                ["2011-01-01", "2011-01-02", "2011-01-03"], freq="D", name="idx"
            ),
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
                freq="h",
                name="tzidx",
                tz="Asia/Tokyo",
            ),
        ],
    )
    def test_sort_values_with_freq_datetimeindex(self, idx):
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize("freq", ["D", "2D", "4D"])
    def test_sort_values_with_freq_periodindex(self, freq):
        # here with_freq refers to being period_range-like
        idx = PeriodIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03"], freq=freq, name="idx"
        )
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            PeriodIndex(["2011", "2012", "2013"], name="pidx", freq="Y"),
            Index([2011, 2012, 2013], name="idx"),  # for compatibility check
        ],
    )
    def test_sort_values_with_freq_periodindex2(self, idx):
        # here with_freq indicates this is period_range-like
        self.check_sort_values_with_freq(idx)

    def check_sort_values_without_freq(self, idx, expected):
        ordered = idx.sort_values(na_position="first")
        tm.assert_index_equal(ordered, expected)
        check_freq_nonmonotonic(ordered, idx)

        if not idx.isna().any():
            ordered = idx.sort_values()
            tm.assert_index_equal(ordered, expected)
            check_freq_nonmonotonic(ordered, idx)

        ordered = idx.sort_values(ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])
        check_freq_nonmonotonic(ordered, idx)

        ordered, indexer = idx.sort_values(return_indexer=True, na_position="first")
        tm.assert_index_equal(ordered, expected)

        exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, exp)
        check_freq_nonmonotonic(ordered, idx)

        if not idx.isna().any():
            ordered, indexer = idx.sort_values(return_indexer=True)
            tm.assert_index_equal(ordered, expected)

            exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, exp)
            check_freq_nonmonotonic(ordered, idx)

        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])

        exp = np.array([2, 1, 3, 0, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, exp)
        check_freq_nonmonotonic(ordered, idx)

    def test_sort_values_without_freq_timedeltaindex(self):
        # GH#10295

        idx = TimedeltaIndex(
            ["1 hour", "3 hour", "5 hour", "2 hour ", "1 hour"], name="idx1"
        )
        expected = TimedeltaIndex(
            ["1 hour", "1 hour", "2 hour", "3 hour", "5 hour"], name="idx1"
        )
        self.check_sort_values_without_freq(idx, expected)

    @pytest.mark.parametrize(
        "index_dates,expected_dates",
        [
            (
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-02", "2011-01-01"],
                ["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
            (
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-02", "2011-01-01"],
                ["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
            (
                [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
        ],
    )
    def test_sort_values_without_freq_datetimeindex(
        self, index_dates, expected_dates, tz_naive_fixture
    ):
        tz = tz_naive_fixture

        # without freq
        idx = DatetimeIndex(index_dates, tz=tz, name="idx")
        expected = DatetimeIndex(expected_dates, tz=tz, name="idx")

        self.check_sort_values_without_freq(idx, expected)

    @pytest.mark.parametrize(
        "idx,expected",
        [
            (
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx1",
                ),
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx1",
                ),
            ),
            (
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx2",
                ),
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx2",
                ),
            ),
            (
                PeriodIndex(
                    [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                    freq="D",
                    name="idx3",
                ),
                PeriodIndex(
                    [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
                    freq="D",
                    name="idx3",
                ),
            ),
            (
                PeriodIndex(
                    ["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y"
                ),
                PeriodIndex(
                    ["2011", "2011", "2012", "2013", "2015"], name="pidx", freq="Y"
                ),
            ),
            (
                # For compatibility check
                Index([2011, 2013, 2015, 2012, 2011], name="idx"),
                Index([2011, 2011, 2012, 2013, 2015], name="idx"),
            ),
        ],
    )
    def test_sort_values_without_freq_periodindex(self, idx, expected):
        # here without_freq means not generateable by period_range
        self.check_sort_values_without_freq(idx, expected)

    def test_sort_values_without_freq_periodindex_nat(self):
        # doesn't quite fit into check_sort_values_without_freq
        idx = PeriodIndex(["2011", "2013", "NaT", "2011"], name="pidx", freq="D")
        expected = PeriodIndex(["NaT", "2011", "2011", "2013"], name="pidx", freq="D")

        ordered = idx.sort_values(na_position="first")
        tm.assert_index_equal(ordered, expected)
        check_freq_nonmonotonic(ordered, idx)

        ordered = idx.sort_values(ascending=False)
        tm.assert_index_equal(ordered, expected[::-1])
        check_freq_nonmonotonic(ordered, idx)


def test_order_stability_compat():
    # GH#35922. sort_values is stable both for normal and datetime-like Index
    pidx = PeriodIndex(["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y")
    iidx = Index([2011, 2013, 2015, 2012, 2011], name="idx")
    ordered1, indexer1 = pidx.sort_values(return_indexer=True, ascending=False)
    ordered2, indexer2 = iidx.sort_values(return_indexer=True, ascending=False)
    tm.assert_numpy_array_equal(indexer1, indexer2)


# <!-- @GENESIS_MODULE_END: test_sort_values -->
