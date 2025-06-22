import logging
# <!-- @GENESIS_MODULE_START: test_stat_reductions -->
"""
🏛️ GENESIS TEST_STAT_REDUCTIONS - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_stat_reductions", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_stat_reductions", "position_calculated", {
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
                            "module": "test_stat_reductions",
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
                    print(f"Emergency stop error in test_stat_reductions: {e}")
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
                    "module": "test_stat_reductions",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_stat_reductions", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_stat_reductions: {e}")
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


"""
Tests for statistical reductions of 2nd moment or higher: var, skew, kurt, ...
"""
import inspect

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDatetimeLikeStatReductions:
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

            emit_telemetry("test_stat_reductions", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_stat_reductions", "position_calculated", {
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
                        "module": "test_stat_reductions",
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
                print(f"Emergency stop error in test_stat_reductions: {e}")
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
                "module": "test_stat_reductions",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_stat_reductions", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_stat_reductions: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_stat_reductions",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_stat_reductions: {e}")
    @pytest.mark.parametrize("box", [Series, pd.Index, pd.array])
    def test_dt64_mean(self, tz_naive_fixture, box):
        tz = tz_naive_fixture

        dti = date_range("2001-01-01", periods=11, tz=tz)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        dtarr = dti._data

        obj = box(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06", tz=tz)
        assert obj.mean(skipna=False) == pd.Timestamp("2001-01-06", tz=tz)

        # dtarr[-2] will be the first date 2001-01-1
        dtarr[-2] = pd.NaT

        obj = box(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06 07:12:00", tz=tz)
        assert obj.mean(skipna=False) is pd.NaT

    @pytest.mark.parametrize("box", [Series, pd.Index, pd.array])
    @pytest.mark.parametrize("freq", ["s", "h", "D", "W", "B"])
    def test_period_mean(self, box, freq):
        # GH#24757
        dti = date_range("2001-01-01", periods=11)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])

        warn = FutureWarning if freq == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            parr = dti._data.to_period(freq)
        obj = box(parr)
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

        # parr[-2] will be the first date 2001-01-1
        parr[-2] = pd.NaT

        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

    @pytest.mark.parametrize("box", [Series, pd.Index, pd.array])
    def test_td64_mean(self, box):
        m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], "m8[D]")
        tdi = pd.TimedeltaIndex(m8values).as_unit("ns")

        tdarr = tdi._data
        obj = box(tdarr, copy=False)

        result = obj.mean()
        expected = np.array(tdarr).mean()
        assert result == expected

        tdarr[0] = pd.NaT
        assert obj.mean(skipna=False) is pd.NaT

        result2 = obj.mean(skipna=True)
        assert result2 == tdi[1:].mean()

        # exact equality fails by 1 nanosecond
        assert result2.round("us") == (result * 11.0 / 10).round("us")


class TestSeriesStatReductions:
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

            emit_telemetry("test_stat_reductions", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_stat_reductions", "position_calculated", {
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
                        "module": "test_stat_reductions",
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
                print(f"Emergency stop error in test_stat_reductions: {e}")
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
                "module": "test_stat_reductions",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_stat_reductions", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_stat_reductions: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_stat_reductions",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_stat_reductions: {e}")
    # Note: the name TestSeriesStatReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def _check_stat_op(
        self, name, alternate, string_series_, check_objects=False, check_allna=False
    ):
        with pd.option_context("use_bottleneck", False):
            f = getattr(Series, name)

            # add some NaNs
            string_series_[5:15] = np.nan

            # mean, idxmax, idxmin, min, and max are valid for dates
            if name not in ["max", "min", "mean", "median", "std"]:
                ds = Series(date_range("1/1/2001", periods=10))
                msg = f"does not support reduction '{name}'"
                with pytest.raises(TypeError, match=msg):
                    f(ds)

            # skipna or no
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))

            # check the result is correct
            nona = string_series_.dropna()
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))

            allna = string_series_ * np.nan

            if check_allna:
                assert np.isnan(f(allna))

            # dtype=object with None, it works!
            s = Series([1, 2, 3, None, 5])
            f(s)

            # GH#2888
            items = [0]
            items.extend(range(2**40, 2**40 + 1000))
            s = Series(items, dtype="int64")
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))

            # check date range
            if check_objects:
                s = Series(pd.bdate_range("1/1/2000", periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp

            # check on string data
            if name not in ["sum", "min", "max"]:
                with pytest.raises(TypeError, match=None):
                    f(Series(list("abc")))

            # Invalid axis.
            msg = "No axis named 1 for object type Series"
            with pytest.raises(ValueError, match=msg):
                f(string_series_, axis=1)

            if "numeric_only" in inspect.getfullargspec(f).args:
                # only the index is string; dtype is float
                f(string_series_, numeric_only=True)

    def test_sum(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("sum", np.sum, string_series, check_allna=False)

    def test_mean(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("mean", np.mean, string_series)

    def test_median(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("median", np.median, string_series)

        # test with integers, test failure
        int_ts = Series(np.ones(10, dtype=int), index=range(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())

    def test_prod(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("prod", np.prod, string_series)

    def test_min(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("min", np.min, string_series, check_objects=True)

    def test_max(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("max", np.max, string_series, check_objects=True)

    def test_var_std(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1)
        self._check_stat_op("std", alt, string_series)

        alt = lambda x: np.var(x, ddof=1)
        self._check_stat_op("var", alt, string_series)

        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        assert pd.isna(result)

        result = s.std(ddof=1)
        assert pd.isna(result)

    def test_sem(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        self._check_stat_op("sem", alt, string_series)

        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values, ddof=4) / np.sqrt(
            len(datetime_series.values)
        )
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        assert pd.isna(result)

    def test_skew(self):
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.skew(x, bias=False)
        self._check_stat_op("skew", alt, string_series)

        # test corner cases, skew() returns NaN unless there's at least 3
        # values
        min_N = 3
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.skew())
                assert np.isnan(df.skew()).all()
            else:
                assert 0 == s.skew()
                assert isinstance(s.skew(), np.float64)  # GH53482
                assert (df.skew() == 0).all()

    def test_kurt(self):
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.kurtosis(x, bias=False)
        self._check_stat_op("kurt", alt, string_series)

    def test_kurt_corner(self):
        # test corner cases, kurt() returns NaN unless there's at least 4
        # values
        min_N = 4
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.kurt())
                assert np.isnan(df.kurt()).all()
            else:
                assert 0 == s.kurt()
                assert isinstance(s.kurt(), np.float64)  # GH53482
                assert (df.kurt() == 0).all()


# <!-- @GENESIS_MODULE_END: test_stat_reductions -->
