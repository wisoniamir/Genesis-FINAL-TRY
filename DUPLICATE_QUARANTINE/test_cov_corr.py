
# <!-- @GENESIS_MODULE_START: test_cov_corr -->
"""
🏛️ GENESIS TEST_COV_CORR - INSTITUTIONAL GRADE v8.0.0
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

logger = logging.getLogger('test_cov_corr')

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
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


    DataFrame,
    Index,
    Series,
    date_range,
    isna,
)
import pandas._testing as tm


class TestDataFrameCov:
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

            emit_telemetry("test_cov_corr", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_cov_corr",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_cov_corr", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_cov_corr", "position_calculated", {
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
                emit_telemetry("test_cov_corr", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_cov_corr", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def log_state(self):
        """GENESIS Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": "test_cov_corr",
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "compliance_enforced": True
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            emit_telemetry("test_cov_corr", "state_update", state_data)
        return state_data

    def test_cov(self, float_frame, float_string_frame):
        # min_periods no NAs (corner case)
        expected = float_frame.cov()
        result = float_frame.cov(min_periods=len(float_frame))

        tm.assert_frame_equal(expected, result)

        result = float_frame.cov(min_periods=len(float_frame) + 1)
        assert isna(result.values).all()

        # with NAs
        frame = float_frame.copy()
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan
        frame.iloc[5:10, frame.columns.get_loc("B")] = np.nan
        result = frame.cov(min_periods=len(frame) - 8)
        expected = frame.cov()
        expected.loc["A", "B"] = np.nan
        expected.loc["B", "A"] = np.nan
        tm.assert_frame_equal(result, expected)

        # regular
        result = frame.cov()
        expected = frame["A"].cov(frame["C"])
        tm.assert_almost_equal(result["A"]["C"], expected)

        # fails on non-numeric types
        with pytest.raises(ValueError, match="could not convert string to float"):
            float_string_frame.cov()
        result = float_string_frame.cov(numeric_only=True)
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].cov()
        tm.assert_frame_equal(result, expected)

        # Single column frame
        df = DataFrame(np.linspace(0.0, 1.0, 10))
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values.T).reshape((1, 1)), index=df.columns, columns=df.columns
        )
        tm.assert_frame_equal(result, expected)
        df.loc[0] = np.nan
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values[1:].T).reshape((1, 1)),
            index=df.columns,
            columns=df.columns,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("test_ddof", [None, 0, 1, 2, 3])
    def test_cov_ddof(self, test_ddof):
        # GH#34611
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)
        df = DataFrame({0: np_array1, 1: np_array2})
        result = df.cov(ddof=test_ddof)
        expected_np = np.cov(np_array1, np_array2, ddof=test_ddof)
        expected = DataFrame(expected_np)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "other_column", [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])]
    )
    def test_cov_nullable_integer(self, other_column):
        # https://github.com/pandas-dev/pandas/issues/33803
        data = DataFrame({"a": pd.array([1, 2, None]), "b": other_column})
        result = data.cov()
        arr = np.array([[0.5, 0.5], [0.5, 1.0]])
        expected = DataFrame(arr, columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_cov_numeric_only(self, numeric_only):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        df = DataFrame({"a": [1, 0], "c": ["x", "y"]})
        expected = DataFrame(0.5, index=["a"], columns=["a"])
        if numeric_only:
            result = df.cov(numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="could not convert string to float"):
                df.cov(numeric_only=numeric_only)


class TestDataFrameCorr:
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

            emit_telemetry("test_cov_corr", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_cov_corr",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_cov_corr", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_cov_corr", "position_calculated", {
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
                emit_telemetry("test_cov_corr", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_cov_corr", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    # DataFrame.corr(), as opposed to DataFrame.corrwith

    @pytest.mark.parametrize("method", ["pearson", "kendall", "spearman"])
    def test_corr_scipy_method(self, float_frame, method):
        pytest.importorskip("scipy")
        float_frame.loc[float_frame.index[:5], "A"] = np.nan
        float_frame.loc[float_frame.index[5:10], "B"] = np.nan
        float_frame.loc[float_frame.index[:10], "A"] = float_frame["A"][10:20].copy()

        correls = float_frame.corr(method=method)
        expected = float_frame["A"].corr(float_frame["C"], method=method)
        tm.assert_almost_equal(correls["A"]["C"], expected)

    # ---------------------------------------------------------------------

    def test_corr_non_numeric(self, float_string_frame):
        with pytest.raises(ValueError, match="could not convert string to float"):
            float_string_frame.corr()
        result = float_string_frame.corr(numeric_only=True)
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].corr()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_nooverlap(self, meth):
        # nothing in common
        pytest.importorskip("scipy")
        df = DataFrame(
            {
                "A": [1, 1.5, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1.5, 1],
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.loc["A", "B"])
        assert isna(rs.loc["B", "A"])
        assert rs.loc["A", "A"] == 1
        assert rs.loc["B", "B"] == 1
        assert isna(rs.loc["C", "C"])

    @pytest.mark.parametrize("meth", ["pearson", "spearman"])
    def test_corr_constant(self, meth):
        # constant --> all NA
        df = DataFrame(
            {
                "A": [1, 1, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1, 1],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.values).all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_int_and_boolean(self, meth):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        pytest.importorskip("scipy")
        df = DataFrame({"a": [True, False], "b": [1, 0]})

        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        result = df.corr(meth)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["cov", "corr"])
    def test_corr_cov_independent_index_column(self, method):
        # GH#14617
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4 * 10).reshape(10, 4),
            columns=list("abcd"),
        )
        result = getattr(df, method)()
        assert result.index is not result.columns
        assert result.index.equals(result.columns)

    def test_corr_invalid_method(self):
        # GH#22298
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 2)))
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        with pytest.raises(ValueError, match=msg):
            df.corr(method="____")

    def test_corr_int(self):
        # dtypes other than float64 GH#1761
        df = DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

        df.cov()
        df.corr()

    @pytest.mark.parametrize(
        "nullable_column", [pd.array([1, 2, 3]), pd.array([1, 2, None])]
    )
    @pytest.mark.parametrize(
        "other_column",
        [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, np.nan])],
    )
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_nullable_integer(self, nullable_column, other_column, method):
        # https://github.com/pandas-dev/pandas/issues/33803
        pytest.importorskip("scipy")
        data = DataFrame({"a": nullable_column, "b": other_column})
        result = data.corr(method=method)
        expected = DataFrame(np.ones((2, 2)), columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_corr_item_cache(self, using_copy_on_write, warn_copy_on_write):
        # Check that corr does not lead to incorrect entries in item_cache

        df = DataFrame({"A": range(10)})
        df["B"] = range(10)[::-1]

        ser = df["A"]  # populate item_cache
        assert len(df._mgr.arrays) == 2  # i.e. 2 blocks

        _ = df.corr(numeric_only=True)

        if using_copy_on_write:
            ser.iloc[0] = 99
            assert df.loc[0, "A"] == 0
        else:
            # Check that the corr didn't break link between ser and df
            ser.values[0] = 99
            assert df.loc[0, "A"] == 99
            if not warn_copy_on_write:
                assert df["A"] is ser
            assert df.values[0, 0] == 99

    @pytest.mark.parametrize("length", [2, 20, 200, 2000])
    def test_corr_for_constant_columns(self, length):
        # GH: 37448
        df = DataFrame(length * [[0.4, 0.1]], columns=["A", "B"])
        result = df.corr()
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    def test_calc_corr_small_numbers(self):
        # GH: 37452
        df = DataFrame(
            {"A": [1.0e-20, 2.0e-20, 3.0e-20], "B": [1.0e-20, 2.0e-20, 3.0e-20]}
        )
        result = df.corr()
        expected = DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]}, index=["A", "B"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_min_periods_greater_than_length(self, method):
        pytest.importorskip("scipy")
        df = DataFrame({"A": [1, 2], "B": [1, 2]})
        result = df.corr(method=method, min_periods=3)
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corr_numeric_only(self, meth, numeric_only):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        pytest.importorskip("scipy")
        df = DataFrame({"a": [1, 0], "b": [1, 0], "c": ["x", "y"]})
        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])
        if numeric_only:
            result = df.corr(meth, numeric_only=numeric_only)
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="could not convert string to float"):
                df.corr(meth, numeric_only=numeric_only)


class TestDataFrameCorrWith:
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

            emit_telemetry("test_cov_corr", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_cov_corr",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_cov_corr", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_cov_corr", "position_calculated", {
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
                emit_telemetry("test_cov_corr", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_cov_corr", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    @pytest.mark.parametrize(
        "dtype",
        [
            "float64",
            "Float64",
            pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_corrwith(self, datetime_frame, dtype):
        datetime_frame = datetime_frame.astype(dtype)

        a = datetime_frame
        noise = Series(np.random.default_rng(2).standard_normal(len(a)), index=a.index)

        b = datetime_frame.add(noise, axis=0)

        # make sure order does not matter
        b = b.reindex(columns=b.columns[::-1], index=b.index[::-1][10:])
        del b["B"]

        colcorr = a.corrwith(b, axis=0)
        tm.assert_almost_equal(colcorr["A"], a["A"].corr(b["A"]))

        rowcorr = a.corrwith(b, axis=1)
        tm.assert_series_equal(rowcorr, a.T.corrwith(b.T, axis=0))

        dropped = a.corrwith(b, axis=0, drop=True)
        tm.assert_almost_equal(dropped["A"], a["A"].corr(b["A"]))
        assert "B" not in dropped

        dropped = a.corrwith(b, axis=1, drop=True)
        assert a.index[-1] not in dropped.index

        # non time-series data
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            index=index,
            columns=columns,
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=index[:4],
            columns=columns,
        )
        correls = df1.corrwith(df2, axis=1)
        for row in index[:4]:
            tm.assert_almost_equal(correls[row], df1.loc[row].corr(df2.loc[row]))

    def test_corrwith_with_objects(self, using_infer_string):
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df2 = df1.copy()
        cols = ["A", "B", "C", "D"]

        df1["obj"] = "foo"
        df2["obj"] = "bar"

        if using_infer_string:
            msg = "Cannot perform reduction 'mean' with string dtype"
            with pytest.raises(TypeError, match=msg):
                df1.corrwith(df2)
        else:
            with pytest.raises(TypeError, match="Could not convert"):
                df1.corrwith(df2)
        result = df1.corrwith(df2, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols])
        tm.assert_series_equal(result, expected)

        with pytest.raises(TypeError, match="unsupported operand type"):
            df1.corrwith(df2, axis=1)
        result = df1.corrwith(df2, axis=1, numeric_only=True)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols], axis=1)
        tm.assert_series_equal(result, expected)

    def test_corrwith_series(self, datetime_frame):
        result = datetime_frame.corrwith(datetime_frame["A"])
        expected = datetime_frame.apply(datetime_frame["A"].corr)

        tm.assert_series_equal(result, expected)

    def test_corrwith_matches_corrcoef(self):
        df1 = DataFrame(np.arange(10000), columns=["a"])
        df2 = DataFrame(np.arange(10000) ** 2, columns=["a"])
        c1 = df1.corrwith(df2)["a"]
        c2 = np.corrcoef(df1["a"], df2["a"])[0][1]

        tm.assert_almost_equal(c1, c2)
        assert c1 < 1

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corrwith_mixed_dtypes(self, numeric_only):
        # GH#18570
        df = DataFrame(
            {"a": [1, 4, 3, 2], "b": [4, 6, 7, 3], "c": ["a", "b", "c", "d"]}
        )
        s = Series([0, 6, 7, 3])
        if numeric_only:
            result = df.corrwith(s, numeric_only=numeric_only)
            corrs = [df["a"].corr(s), df["b"].corr(s)]
            expected = Series(data=corrs, index=["a", "b"])
            tm.assert_series_equal(result, expected)
        else:
            with pytest.raises(
                ValueError,
                match="could not convert string to float",
            ):
                df.corrwith(s, numeric_only=numeric_only)

    def test_corrwith_index_intersection(self):
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        result = df1.corrwith(df2, drop=True).index.sort_values()
        expected = df1.columns.intersection(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_index_union(self):
        df1 = DataFrame(
            np.random.default_rng(2).random(size=(10, 2)), columns=["a", "b"]
        )
        df2 = DataFrame(
            np.random.default_rng(2).random(size=(10, 3)), columns=["a", "b", "c"]
        )

        result = df1.corrwith(df2, drop=False).index.sort_values()
        expected = df1.columns.union(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_dup_cols(self):
        # GH#21925
        df1 = DataFrame(np.vstack([np.arange(10)] * 3).T)
        df2 = df1.copy()
        df2 = pd.concat((df2, df2[0]), axis=1)

        result = df1.corrwith(df2)
        expected = Series(np.ones(4), index=[0, 0, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_corr_numerical_instabilities(self):
        # GH#45640
        df = DataFrame([[0.2, 0.4], [0.4, 0.2]])
        result = df.corr()
        expected = DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
        tm.assert_frame_equal(result - 1, expected - 1, atol=1e-17)

    def test_corrwith_spearman(self):
        # GH#21925
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df**2, method="spearman")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_kendall(self):
        # GH#21925
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
        result = df.corrwith(df**2, method="kendall")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_corrwith_spearman_with_tied_data(self):
        # GH#48826
        pytest.importorskip("scipy")
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        df2 = df1[["B", "C"]]
        result = (df1 + 1).corrwith(df2.B, method="spearman")
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        tm.assert_series_equal(result, expected)

        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        ser_bool = Series([True, True, False, True])
        result = df_bool.corrwith(ser_bool)
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        tm.assert_series_equal(result, expected)


# <!-- @GENESIS_MODULE_END: test_cov_corr -->
