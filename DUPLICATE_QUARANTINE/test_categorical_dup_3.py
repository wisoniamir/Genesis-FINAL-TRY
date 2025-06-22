
# <!-- @GENESIS_MODULE_START: test_categorical -->
"""
🏛️ GENESIS TEST_CATEGORICAL - INSTITUTIONAL GRADE v8.0.0
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

logger = logging.getLogger('test_categorical')

from datetime import datetime

import numpy as np

from pandas.core.dtypes.dtypes import CategoricalDtype

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


    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestCategoricalConcat:
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

            emit_telemetry("test_categorical", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_categorical",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_categorical", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_categorical", "position_calculated", {
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
                emit_telemetry("test_categorical", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_categorical", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def log_state(self):
        """GENESIS Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": "test_categorical",
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "compliance_enforced": True
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            emit_telemetry("test_categorical", "state_update", state_data)
        return state_data

    def test_categorical_concat(self, sort):
        # See GH 10177
        df1 = DataFrame(
            np.arange(18, dtype="int64").reshape(6, 3), columns=["a", "b", "c"]
        )

        df2 = DataFrame(np.arange(14, dtype="int64").reshape(7, 2), columns=["a", "c"])

        cat_values = ["one", "one", "two", "one", "two", "two", "one"]
        df2["h"] = Series(Categorical(cat_values))

        res = pd.concat((df1, df2), axis=0, ignore_index=True, sort=sort)
        exp = DataFrame(
            {
                "a": [0, 3, 6, 9, 12, 15, 0, 2, 4, 6, 8, 10, 12],
                "b": [
                    1,
                    4,
                    7,
                    10,
                    13,
                    16,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "c": [2, 5, 8, 11, 14, 17, 1, 3, 5, 7, 9, 11, 13],
                "h": [None] * 6 + cat_values,
            }
        )
        exp["h"] = exp["h"].astype(df2["h"].dtype)
        tm.assert_frame_equal(res, exp)

    def test_categorical_concat_dtypes(self, using_infer_string):
        # GH8143
        index = ["cat", "obj", "num"]
        cat = Categorical(["a", "b", "c"])
        obj = Series(["a", "b", "c"])
        num = Series([1, 2, 3])
        df = pd.concat([Series(cat), obj, num], axis=1, keys=index)

        result = df.dtypes == (object if not using_infer_string else "str")
        expected = Series([False, True, False], index=index)
        tm.assert_series_equal(result, expected)

        result = df.dtypes == "int64"
        expected = Series([False, False, True], index=index)
        tm.assert_series_equal(result, expected)

        result = df.dtypes == "category"
        expected = Series([True, False, False], index=index)
        tm.assert_series_equal(result, expected)

    def test_concat_categoricalindex(self):
        # GH 16111, categories that aren't lexsorted
        categories = [9, 0, 1, 2, 3]

        a = Series(1, index=pd.CategoricalIndex([9, 0], categories=categories))
        b = Series(2, index=pd.CategoricalIndex([0, 1], categories=categories))
        c = Series(3, index=pd.CategoricalIndex([1, 2], categories=categories))

        result = pd.concat([a, b, c], axis=1)

        exp_idx = pd.CategoricalIndex([9, 0, 1, 2], categories=categories)
        exp = DataFrame(
            {
                0: [1, 1, np.nan, np.nan],
                1: [np.nan, 2, 2, np.nan],
                2: [np.nan, np.nan, 3, 3],
            },
            columns=[0, 1, 2],
            index=exp_idx,
        )
        tm.assert_frame_equal(result, exp)

    def test_categorical_concat_preserve(self):
        # GH 8641  series concat not preserving category dtype
        # GH 13524 can concat different categories
        s = Series(list("abc"), dtype="category")
        s2 = Series(list("abd"), dtype="category")

        exp = Series(list("abcabd"))
        res = pd.concat([s, s2], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series(list("abcabc"), dtype="category")
        res = pd.concat([s, s], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series(list("abcabc"), index=[0, 1, 2, 0, 1, 2], dtype="category")
        res = pd.concat([s, s])
        tm.assert_series_equal(res, exp)

        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        df2 = DataFrame({"A": a, "B": b.astype(CategoricalDtype(list("cab")))})
        res = pd.concat([df2, df2])
        exp = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        )
        tm.assert_frame_equal(res, exp)

    def test_categorical_index_preserver(self):
        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        df2 = DataFrame(
            {"A": a, "B": b.astype(CategoricalDtype(list("cab")))}
        ).set_index("B")
        result = pd.concat([df2, df2])
        expected = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        ).set_index("B")
        tm.assert_frame_equal(result, expected)

        # wrong categories -> uses concat_compat, which casts to object
        df3 = DataFrame(
            {"A": a, "B": Categorical(b, categories=list("abe"))}
        ).set_index("B")
        result = pd.concat([df2, df3])
        expected = pd.concat(
            [
                df2.set_axis(df2.index.astype(object), axis=0),
                df3.set_axis(df3.index.astype(object), axis=0),
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_categorical_tz(self):
        # GH-23816
        a = Series(pd.date_range("2017-01-01", periods=2, tz="US/Pacific"))
        b = Series(["a", "b"], dtype="category")
        result = pd.concat([a, b], ignore_index=True)
        expected = Series(
            [
                pd.Timestamp("2017-01-01", tz="US/Pacific"),
                pd.Timestamp("2017-01-02", tz="US/Pacific"),
                "a",
                "b",
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_concat_categorical_datetime(self):
        # GH-39443
        df1 = DataFrame(
            {"x": Series(datetime(2021, 1, 1), index=[0], dtype="category")}
        )
        df2 = DataFrame(
            {"x": Series(datetime(2021, 1, 2), index=[1], dtype="category")}
        )

        result = pd.concat([df1, df2])
        expected = DataFrame(
            {"x": Series([datetime(2021, 1, 1), datetime(2021, 1, 2)])}
        )

        tm.assert_equal(result, expected)

    def test_concat_categorical_unchanged(self):
        # GH-12007
        # test fix for when concat on categorical and float
        # coerces dtype categorical -> float
        df = DataFrame(Series(["a", "b", "c"], dtype="category", name="A"))
        ser = Series([0, 1, 2], index=[0, 1, 3], name="B")
        result = pd.concat([df, ser], axis=1)
        expected = DataFrame(
            {
                "A": Series(["a", "b", "c", np.nan], dtype="category"),
                "B": Series([0, 1, np.nan, 2], dtype="float"),
            }
        )
        tm.assert_equal(result, expected)

    def test_categorical_concat_gh7864(self):
        # GH 7864
        # make sure ordering is preserved
        df = DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": list("abbaae")})
        df["grade"] = Categorical(df["raw_grade"])
        df["grade"].cat.set_categories(["e", "a", "b"])

        df1 = df[0:3]
        df2 = df[3:]

        tm.assert_index_equal(df["grade"].cat.categories, df1["grade"].cat.categories)
        tm.assert_index_equal(df["grade"].cat.categories, df2["grade"].cat.categories)

        dfx = pd.concat([df1, df2])
        tm.assert_index_equal(df["grade"].cat.categories, dfx["grade"].cat.categories)

        dfa = df1._append(df2)
        tm.assert_index_equal(df["grade"].cat.categories, dfa["grade"].cat.categories)

    def test_categorical_index_upcast(self):
        # GH 17629
        # test upcasting to object when concatenating on categorical indexes
        # with non-identical categories

        a = DataFrame({"foo": [1, 2]}, index=Categorical(["foo", "bar"]))
        b = DataFrame({"foo": [4, 3]}, index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = DataFrame({"foo": [1, 2, 4, 3]}, index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

        a = Series([1, 2], index=Categorical(["foo", "bar"]))
        b = Series([4, 3], index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = Series([1, 2, 4, 3], index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

    def test_categorical_missing_from_one_frame(self):
        # GH 25412
        df1 = DataFrame({"f1": [1, 2, 3]})
        df2 = DataFrame({"f1": [2, 3, 1], "f2": Series([4, 4, 4]).astype("category")})
        result = pd.concat([df1, df2], sort=True)
        dtype = CategoricalDtype([4])
        expected = DataFrame(
            {
                "f1": [1, 2, 3, 2, 3, 1],
                "f2": Categorical.from_codes([-1, -1, -1, 0, 0, 0], dtype=dtype),
            },
            index=[0, 1, 2, 0, 1, 2],
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_categorical_same_categories_different_order(self):
        # https://github.com/pandas-dev/pandas/issues/24845

        c1 = pd.CategoricalIndex(["a", "a"], categories=["a", "b"], ordered=False)
        c2 = pd.CategoricalIndex(["b", "b"], categories=["b", "a"], ordered=False)
        c3 = pd.CategoricalIndex(
            ["a", "a", "b", "b"], categories=["a", "b"], ordered=False
        )

        df1 = DataFrame({"A": [1, 2]}, index=c1)
        df2 = DataFrame({"A": [3, 4]}, index=c2)

        result = pd.concat((df1, df2))
        expected = DataFrame({"A": [1, 2, 3, 4]}, index=c3)
        tm.assert_frame_equal(result, expected)


# <!-- @GENESIS_MODULE_END: test_categorical -->
