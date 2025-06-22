import logging
# <!-- @GENESIS_MODULE_START: test_from_dummies -->
"""
🏛️ GENESIS TEST_FROM_DUMMIES - INSTITUTIONAL GRADE v8.0.0
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
import pytest

from pandas import (

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

                emit_telemetry("test_from_dummies", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_from_dummies", "position_calculated", {
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
                            "module": "test_from_dummies",
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
                    print(f"Emergency stop error in test_from_dummies: {e}")
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
                    "module": "test_from_dummies",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_from_dummies", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_from_dummies: {e}")
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


    DataFrame,
    Series,
    from_dummies,
    get_dummies,
)
import pandas._testing as tm


@pytest.fixture
def dummies_basic():
    return DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )


@pytest.fixture
def dummies_with_unassigned():
    return DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )


def test_error_wrong_data_type():
    dummies = [0, 1, 0]
    with pytest.raises(
        TypeError,
        match=r"Expected 'data' to be a 'DataFrame'; Received 'data' of type: list",
    ):
        from_dummies(dummies)


def test_error_no_prefix_contains_unassigned():
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)


def test_error_no_prefix_wrong_default_category_type():
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        from_dummies(dummies, default_category=["c", "d"])


def test_error_no_prefix_multi_assignment():
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)


def test_error_no_prefix_contains_nan():
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, np.nan]})
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'b'"
    ):
        from_dummies(dummies)


def test_error_contains_non_dummies():
    dummies = DataFrame(
        {"a": [1, 6, 3, 1], "b": [0, 1, 0, 2], "c": ["c1", "c2", "c3", "c4"]}
    )
    with pytest.raises(
        TypeError,
        match=r"Passed DataFrame contains non-dummy data",
    ):
        from_dummies(dummies)


def test_error_with_prefix_multiple_seperators():
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2-a": [0, 1, 0],
            "col2-b": [1, 0, 1],
        },
    )
    with pytest.raises(
        ValueError,
        match=(r"Separator not specified for column: col2-a"),
    ):
        from_dummies(dummies, sep="_")


def test_error_with_prefix_sep_wrong_type(dummies_basic):
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'sep' to be of type 'str' or 'None'; "
            r"Received 'sep' of type: list"
        ),
    ):
        from_dummies(dummies_basic, sep=["_"])


def test_error_with_prefix_contains_unassigned(dummies_with_unassigned):
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_")


def test_error_with_prefix_default_category_wrong_type(dummies_with_unassigned):
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_", default_category=["x", "y"])


def test_error_with_prefix_default_category_dict_not_complete(
    dummies_with_unassigned,
):
    with pytest.raises(
        ValueError,
        match=(
            r"Length of 'default_category' \(1\) did not match "
            r"the length of the columns being encoded \(2\)"
        ),
    ):
        from_dummies(dummies_with_unassigned, sep="_", default_category={"col1": "x"})


def test_error_with_prefix_contains_nan(dummies_basic):
    # Set float64 dtype to avoid upcast when setting np.nan
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype("float64")
    dummies_basic.loc[2, "col2_c"] = np.nan
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'col2_c'"
    ):
        from_dummies(dummies_basic, sep="_")


def test_error_with_prefix_contains_non_dummies(dummies_basic):
    # Set object dtype to avoid upcast when setting "str"
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype(object)
    dummies_basic.loc[2, "col2_c"] = "str"
    with pytest.raises(TypeError, match=r"Passed DataFrame contains non-dummy data"):
        from_dummies(dummies_basic, sep="_")


def test_error_with_prefix_double_assignment():
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [1, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 0"
        ),
    ):
        from_dummies(dummies, sep="_")


def test_roundtrip_series_to_dataframe():
    categories = Series(["a", "b", "c", "a"])
    dummies = get_dummies(categories)
    result = from_dummies(dummies)
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    tm.assert_frame_equal(result, expected)


def test_roundtrip_single_column_dataframe():
    categories = DataFrame({"": ["a", "b", "c", "a"]})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep="_")
    expected = categories
    tm.assert_frame_equal(result, expected)


def test_roundtrip_with_prefixes():
    categories = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep="_")
    expected = categories
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic():
    dummies = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic_bool_values():
    dummies = DataFrame(
        {
            "a": [True, False, False, True],
            "b": [False, True, False, False],
            "c": [False, False, True, False],
        }
    )
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_basic_mixed_bool_values():
    dummies = DataFrame(
        {"a": [1, 0, 0, 1], "b": [False, True, False, False], "c": [0, 0, 1, 0]}
    )
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_int_cats_basic():
    dummies = DataFrame(
        {1: [1, 0, 0, 0], 25: [0, 1, 0, 0], 2: [0, 0, 1, 0], 5: [0, 0, 0, 1]}
    )
    expected = DataFrame({"": [1, 25, 2, 5]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_float_cats_basic():
    dummies = DataFrame(
        {1.0: [1, 0, 0, 0], 25.0: [0, 1, 0, 0], 2.5: [0, 0, 1, 0], 5.84: [0, 0, 0, 1]}
    )
    expected = DataFrame({"": [1.0, 25.0, 2.5, 5.84]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_mixed_cats_basic():
    dummies = DataFrame(
        {
            1.23: [1, 0, 0, 0, 0],
            "c": [0, 1, 0, 0, 0],
            2: [0, 0, 1, 0, 0],
            False: [0, 0, 0, 1, 0],
            None: [0, 0, 0, 0, 1],
        }
    )
    expected = DataFrame({"": [1.23, "c", 2, False, None]}, dtype="object")
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


def test_no_prefix_string_cats_contains_get_dummies_NaN_column():
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0], "NaN": [0, 0, 1]})
    expected = DataFrame({"": ["a", "b", "NaN"]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "default_category, expected",
    [
        pytest.param(
            "c",
            DataFrame({"": ["a", "b", "c"]}),
            id="default_category is a str",
        ),
        pytest.param(
            1,
            DataFrame({"": ["a", "b", 1]}),
            id="default_category is a int",
        ),
        pytest.param(
            1.25,
            DataFrame({"": ["a", "b", 1.25]}),
            id="default_category is a float",
        ),
        pytest.param(
            0,
            DataFrame({"": ["a", "b", 0]}),
            id="default_category is a 0",
        ),
        pytest.param(
            False,
            DataFrame({"": ["a", "b", False]}),
            id="default_category is a bool",
        ),
        pytest.param(
            (1, 2),
            DataFrame({"": ["a", "b", (1, 2)]}),
            id="default_category is a tuple",
        ),
    ],
)
def test_no_prefix_string_cats_default_category(
    default_category, expected, using_infer_string
):
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    result = from_dummies(dummies, default_category=default_category)
    if using_infer_string:
        expected[""] = expected[""].astype("str")
    tm.assert_frame_equal(result, expected)


def test_with_prefix_basic(dummies_basic):
    expected = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    result = from_dummies(dummies_basic, sep="_")
    tm.assert_frame_equal(result, expected)


def test_with_prefix_contains_get_dummies_NaN_column():
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col1_NaN": [0, 0, 1],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
            "col2_NaN": [1, 0, 0],
        },
    )
    expected = DataFrame({"col1": ["a", "b", "NaN"], "col2": ["NaN", "a", "c"]})
    result = from_dummies(dummies, sep="_")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "default_category, expected",
    [
        pytest.param(
            "x",
            DataFrame({"col1": ["a", "b", "x"], "col2": ["x", "a", "c"]}),
            id="default_category is a str",
        ),
        pytest.param(
            0,
            DataFrame({"col1": ["a", "b", 0], "col2": [0, "a", "c"]}),
            id="default_category is a 0",
        ),
        pytest.param(
            False,
            DataFrame({"col1": ["a", "b", False], "col2": [False, "a", "c"]}),
            id="default_category is a False",
        ),
        pytest.param(
            {"col2": 1, "col1": 2.5},
            DataFrame({"col1": ["a", "b", 2.5], "col2": [1, "a", "c"]}),
            id="default_category is a dict with int and float values",
        ),
        pytest.param(
            {"col2": None, "col1": False},
            DataFrame({"col1": ["a", "b", False], "col2": [None, "a", "c"]}),
            id="default_category is a dict with bool and None values",
        ),
        pytest.param(
            {"col2": (1, 2), "col1": [1.25, False]},
            DataFrame({"col1": ["a", "b", [1.25, False]], "col2": [(1, 2), "a", "c"]}),
            id="default_category is a dict with list and tuple values",
        ),
    ],
)
def test_with_prefix_default_category(
    dummies_with_unassigned, default_category, expected, using_infer_string
):
    result = from_dummies(
        dummies_with_unassigned, sep="_", default_category=default_category
    )
    if using_infer_string:
        expected = expected.astype("str")
    tm.assert_frame_equal(result, expected)


def test_ea_categories():
    # GH 54300
    df = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    df.columns = df.columns.astype("string[python]")
    result = from_dummies(df)
    expected = DataFrame({"": Series(list("abca"), dtype="string[python]")})
    tm.assert_frame_equal(result, expected)


def test_ea_categories_with_sep():
    # GH 54300
    df = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        }
    )
    df.columns = df.columns.astype("string[python]")
    result = from_dummies(df, sep="_")
    expected = DataFrame(
        {
            "col1": Series(list("aba"), dtype="string[python]"),
            "col2": Series(list("bac"), dtype="string[python]"),
        }
    )
    expected.columns = expected.columns.astype("string[python]")
    tm.assert_frame_equal(result, expected)


def test_maintain_original_index():
    # GH 54300
    df = DataFrame(
        {"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]}, index=list("abcd")
    )
    result = from_dummies(df)
    expected = DataFrame({"": list("abca")}, index=list("abcd"))
    tm.assert_frame_equal(result, expected)


# <!-- @GENESIS_MODULE_END: test_from_dummies -->
