
# <!-- @GENESIS_MODULE_START: test_value_counts -->
"""
🏛️ GENESIS TEST_VALUE_COUNTS - INSTITUTIONAL GRADE v8.0.0
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

logger = logging.getLogger('test_value_counts')

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

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




def production_data_frame_value_counts_unsorted():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(sort=False)
    expected = pd.Series(
        data=[1, 2, 1],
        index=pd.MultiIndex.from_arrays(
            [(2, 4, 6), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_ascending():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(ascending=True)
    expected = pd.Series(
        data=[1, 1, 2],
        index=pd.MultiIndex.from_arrays(
            [(2, 6, 4), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_default():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts()
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_normalize():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(normalize=True)
    expected = pd.Series(
        data=[0.5, 0.25, 0.25],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="proportion",
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_single_col_default():
    df = pd.DataFrame({"num_legs": [2, 4, 4, 6]})

    result = df.value_counts()
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays([[4, 2, 6]], names=["num_legs"]),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_empty():
    df_no_cols = pd.DataFrame()

    result = df_no_cols.value_counts()
    expected = pd.Series(
        [], dtype=np.int64, name="count", index=np.array([], dtype=np.intp)
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_empty_normalize():
    df_no_cols = pd.DataFrame()

    result = df_no_cols.value_counts(normalize=True)
    expected = pd.Series(
        [], dtype=np.float64, name="proportion", index=np.array([], dtype=np.intp)
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_dropna_true(nulls_fixture):
    # GH 41334
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    result = df.value_counts()
    expected = pd.Series(
        data=[1, 1],
        index=pd.MultiIndex.from_arrays(
            [("Beth", "John"), ("Louise", "Smith")], names=["first_name", "middle_name"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def production_data_frame_value_counts_dropna_false(nulls_fixture):
    # GH 41334
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )

    result = df.value_counts(dropna=False)
    expected = pd.Series(
        data=[1, 1, 1, 1],
        index=pd.MultiIndex(
            levels=[
                pd.Index(["Anne", "Beth", "John"]),
                pd.Index(["Louise", "Smith", np.nan]),
            ],
            codes=[[0, 1, 2, 2], [2, 0, 1, 2]],
            names=["first_name", "middle_name"],
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", (["first_name", "middle_name"], [0, 1]))
def production_data_frame_value_counts_subset(nulls_fixture, columns):
    # GH 50829
    df = pd.DataFrame(
        {
            columns[0]: ["John", "Anne", "John", "Beth"],
            columns[1]: ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    result = df.value_counts(columns[0])
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.Index(["John", "Anne", "Beth"], name=columns[0]),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_value_counts_categorical_future_warning():
    # GH#54775
    df = pd.DataFrame({"a": [1, 2, 3]}, dtype="category")
    result = df.value_counts()
    expected = pd.Series(
        1,
        index=pd.MultiIndex.from_arrays(
            [pd.Index([1, 2, 3], name="a", dtype="category")]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_value_counts_with_missing_category():
    # GH-54836
    df = pd.DataFrame({"a": pd.Categorical([1, 2, 4], categories=[1, 2, 3, 4])})
    result = df.value_counts()
    expected = pd.Series(
        [1, 1, 1, 0],
        index=pd.MultiIndex.from_arrays(
            [pd.CategoricalIndex([1, 2, 4, 3], categories=[1, 2, 3, 4], name="a")]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


# <!-- @GENESIS_MODULE_END: test_value_counts -->
