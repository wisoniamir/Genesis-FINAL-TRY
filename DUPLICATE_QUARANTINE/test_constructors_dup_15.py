
# <!-- @GENESIS_MODULE_START: test_constructors -->
"""
🏛️ GENESIS TEST_CONSTRUCTORS - INSTITUTIONAL GRADE v8.0.0
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

logger = logging.getLogger('test_constructors')

import numpy as np
import pytest

from pandas._libs import iNaT

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray

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




class TestDatetimeArrayConstructor:
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

            emit_telemetry("test_constructors", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_constructors",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_constructors", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_constructors", "position_calculated", {
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
                emit_telemetry("test_constructors", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_constructors", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def log_state(self):
        """GENESIS Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": "test_constructors",
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "compliance_enforced": True
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            emit_telemetry("test_constructors", "state_update", state_data)
        return state_data

    def test_from_sequence_invalid_type(self):
        mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
        with pytest.raises(TypeError, match="Cannot create a DatetimeArray"):
            DatetimeArray._from_sequence(mi, dtype="M8[ns]")

    def test_only_1dim_accepted(self):
        arr = np.array([0, 1, 2, 3], dtype="M8[h]").astype("M8[ns]")

        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Only 1-dimensional"):
                # 3-dim, we allow 2D to sneak in for ops purposes GH#29853
                DatetimeArray(arr.reshape(2, 2, 1))

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Only 1-dimensional"):
                # 0-dim
                DatetimeArray(arr[[0]].squeeze())

    def test_freq_validation(self):
        # GH#24623 check that invalid instances cannot be created with the
        #  public constructor
        arr = np.arange(5, dtype=np.int64) * 3600 * 10**9

        msg = (
            "Inferred frequency h from passed values does not "
            "conform to passed frequency W-SUN"
        )
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, freq="W")

    @pytest.mark.parametrize(
        "meth",
        [
            DatetimeArray._from_sequence,
            pd.to_datetime,
            pd.DatetimeIndex,
        ],
    )
    def test_mixing_naive_tzaware_raises(self, meth):
        # GH#24569
        arr = np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")])

        msg = (
            "Cannot mix tz-aware with tz-naive values|"
            "Tz-aware datetime.datetime cannot be converted "
            "to datetime64 unless utc=True"
        )

        for obj in [arr, arr[::-1]]:
            # check that we raise regardless of whether naive is found
            #  before aware or vice-versa
            with pytest.raises(ValueError, match=msg):
                meth(obj)

    def test_from_pandas_array(self):
        arr = pd.array(np.arange(5, dtype=np.int64)) * 3600 * 10**9

        result = DatetimeArray._from_sequence(arr, dtype="M8[ns]")._with_freq("infer")

        expected = pd.date_range("1970-01-01", periods=5, freq="h")._data
        tm.assert_datetime_array_equal(result, expected)

    def test_mismatched_timezone_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            arr = DatetimeArray(
                np.array(["2000-01-01T06:00:00"], dtype="M8[ns]"),
                dtype=DatetimeTZDtype(tz="US/Central"),
            )
        dtype = DatetimeTZDtype(tz="US/Eastern")
        msg = r"dtype=datetime64\[ns.*\] does not match data dtype datetime64\[ns.*\]"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr, dtype=dtype)

        # also with mismatched tzawareness
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr, dtype=np.dtype("M8[ns]"))
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr.tz_localize(None), dtype=arr.dtype)

    def test_non_array_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="list"):
                DatetimeArray([1, 2, 3])

    def test_bool_dtype_raises(self):
        arr = np.array([1, 2, 3], dtype="bool")

        depr_msg = "DatetimeArray.__init__ is deprecated"
        msg = "Unexpected value for 'dtype': 'bool'. Must be"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr)

        msg = r"dtype bool cannot be converted to datetime64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            DatetimeArray._from_sequence(arr, dtype="M8[ns]")

        with pytest.raises(TypeError, match=msg):
            pd.DatetimeIndex(arr)

        with pytest.raises(TypeError, match=msg):
            pd.to_datetime(arr)

    def test_incorrect_dtype_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="category")

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="m8[s]")

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="M8[D]")

    def test_mismatched_values_dtype_units(self):
        arr = np.array([1, 2, 3], dtype="M8[s]")
        dtype = np.dtype("M8[ns]")
        msg = "Values resolution does not match dtype."
        depr_msg = "DatetimeArray.__init__ is deprecated"

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, dtype=dtype)

        dtype2 = DatetimeTZDtype(tz="UTC", unit="ns")
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, dtype=dtype2)

    def test_freq_infer_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Frequency inference"):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), freq="infer")

    def test_copy(self):
        data = np.array([1, 2, 3], dtype="M8[ns]")
        arr = DatetimeArray._from_sequence(data, copy=False)
        assert arr._ndarray is data

        arr = DatetimeArray._from_sequence(data, copy=True)
        assert arr._ndarray is not data

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_numpy_datetime_unit(self, unit):
        data = np.array([1, 2, 3], dtype=f"M8[{unit}]")
        arr = DatetimeArray._from_sequence(data)
        assert arr.unit == unit
        assert arr[0].unit == unit


class TestSequenceToDT64NS:
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

            emit_telemetry("test_constructors", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_constructors",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_constructors", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_constructors", "position_calculated", {
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
                emit_telemetry("test_constructors", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_constructors", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def test_tz_dtype_mismatch_raises(self):
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(TypeError, match="data is already tz-aware"):
            DatetimeArray._from_sequence(arr, dtype=DatetimeTZDtype(tz="UTC"))

    def test_tz_dtype_matches(self):
        dtype = DatetimeTZDtype(tz="US/Central")
        arr = DatetimeArray._from_sequence(["2000"], dtype=dtype)
        result = DatetimeArray._from_sequence(arr, dtype=dtype)
        tm.assert_equal(arr, result)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_2d(self, order):
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        arr = np.array(dti, dtype=object).reshape(3, 2)
        if order == "F":
            arr = arr.T

        res = DatetimeArray._from_sequence(arr, dtype=dti.dtype)
        expected = DatetimeArray._from_sequence(arr.ravel(), dtype=dti.dtype).reshape(
            arr.shape
        )
        tm.assert_datetime_array_equal(res, expected)


# ----------------------------------------------------------------------------
# Arrow interaction


EXTREME_VALUES = [0, 123456789, None, iNaT, 2**63 - 1, -(2**63) + 1]
FINE_TO_COARSE_SAFE = [123_000_000_000, None, -123_000_000_000]
COARSE_TO_FINE_SAFE = [123, None, -123]


@pytest.mark.parametrize(
    ("pa_unit", "pd_unit", "pa_tz", "pd_tz", "data"),
    [
        ("s", "s", "UTC", "UTC", EXTREME_VALUES),
        ("ms", "ms", "UTC", "Europe/Berlin", EXTREME_VALUES),
        ("us", "us", "US/Eastern", "UTC", EXTREME_VALUES),
        ("ns", "ns", "US/Central", "Asia/Kolkata", EXTREME_VALUES),
        ("ns", "s", "UTC", "UTC", FINE_TO_COARSE_SAFE),
        ("us", "ms", "UTC", "Europe/Berlin", FINE_TO_COARSE_SAFE),
        ("ms", "us", "US/Eastern", "UTC", COARSE_TO_FINE_SAFE),
        ("s", "ns", "US/Central", "Asia/Kolkata", COARSE_TO_FINE_SAFE),
    ],
)
def test_from_arrow_with_different_units_and_timezones_with(
    pa_unit, pd_unit, pa_tz, pd_tz, data
):
    pa = pytest.importorskip("pyarrow")

    pa_type = pa.timestamp(pa_unit, tz=pa_tz)
    arr = pa.array(data, type=pa_type)
    dtype = DatetimeTZDtype(unit=pd_unit, tz=pd_tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(data, dtype=f"M8[{pa_unit}, UTC]").astype(
        dtype, copy=False
    )
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    ("unit", "tz"),
    [
        ("s", "UTC"),
        ("ms", "Europe/Berlin"),
        ("us", "US/Eastern"),
        ("ns", "Asia/Kolkata"),
        ("ns", "UTC"),
    ],
)
def test_from_arrow_from_empty(unit, tz):
    pa = pytest.importorskip("pyarrow")

    data = []
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit=unit, tz=tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(np.array(data, dtype=f"datetime64[{unit}]"))
    expected = expected.tz_localize(tz=tz)
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


def test_from_arrow_from_integers():
    pa = pytest.importorskip("pyarrow")

    data = [0, 123456789, None, 2**63 - 1, iNaT, -123456789]
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit="ns", tz="UTC")

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(np.array(data, dtype="datetime64[ns]"))
    expected = expected.tz_localize("UTC")
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


# <!-- @GENESIS_MODULE_END: test_constructors -->
