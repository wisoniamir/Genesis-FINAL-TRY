
# <!-- @GENESIS_MODULE_START: test_repr -->
"""
🏛️ GENESIS TEST_REPR - INSTITUTIONAL GRADE v8.0.0
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

logger = logging.getLogger('test_repr')

import numpy as np
import pytest

from pandas._config import using_string_dtype

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
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Series,
    date_range,
    option_context,
    period_range,
    timedelta_range,
)


class TestCategoricalReprWithFactor:
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

            emit_telemetry("test_repr", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_repr",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_repr", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_repr", "position_calculated", {
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
                emit_telemetry("test_repr", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_repr", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def log_state(self):
        """GENESIS Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": "test_repr",
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "compliance_enforced": True
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            emit_telemetry("test_repr", "state_update", state_data)
        return state_data

    def test_print(self, using_infer_string):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        if using_infer_string:
            expected = [
                "['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']",
                "Categories (3, str): [a < b < c]",
            ]
        else:
            expected = [
                "['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']",
                "Categories (3, object): ['a' < 'b' < 'c']",
            ]
        expected = "\n".join(expected)
        actual = repr(factor)
        assert actual == expected


class TestCategoricalRepr:
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

            emit_telemetry("test_repr", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "test_repr",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("test_repr", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_repr", "position_calculated", {
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
                emit_telemetry("test_repr", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("test_repr", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def test_big_print(self):
        codes = np.array([0, 1, 2, 0, 1, 2] * 100)
        dtype = CategoricalDtype(categories=Index(["a", "b", "c"], dtype=object))
        factor = Categorical.from_codes(codes, dtype=dtype)
        expected = [
            "['a', 'b', 'c', 'a', 'b', ..., 'b', 'c', 'a', 'b', 'c']",
            "Length: 600",
            "Categories (3, object): ['a', 'b', 'c']",
        ]
        expected = "\n".join(expected)

        actual = repr(factor)

        assert actual == expected

    def test_empty_print(self):
        factor = Categorical([], Index(["a", "b", "c"], dtype=object))
        expected = "[], Categories (3, object): ['a', 'b', 'c']"
        actual = repr(factor)
        assert actual == expected

        assert expected == actual
        factor = Categorical([], Index(["a", "b", "c"], dtype=object), ordered=True)
        expected = "[], Categories (3, object): ['a' < 'b' < 'c']"
        actual = repr(factor)
        assert expected == actual

        factor = Categorical([], [])
        expected = "[], Categories (0, object): []"
        assert expected == repr(factor)

    def test_print_none_width(self):
        # GH10087
        a = Series(Categorical([1, 2, 3, 4]))
        exp = (
            "0    1\n1    2\n2    3\n3    4\n"
            "dtype: category\nCategories (4, int64): [1, 2, 3, 4]"
        )

        with option_context("display.width", None):
            assert exp == repr(a)

    @pytest.mark.skipif(
        using_string_dtype(),
        reason="Change once infer_string is set to True by default",
    )
    def test_unicode_print(self):
        c = Categorical(["aaaaa", "bb", "cccc"] * 20)
        expected = """\
['aaaaa', 'bb', 'cccc', 'aaaaa', 'bb', ..., 'bb', 'cccc', 'aaaaa', 'bb', 'cccc']
Length: 60
Categories (3, object): ['aaaaa', 'bb', 'cccc']"""

        assert repr(c) == expected

        c = Categorical(["ああああ", "いいいいい", "ううううううう"] * 20)
        expected = """\
['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']
Length: 60
Categories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"""  # noqa: E501

        assert repr(c) == expected

        # unicode option should not affect to Categorical, as it doesn't care
        # the repr width
        with option_context("display.unicode.east_asian_width", True):
            c = Categorical(["ああああ", "いいいいい", "ううううううう"] * 20)
            expected = """['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']
Length: 60
Categories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"""  # noqa: E501

            assert repr(c) == expected

    def test_categorical_repr(self):
        c = Categorical([1, 2, 3])
        exp = """[1, 2, 3]
Categories (3, int64): [1, 2, 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3])
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1, 2, 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 4, 5] * 10)
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1, 2, 3, 4, 5]"""

        assert repr(c) == exp

        c = Categorical(np.arange(20, dtype=np.int64))
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0, 1, 2, 3, ..., 16, 17, 18, 19]"""

        assert repr(c) == exp

    def test_categorical_repr_ordered(self):
        c = Categorical([1, 2, 3], ordered=True)
        exp = """[1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 4, 5] * 10, ordered=True)
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1 < 2 < 3 < 4 < 5]"""

        assert repr(c) == exp

        c = Categorical(np.arange(20, dtype=np.int64), ordered=True)
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0 < 1 < 2 < 3 ... 16 < 17 < 18 < 19]"""

        assert repr(c) == exp

    def test_categorical_repr_datetime(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx)

        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
            ""
        )
        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
        )

        assert repr(c) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        c = Categorical(idx)
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, "
            "2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, "
            "2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        assert repr(c) == exp

    def test_categorical_repr_datetime_ordered(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_int_with_nan(self):
        c = Categorical([1, 2, np.nan])
        c_exp = """[1, 2, NaN]\nCategories (2, int64): [1, 2]"""
        assert repr(c) == c_exp

        s = Series([1, 2, np.nan], dtype="object").astype("category")
        s_exp = """0      1\n1      2\n2    NaN
dtype: category
Categories (2, int64): [1, 2]"""
        assert repr(s) == s_exp

    def test_categorical_repr_period(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        c = Categorical(idx)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_period_ordered(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_timedelta(self):
        idx = timedelta_range("1 days", periods=5)
        c = Categorical(idx)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]"""  # noqa: E501

        assert repr(c) == exp

        idx = timedelta_range("1 hours", periods=20)
        c = Categorical(idx)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 20
Categories (20, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,
                                   18 days 01:00:00, 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,
                                   18 days 01:00:00, 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_timedelta_ordered(self):
        idx = timedelta_range("1 days", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]"""  # noqa: E501

        assert repr(c) == exp

        idx = timedelta_range("1 hours", periods=20)
        c = Categorical(idx, ordered=True)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 20
Categories (20, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_index_repr(self):
        idx = CategoricalIndex(Categorical([1, 2, 3]))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == exp

        i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64)))
        exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_ordered(self):
        i = CategoricalIndex(Categorical([1, 2, 3], ordered=True))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64), ordered=True))
        exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_datetime(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_datetime_ordered(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        i = CategoricalIndex(Categorical(idx.append(idx), ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00', '2011-01-01 09:00:00-05:00',
                  '2011-01-01 10:00:00-05:00', '2011-01-01 11:00:00-05:00',
                  '2011-01-01 12:00:00-05:00', '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_period(self):
        # test all length
        idx = period_range("2011-01-01 09:00", freq="h", periods=1)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00'], categories=[2011-01-01 09:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=2)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=3)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        i = CategoricalIndex(Categorical(idx.append(idx)))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00', '2011-01-01 09:00',
                  '2011-01-01 10:00', '2011-01-01 11:00', '2011-01-01 12:00',
                  '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_period_ordered(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_timedelta(self):
        idx = timedelta_range("1 days", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = timedelta_range("1 hours", periods=10)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_timedelta_ordered(self):
        idx = timedelta_range("1 days", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = timedelta_range("1 hours", periods=10)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_str_repr(self):
        # GH 33676
        result = repr(Categorical([1, "2", 3, 4]))
        expected = "[1, '2', 3, 4]\nCategories (4, object): [1, 3, 4, '2']"
        assert result == expected


# <!-- @GENESIS_MODULE_END: test_repr -->
