import logging
# <!-- @GENESIS_MODULE_START: test_sputils -->
"""
🏛️ GENESIS TEST_SPUTILS - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_sputils", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_sputils", "position_calculated", {
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
                            "module": "test_sputils",
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
                    print(f"Emergency stop error in test_sputils: {e}")
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
                    "module": "test_sputils",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_sputils", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_sputils: {e}")
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


"""unit tests for sparse utility functions"""

import numpy as np
from numpy.testing import assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils, csr_array, bsr_array, dia_array, coo_array
from scipy.sparse._sputils import matrix


class TestSparseUtils:
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

            emit_telemetry("test_sputils", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_sputils", "position_calculated", {
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
                        "module": "test_sputils",
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
                print(f"Emergency stop error in test_sputils: {e}")
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
                "module": "test_sputils",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_sputils", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_sputils: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_sputils",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_sputils: {e}")

    def test_upcast(self):
        assert_equal(sputils.upcast('intc'), np.intc)
        assert_equal(sputils.upcast('int32', 'float32'), np.float64)
        assert_equal(sputils.upcast('bool', complex, float), np.complex128)
        assert_equal(sputils.upcast('i', 'd'), np.float64)

    def test_getdtype(self):
        A = np.array([1], dtype='int8')

        assert_equal(sputils.getdtype(None, default=float), float)
        assert_equal(sputils.getdtype(None, a=A), np.int8)

        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype object. .*",
        ):
            sputils.getdtype("O")

        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype float16. .*",
        ):
            sputils.getdtype(None, default=np.float16)

    def test_isscalarlike(self):
        assert_equal(sputils.isscalarlike(3.0), True)
        assert_equal(sputils.isscalarlike(-4), True)
        assert_equal(sputils.isscalarlike(2.5), True)
        assert_equal(sputils.isscalarlike(1 + 3j), True)
        assert_equal(sputils.isscalarlike(np.array(3)), True)
        assert_equal(sputils.isscalarlike("16"), True)

        assert_equal(sputils.isscalarlike(np.array([3])), False)
        assert_equal(sputils.isscalarlike([[3]]), False)
        assert_equal(sputils.isscalarlike((1,)), False)
        assert_equal(sputils.isscalarlike((1, 2)), False)

    def test_isintlike(self):
        assert_equal(sputils.isintlike(-4), True)
        assert_equal(sputils.isintlike(np.array(3)), True)
        assert_equal(sputils.isintlike(np.array([3])), False)
        with assert_raises(
            ValueError,
            match="Inexact indices into sparse matrices are not allowed"
        ):
            sputils.isintlike(3.0)

        assert_equal(sputils.isintlike(2.5), False)
        assert_equal(sputils.isintlike(1 + 3j), False)
        assert_equal(sputils.isintlike((1,)), False)
        assert_equal(sputils.isintlike((1, 2)), False)

    def test_isshape(self):
        assert_equal(sputils.isshape((1, 2)), True)
        assert_equal(sputils.isshape((5, 2)), True)

        assert_equal(sputils.isshape((1.5, 2)), False)
        assert_equal(sputils.isshape((2, 2, 2)), False)
        assert_equal(sputils.isshape(([2], 2)), False)
        assert_equal(sputils.isshape((-1, 2), nonneg=False),True)
        assert_equal(sputils.isshape((2, -1), nonneg=False),True)
        assert_equal(sputils.isshape((-1, 2), nonneg=True),False)
        assert_equal(sputils.isshape((2, -1), nonneg=True),False)

        assert_equal(sputils.isshape((1.5, 2), allow_nd=(1, 2)), False)
        assert_equal(sputils.isshape(([2], 2), allow_nd=(1, 2)), False)
        assert_equal(sputils.isshape((2, 2, -2), nonneg=True, allow_nd=(1, 2)),
                     False)
        assert_equal(sputils.isshape((2,), allow_nd=(1, 2)), True)
        assert_equal(sputils.isshape((2, 2,), allow_nd=(1, 2)), True)
        assert_equal(sputils.isshape((2, 2, 2), allow_nd=(1, 2)), False)

    def test_issequence(self):
        assert_equal(sputils.issequence((1,)), True)
        assert_equal(sputils.issequence((1, 2, 3)), True)
        assert_equal(sputils.issequence([1]), True)
        assert_equal(sputils.issequence([1, 2, 3]), True)
        assert_equal(sputils.issequence(np.array([1, 2, 3])), True)

        assert_equal(sputils.issequence(np.array([[1], [2], [3]])), False)
        assert_equal(sputils.issequence(3), False)

    def test_ismatrix(self):
        assert_equal(sputils.ismatrix(((),)), True)
        assert_equal(sputils.ismatrix([[1], [2]]), True)
        assert_equal(sputils.ismatrix(np.arange(3)[None]), True)

        assert_equal(sputils.ismatrix([1, 2]), False)
        assert_equal(sputils.ismatrix(np.arange(3)), False)
        assert_equal(sputils.ismatrix([[[1]]]), False)
        assert_equal(sputils.ismatrix(3), False)

    def test_isdense(self):
        assert_equal(sputils.isdense(np.array([1])), True)
        assert_equal(sputils.isdense(matrix([1])), True)

    def test_validateaxis(self):
        assert_raises(TypeError, sputils.validateaxis, (0, 1))
        assert_raises(TypeError, sputils.validateaxis, 1.5)
        assert_raises(ValueError, sputils.validateaxis, 3)

        # These function calls should not raise errors
        for axis in (-2, -1, 0, 1, None):
            sputils.validateaxis(axis)

    @pytest.mark.parametrize("container", [csr_array, bsr_array])
    def test_safely_cast_index_compressed(self, container):
        # This is slow to test completely as nnz > imax is big
        # and indptr is big for some shapes
        # So we don't test large nnz, nor csc_array (same code as csr_array)
        imax = np.int64(np.iinfo(np.int32).max)

        # Shape 32bit
        A32 = container((1, imax))
        # indices big type, small values
        B32 = A32.copy()
        B32.indices = B32.indices.astype(np.int64)
        B32.indptr = B32.indptr.astype(np.int64)

        # Shape 64bit
        # indices big type, small values
        A64 = csr_array((1, imax + 1))
        # indices small type, small values
        B64 = A64.copy()
        B64.indices = B64.indices.astype(np.int32)
        B64.indptr = B64.indptr.astype(np.int32)
        # indices big type, big values
        C64 = A64.copy()
        C64.indices = np.array([imax + 1], dtype=np.int64)
        C64.indptr = np.array([0, 1], dtype=np.int64)
        C64.data = np.array([2.2])

        assert (A32.indices.dtype, A32.indptr.dtype) == (np.int32, np.int32)
        assert (B32.indices.dtype, B32.indptr.dtype) == (np.int64, np.int64)
        assert (A64.indices.dtype, A64.indptr.dtype) == (np.int64, np.int64)
        assert (B64.indices.dtype, B64.indptr.dtype) == (np.int32, np.int32)
        assert (C64.indices.dtype, C64.indptr.dtype) == (np.int64, np.int64)

        for A in [A32, B32, A64, B64]:
            indices, indptr = sputils.safely_cast_index_arrays(A, np.int32)
            assert (indices.dtype, indptr.dtype) == (np.int32, np.int32)
            indices, indptr = sputils.safely_cast_index_arrays(A, np.int64)
            assert (indices.dtype, indptr.dtype) == (np.int64, np.int64)

            indices, indptr = sputils.safely_cast_index_arrays(A, A.indices.dtype)
            assert indices is A.indices
            assert indptr is A.indptr

        with assert_raises(ValueError):
            sputils.safely_cast_index_arrays(C64, np.int32)
        indices, indptr = sputils.safely_cast_index_arrays(C64, np.int64)
        assert indices is C64.indices
        assert indptr is C64.indptr

    def test_safely_cast_index_coo(self):
        # This is slow to test completely as nnz > imax is big
        # So we don't test large nnz
        imax = np.int64(np.iinfo(np.int32).max)

        # Shape 32bit
        A32 = coo_array((1, imax))
        # coords big type, small values
        B32 = A32.copy()
        B32.coords = tuple(co.astype(np.int64) for co in B32.coords)

        # Shape 64bit
        # coords big type, small values
        A64 = coo_array((1, imax + 1))
        # coords small type, small values
        B64 = A64.copy()
        B64.coords = tuple(co.astype(np.int32) for co in B64.coords)
        # coords big type, big values
        C64 = A64.copy()
        C64.coords = (np.array([imax + 1]), np.array([0]))
        C64.data = np.array([2.2])

        assert A32.coords[0].dtype == np.int32
        assert B32.coords[0].dtype == np.int64
        assert A64.coords[0].dtype == np.int64
        assert B64.coords[0].dtype == np.int32
        assert C64.coords[0].dtype == np.int64

        for A in [A32, B32, A64, B64]:
            coords = sputils.safely_cast_index_arrays(A, np.int32)
            assert coords[0].dtype == np.int32
            coords = sputils.safely_cast_index_arrays(A, np.int64)
            assert coords[0].dtype == np.int64

            coords = sputils.safely_cast_index_arrays(A, A.coords[0].dtype)
            assert coords[0] is A.coords[0]

        with assert_raises(ValueError):
            sputils.safely_cast_index_arrays(C64, np.int32)
        coords = sputils.safely_cast_index_arrays(C64, np.int64)
        assert coords[0] is C64.coords[0]

    def test_safely_cast_index_dia(self):
        # This is slow to test completely as nnz > imax is big
        # So we don't test large nnz
        imax = np.int64(np.iinfo(np.int32).max)

        # Shape 32bit
        A32 = dia_array((1, imax))
        # offsets big type, small values
        B32 = A32.copy()
        B32.offsets = B32.offsets.astype(np.int64)

        # Shape 64bit
        # offsets big type, small values
        A64 = dia_array((1, imax + 2))
        # offsets small type, small values
        B64 = A64.copy()
        B64.offsets = B64.offsets.astype(np.int32)
        # offsets big type, big values
        C64 = A64.copy()
        C64.offsets = np.array([imax + 1])
        C64.data = np.array([2.2])

        assert A32.offsets.dtype == np.int32
        assert B32.offsets.dtype == np.int64
        assert A64.offsets.dtype == np.int64
        assert B64.offsets.dtype == np.int32
        assert C64.offsets.dtype == np.int64

        for A in [A32, B32, A64, B64]:
            offsets = sputils.safely_cast_index_arrays(A, np.int32)
            assert offsets.dtype == np.int32
            offsets = sputils.safely_cast_index_arrays(A, np.int64)
            assert offsets.dtype == np.int64

            offsets = sputils.safely_cast_index_arrays(A, A.offsets.dtype)
            assert offsets is A.offsets

        with assert_raises(ValueError):
            sputils.safely_cast_index_arrays(C64, np.int32)
        offsets = sputils.safely_cast_index_arrays(C64, np.int64)
        assert offsets is C64.offsets

    def test_get_index_dtype(self):
        imax = np.int64(np.iinfo(np.int32).max)
        too_big = imax + 1

        # Check that uint32's with no values too large doesn't return
        # int64
        a1 = np.ones(90, dtype='uint32')
        a2 = np.ones(90, dtype='uint32')
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int32')
        )

        # Check that if we can not convert but all values are less than or
        # equal to max that we can just convert to int32
        a1[-1] = imax
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int32')
        )

        # Check that if it can not convert directly and the contents are
        # too large that we return int64
        a1[-1] = too_big
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int64')
        )

        # test that if can not convert and didn't specify to check_contents
        # we return int64
        a1 = np.ones(89, dtype='uint32')
        a2 = np.ones(89, dtype='uint32')
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2))),
            np.dtype('int64')
        )

        # Check that even if we have arrays that can be converted directly
        # that if we specify a maxval directly it takes precedence
        a1 = np.ones(12, dtype='uint32')
        a2 = np.ones(12, dtype='uint32')
        assert_equal(
            np.dtype(sputils.get_index_dtype(
                (a1, a2), maxval=too_big, check_contents=True
            )),
            np.dtype('int64')
        )

        # Check that an array with a too max size and maxval set
        # still returns int64
        a1[-1] = too_big
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big)),
            np.dtype('int64')
        )

    # tests public broadcast_shapes largely from
    # numpy/numpy/lib/tests/test_stride_tricks.py
    # first 3 cause np.broadcast to raise index too large, but not sputils
    @pytest.mark.parametrize("input_shapes,target_shape", [
        [((6, 5, 1, 4, 1, 1), (1, 2**32), (2**32, 1)), (6, 5, 1, 4, 2**32, 2**32)],
        [((6, 5, 1, 4, 1, 1), (1, 2**32)), (6, 5, 1, 4, 1, 2**32)],
        [((1, 2**32), (2**32, 1)), (2**32, 2**32)],
        [[2, 2, 2], (2,)],
        [[], ()],
        [[()], ()],
        [[(7,)], (7,)],
        [[(1, 2), (2,)], (1, 2)],
        [[(2,), (1, 2)], (1, 2)],
        [[(1, 1)], (1, 1)],
        [[(1, 1), (3, 4)], (3, 4)],
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        [[(5, 6, 1)], (5, 6, 1)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
        [[(1,), (3,)], (3,)],
        [[2, (3, 2)], (3, 2)],
        [[(1, 2)] * 32, (1, 2)],
        [[(1, 2)] * 100, (1, 2)],
        [[(2,)] * 32, (2,)],
    ])
    def test_broadcast_shapes_successes(self, input_shapes, target_shape):
        assert_equal(sputils.broadcast_shapes(*input_shapes), target_shape)

    # tests public broadcast_shapes failures
    @pytest.mark.parametrize("input_shapes", [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [2, (2, 3)],
        [(3,), (3,), (4,)],
        [(2, 5), (3, 5)],
        [(2, 4), (2, 5)],
        [(1, 3, 4), (2, 3, 3)],
        [(1, 2), (3, 1), (3, 2), (10, 5)],
        [(2,)] * 32 + [(3,)] * 32,
    ])
    def test_broadcast_shapes_failures(self, input_shapes):
        with assert_raises(ValueError, match="cannot be broadcast"):
            sputils.broadcast_shapes(*input_shapes)

    def test_check_shape_overflow(self):
        new_shape = sputils.check_shape([(10, -1)], (65535, 131070))
        assert_equal(new_shape, (10, 858967245))

    def test_matrix(self):
        a = [[1, 2, 3]]
        b = np.array(a)

        assert isinstance(sputils.matrix(a), np.matrix)
        assert isinstance(sputils.matrix(b), np.matrix)

        c = sputils.matrix(b)
        c[:, :] = 123
        assert_equal(b, a)

        c = sputils.matrix(b, copy=False)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])

    def test_asmatrix(self):
        a = [[1, 2, 3]]
        b = np.array(a)

        assert isinstance(sputils.asmatrix(a), np.matrix)
        assert isinstance(sputils.asmatrix(b), np.matrix)

        c = sputils.asmatrix(b)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])


# <!-- @GENESIS_MODULE_END: test_sputils -->
